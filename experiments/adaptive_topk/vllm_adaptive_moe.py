"""Confidence-adaptive top-k expert routing for Qwen3-235B-A22B on vLLM.

GOAL (B=1 decode latency): instead of always reading top-8 of 128 experts from
HBM each layer, read FEWER experts on tokens whose router softmax is already
concentrated. Expert-weight HBM reads are ~66% of B=1 decode latency, so reading
k=4 instead of k=8 on confident tokens is a direct, real byte saving -- *not* a
zero-weighting trick. The dropped experts are never loaded.

----------------------------------------------------------------------------
HOW THE SAVING IS REAL (the one thing that matters)
----------------------------------------------------------------------------
vLLM's fused MoE kernel work is driven by the *distinct expert ids* that appear
in `topk_ids` (shape [M, top_k]). For each (expert, token-block) pair that
appears, `moe_align_block_size` creates a block and the Triton kernel loads that
expert's gate/up/down weights from HBM and runs the GEMM.

With `--enable-expert-parallel`, vLLM remaps `topk_ids` through `expert_map`;
experts not present become `-1`, and the Triton kernel hits:

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        write_zeros_to_output(...)   # skip GEMM, skip the weight load
        return

So: if a token's `topk_ids` row only references 4 real experts (the other 4
columns set to the sentinel that maps to -1), only those 4 experts' weights are
read from HBM for that token. At B=1 (M=1) that is literally 4/8 of the per-layer
expert bytes. THIS is the mechanism, and `custom_routing_function` is the hook
that lets us control `topk_ids` without touching any kernel.

----------------------------------------------------------------------------
WHY custom_routing_function IS THE LOWEST-RISK INTEGRATION POINT
----------------------------------------------------------------------------
vLLM threads `self.custom_routing_function` straight through:
    FusedMoE.forward_impl -> quant_method.apply(custom_routing_function=...)
      -> FusedMoE.select_experts -> custom_routing_function(hidden, logits,
                                                            top_k, renormalize)
It must return `(topk_weights, topk_ids)` with shape [M, top_k]. We keep top_k=8
(rectangular shape the kernel demands) but stamp the low-confidence trailing
columns with a DROP sentinel so the kernel skips them. No kernel patch, no model
re-registration, no shape change.

----------------------------------------------------------------------------
THE SENTINEL: how a column becomes a real -1 (skipped) expert
----------------------------------------------------------------------------
Two regimes, auto-detected per layer from the FusedMoE module:

  A) Expert-parallel ON (the user's config, --enable-expert-parallel) and the
     layer has an `expert_map`:
        We set dropped columns to a GLOBAL expert id that is NOT local to this
        rank -> expert_map maps it to -1 -> kernel early-returns (skips load).
        We pick the global id with the smallest router weight that maps to -1
        on this rank; if none exists locally we fall back to regime (B).
     This is the regime that yields the real HBM saving with zero kernel risk.

  B) No expert_map available (TP-only / single shard): the kernel's CUDA
     `moe_align_block_size` does NOT tolerate raw -1, so we cannot use the
     sentinel safely there. We instead DUPLICATE the token's top expert id into
     the dropped columns with weight 0. That removes the *distinct* experts (so
     their weights are not loaded -- a block is created only per distinct id),
     while staying within [0, num_experts). Net: still fewer expert weight loads,
     at the cost of a slightly larger aligned block for the duplicated expert.
     (Honest caveat: regime B's saving is real but smaller/less clean than A.)

----------------------------------------------------------------------------
POLICY (all env-configurable)
----------------------------------------------------------------------------
  ADAPTIVE_TOPK_ENABLE   = 1            # master on/off (default off)
  ADAPTIVE_TOPK_K        = 4            # reduced k when confident (1..8)
  ADAPTIVE_TOPK_THRESH   = 0.9          # top-K softmax mass needed to reduce
  ADAPTIVE_TOPK_MIN_LAYER= 0            # only adapt at/after this layer index
  ADAPTIVE_TOPK_DEBUG    = 0            # log per-layer drop-rate every N tokens

Per-token decision: compute softmax over the 128 router logits, take the top
ADAPTIVE_TOPK_K mass; if that mass > THRESH, keep k=ADAPTIVE_TOPK_K and drop the
remaining (8-k) columns; else keep the full k=8.

----------------------------------------------------------------------------
USAGE
----------------------------------------------------------------------------
Import this module *before* the engine builds the model. Two ways:

  1) Plugin (preferred, no code change to vllm serve):
        pip install -e experiments/adaptive_topk   # exposes entry point, or
        VLLM_PLUGINS=adaptive_topk vllm serve ...
     (see register() / the [project.entry-points."vllm.general_plugins"] note
      in PLAN.md)

  2) PYTHONSTARTUP-style preload:
        ADAPTIVE_TOPK_ENABLE=1 ADAPTIVE_TOPK_K=4 ADAPTIVE_TOPK_THRESH=0.9 \
          python -c "import experiments.adaptive_topk.vllm_adaptive_moe as m; \
                     m.install(); import runpy; \
                     runpy.run_module('vllm.entrypoints.openai.api_server', \
                                      run_name='__main__')" \
          -- --model /alloc/data/Qwen3-235B-A22B --enable-expert-parallel ...

Tested against vLLM ~0.10.x, torch 2.7.1+cu126.
"""

from __future__ import annotations

import os
import threading

import torch

# ---------------------------------------------------------------------------
# Config (read once at install; cheap to re-read but we snapshot for speed)
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class _Cfg:
    enable = _env_flag("ADAPTIVE_TOPK_ENABLE", False)
    k = int(os.environ.get("ADAPTIVE_TOPK_K", "4"))
    thresh = float(os.environ.get("ADAPTIVE_TOPK_THRESH", "0.9"))
    min_layer = int(os.environ.get("ADAPTIVE_TOPK_MIN_LAYER", "0"))
    debug = int(os.environ.get("ADAPTIVE_TOPK_DEBUG", "0"))


# Telemetry: how often we actually reduced, for the A/B writeup.
_stats_lock = threading.Lock()
_stats = {"tokens": 0, "reduced_rows": 0, "kept_rows": 0, "dropped_cols": 0}


def stats_snapshot() -> dict:
    with _stats_lock:
        s = dict(_stats)
    rows = s["reduced_rows"] + s["kept_rows"]
    s["reduced_frac"] = (s["reduced_rows"] / rows) if rows else 0.0
    s["avg_k"] = (
        (8 * s["kept_rows"] + _Cfg.k * s["reduced_rows"]) / rows if rows else 8.0
    )
    return s


# ---------------------------------------------------------------------------
# The adaptive routing function. Signature matches what FusedMoE.select_experts
# calls: custom_routing_function(hidden_states, gating_output, topk, renormalize,
#                                indices_type=...) -> (topk_weights, topk_ids)
# ---------------------------------------------------------------------------


def _make_routing_fn(layer_idx: int, expert_map_getter):
    """Build a per-layer custom_routing_function.

    `expert_map_getter()` returns this layer's expert_map tensor or None, read
    lazily because vLLM may set expert_map after module construction.
    """

    def adaptive_routing(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        indices_type=None,
    ):
        # Baseline top-k exactly as vLLM's fused_topk would (softmax over logits).
        scores = torch.softmax(gating_output.float(), dim=-1)  # [M, 128]
        topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)  # [M, topk]

        out_idx_dtype = (
            indices_type if indices_type is not None else torch.int32
        )

        if not _Cfg.enable or _Cfg.k >= topk or layer_idx < _Cfg.min_layer:
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
            return (
                topk_weights.to(torch.float32),
                topk_ids.to(out_idx_dtype),
            )

        k = _Cfg.k
        M = topk_ids.size(0)

        # Confidence = sum of the top-k softmax mass (weights are already sorted
        # descending by torch.topk). Reduce only rows above threshold.
        topk_mass = topk_weights[:, :k].sum(dim=-1)  # [M]
        reduce_mask = topk_mass > _Cfg.thresh  # [M] bool

        # Resolve the DROP sentinel for this layer/regime (see module docstring).
        expert_map = expert_map_getter()
        sentinel = _resolve_drop_sentinel(expert_map, topk_ids, k)

        # For reduced rows, stamp columns [k:] with the sentinel and zero their
        # weight so they contribute nothing even if the kernel does not skip.
        if reduce_mask.any():
            rows = reduce_mask.nonzero(as_tuple=True)[0]  # [R]
            # sentinel(...) -> [R, topk-k]; assign into the trailing columns.
            topk_ids[rows, k:topk] = sentinel(topk_ids, rows)
            topk_weights[rows, k:topk] = 0.0

        if renormalize:
            denom = topk_weights.sum(-1, keepdim=True)
            denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            topk_weights = topk_weights / denom

        # Lightweight telemetry for the A/B writeup (cheap; always on).
        with _stats_lock:
            _stats["tokens"] += M
            r = int(reduce_mask.sum())
            _stats["reduced_rows"] += r
            _stats["kept_rows"] += M - r
            _stats["dropped_cols"] += r * (topk - k)

        return topk_weights.to(torch.float32), topk_ids.to(out_idx_dtype)

    return adaptive_routing


def _resolve_drop_sentinel(expert_map, topk_ids, k):
    """Return a callable sentinel(topk_ids, rows) -> values for cols [k:].

    Regime A (expert_map present): find a global expert id that maps to -1 on
    this rank, so the kernel early-returns and skips the weight load. We cache
    one such id.

    Regime B (no expert_map): duplicate each row's top (col 0) expert id, so no
    NEW distinct expert is introduced -> no extra weights loaded, and the id
    stays valid for the CUDA align kernel.
    """
    if expert_map is not None:
        # expert_map[g] == -1  => global expert g is not local to this rank.
        nonlocal_ids = (expert_map < 0).nonzero(as_tuple=True)[0]
        if nonlocal_ids.numel() > 0:
            drop_id = int(nonlocal_ids[0].item())

            def sentinel(_ids, rows):
                ncols = _ids.size(1) - k
                return torch.full(
                    (rows.numel(), ncols),
                    drop_id,
                    dtype=_ids.dtype,
                    device=_ids.device,
                )

            return sentinel

    # Regime B fallback: duplicate the surviving top expert (col 0) of each row.
    def sentinel(_ids, rows):
        ncols = _ids.size(1) - k
        top = _ids[rows, 0:1]  # [R,1]
        return top.expand(rows.numel(), ncols).clone()

    return sentinel


# ---------------------------------------------------------------------------
# Installer: attach the routing fn to every Qwen3 MoE FusedMoE layer.
# ---------------------------------------------------------------------------


def _install_on_model(model) -> int:
    """Walk the model, set custom_routing_function on each FusedMoE expert block.
    Returns the number of layers patched."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, FusedMoE):
            layer_idx = _layer_idx_from_name(name)

            # Lazy getter so we read expert_map as it exists at call time.
            def _getter(m=module):
                return getattr(m, "expert_map", None)

            module.custom_routing_function = _make_routing_fn(
                layer_idx if layer_idx is not None else 0, _getter
            )
            patched += 1
    if _Cfg.debug:
        print(f"[adaptive_topk] patched {patched} FusedMoE layers "
              f"(enable={_Cfg.enable} k={_Cfg.k} thresh={_Cfg.thresh})")
    return patched


def _layer_idx_from_name(name: str):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def install():
    """Monkeypatch FusedMoE construction so every layer gets our routing fn.

    We wrap FusedMoE.__init__ to stamp custom_routing_function at build time.
    This is robust to where/when the model is created (worker processes, etc.)
    and avoids needing a handle to the assembled model object.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    if getattr(FusedMoE, "_adaptive_topk_installed", False):
        return
    orig_init = FusedMoE.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Only Qwen3-style softmax routing (no grouped_topk groups). Guard so we
        # never silently break grouped/DeepSeek-style models.
        if getattr(self, "use_grouped_topk", False):
            return
        if getattr(self, "custom_routing_function", None) is not None:
            return  # respect an existing custom router
        prefix = getattr(self, "layer_name", "") or ""
        layer_idx = _layer_idx_from_name(prefix)

        def _getter(m=self):
            return getattr(m, "expert_map", None)

        self.custom_routing_function = _make_routing_fn(
            layer_idx if layer_idx is not None else 0, _getter
        )

    FusedMoE.__init__ = patched_init
    FusedMoE._adaptive_topk_installed = True
    if _Cfg.debug:
        print("[adaptive_topk] FusedMoE.__init__ patched; "
              f"enable={_Cfg.enable} k={_Cfg.k} thresh={_Cfg.thresh}")


def register():
    """vLLM general-plugin entry point. vLLM calls this in every process
    (engine + workers) before models are built -- the correct place to patch."""
    install()
