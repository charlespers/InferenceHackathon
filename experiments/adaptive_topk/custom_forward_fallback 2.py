"""B=1 custom-forward fallback for confidence-adaptive top-k MoE (Qwen3-235B-A22B).

This is the DOCUMENTED FALLBACK to `vllm_adaptive_moe.py`. Same env config
(`ADAPTIVE_TOPK_ENABLE/K/THRESH/MIN_LAYER/DEBUG`), same `install()`/`register()`
drop-in plugin shape -- but a *different mechanism*.

----------------------------------------------------------------------------
WHY THIS EXISTS (when to prefer it over the fused-skip plugin)
----------------------------------------------------------------------------
`vllm_adaptive_moe.py` keeps the fused Triton MoE kernel and merely stamps the
low-confidence columns with a sentinel expert id that maps to -1 under
`--enable-expert-parallel`, so the kernel early-returns for that (expert, block)
pair and skips the weight load. That is the lowest-risk path and the *byte* read
genuinely drops.

The open question that path can't escape is whether the fused kernel's FIXED
overhead lets the skip become wall-clock time:
  * `moe_align_block_size` pads `EM` (= topk_ids.numel() + num_experts*(block-1))
    to a fixed grid -- the grid does NOT shrink when fewer distinct experts
    appear. At B=1/M=1 the grid is already tiny, so the early-returned -1 blocks
    are a handful of no-op block iterations; the *load* is skipped but the kernel
    launch / align / write-zeros scaffolding is not.
  * So the fused-skip saving is real on bytes but can be partially eaten by fixed
    per-layer kernel overhead.

This fallback removes that uncertainty at B=1 by BYPASSING the fused kernel
entirely: it gathers exactly the surviving k expert ids for the single token and
runs a plain Python loop of k dense expert MLPs (gate_up -> SiLU*gate -> down).
The dropped experts are never touched -- no sentinel, no -1, no align kernel, no
write-zeros blocks. Exactly k weight reads, guaranteed.

TRADEOFF (honest):
  + Pro: loads *exactly* k experts' weights; true per-token variable k; zero
    kernel/padding/align overhead for skipped OR surviving experts. Cleanest
    possible saving and the result is unambiguous to measure.
  - Con: loses the fused kernel's launch-coalescing across the k surviving
    experts (k separate small GEMMs instead of one grouped GEMM). At B=1 decode
    this is memory-bound and compute has slack, so the loop is usually a net win
    -- but ONLY at B=1. This path deliberately falls back to the original fused
    `forward` for any M>1 batch (prefill, or batched decode), where the fused
    kernel wins decisively.
  - Con: this reference implementation handles the UNQUANTIZED expert weights
    (w13_weight / w2_weight on the FusedMoE module, bf16/fp16). FP8/quantized
    expert weight layouts are NOT dequantized here; on a quantized layer it
    safely no-ops back to the fused path (see `_supported_layer`). For the
    weight-bound FP8 A/B, prefer the fused-skip plugin and only switch arms here
    if you build a bf16 layer or extend `_expert_mlp` to dequant.

Net: use this fallback for the weight-bound test ONLY IF the fused-skip plugin's
measured speedup is shown to be swallowed by EM-padding / fixed kernel overhead.
Otherwise ship the fused-skip plugin (lower risk, supports quantized experts).

----------------------------------------------------------------------------
USAGE (identical to vllm_adaptive_moe.py)
----------------------------------------------------------------------------
  VLLM_PLUGINS=adaptive_topk_fallback ADAPTIVE_TOPK_ENABLE=1 \
    ADAPTIVE_TOPK_K=4 ADAPTIVE_TOPK_THRESH=0.9 vllm serve ... --max-num-seqs 1

or preload: `import ...custom_forward_fallback as m; m.install()` before serve.

Tested against vLLM ~0.10.x, torch 2.7.1+cu126.
"""

from __future__ import annotations

import os
import threading

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config -- identical knobs to vllm_adaptive_moe.py so the two are interchangeable.
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


_stats_lock = threading.Lock()
_stats = {"tokens": 0, "reduced_rows": 0, "kept_rows": 0, "dropped_experts": 0,
          "fused_fallbacks": 0}


def stats_snapshot() -> dict:
    with _stats_lock:
        s = dict(_stats)
    rows = s["reduced_rows"] + s["kept_rows"]
    s["reduced_frac"] = (s["reduced_rows"] / rows) if rows else 0.0
    return s


# ---------------------------------------------------------------------------
# The selection policy -- mirrors vllm_adaptive_moe.adaptive_routing, but here we
# return the *list of surviving expert ids* (variable length) rather than a
# rectangular [M, top_k] tensor, because we drive a plain loop, not the kernel.
# ---------------------------------------------------------------------------


def _select_experts_b1(router_logits: torch.Tensor, full_topk: int,
                       renormalize: bool, layer_idx: int):
    """B=1 selection. router_logits: [1, num_experts].

    Returns (ids: LongTensor[k_eff], weights: FloatTensor[k_eff], reduced: bool).
    weights are renormalized over the surviving k_eff experts iff `renormalize`.
    """
    scores = torch.softmax(router_logits.float(), dim=-1)          # [1, E]
    topw, topi = torch.topk(scores, full_topk, dim=-1)            # [1, full_topk]
    topw = topw[0]
    topi = topi[0]

    k_eff = full_topk
    reduced = False
    if _Cfg.enable and _Cfg.k < full_topk and layer_idx >= _Cfg.min_layer:
        mass = float(topw[: _Cfg.k].sum())
        if mass > _Cfg.thresh:
            k_eff = _Cfg.k
            reduced = True

    ids = topi[:k_eff].to(torch.long)
    weights = topw[:k_eff].clone()
    if renormalize:
        denom = weights.sum()
        if float(denom) > 0:
            weights = weights / denom
    return ids, weights, reduced


# ---------------------------------------------------------------------------
# Per-expert dense MLP for the unquantized Qwen3 FusedMoE weight layout.
#
# vLLM stores fused expert weights on the FusedMoE module as:
#   w13_weight : [num_local_experts, 2*intermediate, hidden]   (gate||up stacked)
#   w2_weight  : [num_local_experts, hidden, intermediate]     (down)
# Qwen3 MoE activation is SiLU(gate) * up  (act_fn == "silu", standard SwiGLU).
# ---------------------------------------------------------------------------


def _expert_mlp(experts_module, local_eid: int, x: torch.Tensor) -> torch.Tensor:
    """Run a single expert's dense MLP. x: [hidden]. Returns [hidden]."""
    w13 = experts_module.w13_weight[local_eid]   # [2*inter, hidden]
    w2 = experts_module.w2_weight[local_eid]     # [hidden, inter]
    gate_up = F.linear(x, w13)                    # [2*inter]
    gate, up = gate_up.chunk(2, dim=-1)
    inter = F.silu(gate) * up                     # [inter]
    return F.linear(inter, w2)                     # [hidden]


def _supported_layer(experts_module) -> bool:
    """Only the unquantized bf16/fp16 path is handled by this reference loop.

    On quantized (FP8 etc.) layers the stored tensors are not plain weights, so
    we decline and let the original fused forward run (safe no-op fallback)."""
    w13 = getattr(experts_module, "w13_weight", None)
    w2 = getattr(experts_module, "w2_weight", None)
    if w13 is None or w2 is None:
        return False
    if w13.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    # An expert_map means EP remap to global ids; we index local rows directly,
    # so a non-trivial expert_map would mismatch. Only support the no-map case
    # (the loop already loads exactly k experts, so EP byte-skipping is moot).
    if getattr(experts_module, "expert_map", None) is not None:
        return False
    return True


# ---------------------------------------------------------------------------
# The monkeypatched Qwen3MoeSparseMoeBlock.forward.
# ---------------------------------------------------------------------------


def _make_forward(orig_forward, layer_idx: int):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Shape handling: Qwen3MoeSparseMoeBlock flattens to [num_tokens, hidden]
        # internally; here we inspect the incoming shape. Only B=1 / single-token
        # decode takes the custom loop; everything else uses the fused path.
        orig_shape = hidden_states.shape
        num_tokens = hidden_states.numel() // orig_shape[-1]

        experts = getattr(self, "experts", None)
        if (not _Cfg.enable or num_tokens != 1 or experts is None
                or not _supported_layer(experts)):
            with _stats_lock:
                _stats["fused_fallbacks"] += 1
            return orig_forward(hidden_states)

        hidden_dim = orig_shape[-1]
        x = hidden_states.reshape(-1, hidden_dim)[0]   # [hidden], the one token

        # Router. Qwen3MoeSparseMoeBlock holds `self.gate` (a ReplicatedLinear).
        router_logits, _ = self.gate(x.unsqueeze(0))   # [1, num_experts]
        full_topk = int(getattr(experts, "top_k", 8))
        renorm = bool(getattr(experts, "renormalize",
                              getattr(self, "norm_topk_prob", True)))

        ids, weights, reduced = _select_experts_b1(
            router_logits, full_topk, renorm, layer_idx)

        out = torch.zeros_like(x)
        for j in range(ids.numel()):
            eid = int(ids[j].item())
            out = out + weights[j].to(x.dtype) * _expert_mlp(experts, eid, x)

        with _stats_lock:
            _stats["tokens"] += 1
            if reduced:
                _stats["reduced_rows"] += 1
                _stats["dropped_experts"] += full_topk - ids.numel()
            else:
                _stats["kept_rows"] += 1

        return out.reshape(orig_shape)

    return forward


# ---------------------------------------------------------------------------
# Installer -- patches Qwen3MoeSparseMoeBlock.forward at class level.
# ---------------------------------------------------------------------------


def install():
    """Monkeypatch Qwen3MoeSparseMoeBlock.forward (drop-in plugin shape)."""
    try:
        from vllm.model_executor.models.qwen3_moe import (
            Qwen3MoeSparseMoeBlock as Block,
        )
    except Exception as e:  # pragma: no cover - import guard
        if _Cfg.debug:
            print(f"[adaptive_topk_fallback] qwen3_moe import failed: {e}")
        return

    if getattr(Block, "_adaptive_topk_fallback_installed", False):
        return

    orig_forward = Block.forward

    def patched_forward(self, hidden_states):
        # Resolve this block's layer index once and cache it on the instance.
        li = getattr(self, "_adaptive_topk_layer_idx", None)
        if li is None:
            li = _layer_idx_from_name(getattr(self, "prefix", "")
                                      or getattr(self, "layer_name", ""))
            li = li if li is not None else 0
            self._adaptive_topk_layer_idx = li
        return _make_forward(orig_forward, li)(self, hidden_states)

    Block.forward = patched_forward
    Block._adaptive_topk_fallback_installed = True
    if _Cfg.debug:
        print("[adaptive_topk_fallback] Qwen3MoeSparseMoeBlock.forward patched; "
              f"enable={_Cfg.enable} k={_Cfg.k} thresh={_Cfg.thresh} "
              "(B=1 custom expert loop, bypasses fused kernel)")


def _layer_idx_from_name(name: str):
    if not name:
        return None
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def register():
    """vLLM general-plugin entry point (engine + every worker process)."""
    install()
