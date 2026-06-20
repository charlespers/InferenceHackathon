"""LOOP-C — Speculative/Stale Tensor Parallelism probe for Qwen3-235B-A22B on vLLM.

GOAL (B=1 decode latency): the dominant B=1 floor term is the ~188 serial TP
all-reduces/token (2/layer x 94 layers). They are serial because each layer's
all-reduce produces the input the *same* layer needs next, so compute blocks on
the collective (comms_floor.md S3 showed lossless overlap is therefore infeasible
at B=1). Stale-TP breaks that dependency: on most layers, instead of doing the real
all-reduce we substitute a STALE/approximate reduced activation, so the collective
could be overlapped (or skipped) and the critical path no longer waits on it.

This module is the QUALITY PROBE only -- it answers the single go/no-go question:
**how much all-reduce staleness does Qwen3-235B tolerate before greedy decode
output diverges from exact?** It does NOT itself make decode faster (it still calls
the real reduce on refresh layers); it MEASURES the quality ceiling so we know
whether a real stale-TP kernel is worth building. See research/n4_speculative_stale_tp.md.

Prior art (research run wf_07e8b2cc-0b1, 23/25 verified): Ladder-Residual (ICML'25)
proves depth-1 stale-residual pays at B=1/TP=8/8xH100 (23.7% latency) but needs
RETRAINING; Kog "Delayed TP" is approximate + pretrained. The open question this
probe settles is whether a *no-retrain, runtime-only* K-layer variant keeps quality.

----------------------------------------------------------------------------
WHAT WE SUBSTITUTE (modes + policies, all env-configurable)
----------------------------------------------------------------------------
Per forward pass there are `period` TP all-reduces (default 2*94=188). Index them
0..period-1; layer = idx // collectives_per_layer, slot = idx % collectives_per_layer
(slot 0 = post-attention o_proj, slot 1 = post-MoE down_proj).

  STALE_TP_MODE = layer     (default; the Ladder/N4 within-token depth-staleness)
      Refresh layers (layer % K == 0) do the REAL all-reduce and cache the result
      per slot. Non-refresh layers reuse the most recent real reduced value for
      that slot ("activations change slowly across layers" -- Ladder-Residual).
  STALE_TP_MODE = temporal  (across-token staleness)
      Every K-th decode step does a full real pass (and caches per idx); the K-1
      steps in between reuse the previous step's cached reduced value per idx.

  STALE_TP_POLICY = proxy   (default; reuse the cached real reduced activation)
                  = local   (return the un-reduced local partial -- crude lower
                             bound; expected to "heavily degrade", matches Kog's
                             naive-removal finding; use as a sanity floor)

  STALE_TP_K            = 2     (refresh period in layers (mode=layer) or steps
                                 (mode=temporal); K=1 == exact baseline)
  STALE_TP_DECODE_ONLY  = 1     (only stale at B=1 decode (token_count==1);
                                 keep prefill exact so the prompt encodes cleanly)
  STALE_TP_PERIOD       = 188   (all-reduces per forward; auto-wraps the counter)
  STALE_TP_COLLECTIVES_PER_LAYER = 2
  STALE_TP_ENABLE       = 0     (master off; K=1 also == exact)
  STALE_TP_DEBUG        = 0

The scheduler decision logic (StaleScheduler) is pure Python (no torch), so it is
unit-tested with no GPU (see test_stale_tp.py). Only install() touches vLLM.
"""

from __future__ import annotations

import os
import threading


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class Cfg:
    enable = _env_flag("STALE_TP_ENABLE", False)
    K = int(os.environ.get("STALE_TP_K", "2"))
    mode = os.environ.get("STALE_TP_MODE", "layer").strip().lower()
    policy = os.environ.get("STALE_TP_POLICY", "proxy").strip().lower()
    decode_only = _env_flag("STALE_TP_DECODE_ONLY", True)
    period = int(os.environ.get("STALE_TP_PERIOD", "188"))
    collectives_per_layer = int(os.environ.get("STALE_TP_COLLECTIVES_PER_LAYER", "2"))
    world = int(os.environ.get("STALE_TP_WORLD", "8"))  # TP size; predicted = local x world
    debug = int(os.environ.get("STALE_TP_DEBUG", "0"))
    # Optional control file: JSON {"enable","K","mode","policy","decode_only"} re-read
    # at the start of each forward pass, so ONE vLLM launch can sweep K/policy (edit
    # the file between quality_probe runs) instead of reloading the 235B per config.
    ctl = os.environ.get("STALE_TP_CTL", "")


# ---------------------------------------------------------------------------
# PURE decision logic (no torch / no vLLM) -- unit-tested without a GPU.
# It treats reduced activations as opaque "value" handles; it only decides
# real-vs-substitute and which cached handle to return. The vLLM wrapper
# supplies a real_reduce() thunk and the local-partial handle.
# ---------------------------------------------------------------------------

# decision kinds
REAL = "real"            # did the true all-reduce
REAL_FALLBACK = "real_fallback"  # wanted to substitute but had no cache yet
STALE = "stale"          # reused a cached real reduced value (proxy policy)
LOCAL = "local"          # returned the un-reduced local partial (local policy)
PREDICTED = "predicted"  # estimate the sum from THIS layer's local partial x world_size
EXACT = "exact"          # disabled / refresh / prefill -> true reduce, no caching role


class StaleScheduler:
    """Decides, per TP all-reduce call, whether to do the real collective or
    substitute a stale value. Pure; deterministic; no tensor introspection."""

    def __init__(self, cfg=Cfg):
        self.K = max(1, cfg.K)
        self.mode = cfg.mode
        self.policy = cfg.policy
        self.enable = cfg.enable
        self.decode_only = cfg.decode_only
        self.period = max(1, cfg.period)
        self.cpl = max(1, cfg.collectives_per_layer)
        self.world = getattr(cfg, "world", 8)
        self.debug = cfg.debug
        self.ctl_path = getattr(cfg, "ctl", "") or getattr(cfg, "ctl_path", "")
        self._ctl_mtime = 0.0

        self._call = 0          # index within current forward pass
        self._step = 0          # decode step counter (for temporal mode)
        self._last_by_slot = {}  # mode=layer: slot -> last real reduced handle (this step)
        self._cache_by_idx = {}  # mode=temporal: idx -> last real reduced handle (prev step)
        self._lock = threading.Lock()
        self.stats = {k: 0 for k in (REAL, REAL_FALLBACK, STALE, LOCAL, EXACT)}
        self.observed_calls_per_pass = 0  # for period calibration / sanity

    def is_refresh_layer(self, layer: int) -> bool:
        return (layer % self.K) == 0

    def _wrap_step(self):
        """Called when a forward pass completes (counter hit period)."""
        self.observed_calls_per_pass = self._call
        self._call = 0
        if self.mode == "layer":
            self._last_by_slot.clear()   # within-token staleness only
        self._step += 1

    def maybe_reload_ctl(self):
        """Re-read the control file (if configured) to allow live K/policy sweeps
        without relaunching vLLM. Cheap: called once per forward pass. No-op when
        ctl_path is unset (so unit tests never touch disk)."""
        if not self.ctl_path:
            return
        try:
            import json
            mt = os.path.getmtime(self.ctl_path)
            if mt == self._ctl_mtime:
                return
            self._ctl_mtime = mt
            with open(self.ctl_path) as f:
                d = json.load(f)
            self.enable = bool(d.get("enable", self.enable))
            self.K = max(1, int(d.get("K", self.K)))
            self.mode = str(d.get("mode", self.mode)).lower()
            self.policy = str(d.get("policy", self.policy)).lower()
            self.decode_only = bool(d.get("decode_only", self.decode_only))
            # reset caches on reconfig so a new sweep point starts clean
            self._last_by_slot.clear()
            self._cache_by_idx.clear()
            if self.debug:
                print(f"[stale_tp] ctl reload: enable={self.enable} K={self.K} "
                      f"mode={self.mode} policy={self.policy}")
        except Exception:
            pass

    def route(self, real_reduce, local_value, token_count: int, shape=None):
        """Core hook. `real_reduce()` -> true reduced handle; `local_value` is the
        un-reduced local partial handle; `shape` is the input's shape (a hashable
        tuple) used to guarantee a substituted value matches the current call (so a
        prefill-shaped cache can never leak into a decode call). Returns (kind, value)."""
        with self._lock:
            idx = self._call
            if idx == 0:
                self.maybe_reload_ctl()
            self._call += 1
            wrap = self._call >= self.period
            layer = idx // self.cpl
            slot = idx % self.cpl

            disabled = (not self.enable) or self.K <= 1
            is_prefill = token_count > 1
            force_real = disabled or (self.decode_only and is_prefill)

            if self.mode == "temporal":
                kind, value = self._route_temporal(idx, force_real, is_prefill,
                                                   real_reduce, local_value, shape)
            else:
                kind, value = self._route_layer(layer, slot, force_real, is_prefill,
                                                real_reduce, local_value, shape)

            self.stats[kind] += 1
            if wrap:
                self._wrap_step()
            return kind, value

    def _real(self, real_reduce, store, key, shape, is_prefill, kind):
        out = real_reduce()
        if not is_prefill:                 # never cache prefill tensors for reuse
            store[key] = (shape, out)
        return kind, out

    def _route_layer(self, layer, slot, force_real, is_prefill, real_reduce, local_value, shape):
        if force_real:
            kind = EXACT if (not self.enable or self.K <= 1) else REAL
            return self._real(real_reduce, self._last_by_slot, slot, shape, is_prefill, kind)
        if self.is_refresh_layer(layer):
            return self._real(real_reduce, self._last_by_slot, slot, shape, is_prefill, REAL)
        # non-refresh: substitute
        if self.policy == "local":
            return LOCAL, local_value      # local partial always matches shape
        if self.policy == "predicted":
            return PREDICTED, local_value  # wrapper scales by world_size (cheap sum estimate)
        cached = self._last_by_slot.get(slot)
        if cached is None or cached[0] != shape:   # no cache / shape mismatch -> real
            return self._real(real_reduce, self._last_by_slot, slot, shape, is_prefill, REAL_FALLBACK)
        return STALE, cached[1]

    def _route_temporal(self, idx, force_real, is_prefill, real_reduce, local_value, shape):
        full_real_step = (self._step % self.K) == 0
        if force_real or full_real_step:
            kind = EXACT if (not self.enable or self.K <= 1) else REAL
            return self._real(real_reduce, self._cache_by_idx, idx, shape, is_prefill, kind)
        if self.policy == "local":
            return LOCAL, local_value
        if self.policy == "predicted":
            return PREDICTED, local_value
        cached = self._cache_by_idx.get(idx)
        if cached is None or cached[0] != shape:
            return self._real(real_reduce, self._cache_by_idx, idx, shape, is_prefill, REAL_FALLBACK)
        return STALE, cached[1]

    def snapshot(self):
        with self._lock:
            s = dict(self.stats)
            s["step"] = self._step
            s["observed_calls_per_pass"] = self.observed_calls_per_pass
            return s


# module-level scheduler (built at install)
_SCHED: StaleScheduler | None = None


def get_scheduler() -> StaleScheduler:
    global _SCHED
    if _SCHED is None:
        _SCHED = StaleScheduler(Cfg)
    return _SCHED


# ---------------------------------------------------------------------------
# vLLM install: rebind tensor_model_parallel_all_reduce everywhere it is used.
# RowParallelLinear (attn o_proj + MLP/MoE down) imports the symbol by name, so
# we replace it in every module that holds a reference to the original.
# ---------------------------------------------------------------------------


def install():
    import sys
    import torch  # noqa: F401  (only needed in the real engine)

    try:
        from vllm.distributed import communication_op as cop
    except Exception as e:  # pragma: no cover - import guard
        print(f"[stale_tp] vLLM not importable, install skipped: {e}")
        return 0

    orig = cop.tensor_model_parallel_all_reduce
    if getattr(orig, "_stale_tp_wrapped", False):
        return 0

    sched = get_scheduler()

    def wrapper(input_):
        # token_count = first dim (B=1 decode -> 1; prefill -> seq_len)
        try:
            token_count = int(input_.shape[0]) if input_.dim() >= 1 else 1
        except Exception:
            token_count = 1

        try:
            shape = tuple(input_.shape)
        except Exception:
            shape = None
        # clone-on-return for stale values so downstream in-place residual adds
        # don't corrupt the cache. The scheduler guarantees a STALE value's shape
        # matches `shape` (else it falls back to a real reduce), so a prefill-shaped
        # tensor can never leak into a decode call.
        kind, value = sched.route(
            real_reduce=lambda: orig(input_),
            local_value=input_,
            token_count=token_count,
            shape=shape,
        )
        if kind == PREDICTED:
            # cheapest sum estimate from local info: E[sum] = world_size * E[partial].
            # Right expected magnitude, but local DIRECTION (missing the other ranks'
            # contributions) -- the info-barrier test.
            try:
                return value * sched.world
            except Exception:
                return orig(input_)
        if kind in (STALE, LOCAL):
            return value.clone() if hasattr(value, "clone") else value
        return value

    wrapper._stale_tp_wrapped = True

    # Rebind in EVERY already-imported module that holds the original symbol --
    # the defining module (vllm.distributed.communication_op), its re-exports
    # (vllm.distributed), and every consumer that imported it by name
    # (e.g. vllm.model_executor.layers.linear, where RowParallelLinear looks it up).
    # Single mechanism = the count reflects all rebinds.
    patched_mods = []
    for modname, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            if getattr(mod, "tensor_model_parallel_all_reduce", None) is orig:
                setattr(mod, "tensor_model_parallel_all_reduce", wrapper)
                patched_mods.append(modname)
        except Exception:
            continue
    # belt-and-suspenders: ensure the defining module is rebound even if it was
    # not enumerable for some reason.
    if getattr(cop, "tensor_model_parallel_all_reduce", None) is orig:
        cop.tensor_model_parallel_all_reduce = wrapper
        patched_mods.append("vllm.distributed.communication_op")

    print(f"[stale_tp] installed: enable={Cfg.enable} K={Cfg.K} mode={Cfg.mode} "
          f"policy={Cfg.policy} decode_only={Cfg.decode_only} period={Cfg.period} "
          f"-> rebound in {len(patched_mods)} module(s): {patched_mods}")
    return len(patched_mods)


def register():
    """vLLM general-plugin entry point (called in engine + worker processes)."""
    install()
