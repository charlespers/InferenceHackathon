from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..model import MoEConfig
from ..hardware import Cluster
from ..latency import decode_latency
from .prefill import prefill_latency


@dataclass(frozen=True)
class ExpertRoute:
    layer: int
    expert_id: int
    gpu: int


@dataclass(frozen=True)
class StepBreakdown:
    weight_s: float
    kv_s: float
    comms_s: float
    compute_s: float


@dataclass(frozen=True)
class PrefillResult:
    n_prompt_tokens: int
    seconds: float                 # prompt-processing wall-time
    first_token_seconds: float     # time to emit the first decoded token


@dataclass(frozen=True)
class DecodeStep:
    index: int
    seconds: float
    routes: tuple = ()             # tuple[ExpertRoute, ...]; empty if not exposed
    breakdown: "StepBreakdown | None" = None
    token_id: "int | None" = None


@runtime_checkable
class Engine(Protocol):
    def reset(self, *, plan: str, dtype_bytes: int, kv_dtype_bytes: int,
              tp: int, ep: int, seq_len: int) -> None: ...
    def prefill(self, token_ids: list) -> PrefillResult: ...
    def decode_step(self) -> DecodeStep: ...


class MockEngine:
    """Engine whose timings are seeded from the analytical roofline.

    With efficiency=1.0 and jitter=0.0 each decode step takes exactly
    decode_latency(...).total_s, so an end-to-end run reports 100% of floor —
    the wiring sanity check. efficiency<1 models real kernels leaving BW on the
    table; jitter adds seeded per-step noise for realistic p50/p95.
    """

    def __init__(self, cfg: MoEConfig, cluster: Cluster, *, efficiency: float = 1.0,
                 jitter: float = 0.0, seed: int = 0, expose_routes: bool = False,
                 quality_offset: int = 0):
        self.cfg = cfg
        self.cluster = cluster
        self.efficiency = efficiency
        self.jitter = jitter
        self.seed = seed
        self.expose_routes = expose_routes
        self.quality_offset = quality_offset
        self._floor_s = 0.0
        self._shares = (0.0, 0.0, 0.0, 0.0)
        self._dtype_bytes = 2
        self._kv_dtype_bytes = 2
        self._index = 0
        self._rng = random.Random(seed)

    def reset(self, *, plan, dtype_bytes, kv_dtype_bytes, tp, ep, seq_len):
        bd = decode_latency(
            self.cfg, self.cluster, plan=plan, dtype_bytes=dtype_bytes,
            kv_dtype_bytes=kv_dtype_bytes, seq_len=seq_len, tp=tp, ep=ep)
        self._floor_s = bd.total_s
        self._shares = (bd.weight_read_s, bd.kv_read_s, bd.comms_s, bd.compute_s)
        self._dtype_bytes = dtype_bytes
        self._kv_dtype_bytes = kv_dtype_bytes
        self._index = 0
        self._rng = random.Random(self.seed)

    def _step_seconds(self) -> float:
        s = self._floor_s / self.efficiency
        if self.jitter:
            s *= 1.0 + self._rng.uniform(-self.jitter, self.jitter)
        return s

    def prefill(self, token_ids) -> PrefillResult:
        n = len(token_ids)
        # Batched prefill roofline (compute vs full-weight-read), inflated by the
        # same kernel-efficiency knob as decode for consistency.
        floor = prefill_latency(self.cfg, self.cluster, dtype_bytes=self._dtype_bytes,
                                kv_dtype_bytes=self._kv_dtype_bytes, prompt_tokens=n)
        seconds = floor / self.efficiency if self.efficiency else floor
        return PrefillResult(n_prompt_tokens=n, seconds=seconds,
                             first_token_seconds=self._step_seconds())

    def decode_step(self) -> DecodeStep:
        i = self._index
        self._index += 1
        routes = ()
        if self.expose_routes:
            routes = tuple(
                ExpertRoute(
                    layer=i % self.cfg.n_layers,
                    expert_id=(i * 7 + k * 13) % self.cfg.n_experts,
                    gpu=((i * 7 + k * 13) % self.cfg.n_experts) % self.cluster.n_gpus,
                )
                for k in range(self.cfg.top_k)
            )
        s = self._step_seconds()
        w, k, c, comp = self._shares
        scale = (s / self._floor_s) if self._floor_s else 0.0
        breakdown = StepBreakdown(weight_s=w * scale, kv_s=k * scale,
                                  comms_s=c * scale, compute_s=comp * scale)
        token_id = ((i * 2654435761) + self.quality_offset * (i + 1)) % self.cfg.vocab
        return DecodeStep(index=i, seconds=s, routes=routes, breakdown=breakdown,
                          token_id=token_id)
