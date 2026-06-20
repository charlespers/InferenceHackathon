"""Analytical sweeps — explore how the bottleneck and the best lever change with
context depth and quantization, without touching a GPU.

Two sweeps answer two different "what next?" questions:
  - depth_sweep  : as context grows, when does KV-bandwidth overtake weights?
                   (decides whether KV quant or weight quant is the next lever)
  - config_sweep : rank candidate (dtype, kv_dtype, plan, tp, ep) configs by
                   predicted decode tok/s, each annotated with its bottleneck.

Both use the analytical roofline model (`latency.decode_latency`), so they run on
a laptop and produce the same metric vocabulary a measured run does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..model import MoEConfig
from ..hardware import Cluster
from ..latency import decode_latency
from .config import BenchConfig
from .efficiency import compute_efficiency, peak_flops_for_dtype
from .attribution import dominant_of_breakdown, lever_hint


@dataclass(frozen=True)
class SweepPoint:
    label: str
    seq_len: int
    plan: str
    dtype_bytes: float
    kv_dtype_bytes: int
    tp: int
    ep: int
    decode_tok_s: float
    tpot_ms: float
    mbu_decode: Optional[float]
    mfu_decode: Optional[float]
    kv_byte_share: Optional[float]
    regime: str
    dominant_term: str
    hint: str


def realized_efficiency(cfg: MoEConfig, cluster: Cluster, config: BenchConfig,
                        measured_tok_s: float):
    """Whole-model realized efficiency e = measured_tok_s / analytical_floor_tok_s.
    Feed e back into sweeps/plan (--efficiency) so predictions are calibrated to
    reality instead of the optimistic floor (e=1.0). Returns (e, floor_tok_s)."""
    floor = decode_latency(
        cfg, cluster, plan=config.plan, dtype_bytes=config.dtype_bytes,
        kv_dtype_bytes=config.kv_dtype_bytes, seq_len=config.seq_len,
        tp=config.tp, ep=config.ep).tokens_per_s
    e = (measured_tok_s / floor) if floor else None
    return e, floor


def _point(cfg: MoEConfig, cluster: Cluster, *, label: str, plan: str,
           dtype_bytes: float, kv_dtype_bytes: int, tp: int, ep: int,
           seq_len: int, efficiency: float = 1.0) -> SweepPoint:
    bd = decode_latency(cfg, cluster, plan=plan, dtype_bytes=dtype_bytes,
                        kv_dtype_bytes=kv_dtype_bytes, seq_len=seq_len, tp=tp, ep=ep)
    # e<1 = realized whole-model efficiency (kernels/launch leave the floor on the
    # table). e=1.0 is the analytical floor (optimistic upper bound).
    tok_s = bd.tokens_per_s * efficiency
    weight_bytes = cfg.active_params * dtype_bytes
    kv_bytes = seq_len * cfg.kv_bytes_per_token(kv_dtype_bytes)
    bpt = weight_bytes + kv_bytes
    peak_flops = peak_flops_for_dtype(
        cluster.gpu.bf16_flops, cluster.gpu.fp8_flops, dtype_bytes) * cluster.n_gpus
    eff = compute_efficiency(
        active_params=cfg.active_params, weight_bytes=weight_bytes,
        bytes_per_token=bpt, kv_bytes=kv_bytes, prompt_tokens=seq_len,
        prefill_tok_s=None, decode_tok_s=tok_s,
        peak_flops=peak_flops, peak_bw=cluster.aggregate_hbm_bw)
    dom = dominant_of_breakdown(bd.weight_read_s, bd.kv_read_s, bd.comms_s, bd.compute_s)
    return SweepPoint(
        label=label, seq_len=seq_len, plan=plan, dtype_bytes=dtype_bytes,
        kv_dtype_bytes=kv_dtype_bytes, tp=tp, ep=ep,
        decode_tok_s=tok_s, tpot_ms=(1000.0 / tok_s) if tok_s else float("inf"),
        mbu_decode=eff.mbu_decode, mfu_decode=eff.mfu_decode,
        kv_byte_share=eff.kv_byte_share, regime=eff.regime_decode,
        dominant_term=dom, hint=lever_hint(dom))


def depth_sweep(cfg: MoEConfig, cluster: Cluster, config: BenchConfig,
                depths: List[int], efficiency: float = 1.0) -> List[SweepPoint]:
    """Decode behaviour vs context depth at a fixed config (KV-decay curve)."""
    return [_point(cfg, cluster, label=f"ctx{d}", plan=config.plan,
                   dtype_bytes=config.dtype_bytes, kv_dtype_bytes=config.kv_dtype_bytes,
                   tp=config.tp, ep=config.ep, seq_len=d, efficiency=efficiency)
            for d in depths]


def quant_grid(base: BenchConfig,
               dtypes=(2, 1, 0.5), kv_dtypes=(2, 1)) -> List[BenchConfig]:
    """Quantization variants of a base config (weight dtype x KV dtype)."""
    out = []
    for d in dtypes:
        for kv in kv_dtypes:
            out.append(BenchConfig(
                name=f"w{d}b-kv{kv}b", plan=base.plan, dtype_bytes=d,
                kv_dtype_bytes=kv, tp=base.tp, ep=base.ep,
                prompt_tokens=base.prompt_tokens, decode_tokens=base.decode_tokens))
    return out


def _divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def layout_grid(base: BenchConfig, n_gpus: int) -> List[BenchConfig]:
    """Parallelism variants of a base config: every (tp, ep) over divisors of
    n_gpus. Layout doesn't change outputs, so this is a pure-speed search."""
    out = []
    for tp in _divisors(n_gpus):
        for ep in _divisors(n_gpus):
            out.append(BenchConfig(
                name=f"tp{tp}-ep{ep}", plan=base.plan, dtype_bytes=base.dtype_bytes,
                kv_dtype_bytes=base.kv_dtype_bytes, tp=tp, ep=ep,
                prompt_tokens=base.prompt_tokens, decode_tokens=base.decode_tokens))
    return out


def full_grid(base: BenchConfig, n_gpus: int,
              dtypes=(2, 1, 0.5), kv_dtypes=(2, 1)) -> List[BenchConfig]:
    """The full quant x layout space (quant changes quality; layout doesn't)."""
    out = []
    for d in dtypes:
        for kv in kv_dtypes:
            for tp in _divisors(n_gpus):
                for ep in _divisors(n_gpus):
                    out.append(BenchConfig(
                        name=f"w{d}b-kv{kv}b-tp{tp}-ep{ep}", plan=base.plan,
                        dtype_bytes=d, kv_dtype_bytes=kv, tp=tp, ep=ep,
                        prompt_tokens=base.prompt_tokens, decode_tokens=base.decode_tokens))
    return out


def config_sweep(cfg: MoEConfig, cluster: Cluster, configs: List[BenchConfig],
                 efficiency: float = 1.0) -> List[SweepPoint]:
    """Rank candidate configs by predicted decode tok/s (descending)."""
    pts = [_point(cfg, cluster, label=c.name, plan=c.plan, dtype_bytes=c.dtype_bytes,
                  kv_dtype_bytes=c.kv_dtype_bytes, tp=c.tp, ep=c.ep, seq_len=c.seq_len,
                  efficiency=efficiency)
           for c in configs]
    pts.sort(key=lambda p: -p.decode_tok_s)
    return pts
