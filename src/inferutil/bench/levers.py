"""Next-optimization recommender — rank candidate levers by predicted speedup.

Each lever is scored by re-running the analytical decode model
(`latency.decode_latency`) with the lever applied and comparing tokens/s to the
current configuration's baseline. This turns "what should we work on next?" into
a ranked, quantified list instead of a hunch. Speedups are *predictions* from the
roofline model; a real measured run then validates them (see attribution's
`kernel_gap`).

Levers considered:
  - int4 experts        weights -> half the bytes      (effort S, kernel exists: k5_int4)
  - fp8/int8 KV         KV dtype 2B -> 1B              (effort S)
  - ideal routing       remove expert load imbalance   (effort M, engine/src/routing)
  - EP/TP relayout      best (tp, ep) split            (effort S)
  - speculative decode  amortize verify over a draft tree (effort L, engine/src/spec)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..model import MoEConfig
from ..hardware import Cluster
from ..latency import decode_latency
from .config import BenchConfig

_EFFORT_RANK = {"S": 0, "M": 1, "L": 2}


@dataclass(frozen=True)
class Lever:
    name: str
    predicted_tok_s: float
    speedup: float            # predicted / baseline
    effort: str               # S | M | L
    rationale: str


def _tok_s(cfg, cluster, *, plan, dtype_bytes, kv_dtype_bytes, seq_len, tp, ep,
           ideal_routing=False) -> float:
    return decode_latency(
        cfg, cluster, plan=plan, dtype_bytes=dtype_bytes,
        kv_dtype_bytes=kv_dtype_bytes, seq_len=seq_len, tp=tp, ep=ep,
        ideal_routing=ideal_routing).tokens_per_s


def _spec_speedup(accept_rate: float, draft_len: int, verify_overhead: float) -> float:
    """Expected accepted tokens per target step / verify tax.

    E[accepted draft tokens before first reject] = a(1-a^d)/(1-a); plus the 1
    free token the target always emits. Divide by the verify overhead factor
    (running the small draft model d times costs a little)."""
    a = max(0.0, min(0.999, accept_rate))
    geom = a * (1.0 - a ** draft_len) / (1.0 - a) if a else 0.0
    t_eff = 1.0 + geom
    return t_eff / max(1.0, verify_overhead)


def recommend(cfg: MoEConfig, cluster: Cluster, config: BenchConfig, *,
              bottleneck=None, accept_rate: float = 0.7, draft_len: int = 4,
              verify_overhead: float = 1.15, min_speedup: float = 1.02) -> List[Lever]:
    """Ranked levers for this config. `bottleneck` (optional) annotates the lever
    that directly targets the diagnosed dominant term."""
    base = _tok_s(cfg, cluster, plan=config.plan, dtype_bytes=config.dtype_bytes,
                  kv_dtype_bytes=config.kv_dtype_bytes, seq_len=config.seq_len,
                  tp=config.tp, ep=config.ep)
    dom = getattr(bottleneck, "dominant_term", None)
    levers: List[Lever] = []

    def add(name, pred, effort, rationale, targets=None):
        if base <= 0 or pred <= 0:
            return
        sp = pred / base
        if sp < min_speedup:
            return
        if targets and dom in targets:
            rationale += "  [targets diagnosed bottleneck]"
        levers.append(Lever(name=name, predicted_tok_s=pred, speedup=sp,
                            effort=effort, rationale=rationale))

    # int4 experts: halve weight bytes (only if currently above int4).
    if config.dtype_bytes > 0.5:
        pred = _tok_s(cfg, cluster, plan=config.plan,
                      dtype_bytes=config.dtype_bytes / 2.0,
                      kv_dtype_bytes=config.kv_dtype_bytes, seq_len=config.seq_len,
                      tp=config.tp, ep=config.ep)
        add("int4 experts", pred, "S",
            f"weights {config.dtype_bytes}B -> {config.dtype_bytes/2.0}B/param",
            targets={"weight_bw"})

    # fp8/int8 KV: halve KV bytes (only if currently above 1 byte).
    if config.kv_dtype_bytes > 1:
        pred = _tok_s(cfg, cluster, plan=config.plan, dtype_bytes=config.dtype_bytes,
                      kv_dtype_bytes=1, seq_len=config.seq_len, tp=config.tp, ep=config.ep)
        add("fp8/int8 KV", pred, "S", "KV cache 2B -> 1B/elem", targets={"kv_bw"})

    # ideal routing: remove expert load imbalance (placement optimizer payoff).
    if config.plan in ("ep", "hybrid"):
        pred = _tok_s(cfg, cluster, plan=config.plan, dtype_bytes=config.dtype_bytes,
                      kv_dtype_bytes=config.kv_dtype_bytes, seq_len=config.seq_len,
                      tp=config.tp, ep=config.ep, ideal_routing=True)
        add("ideal expert routing", pred, "M",
            "eliminate busiest-GPU expert imbalance (routing/placement optimizer)",
            targets={"weight_bw", "comms"})

    # EP/TP relayout: best (tp, ep) over divisors of n_gpus.
    n = cluster.n_gpus
    best_pred, best_split = base, (config.tp, config.ep)
    for tp in [d for d in range(1, n + 1) if n % d == 0]:
        for ep in [d for d in range(1, n + 1) if n % d == 0]:
            pred = _tok_s(cfg, cluster, plan=config.plan, dtype_bytes=config.dtype_bytes,
                          kv_dtype_bytes=config.kv_dtype_bytes, seq_len=config.seq_len,
                          tp=tp, ep=ep)
            if pred > best_pred:
                best_pred, best_split = pred, (tp, ep)
    if best_split != (config.tp, config.ep):
        add("relayout tp/ep", best_pred, "S",
            f"tp={config.tp},ep={config.ep} -> tp={best_split[0]},ep={best_split[1]}",
            targets={"weight_bw", "comms"})

    # speculative decode: amortize verify over a draft tree.
    sp = _spec_speedup(accept_rate, draft_len, verify_overhead)
    add("speculative decode", base * sp, "L",
        f"est. accept~{accept_rate:.2f}, tree={draft_len}, verify x{verify_overhead:.2f}")

    levers.sort(key=lambda lv: (-lv.speedup, _EFFORT_RANK.get(lv.effort, 9)))
    return levers
