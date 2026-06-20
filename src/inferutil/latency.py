"""B=1 decode latency model for MoE inference on a multi-GPU node.

The whole point: at batch size 1 the arithmetic intensity of a matmul is ~1
FLOP per byte read, so decode is **memory-bandwidth bound**, not compute bound.
A token's latency is essentially:

    time to read the active weights from HBM
  + time to read the KV cache from HBM
  + per-layer collective latency (tiny payloads => latency, not BW, bound)
  + (compute, which we show is negligible)

This module computes that breakdown under different parallelism plans so we can
see which term dominates and therefore what is worth optimizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb

from .hardware import GPU, Cluster
from .model import MoEConfig


# ---------------------------------------------------------------------------
# Expert load imbalance: top_k distinct experts scatter across EP GPUs. The
# busiest GPU sets the per-layer expert latency, not the average. Model expert
# choices as `top_k` balls dropped uniformly into `ep` bins and take E[max].
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _binom_pmf(k: int, n: int, p: float) -> float:
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


@lru_cache(maxsize=None)
def _emax(balls: int, bins: int, cur_max: int) -> float:
    if bins == 1:
        return float(max(cur_max, balls))
    p = 1.0 / bins
    total = 0.0
    for k in range(balls + 1):
        total += _binom_pmf(k, balls, p) * _emax(balls - k, bins - 1, max(cur_max, k))
    return total


def expected_max_experts_per_gpu(top_k: int, ep: int) -> float:
    """E[busiest GPU's active-expert count] for uniform routing."""
    if ep <= 1:
        return float(top_k)
    return _emax(top_k, ep, 0)


@dataclass
class DecodeBreakdown:
    plan: str
    dtype_bytes: int
    seq_len: int
    # all times in seconds
    weight_read_s: float
    kv_read_s: float
    compute_s: float
    comms_s: float
    imbalance: float          # busiest-GPU expert multiplier vs ideal

    @property
    def total_s(self) -> float:
        return self.weight_read_s + self.kv_read_s + self.compute_s + self.comms_s

    @property
    def tokens_per_s(self) -> float:
        return 1.0 / self.total_s if self.total_s else float("inf")

    def as_row(self) -> dict:
        ms = 1e3
        return {
            "plan": self.plan,
            "dtype": f"{self.dtype_bytes*8}b" if self.dtype_bytes < 2 else "bf16",
            "seq": self.seq_len,
            "weight_ms": round(self.weight_read_s * ms, 3),
            "kv_ms": round(self.kv_read_s * ms, 3),
            "compute_ms": round(self.compute_s * ms, 3),
            "comms_ms": round(self.comms_s * ms, 3),
            "total_ms": round(self.total_s * ms, 3),
            "tok_per_s": round(self.tokens_per_s, 1),
            "expert_imbalance": round(self.imbalance, 2),
        }


def decode_latency(
    cfg: MoEConfig,
    cluster: Cluster,
    *,
    plan: str = "hybrid",          # "tp" | "ep" | "hybrid" | "floor"
    dtype_bytes: int = 2,          # 2=bf16, 1=fp8 weights
    kv_dtype_bytes: int = 2,
    seq_len: int = 32768,
    tp: int | None = None,         # attention tensor-parallel degree
    ep: int | None = None,         # expert-parallel degree
    ideal_routing: bool = False,   # ignore expert imbalance (placement-perfect)
    measured_max_experts: float | None = None,  # busiest-GPU expert count from a
                                   # real trace (overrides the theoretical E[max]);
                                   # lets a placement/prefetch result drive the model
) -> DecodeBreakdown:
    gpu: GPU = cluster.gpu
    n = cluster.n_gpus
    tp = tp or n
    ep = ep or n

    # --- weights: per-token active bytes, sharded according to plan ---
    attn_bytes = cfg.active_attn_params * dtype_bytes
    expert_bytes_per_layer = cfg.top_k * cfg.one_expert_params * dtype_bytes
    router_bytes = cfg.router_params * dtype_bytes
    lm_head_bytes = cfg.vocab * cfg.hidden * dtype_bytes

    if plan == "floor":
        # absolute lower bound: everything perfectly split across all GPUs,
        # no imbalance, no comms.
        total_w = cfg.active_params * dtype_bytes
        weight_read = total_w / cluster.aggregate_hbm_bw
        imbalance = 1.0
        comms = 0.0
    elif plan == "tp":
        # every weight split across all n GPUs; experts also column/row split.
        per_gpu_layer = (attn_bytes + expert_bytes_per_layer + router_bytes) / n
        weight_read = (per_gpu_layer * cfg.n_layers + lm_head_bytes / n) / gpu.hbm_bw
        imbalance = 1.0
        # 2 all-reduces / layer (post-attn, post-MoE), latency bound at B=1
        comms = 2 * cfg.n_layers * gpu.collective_latency_s
    elif plan in ("ep", "hybrid"):
        # attention tensor-parallel across tp GPUs; experts expert-parallel.
        if measured_max_experts is not None:
            max_experts = measured_max_experts
        elif ideal_routing:
            max_experts = cfg.top_k / ep
        else:
            max_experts = expected_max_experts_per_gpu(cfg.top_k, ep)
        imbalance = max_experts / (cfg.top_k / ep)
        per_gpu_attn = attn_bytes / tp
        per_gpu_expert = max_experts * cfg.one_expert_params * dtype_bytes
        per_gpu_router = router_bytes  # router replicated, it's tiny
        per_gpu_layer = per_gpu_attn + per_gpu_expert + per_gpu_router
        weight_read = (per_gpu_layer * cfg.n_layers + lm_head_bytes / n) / gpu.hbm_bw
        # comms: all-to-all dispatch + combine for experts, plus attn all-reduce
        comms = (2 + 1) * cfg.n_layers * gpu.collective_latency_s
    else:
        raise ValueError(f"unknown plan {plan!r}")

    # --- KV cache reads (attention) ---
    kv_per_token = cfg.kv_bytes_per_token(kv_dtype_bytes)
    kv_shards = min(tp, cfg.n_kv_heads) if plan != "floor" else n
    kv_total = seq_len * kv_per_token / kv_shards
    kv_read = kv_total / gpu.hbm_bw

    # --- compute (shown to prove it's negligible) ---
    flops = 2 * cfg.active_params  # 1 token, fwd MAC ~ 2*params
    flops += 4 * cfg.n_heads * cfg.head_dim * seq_len  # attention scores+values
    peak = gpu.bf16_flops if dtype_bytes >= 2 else gpu.fp8_flops
    compute = flops / (peak * (n if plan != "floor" else n))

    return DecodeBreakdown(
        plan=plan, dtype_bytes=dtype_bytes, seq_len=seq_len,
        weight_read_s=weight_read, kv_read_s=kv_read,
        compute_s=compute, comms_s=comms, imbalance=imbalance,
    )
