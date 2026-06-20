"""Analytical prefill (prompt-processing) latency model.

Decode is memory-bound (one token rereads the active weights); prefill is the
opposite. It processes all P prompt tokens in one batched pass, so the weights
are read *once* and amortized over P tokens — for a long-enough prompt that makes
prefill **compute-bound**, which is the regime where MFU is the meaningful metric
(see efficiency.py / the PaLM definition).

Roofline prefill time:

    weight_read = prefill_weight_bytes / aggregate_bw     (read once for the batch)
    compute     = 2 * N_active * P / aggregate_peak_FLOPS
    kv_write    = P * kv_bytes_per_token / aggregate_bw
    prefill_s   = max(weight_read, compute) + kv_write

Two refinements over a naive "2N*P / peak":
  - Prefill activates *many* experts (each of the P tokens routes to its own
    top-k), so unlike decode it does NOT read just the active experts. We model
    the expected number of distinct experts touched and read that fraction of the
    expert weights — for P >> n_experts/top_k this approaches all experts.
  - We take max(compute, weight_read): short prompts are weight-read bound (few
    tokens to amortize over), long prompts are compute bound.
"""

from __future__ import annotations

from ..model import MoEConfig
from ..hardware import Cluster
from .efficiency import peak_flops_for_dtype


def expected_distinct_experts(n_experts: int, top_k: int, prompt_tokens: int) -> float:
    """E[distinct experts touched per layer] when P tokens each pick top_k of
    n_experts. Approaches n_experts as P grows."""
    if prompt_tokens <= 0 or n_experts <= 0:
        return 0.0
    miss = (n_experts - top_k) / n_experts          # P(one token skips a given expert)
    return n_experts * (1.0 - miss ** prompt_tokens)


def prefill_weight_bytes(cfg: MoEConfig, dtype_bytes: float,
                         prompt_tokens: int) -> float:
    """Bytes read once during prefill: all non-expert weights + the fraction of
    expert weights the prompt actually activates."""
    all_expert_params = cfg.n_layers * cfg.n_experts * cfg.one_expert_params
    nonexpert_params = cfg.total_params - all_expert_params
    distinct = expected_distinct_experts(cfg.n_experts, cfg.top_k, prompt_tokens)
    expert_params = cfg.n_layers * distinct * cfg.one_expert_params
    return (nonexpert_params + expert_params) * dtype_bytes


def prefill_latency(cfg: MoEConfig, cluster: Cluster, *, dtype_bytes: float,
                    kv_dtype_bytes: int, prompt_tokens: int) -> float:
    """Seconds to process a `prompt_tokens`-long prompt in one batched pass."""
    if prompt_tokens <= 0:
        return 0.0
    agg_bw = cluster.aggregate_hbm_bw
    peak = peak_flops_for_dtype(
        cluster.gpu.bf16_flops, cluster.gpu.fp8_flops, dtype_bytes) * cluster.n_gpus
    weight_read = prefill_weight_bytes(cfg, dtype_bytes, prompt_tokens) / agg_bw
    compute = (2.0 * cfg.active_params * prompt_tokens) / peak
    kv_write = (prompt_tokens * cfg.kv_bytes_per_token(kv_dtype_bytes)) / agg_bw
    return max(weight_read, compute) + kv_write
