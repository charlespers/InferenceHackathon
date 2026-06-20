"""Efficiency math — the single source of truth for MFU / MBU / roofline.

Clean-room reimplementation of public metric definitions:
  - MFU (Model FLOPs Utilization)  — PaLM paper (Chowdhery et al., 2022)
  - MBU (Model Bandwidth Utilization) — Databricks LLM-inference blog
  - Roofline / arithmetic intensity — Williams, Waterman & Patterson (2009)

Two regimes for autoregressive transformer inference:
  - PREFILL is a single batched pass over the prompt -> compute-bound -> MFU.
  - DECODE is one token at a time -> each step rereads the (quantized) weights
    plus the KV cache once -> memory-bandwidth bound -> MBU.

The dominant matmul cost is ~2 FLOPs per active parameter per token (one
multiply + one add). Attention adds an O(seq) term that is second-order at our
batch=1 / fixed-window regime; it is available via `include_attention=True` but
off by default (matches the 2N convention).

All functions take raw numbers (no model/hardware imports) so this module stays
pure and trivially testable. `compute_efficiency` packages the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# --------------------------------------------------------------------------- #
# Pure scalar formulas                                                         #
# --------------------------------------------------------------------------- #
def flops_per_token(active_params: int, *, seq_len: int = 0, n_layers: int = 0,
                    d_model: int = 0, include_attention: bool = False) -> float:
    """Forward-pass FLOPs to process one token. 2N for the matmuls; the optional
    attention term is 2 * n_layers * seq_len * d_model (off by default)."""
    base = 2.0 * float(active_params)
    if include_attention and seq_len and n_layers and d_model:
        base += 2.0 * n_layers * seq_len * d_model
    return base


def mfu(tok_s: Optional[float], active_params: int,
        peak_flops: float) -> Optional[float]:
    """Model FLOPs Utilization in [0, 1] for a throughput (tok/s)."""
    if not peak_flops or tok_s is None or not active_params:
        return None
    return (flops_per_token(active_params) * tok_s) / peak_flops


def mbu(tok_s: Optional[float], bytes_per_token: Optional[float],
        peak_bw: float) -> Optional[float]:
    """Model Bandwidth Utilization in [0, 1]. bytes_per_token = weights + KV
    actually streamed per decoded token; peak_bw in bytes/s."""
    if not peak_bw or tok_s is None or bytes_per_token is None:
        return None
    return (bytes_per_token * tok_s) / peak_bw


def achieved_tflops(tok_s: Optional[float], active_params: int) -> Optional[float]:
    """Actual TFLOP/s sustained at this throughput (2N * tok/s)."""
    if tok_s is None or not active_params:
        return None
    return flops_per_token(active_params) * tok_s / 1e12


def arithmetic_intensity_decode(active_params: int,
                                bytes_per_token: Optional[float]) -> Optional[float]:
    """FLOPs per byte during decode (one token at a time) — low => memory-bound."""
    if not bytes_per_token:
        return None
    return flops_per_token(active_params) / bytes_per_token


def arithmetic_intensity_prefill(active_params: int, weight_bytes: Optional[float],
                                 prompt_tokens: int) -> Optional[float]:
    """FLOPs per byte during prefill. Weights are read once for the whole prompt,
    so intensity is prompt_tokens-times higher than decode -> compute-bound."""
    if not weight_bytes:
        return None
    return flops_per_token(active_params) * prompt_tokens / weight_bytes


def roofline_ridge(peak_flops: float, peak_bw: float) -> Optional[float]:
    """FLOPs/byte at the roofline knee = peak_FLOPS / peak_bandwidth. Workloads
    below this intensity are memory-bound; above it, compute-bound."""
    if not peak_bw:
        return None
    return peak_flops / peak_bw


def classify_regime(ai: Optional[float], ridge: Optional[float]) -> str:
    if ai is None or ridge is None:
        return "unknown"
    return "memory-bound" if ai < ridge else "compute-bound"


def peak_flops_for_dtype(bf16_flops: float, fp8_flops: float,
                         dtype_bytes: float) -> float:
    """Tensor-core peak for the compute dtype. <2 bytes => fp8 path."""
    return fp8_flops if dtype_bytes < 2 else bf16_flops


# --------------------------------------------------------------------------- #
# Packaged result                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Efficiency:
    mfu_prefill: Optional[float]
    mfu_decode: Optional[float]
    mbu_decode: Optional[float]
    achieved_tflops_prefill: Optional[float]
    achieved_tflops_decode: Optional[float]
    ai_decode: Optional[float]
    ai_prefill: Optional[float]
    roofline_ridge: Optional[float]
    regime_decode: str = "unknown"
    regime_prefill: str = "unknown"
    kv_byte_share: Optional[float] = None


def compute_efficiency(*, active_params: int, weight_bytes: float,
                       bytes_per_token: float, kv_bytes: float,
                       prompt_tokens: int, prefill_tok_s: Optional[float],
                       decode_tok_s: Optional[float], peak_flops: float,
                       peak_bw: float) -> Efficiency:
    """Bundle MFU/MBU/AI/ridge/regime from one run's raw numbers.

    peak_flops / peak_bw are AGGREGATE (per-GPU * n_gpus) so utilization is
    against the whole node. bytes_per_token already includes KV; kv_bytes is the
    KV portion (for kv_byte_share).
    """
    ai_d = arithmetic_intensity_decode(active_params, bytes_per_token)
    ai_p = arithmetic_intensity_prefill(active_params, weight_bytes, prompt_tokens)
    ridge = roofline_ridge(peak_flops, peak_bw)
    kv_share = (kv_bytes / bytes_per_token) if bytes_per_token else None
    return Efficiency(
        mfu_prefill=mfu(prefill_tok_s, active_params, peak_flops),
        mfu_decode=mfu(decode_tok_s, active_params, peak_flops),
        mbu_decode=mbu(decode_tok_s, bytes_per_token, peak_bw),
        achieved_tflops_prefill=achieved_tflops(prefill_tok_s, active_params),
        achieved_tflops_decode=achieved_tflops(decode_tok_s, active_params),
        ai_decode=ai_d, ai_prefill=ai_p, roofline_ridge=ridge,
        regime_decode=classify_regime(ai_d, ridge),
        regime_prefill=classify_regime(ai_p, ridge),
        kv_byte_share=kv_share)
