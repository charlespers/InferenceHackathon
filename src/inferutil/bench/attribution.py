"""Bottleneck attribution — which term sets the per-token decode latency.

Replaces the hand-tuned 0.7/0.5 thresholds in `bench/roofline.py` with a
principled decomposition. We split the *actual* per-token time into:

    weight_bw  + kv_bw + comms + compute      (the analytical floor terms)
  + kernel_gap                                (time lost below the floor:
                                               real kernels / launch / dequant)

The dominant term names the bottleneck; `levers.recommend` maps it to the next
optimization. `kernel_gap` is the share of time the run spends *above* the
analytical floor — when it dominates, the win is in the implementation
(CUDA graphs, fused dequant) rather than in the algorithm or quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# term -> short lever hint (one line; full ranking lives in levers.py)
_HINTS = {
    "weight_bw": "weights dominate HBM reads -> quantize harder (fp8->int4 experts)",
    "kv_bw": "KV cache dominates -> fp8/int8 KV, prefix reuse, shorter context",
    "comms": "collective/launch latency -> TP-heavier layout, CUDA graphs, fused all-to-all",
    "compute": "compute-bound (unusual at B=1) -> check dequant isn't on the math path",
    "kernel_gap": "running below the analytical floor -> CUDA graphs, fused dequant, kernel tuning",
}


@dataclass(frozen=True)
class Bottleneck:
    dominant_term: str
    share: float                  # dominant fraction of actual per-token time
    second_term: str
    second_share: float
    confidence: float             # dominant/(dominant+second) in [0.5, 1.0]
    headroom_to_floor: float      # 1 - pct_of_floor (0 if at/above floor)
    regime: str
    ai_decode: Optional[float]
    ridge: Optional[float]
    note: str


def _shape(result) -> dict:
    """Proportions of the floor work across terms (sum to 1.0). Prefer the
    measured breakdown's shape; otherwise split weight vs KV by the byte share."""
    mb = result.measured_breakdown
    if mb is not None:
        raw = {"weight_bw": mb.weight_s, "kv_bw": mb.kv_s,
               "comms": mb.comms_s, "compute": mb.compute_s}
        tot = sum(raw.values())
        if tot > 0:
            return {k: v / tot for k, v in raw.items()}
    kv_share = (result.efficiency.kv_byte_share if result.efficiency else None) or 0.0
    return {"weight_bw": 1.0 - kv_share, "kv_bw": kv_share,
            "comms": 0.0, "compute": 0.0}


def diagnose(result) -> Bottleneck:
    """Diagnose the dominant decode bottleneck for one BenchResult.

    Splits the *actual* per-token time into floor terms (weight/kv/comms/compute,
    apportioned by the measured shape) plus `kernel_gap` (time spent above the
    analytical floor). The breakdown reports actual per-term times that sum to
    TPOT, so we re-derive the floor from `analytical_floor_tok_per_s` and add the
    gap exactly once — no double counting.
    """
    floor_tok_s = result.analytical_floor_tok_per_s or 0.0
    actual_tok_s = result.decode_tok_per_s or 0.0
    pct = result.pct_of_floor or 0.0
    floor_s = (1.0 / floor_tok_s) if floor_tok_s else 0.0
    actual_s = (1.0 / actual_tok_s) if actual_tok_s else floor_s
    gap_s = max(0.0, actual_s - floor_s)
    parts = {term: floor_s * frac for term, frac in _shape(result).items()}
    parts["kernel_gap"] = gap_s

    ranked = sorted(parts.items(), key=lambda kv: kv[1], reverse=True)
    (d_term, d_val), (s_term, s_val) = ranked[0], ranked[1]
    denom = sum(parts.values()) or 1.0
    pair = (d_val + s_val) or 1.0
    eff = result.efficiency
    return Bottleneck(
        dominant_term=d_term, share=d_val / denom,
        second_term=s_term, second_share=s_val / denom,
        confidence=d_val / pair,
        headroom_to_floor=max(0.0, 1.0 - pct),
        regime=(eff.regime_decode if eff else "unknown"),
        ai_decode=(eff.ai_decode if eff else None),
        ridge=(eff.roofline_ridge if eff else None),
        note=_HINTS.get(d_term, d_term))
