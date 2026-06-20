"""Derive the B=1 optimization telemetry the console should show, from the measured per-token latency.

Turns a raw `tpot_ms` (+ config) into the `x_summary` fields `docs/console-telemetry-spec.md` calls for:
the **floor breakdown** (weight / comms / kv / overhead ms), the **regime** (floor-bound vs weight-bound →
"what to pull next"), and the **ceiling/roofline %** (how close to physics). Both `mock_engine` (synthesize
for the GPU-free demo) and `VLLMBackend` (from the live measured TPOT) can call `summary_fields(...)`.

Grounded in this session's measured/computed constants (8×H100, Qwen3-235B-A22B, 22B active, 94 layers,
GQA-4): weight rooflines `absolute-ceiling.md`; comms = 2·94 all-reduces × measured 16µs `overhead-attribution.md`;
ceiling ~2000 tok/s (fp8+spec). Self-contained (no engine import) so it can't drift with the model crate.
"""
from __future__ import annotations

# e=1 weight-read (ms) for 22B active over 8×3.35TB/s, by precision (absolute-ceiling.md rooflines).
WEIGHT_MS = {"bf16": 1.61, "fp8": 0.80, "int4": 0.40}
ROOFLINE_TOK_S = {"bf16": 618.0, "fp8": 1236.0, "int4": 2457.0}   # = 1000/WEIGHT_MS
CEILING_TOK_S = 2000.0          # fp8 + optimal spec (absolute-ceiling.md)
N_COLLECTIVES = 188             # 2 all-reduces/layer × 94 layers (TP); the per-step serial chain
KV_BYTES = {"bf16": 2, "fp8": 1}
# KV read per token: 2(k+v)·4 kv-heads·128 head_dim·94 layers = 192 KB/token (bf16), streamed over 8 GPUs.
_KV_BYTES_PER_TOKEN = 2 * 4 * 128 * 94
_AGG_BW = 8 * 3.35e12           # 8×H100 usable HBM BW (bytes/s)


def kv_ms(ctx: int, kv_dtype: str = "bf16") -> float:
    by = KV_BYTES.get(kv_dtype, 2)
    return (_KV_BYTES_PER_TOKEN * by * max(ctx, 0)) / _AGG_BW * 1e3


def floor_breakdown(tpot_ms: float, weight_dtype: str = "bf16", kv_dtype: str = "bf16",
                    collective_us: float = 16.0, ctx: int = 512) -> dict:
    """Split a measured per-token latency into weight / comms / kv / overhead (ms). 'overhead' = the residual
    (kernel inefficiency e<1 + launch + host/scheduler) — the dominant floor term while floor-bound."""
    weight = WEIGHT_MS.get(weight_dtype, WEIGHT_MS["bf16"])
    comms = N_COLLECTIVES * collective_us / 1e3
    kv = kv_ms(ctx, kv_dtype)
    overhead = max(tpot_ms - weight - comms - kv, 0.0)
    return {"weight": round(weight, 3), "comms": round(comms, 3), "kv": round(kv, 3),
            "overhead": round(overhead, 3)}


def summary_fields(tpot_ms: float, decode_tok_s: float, weight_dtype: str = "bf16",
                   kv_dtype: str = "bf16", collective_us: float = 16.0, ctx: int = 512) -> dict:
    """The x_summary optimization block: floor_breakdown_ms + regime + pct_of_ceiling + pct_of_roofline."""
    bd = floor_breakdown(tpot_ms, weight_dtype, kv_dtype, collective_us, ctx)
    floor = bd["comms"] + bd["overhead"] + bd["kv"]
    regime = "floor-bound" if floor >= bd["weight"] else "weight-bound"
    roof = ROOFLINE_TOK_S.get(weight_dtype, ROOFLINE_TOK_S["bf16"])
    return {
        "floor_breakdown_ms": bd,
        "regime": regime,
        "next_lever": ("spec + comms + kernels (fix the floor)" if regime == "floor-bound"
                       else "quant / route-aware (weight now pays)"),
        "pct_of_ceiling": round(100.0 * decode_tok_s / CEILING_TOK_S, 1),
        "pct_of_roofline": round(100.0 * decode_tok_s / roof, 1),
    }
