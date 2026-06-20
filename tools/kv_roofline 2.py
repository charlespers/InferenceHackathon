#!/usr/bin/env python3
"""Roofline predictor for the KV-cache FP8 decode win on Qwen3-235B-A22B (B=1).

At B=1 decode each step is HBM-bandwidth bound: it re-reads the active weights
once plus the whole KV cache (which scales with context length L). FP8 KV halves
the KV bytes. The *fraction* of per-token bytes that is KV decides how much the
halving helps — and Qwen3-235B uses aggressive GQA (4 KV heads vs 64 query heads,
94 layers), so its KV cache is SMALL relative to the 22B-active FP8 weight read.

This computes the bandwidth-bound bytes/token for kv=auto(bf16) vs fp8(e4m3) across
context lengths, the implied best-case TPOT improvement, and flags the caveat that
real TPOT also carries TP/MoE comms overhead that dilutes the bandwidth win.

    python3 tools/kv_roofline.py            # default Qwen3-235B-A22B on 8xH100
"""
import argparse

# --- Qwen3-235B-A22B-Instruct-2507-FP8 (from HF config.json on the box) ---
N_LAYERS = 94
N_KV_HEADS = 4          # GQA: 4 KV heads vs 64 query heads (16:1)
HEAD_DIM = 128
ACTIVE_PARAMS = 22e9    # ~22B active params per token (MoE A22B)
WEIGHT_BYTES_PER_PARAM = 1.0   # FP8 weights

# --- 8x H100 SXM ---
HBM_BW_PER_GPU = 3.35e12        # bytes/s
N_GPU = 8
BW_AGG = HBM_BW_PER_GPU * N_GPU  # tensor-parallel aggregate read bandwidth


def kv_bytes_per_token(kv_dtype_bytes: float) -> float:
    # K and V, all layers, GQA KV heads only
    return 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * kv_dtype_bytes


def predict(ctx_lengths, mbu=1.0):
    """mbu = model-bandwidth-utilisation fudge (1.0 = pure roofline). The KV win
    is invariant to mbu in % terms, but absolute ms scale with 1/mbu."""
    w = ACTIVE_PARAMS * WEIGHT_BYTES_PER_PARAM
    kv16 = kv_bytes_per_token(2.0)   # bf16 / auto
    kv8 = kv_bytes_per_token(1.0)    # fp8 e4m3
    print(f"Qwen3-235B-A22B  weight read/tok (fp8) = {w/1e9:.1f} GB")
    print(f"KV bytes/token: auto(bf16)={kv16/1024:.1f} KB  fp8={kv8/1024:.1f} KB  "
          f"(GQA {N_KV_HEADS} KV heads x {N_LAYERS} layers)")
    print(f"Aggregate HBM BW (8xH100) = {BW_AGG/1e12:.1f} TB/s, MBU={mbu}\n")
    print(f"{'ctx':>7} | {'KV/wt %':>8} | {'TPOT auto':>10} | {'TPOT fp8':>9} | "
          f"{'win %':>6} | note")
    print("-" * 72)
    for L in ctx_lengths:
        bytes_auto = w + kv16 * L
        bytes_fp8 = w + kv8 * L
        t_auto = bytes_auto / (BW_AGG * mbu) * 1e3   # ms
        t_fp8 = bytes_fp8 / (BW_AGG * mbu) * 1e3
        kv_frac = (kv16 * L) / bytes_auto * 100
        win = (1 - t_fp8 / t_auto) * 100
        note = "negligible" if win < 1 else ("modest" if win < 5 else "clear")
        print(f"{L:>7} | {kv_frac:>7.1f}% | {t_auto:>8.3f}ms | {t_fp8:>7.3f}ms | "
              f"{win:>5.1f}% | {note}")
    print()
    print("CAVEAT: this is the bandwidth-bound CEILING. Real B=1 TPOT on this model is")
    print("dominated by TP all-reduce + MoE all-to-all comms (CUDA graphs help but don't")
    print("erase it), so measured TPOT >> roofline and the KV win as a fraction of WALL")
    print("TPOT is SMALLER than the % above. Expect the latency win to be marginal until")
    print("long ctx; the bigger, more certain payoff is MEMORY (half KV footprint -> longer")
    print("context fits, or headroom for the other levers). The A/B measures which it is.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="*",
                    default=[128, 2048, 8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--mbu", type=float, default=1.0)
    a = ap.parse_args()
    predict(a.ctx, a.mbu)
