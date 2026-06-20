#!/usr/bin/env python3
"""Roofline + MFU/MBU + principled bottleneck attribution for Qwen3-235B-A22B
B=1 decode on 8xH100.

Standalone (public model facts only, no inferutil import) so it runs on the bare
remote box. Mirrors the math in src/inferutil/bench/{efficiency,attribution}.py:

  - MFU = 2N * tok/s / peak_FLOPS          (PaLM)
  - MBU = bytes/token * tok/s / peak_BW    (Databricks)
  - regime via the roofline ridge = peak_FLOPS / peak_BW
  - dominant term = argmax(weight_ms, kv_ms, kernel_gap_ms)   (NOT a threshold cascade)

Usage:
    python bench/roofline.py --ctx 4096 --weight-bytes 2 --kv-bytes 2 [--tpot-ms 2.3]
"""
import argparse

# Verified Qwen3-235B-A22B facts
ACTIVE_PARAMS = 21.57e9          # per-token active params (attn+router ~6.8B, experts ~14.2B, lm_head 0.62B)
KV_ELEMS_PER_TOKEN = 96_256      # 94 layers * 4 KV heads * 128 * 2 (K&V)
N_GPU = 8
HBM_BW = 3.35e12                 # bytes/s per H100 (HBM3)
AGG_BW = HBM_BW * N_GPU          # 26.8 TB/s
BF16_FLOPS = 989.4e12            # dense bf16 tensor-core FLOP/s per H100
FP8_FLOPS = 1978.9e12            # dense fp8 tensor-core FLOP/s per H100

FLOPS_PER_TOKEN = 2.0 * ACTIVE_PARAMS

_HINTS = {
    "weight_bw": "weights dominate -> quantize harder (fp8->int4 experts)",
    "kv_bw": "KV dominates -> fp8/int8 KV, prefix reuse, shorter ctx",
    "kernel_gap": "below the floor -> CUDA graphs, fused dequant, kernel tuning",
}


def peak_flops(weight_bytes):
    return (BF16_FLOPS if weight_bytes >= 2 else FP8_FLOPS) * N_GPU


def budget(ctx, weight_bytes=1, kv_bytes=1):
    w = ACTIVE_PARAMS * weight_bytes
    kv = KV_ELEMS_PER_TOKEN * ctx * kv_bytes
    return w, kv, w + kv


def roofline_tok_s(ctx, weight_bytes=1, kv_bytes=1):
    _, _, total = budget(ctx, weight_bytes, kv_bytes)
    return AGG_BW / total


def analyze(ctx, tpot_ms, weight_bytes=1, kv_bytes=1):
    w, kv, total = budget(ctx, weight_bytes, kv_bytes)
    ceil = roofline_tok_s(ctx, weight_bytes, kv_bytes)
    tok_s = 1000.0 / tpot_ms
    achieved = tok_s / ceil
    read_ms = total / AGG_BW * 1000.0
    weight_ms = w / AGG_BW * 1000.0
    kv_ms = kv / AGG_BW * 1000.0
    overhead_ms = max(0.0, tpot_ms - read_ms)         # time spent below the floor
    pf = peak_flops(weight_bytes)
    ridge = pf / AGG_BW
    ai = FLOPS_PER_TOKEN / total
    # Principled attribution: largest actual time term wins (no 0.7/0.5 cascade).
    terms = {"weight_bw": weight_ms, "kv_bw": kv_ms, "kernel_gap": overhead_ms}
    dom = max(terms, key=terms.get)
    return dict(
        weight_GB=w / 1e9, kv_GB=kv / 1e9, total_GB=total / 1e9,
        roofline_tok_s=ceil, ideal_TPOT_ms=read_ms, achieved_frac=achieved,
        mfu=FLOPS_PER_TOKEN * tok_s / pf, mbu=total * tok_s / AGG_BW,
        arithmetic_intensity=ai, roofline_ridge=ridge,
        regime=("memory-bound" if ai < ridge else "compute-bound"),
        overhead_ms=overhead_ms, dominant_term=dom,
        recommendation=_HINTS[dom])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--weight-bytes", type=float, default=1.0, help="1=fp8, 0.5=int4, 2=bf16")
    ap.add_argument("--kv-bytes", type=float, default=1.0, help="1=fp8/int8, 2=fp16")
    ap.add_argument("--tpot-ms", type=float, default=None)
    a = ap.parse_args()
    w, kv, total = budget(a.ctx, a.weight_bytes, a.kv_bytes)
    print(f"ctx={a.ctx}  weights={w/1e9:.1f}GB  KV={kv/1e9:.2f}GB  total={total/1e9:.1f}GB/token")
    print(f"roofline = {roofline_tok_s(a.ctx, a.weight_bytes, a.kv_bytes):.0f} tok/s  "
          f"(ideal TPOT {total/AGG_BW*1000:.2f} ms)")
    if a.tpot_ms:
        for k, v in analyze(a.ctx, a.tpot_ms, a.weight_bytes, a.kv_bytes).items():
            print(f"  {k}: {v}")
