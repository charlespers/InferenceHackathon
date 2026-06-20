#!/usr/bin/env python3
"""Roofline + dominant-term analysis for Qwen3-235B-A22B B=1 decode on 8xH100.

Computes the bytes-moved-per-token budget and the memory-bandwidth ceiling, and — given a
measured TPOT — reports the achieved fraction and which term most likely dominates.

Public model facts only.  Usage:
    python bench/roofline.py --ctx 32768 --weight-bytes 1 --kv-bytes 1 [--tpot-ms 2.3]
"""
import argparse

# Verified Qwen3-235B-A22B facts
ACTIVE_PARAMS = 21.57e9          # per-token active params (attn+router ~6.8B, experts ~14.2B, lm_head 0.62B)
KV_ELEMS_PER_TOKEN = 96_256      # 94 layers * 4 KV heads * 128 * 2 (K&V)
N_GPU = 8
HBM_BW = 3.35e12                 # bytes/s per H100 (HBM3)
AGG_BW = HBM_BW * N_GPU          # 26.8 TB/s


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
    achieved = (1000.0 / tpot_ms) / ceil
    read_ms = total / AGG_BW * 1000.0
    overhead_ms = max(0.0, tpot_ms - read_ms)
    # Heuristic dominant term:
    if achieved >= 0.7:
        term = "memory-bandwidth (near roofline) -> quantize harder (fp8->int4 experts) or add speculation"
    elif kv > w * 0.5:
        term = "KV-bandwidth -> fp8/int8 KV, prefix reuse, shorter ctx"
    elif overhead_ms > read_ms * 0.5:
        term = "comms/launch overhead -> TP-heavier layout, low-latency all-to-all, CUDA graph"
    else:
        term = "weight-bandwidth -> quantize weights, verify dequant not compute-bound"
    return dict(weight_GB=w/1e9, kv_GB=kv/1e9, total_GB=total/1e9,
                roofline_tok_s=ceil, ideal_TPOT_ms=read_ms,
                achieved_frac=achieved, overhead_ms=overhead_ms, dominant_term=term)


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
