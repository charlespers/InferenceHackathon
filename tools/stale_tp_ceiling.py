#!/usr/bin/env python3
"""
stale_tp_ceiling.py — LOOP-C (speculative/stale tensor parallelism)

Offline (NO GPU) ceiling model for the ONE thing stale-TP changes at B=1:
it breaks the serial dependency on the per-layer all-reduce, so the AR of layer L
can be HIDDEN behind the weight-read of layer L+1 (NVLink engine vs HBM engine run
concurrently). comms_floor.md §3 showed this overlap is infeasible *losslessly* at B=1
because the GEMM can't start until the AR lands; stale-TP removes that wait by feeding
layer L+1 a stale/predicted activation.

This model answers: IF quality holds (the separate, GPU-gated question), what is the
performance ceiling, and does it stack with the multimem-one-shot AR lever (comms_floor §2)?

Numbers are taken from research/comms_floor.md and docs/DESIGN.md (the team's own model)
so this is directly comparable to their tables. Pure stdlib.

Usage:
    python tools/stale_tp_ceiling.py
    python tools/stale_tp_ceiling.py --weight-read-us 16.6 --kv-ms 0.10
"""
import argparse

# --- team model constants (research/comms_floor.md, docs/DESIGN.md) ---
N_LAYERS = 94
COLLECTIVES_PER_LAYER = 2          # post-attn AR + post-MoE AR (TP=8)
WEIGHT_READ_US_PER_LAYER = 16.6    # active weights /8 per layer @ B=1 (1.56 ms/token / 94)
KV_MS = 0.10                       # negligible at short ctx (comms_floor uses ~KV negligible)
# per-collective latency scenarios (us) from comms_floor §2
C_SCENARIOS = {
    "16us baseline (eager/un-tuned)": 16.0,
    "10us (CUDA-graph, default custom AR)": 10.0,
    "7us  (multimem one-shot + LL, lever 2)": 7.0,
    "5us  (best case)": 5.0,
}


def model(c_us, weight_read_us, kv_ms, stale=False):
    """Return (comms_ms, total_ms, tok_s) for per-collective latency c_us."""
    weight_ms = weight_read_us * N_LAYERS / 1000.0
    comms_per_layer_us = COLLECTIVES_PER_LAYER * c_us
    if not stale:
        exposed_us = comms_per_layer_us                      # fully serial (blocks)
    else:
        # stale-TP: AR(L) overlaps weight-read(L+1). Different HW engines (NVLink vs HBM)
        # run concurrently, so per layer only the part of comms exceeding the weight read
        # stays on the critical path. (Conservative: one layer of weight-read as cover.)
        exposed_us = max(0.0, comms_per_layer_us - weight_read_us)
    comms_ms = exposed_us * N_LAYERS / 1000.0
    total_ms = weight_ms + comms_ms + kv_ms
    return comms_ms, total_ms, 1000.0 / total_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-read-us", type=float, default=WEIGHT_READ_US_PER_LAYER)
    ap.add_argument("--kv-ms", type=float, default=KV_MS)
    args = ap.parse_args()

    wr, kv = args.weight_read_us, args.kv_ms
    weight_ms = wr * N_LAYERS / 1000.0
    print(f"\nstale-TP overlap CEILING (B=1, TP=8, {N_LAYERS} layers)")
    print(f"  weight-read = {wr:.1f} us/layer = {weight_ms:.2f} ms/token (the overlap 'cover')")
    print(f"  KV = {kv:.2f} ms ; {COLLECTIVES_PER_LAYER} collectives/layer\n")
    hdr = f"{'per-collective C':<40} {'comms ms':>9} {'tok/s':>8}   {'+stale comms ms':>15} {'+stale tok/s':>12}   {'stale gain':>10}"
    print(hdr); print("-" * len(hdr))
    for name, c in C_SCENARIOS.items():
        cb, tb, sb = model(c, wr, kv, stale=False)
        cs, ts, ss = model(c, wr, kv, stale=True)
        gain = ss / sb
        flag = "  <- comms FULLY hidden" if cs <= 1e-6 else ""
        print(f"{name:<40} {cb:>9.2f} {sb:>8.0f}   {cs:>15.2f} {ss:>12.0f}   {gain:>9.2f}x{flag}")

    print(f"\nReadout:")
    print(f"  * Stale-TP's marginal value GROWS as the per-collective constant shrinks.")
    print(f"    At C=16us it trims comms ~2x; at C<=8.3us (=weight-read/2) the ENTIRE comms")
    print(f"    term hides behind weight reads -> decode hits the ~weight+KV roofline.")
    print(f"  * => Stale-TP STACKS with the multimem one-shot lever (comms_floor §2): lever 2")
    print(f"    alone exposes ~1.3 ms comms (~340 tok/s); lever 2 + stale-TP overlap -> ~0")
    print(f"    exposed comms (~roofline). Stale-TP converts 'cheaper comms' into 'free comms'.")
    print(f"  * ALL of this is GATED ON QUALITY (no-retrain staleness tolerance). Literature")
    print(f"    (Ladder-Residual ICML'25, Kog DTP) says quality recovery needs retraining;")
    print(f"    the GPU staleness-probe (research/n4_speculative_stale_tp.md §4) is the go/no-go.")
    print()


if __name__ == "__main__":
    main()
