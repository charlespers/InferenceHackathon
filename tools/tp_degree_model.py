#!/usr/bin/env python3
"""
tp_degree_model.py — LOOP-C: is running B=1 on FEWER GPUs (lower TP) a latency win?

The last unclaimed structural comms lever. At B=1 a single stream on TP=N:
  - weight-read time SCALES AS 1/N  (N GPUs each read 1/N of the active weights in
    parallel) -> fewer GPUs = SLOWER weight reads.
  - per-collective latency C(N) depends on the algorithm:
      * one-shot / NVLS in-switch (what vLLM uses at 8KB): ~CONSTANT in N
        (broadcast + local reduce; NVSwitch does the fan-in) -> lowering N saves ~nothing.
      * ring: ~ (N-1) steps -> lowering N cuts comms ~linearly.
  - collective COUNT is 2/layer regardless of N.

So TP-reduction only wins if the comms saving (algorithm-dependent) beats the weight-read
penalty (always 1/N). This model quantifies the trade at fp8 on 8xH100 and finds the
break-even. Pure stdlib, no GPU. Anchors: docs/DESIGN.md + research/comms_floor.md.

Usage: python tools/tp_degree_model.py
"""
N_LAYERS = 94
COLL_PER_LAYER = 2
WEIGHT_MS_TP8_FP8 = 0.78      # active-weight read at TP=8, fp8 (DESIGN.md / ceiling tool)
KV_MS = 0.10
C8_US = 16.0                  # measured 8-way one-shot AR (comms_floor.md / Charles)


def weight_ms(tp):
    return WEIGHT_MS_TP8_FP8 * 8.0 / tp     # 1/N scaling (fewer GPUs share the read)


def comms_ms(tp, model):
    if model == "oneshot":         # ~constant in N (NVSwitch fan-in); vLLM's actual path
        c = C8_US
    elif model == "ring":          # ~(N-1) steps; C8 anchors N=8 -> per-step = C8/7
        c = C8_US * (tp - 1) / 7.0
    elif model == "log":           # tree/NVLS depth ~ log2(N)
        import math
        c = C8_US * math.log2(tp) / 3.0     # log2(8)=3 anchors N=8
    return COLL_PER_LAYER * N_LAYERS * c / 1000.0


def total(tp, model):
    return weight_ms(tp) + comms_ms(tp, model) + KV_MS


def main():
    print(f"\nTP-degree trade at B=1, fp8, 8xH100 ({N_LAYERS} layers, 2 coll/layer)")
    print(f"  weight-read scales 1/TP (TP8={WEIGHT_MS_TP8_FP8}ms); collective C8={C8_US}us\n")
    for model in ("oneshot", "ring", "log"):
        print(f"--- collective model: {model} "
              f"({'vLLM actual path' if model=='oneshot' else 'hypothetical'}) ---")
        hdr = f"  {'TP':>3} {'weight ms':>10} {'comms ms':>9} {'total ms':>9} {'tok/s':>7}"
        print(hdr)
        best = None
        for tp in (8, 4, 2):
            w, c, t = weight_ms(tp), comms_ms(tp, model), total(tp, model)
            ts = 1000.0 / t
            print(f"  {tp:>3} {w:>10.2f} {c:>9.2f} {t:>9.2f} {ts:>7.0f}")
            if best is None or t < best[1]:
                best = (tp, t)
        print(f"  -> best TP = {best[0]} ({1000.0/best[1]:.0f} tok/s)\n")

    print("Readout:")
    print("  * one-shot/NVLS (vLLM's real 8KB path): comms ~constant in N, so dropping to")
    print("    TP4 only DOUBLES the weight read for ~no comms saving -> TP8 wins. KILL.")
    print("  * ring: comms falls ~linearly, so TP4 can win IF the box actually uses ring")
    print("    at 8KB -- but it doesn't (custom one-shot AR is on by default, <256KB cutoff).")
    print("  * Verdict: with the collective vLLM actually runs, TP-degree reduction LOSES at")
    print("    B=1 (same weight-read-penalty reason DP-attn loses, comms_floor.md §1).")
    print("  * Cheap confirm if ever in doubt: measure 4-way vs 8-way one-shot AR latency")
    print("    (Charles's measure_collective.sh) -- if C(4) ~= C(8), the model's KILL holds.\n")


if __name__ == "__main__":
    main()
