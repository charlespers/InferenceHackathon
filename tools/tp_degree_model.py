#!/usr/bin/env python3
"""
tp_degree_model.py — LOOP-C: is running B=1 on FEWER GPUs (lower TP) a latency win?

SCOPE — READ THIS FIRST. This models PURE TP-degree reduction: run the single stream on
N<8 GPUs and LEAVE THE REST IDLE (so each active GPU reads 1/N of the weights, 2x at N=4).
That degenerate variant is what's modeled/killed here.

This is NOT TP4xEP2 — a DIFFERENT layout that uses ALL 8 GPUs (TP=4 on attention/non-expert
x EP=2 on experts). TP4xEP2 does NOT idle GPUs and does NOT pay a flat 2x weight read; the
team analyzes it in depth in `docs/b1-tp8-moe-rearchitecture-h200.md` and considers it the
**day-0 default that WINS** in real regimes (ctx >= ~8K: +22% at 28K via KV-splitting; and
because TP8-FP8 is UNLAUNCHABLE on the stock block-128 checkpoint, vLLM #17569). **Nothing
here kills TP4xEP2** — that's the team's live layout decision (jminding/Charles), not mine.

At B=1 a single stream on PURE TP=N (idle the rest):
  - weight-read time SCALES AS 1/N  (N GPUs each read 1/N of the active weights in
    parallel) -> fewer ACTIVE GPUs = SLOWER weight reads.
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

    # END-STATE (the rigorous, engine-independent kill): once comms is HIDDEN behind
    # the weight read (NVLS + exact-overlap, the path-to-1000 plan), the weight read is
    # the SOLE bottleneck -- and TP-reduction doubles it.
    print("--- END-STATE: comms hidden (NVLS + exact-overlap, the team's plan) ---")
    print(f"  {'TP':>3} {'weight ms':>10} {'comms ms':>9} {'total ms':>9} {'tok/s':>7}")
    for tp in (8, 4):
        w = weight_ms(tp); t = w + KV_MS
        print(f"  {tp:>3} {w:>10.2f} {0.0:>9.2f} {t:>9.2f} {1000.0/t:>7.0f}")
    ratio = (weight_ms(4) + KV_MS) / (weight_ms(8) + KV_MS)
    print(f"  -> TP8 wins ~{ratio:.1f}x (comms gone; only weights matter, and TP4 doubles them)\n")

    print("Readout (the rigorous kill is the END-STATE argument):")
    print("  * THE KILL: in the target engine comms is HIDDEN (NVLS + exact-overlap), so the")
    print("    weight read is the only bottleneck -- TP4 DOUBLES it (0.78->1.56ms) -> TP8 wins")
    print("    ~2x. Engine-INDEPENDENT (pure HBM bandwidth / GPU count); holds for our own engine.")
    print("  * 'But we write our own engine, so we pick the collective' (fair): the latency-optimal")
    print("    8KB collective is NVLS in-switch, ~CONSTANT in N (the switch does the fan-in) -> fewer")
    print("    ranks save ~nothing on comms. And weight-shard CAN'T be decoupled from collective-width")
    print("    on one switch (an 8-rank reduce needs 8 ranks). Picking ring (scales with N) to favor")
    print("    TP4 = deliberately choosing a WORSE collective; NVLS@TP8 still beats it.")
    print("  * HONEST nuance: TODAY (comms-bound, 16us barrier) TP4 could TIE/WIN *iff* a 4-way barrier")
    print("    is <=~12us (unmeasured; see 'oneshot' table). But that's the regime we're LEAVING --")
    print("    optimizing it sabotages the destination. Verdict: NOT worth pursuing.")
    print("  * Contingency only: if the megakernel/NVLS path stalls and we stay comms-bound, THEN")
    print("    measure 4-way vs 8-way NVLS latency (measure_collective.sh) to revisit pure-TP4.")
    print("  * SCOPE: this kills PURE TP-reduction (idle GPUs). It does NOT touch TP4xEP2 (all 8")
    print("    GPUs, TP4-nonexpert x EP2-expert) -- the team's day-0 default that WINS at ctx>=8K")
    print("    and when TP8-FP8 won't launch. See docs/b1-tp8-moe-rearchitecture-h200.md.\n")


if __name__ == "__main__":
    main()
