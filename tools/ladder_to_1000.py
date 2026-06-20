#!/usr/bin/env python3
"""The quantitative ladder from 85.7 → ~1000+ tok/s (B=1, Qwen3-235B-A22B, 8×H100).

Builds the per-token cost from its terms and applies each lever in dependency order, showing the tok/s at every
rung. This is the progress tracker for docs/path-to-1000.md — when a real measurement lands, replace the assumed
delta and re-run. Each rung names the term it attacks; the floor (overhead+comms) must come DOWN before the
weight (fp8) is visible, and spec is the multiplier that rides whatever floor remains.

  python3 tools/ladder_to_1000.py
"""
# Measured starting decomposition (overhead-attribution.md): TPOT 11.67ms @ 85.7 tok/s.
# overhead = launch + host + kernel-sub-roofline (e=0.46) ; comms = 188 x 16us ; weight = fp8-roofline-equiv.
overhead = 7.0      # ms  (launch ~3.5 + host ~1.5 + kernel-inefficiency ~2.0)
comms    = 3.0      # ms  (188 collectives x 16us)
weight   = 1.6      # ms  (bf16 active weight at roofline; e<1 inefficiency is in `overhead`)
kv       = 0.07     # ms  (short ctx)

def tput(o, c, w, k, spec=1.0):
    tpot = (o + c + w + k) / spec
    return tpot, 1000.0 / tpot

rungs = []
def add(label, o, c, w, k, spec, note):
    tpot, tk = tput(o, c, w, k, spec)
    rungs.append((label, tpot, tk, note))

add("0. baseline bf16-TP8 (measured)",            overhead, comms, weight, kv, 1.0,  "the real 85.7 / 11.67ms")
# Lever order: floor first (graphs->fast-path->K5), then fp8 (weight), then NVLS (comms), then spec.
add("1. + CUDA graphs (launch~3.5ms -> 0)",       overhead-3.5, comms, weight, kv, 1.0, "removes per-kernel launch")
add("2. + scheduler-free B=1 loop (host~1.5 ->0)",overhead-3.5-1.5, comms, weight, kv, 1.0, "fast-path; b1-fast-path-design")
add("3. + fp8 K5 at e->1 (kernel-ineff~2 ->0)",   overhead-3.5-1.5-2.0, comms, 0.78, kv, 1.0, "weight 1.6->0.78 fp8 AND e=0.46->1")
add("4. + NVLS all-reduce @2us (comms 3.0->0.38)",0.0, 188*2/1e3, 0.78, kv, 1.0, "the make-or-break kernel")
add("5. + small-tree EAGLE3 spec (x1.35 @F=0)",   0.0, 188*2/1e3, 0.78, kv, 1.35, "amortizes comms+nonexpert; tree SHRINKS at F=0")

print(f"{'rung':46} {'TPOT ms':>8} {'tok/s':>7}   note")
print("-"*95)
for label, tpot, tk, note in rungs:
    flag = "  <<< 1000+" if tk >= 1000 else ""
    print(f"{label:46} {tpot:8.2f} {tk:7.0f}{flag}   {note}")

print("\nAlternatives at rung 4 (the comms make-or-break):")
for label, c in (("NVLS @3us", 188*3/1e3), ("NVLS @4us", 188*4/1e3),
                 ("stale-TP hides comms (LOOP-C, quality-gated)", 0.0)):
    tpot, tk = tput(0.0, c, 0.78, kv, 1.35)
    print(f"  rung5 with {label:46} -> {tk:5.0f} tok/s")
print("\nLossy cushion if comms/spec slip: int4 experts (weight 0.78->0.51) or depth-reduction (cuts weight AND comms).")
print("Make-or-break = rung 4 (NVLS C). Banked-now = rung... none yet; rung 0+big-tree-spec (~300) is the cheap first ship.")
