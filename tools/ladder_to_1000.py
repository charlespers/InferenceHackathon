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

# ---- LIVE diagnostic: plug in the real measurements as they land ----
import argparse
ap = argparse.ArgumentParser(description="plug measured C/e/tau -> projected 1000-path tok/s + gap")
ap.add_argument("--C", type=float, default=2.0, help="measured per-collective NVLS latency (us); make-or-break")
ap.add_argument("--e", type=float, default=1.0, help="measured fp8-K5 kernel efficiency (0.46 today -> target ~0.85)")
ap.add_argument("--tau-mult", type=float, default=1.35, help="measured small-tree spec multiplier at F=0")
ap.add_argument("--weight", choices=["fp8", "int4exp"], default="fp8", help="int4exp = int4 experts + fp8 rest (cushion)")
ap.add_argument("--host-ms", type=float, default=0.0, help="residual host/overhead after graphs+fast-path (E-attr)")
ap.add_argument("--stale-tp", action="store_true", help="LOOP-C stale-TP hides comms (quality-gated) -> comms=0")
a = ap.parse_args()
w = (0.78 if a.weight == "fp8" else 0.51) / max(a.e, 0.05)   # weight read at measured efficiency
c = 0.0 if a.stale_tp else 188 * a.C / 1e3
tpot, tk = tput(a.host_ms, c, w, kv, a.tau_mult)
print(f"\n=== LIVE (C={a.C}us e={a.e} tau×{a.tau_mult} weight={a.weight}"
      f"{' +stale-TP' if a.stale_tp else ''} host={a.host_ms}ms) ===")
print(f"  weight {w:.2f} + comms {c:.2f} + host {a.host_ms:.2f}  / spec {a.tau_mult}  = {tpot:.2f} ms -> {tk:.0f} tok/s"
      f"  {'>>> 1000 CLEARED' if tk>=1000 else f'(gap {1000-tk:.0f})'}")
if tk < 1000:
    gap = 1000 - tk
    print(f"  next lever for the {gap:.0f} gap:", end=" ")
    if a.C > 3 and not a.stale_tp: print("comms still high -> push NVLS C down, or --stale-tp, or --weight int4exp.")
    elif a.e < 0.8: print("kernel under roofline -> tune fp8-K5 e (cp.async, k5-tuning-roadmap).")
    elif a.tau_mult < 1.3: print("spec under-delivering -> check draft_tp=8 / small tree / accept-rate.")
    else: print("close -> --weight int4exp (cushion) or --stale-tp to hide the last comms.")
print("\nMake-or-break = the NVLS C (run bench/measure_collective.sh first). Cheap first ship: spec+prefix ~300.")
