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
add("4a. comms via COUNT: EP 94 coll @16us barrier",0.0, 94*16/1e3, 0.78, kv, 1.0, "reaction-04: per-coll barrier-floored ~16us; cut COUNT 188->94")
add("4b. (OPTIMISTIC) multimem in-switch @2us",   0.0, 188*2/1e3, 0.78, kv, 1.0, "ONLY if measure_collective.sh shows in-switch beats 16us")
add("5. + batched spec (team EAGLE3 ÷2.77)",      0.0, 94*16/1e3, 0.78, kv, 2.77, "the dominant lever; on the 4a (EP-count) base")

print(f"{'rung':46} {'TPOT ms':>8} {'tok/s':>7}   note")
print("-"*95)
for label, tpot, tk, note in rungs:
    flag = "  <<< 1000+" if tk >= 1000 else ""
    print(f"{label:46} {tpot:8.2f} {tk:7.0f}{flag}   {note}")

# ---- LIVE diagnostic: plug in the real measurements as they land ----
import argparse
ap = argparse.ArgumentParser(description="plug measured C/e/tau -> projected 1000-path tok/s + gap")
# reaction-04: per-collective C is BARRIER-floored at ~16-17us (squeeze round). Recursive-doubling=3 barriers
# (51us). So the comms attack is the COUNT (--ncoll: 188 TP -> ~94 EP-1-barrier) + spec, unless multimem
# in-switch (measure_collective.sh) beats the barrier. int4 is RULED OUT at B=1 (0.58x, unpack-bound).
ap.add_argument("--C", type=float, default=16.0, help="per-collective latency (us). MEASURED floor ~16us (1 barrier); <16 only if multimem in-switch beats it")
ap.add_argument("--ncoll", type=int, default=188, help="collective count: 188 (TP 2/layer) -> ~94 (EP 1-barrier/layer) is the #1 comms lever now")
ap.add_argument("--e", type=float, default=1.0, help="measured fp8-K5 kernel efficiency (0.46 today -> target ~0.85)")
ap.add_argument("--tau-mult", type=float, default=2.77, help="batched-spec multiplier (team EAGLE3 ~2.77-3.8); the dominant lever")
ap.add_argument("--weight", choices=["fp8"], default="fp8", help="fp8 only — int4 RULED OUT at B=1 (reaction-04, 0.58x unpack-bound)")
ap.add_argument("--host-ms", type=float, default=0.0, help="residual host/overhead after graphs+fast-path (E-attr)")
ap.add_argument("--overlap", action="store_true", help="LOOP-C exact deferred-overlap (LOSSLESS): hide C behind the ~8.3us fp8 weight read -> exposed comms = max(0,C-8.3) (stale/proxy-TP is DEAD, reaction-05)")
a = ap.parse_args()
w = 0.78 / max(a.e, 0.05)   # fp8 weight read at measured efficiency
COVER_US = 4.3   # fp8 per-collective NEXT-OP weight-read cover (AR-A->MoE gate/up ~3.7us; AR-M->next QKV ~1.75us;
                 # avg ~4.3. LOOP-C's 8.3us = the full-LAYER read, optimistic; the dependency limits it to the
                 # next op. deferred-overlap hides up to COVER of C -> FULL hide needs C<=~4us (multimem in-switch).)
c = (a.ncoll * max(0.0, a.C - COVER_US) / 1e3) if a.overlap else (a.ncoll * a.C / 1e3)
tpot, tk = tput(a.host_ms, c, w, kv, a.tau_mult)
print(f"\n=== LIVE (C={a.C}us e={a.e} tau×{a.tau_mult} weight={a.weight}"
      f"{' +overlap(lossless)' if a.overlap else ''} host={a.host_ms}ms) ===")
print(f"  weight {w:.2f} + comms {c:.2f} + host {a.host_ms:.2f}  / spec {a.tau_mult}  = {tpot:.2f} ms -> {tk:.0f} tok/s"
      f"  {'>>> 1000 CLEARED' if tk>=1000 else f'(gap {1000-tk:.0f})'}")
if tk < 1000:
    gap = 1000 - tk
    print(f"  next lever for the {gap:.0f} gap:", end=" ")
    if c > 1.0 and not a.overlap:
        print(f"comms {c:.1f}ms dominates -> --overlap (lossless deferred-overlap), or push multimem in-switch C<=4us (then overlap fully hides it).")
    elif a.tau_mult < 3.0: print("spec under-delivering -> bigger/better batched verify (team EAGLE3 ~3.8); fix the batched-verify kernel.")
    elif a.e < 0.85: print("kernel under roofline -> tune fp8-K5 e (cp.async, k5-tuning-roadmap).")
    else: print("close -> --overlap (lossless) hides the last comms, or push multimem C<=4us / --tau-mult.")
print("\nreaction-04: per-collective C is BARRIER-floored ~16us. Comms attack = COUNT (--ncoll) + batched spec,")
print("NOT per-collective latency (unless measure_collective.sh shows multimem in-switch beats 16us). int4 ruled out.")
print("Team's measured path: --C 16 --ncoll 94 --tau-mult 2.77 -> ~960.  Cheap first ship: spec+prefix ~300.")
