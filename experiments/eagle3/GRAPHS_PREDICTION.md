# EAGLE3 CUDA-graphs headline — PRE-REGISTERED prediction (before the measurement landed)

Written 2026-06-20 ~12:53 UTC, while the `slot_eagle3_graphs` run (k=3, RedHat head, FP8 target,
TP8+EP, graphs mode `enforce_eager=False`) was still measuring `baseline_graphs`. Committed BEFORE
the S number landed so the prediction can't be back-fit. The model is `engine/src/spec/projection.rs`.

## The setup
- Run: EAGLE3 (`num_speculative_tokens=3`, draft_tp=1, RedHat head) vs matched FP8 baseline, both
  graphs mode, decode 256 × 3 repeats (`measure_baseline.py`). S = tok/s(eagle3)/tok/s(baseline).
- Live SpecDecoding metrics already observed during eagle3_graphs serving: **mean acceptance length
  τ_graphs ≈ 2.5–2.67**, per-position accept ≈ 0.75 / 0.5 / 0.33 (first-pos ~0.7–0.75). This MATCHES
  the eager measurement (τ≈2.7, first-pos~0.75) — acceptance is a model property, mode-independent,
  exactly as expected. So τ is NOT the open question; **S (the tok/s ratio) is.**
- The graph-capture phase **did not crash** (the documented EAGLE3+graphs crash history, INTEGRATION
  §3, did not recur at k=3) — itself a bankable deployment result.

## Pre-registered prediction (projection.rs cost model)
Eager anchor is pinned: EAGLE3 eager S≈1.0 (measured ~10 tok/s ≈ baseline eager). Collapsing the
launch floor with graphs should lift S. The model says the graphs S is set by the **verify expert
union**, because at low floor F the routed-expert weight term dominates the verify:

- If the verify union stays small (~8–12; consecutive EAGLE3 tokens route similarly), **S ≈ 1.8–2.4×**
  (literature/Alyssa-consistent), and V=τ/S ≈ 1.1–1.5 (mild union tax).
- If the union is wide (naive divergent routing, ≥40), S collapses toward ~1.0–1.3× even with graphs.

**Pre-registered band: S_graphs ∈ [1.6, 2.6].** Falsification: if the measured S lands outside
[1.6, 2.6], the model or an input (F_graphs, union) is wrong — a finding either way.

## What I'll compute the moment `slot_graphs.DONE` lands
1. `eagle3_analyze.py --dir /alloc/data/eagle3_graphs --mode graphs` → S, τ, **V=τ/S**, parity,
   plus EAGLE3-absolute vs the bf16-best 85.7 tok/s.
2. Feed measured (τ_graphs, first-pos, S) into `projection.rs::back_solve_graphs_union` → the verify
   union the box actually achieved → how much headroom route-aware verification has.
3. Route-aware go/no-go: **V≈1 ⇒ floor-bound, graphs already hides the union ⇒ route-aware NO-GO in
   this regime; V≫1 ⇒ union taxes ⇒ route-aware GO.** (This is the decision the whole route-aware
   add-on hinges on — let the number decide; kill the lever here if V≈1.)

## Honest caveats (inputs, not measured by the model)
- F_eager≈0.86, F_graphs≈0.40 are estimates (spec_floor_model + "graphs ~5× eager", Alyssa).
- Minimum verify union is 8 (one token's top-8); at union=8 verify_cost==1 ∀F, so the graphs win at
  the ideal union is the **draft-launch collapse**, not a verify-floor effect.
- The projection band was registered at depth=5; this run is k=3 (τ lower) — I'll re-evaluate the
  band at depth=3 with the measured τ when I process S, but the structural claim (union sets S) holds.
