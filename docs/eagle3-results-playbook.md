# Reading the 08:45 EAGLE3 results — decision tree (so interpretation is immediate)

The first real EAGLE3 data lands from LOOP-A's FP8+EP slot (+ my bf16-TP8 arm, `run_eagle3.sh`). Inputs per
mode (eager/graphs): **S** = tok/s(EAGLE3)/tok/s(baseline), **τ** = accept-length (1+accepted/drafts, from
`/metrics`), **V = τ/S** = effective verify cost in decode-steps. Tools: `tools/eagle3_analyze.py` (S, τ, V),
`tools/backout_floor.py` (F from V at ≥2 tree sizes), `tools/spec_floor_model.py` (the model). Here's what each
outcome means and the next move.

## Branch on V = τ/S (graphs mode = the headline)
- **V ≈ 1.0–1.3 (the prediction):** the N·k verify is hidden under the floor → **floor-bound confirmed**, and
  **S ≈ τ (~2.5–3×) — my over-delivery claim holds** (EAGLE3 beats its published ~1.9× *because* this engine's
  floor is bigger). → Conclusions: (a) spec is THE lever; (b) **route-aware tree-shaping = NO-GO** (the union
  tax is on the ~14% weight — LOOP-A defers it); (c) **go BIG on the tree** (`tree_spec_optimizer.py`); (d) next
  decode lever = push the floor down (E0b comms, K5 kernels) — once the floor falls, *re-test*, because route-
  aware turns GO as F→0.
- **V noticeably > 1 and rising with tree size:** the union/weight term is real → the floor is *lower* than the
  86% I measured (good — comms/graphs already cut it). → **route-aware shaping = GO** (LOOP-A's lever has
  headroom), and shrink the tree toward the `spec_floor_model` optimum for the measured F. Run `backout_floor.py`
  on the k-sweep to get the exact F.

## Sanity gates (check before trusting S)
- **Lossless?** Parity gate must pass (EAGLE3 is lossless by construction; a fail = a config/impl bug, not a
  real speedup). If parity fails, S is meaningless — fix first.
- **draft_tp?** If the runner used **`draft_tp=1`**, the ~3ms draft caps S at ~2.5× *regardless of the floor*
  (`eagle3-draft-tp.md`). A surprisingly low S with a healthy τ ⇒ suspect the draft cost → re-run `draft_tp=8`.
  (Tell: τ is good but S < τ/1.3 even though V-from-verify should be ~1.)
- **τ low (~1.5–2.0)?** Either greedy-vs-temp (this is greedy; temp 0.7 would be *lower* — `spec-in-production.md`)
  or the draft head mis-loaded (check the `/metrics` counters are nonzero). Healthy greedy τ for this head ≈ 3–3.5.
- **eager vs graphs:** graphs should *raise* S (lower floor → but also less floor to amortize). If graphs-S <
  eager-S, the floor is already small under graphs (→ closer to weight-bound → route-aware more likely GO).

## The bf16-TP8 (mine) vs FP8+EP (LOOP-A) comparison
- Same head, both run the k-sweep → `backout_floor.py` gives **F_bf16-TP8** and **F_fp8+EP**.
- **ΔF = F_bf16TP8 − F_fp8EP = the floor reduction FP8+graphs buys** — exactly what decides LOOP-A's route-aware
  lever (lower F ⇒ closer to weight-bound ⇒ route-aware GO).
- **EP-verify check:** on FP8+EP, V(k) should grow *sublinearly* in the union (the big-tree verify rebalances
  EP — `ep-balance-spec-verify.md`); if EP-verify's V is *higher* than TP-verify's at the same k, the all-to-all
  is costing more than TP's all-reduce → prefer TP8 for the verify too.

## What goes in the Results Log + next queue
Record per (layout, mode, k): tok/s, τ, S, V, parity, draft_tp. Then: confirm/deny the over-delivery prediction;
set the route-aware GO/NO-GO from F; pick the next decode lever (floor-push if floor-bound, quant/route-aware if
weight-bound); update `b1-optimization-atlas.md` row 2 with the *measured* spec multiplier (replaces the
projected ~2–3×). This is the moment the whole projected ladder (`absolute-ceiling.md`) gets its first real rung.
