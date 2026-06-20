# The EP→TP inversion REVERSES for the big-tree spec verify (EP is fine for LOOP-A's FP8+EP run)

My EP→TP inversion (`b1-tp8-moe-rearchitecture-h200.md`, measured fp8-EP8 64.5 < bf16-TP8 85.7) is a **B=1
*decode* finding**: one token activates 8 experts, which land unevenly across 8 EP ranks (balls-in-bins
E[max]≈2.6), so the busiest rank gates the step. **The spec verify is different** — and the difference matters
for LOOP-A's FP8+EP EAGLE3 run.

## The mechanism
The verify runs the MoE over a **tree of W×D positions** in one forward. The expert weight-read per EP rank is
set by **how many of that rank's 16 experts (128/8) the tree touches** — i.e. the union restricted to the rank.

| regime | active experts | per-rank load | EP imbalance |
|---|---|---|---|
| B=1 decode (1 token) | 8 of 128 | 0–3 of a rank's 16 | **~2.6× (busiest gates)** — the penalty I measured |
| spec verify, small tree (union~30) | ~30 of 128 | ~2–6 of 16 | moderate (~1.4×) |
| **spec verify, BIG tree (union→128)** | **~all 128** | **all 16 on every rank** | **~1.0× — imbalance VANISHES** |

For a big-enough tree the union approaches all 128 experts, so **every EP rank reads all 16 of its experts** →
perfectly balanced → the busiest-rank penalty that kills plain fp8-EP decode is **gone for the verify.**

## Why this matters (three things compose)
1. **LOOP-A's FP8+EP layout is well-suited to big-tree spec** — the verify doesn't pay the EP imbalance my
   decode measurement showed. The fp8-EP 64.5 number is a *plain-decode* penalty; the *verify* on EP is balanced.
2. **It stacks with the floor-amortization the same direction:** `tree_spec_optimizer.py` already says big
   trees win in the floor-bound regime (the union tax is on the 14% weight). Now there's a *second* reason to
   go big on EP: the big tree also **balances the EP weight-read**. Big tree = wins the floor *and* balances EP.
3. **It refines the F-backout** (`backout_floor.py`): on EP, `verify_cost(k)` has an extra rank-imbalance factor
   that *falls* as k grows (toward 1.0), whereas on TP it's flat. So FP8+EP's V(k) may flatten faster than the
   pure union model predicts — a signature LOOP-A can look for (V growing sublinearly in union → EP rebalancing).

## The caveat (be honest)
- The verify still does an **all-to-all dispatch/combine** (EP comms) instead of TP's all-reduce — at B=1 the
  *verify's* payload is W×D tokens (bigger than decode's 1), so the EP all-to-all is no longer a tiny-message
  latency game; it's closer to bandwidth. Whether EP-verify beats TP-verify is the empirical question — but the
  *expert-imbalance* half of my EP penalty is neutralized by the big tree, so EP is far more competitive for the
  verify than for plain decode.
- For the **draft** (1B head, sequential, tiny), TP8 still wins (`eagle3-draft-tp.md`) — that's `draft_tp=8`.

## Net for the collaboration
LOOP-A's FP8+EP + **big tree** is a coherent, strong config: fp8 ½-weight + balanced-verify EP + floor-amortized
big tree. My bf16-**TP8** run is the clean control (no EP imbalance at all). Comparing the two `backout_floor.py`
F's (and the V(k) curvature) tells us whether EP-verify's all-to-all costs more than TP-verify's all-reduce once
the expert-imbalance is gone — a real, measurable question the 08:45 pair answers.
