# Route-aware drafting — the mechanism for `spec_moe_model.py`'s "unclaimed lever"

`spec_moe_model.py` shows that if the draft tokens **overlap in expert space**, the verify union shrinks and
wider trees become net-positive on the MoE — "the unclaimed lever." It quantifies the *payoff* (overlap 0.45 →
+20–60% in weight-bound) but doesn't design *how* to make a drafter route-aware. This is that mechanism,
connecting the spec engine (`engine/spec/`) to the route predictor (`engine/routing/predictor.rs`).

## When it matters (set expectations first) — REFINED for the 1000 path
Per `tools/tree_spec_optimizer.py`: route-awareness is moot for **plain decode** at the floor-bound F=0.86 (union
tax on the 14% weight). **But its real home is the BIG-TREE SPEC VERIFY on the 1000-path engine** (graphs +
fast-path + fp8-K5, comms-bound): there the verify reads the union of a *big* tree, and the union **dominates**
— `verify_cost_check.py`: a W4×D8 tree (M=32) reads union≈112 experts = **7.66 ms ≫ the 3.0 ms comms**. So
shrinking that union is first-order *there*. Net (after the acceptance trade-off, since biasing for overlap costs
some τ): route-aware drafting lifts the big-tree verify ~**15%** on the comms-bound engine (e.g. ~757 → ~870 toward
1000) — a real contributor to closing the lossless gap, **not** the "second-order, defer it" of the plain-decode
floor-bound analysis. **So for the 1000 big tree: route-awareness is on the critical path** (it makes the high-τ
wide tree *affordable* by keeping its union from exploding). LOOP-A's route-aware tree-shaping is first-order for
the 1000 verify, even though it was second-order for plain floor-bound decode. (Plain decode: still defer it.)

## The mechanism
The drafter normally extends the tree by the **highest-probability** next tokens. Route-aware drafting
re-ranks/prunes those candidates to favor ones whose **experts overlap the tree's already-committed experts**,
shrinking the union the verify must read.

```
# per draft position, given the candidate set C (top-m by draft prob) and the set U of experts already
# touched by the committed tree prefix:
for c in C:
    h_c   = predictor.hidden_after(c)              # DirectProxy: residual-stream estimate of c's hidden
    e_c   = topk(router_{next} @ h_c, 8)           # predicted experts for c  (predictor.rs, ~free)
    overlap_c = |e_c ∩ U| / 8                       # how much c reuses already-loaded experts
    score_c   = log p_draft(c) + λ · overlap_c      # trade draft-prob for route-overlap
keep argmax(score_c) (or the top-b for a width-b tree); U |= e_chosen
```

- **`λ` is the knob:** λ=0 → plain spec (max acceptance); large λ → minimal union (max overlap) but lower
  acceptance. The optimum balances `E[accepted]` (falls with λ) against `verify_cost` (falls with λ via a
  smaller union). Sweep λ to maximize `E[accepted]/verify_cost` from `tree_spec_optimizer.py`.
- **The predictor is the enabler:** `predictor.rs` DirectProxy already estimates a token's experts from the
  residual stream *before* the layer runs (persistence 0.45 rising by layer — `routing_predict_early.json`),
  and it's ~free. Route-aware drafting is **the spec engine consuming the route predictor's output** — the
  two `engine/` modules compose exactly here.

## The honest tension
Route-awareness **trades acceptance for a smaller union.** It only wins when the union reduction outweighs the
acceptance loss — i.e. when the union tax actually bites (weight-bound). `tree_spec_optimizer.py` makes this
explicit: at F=0 the route-aware column beats naive by 20–60%; at F=0.86 the columns nearly coincide (the tax
is negligible, so biasing away from the best token just costs acceptance for no union benefit). **So λ should
itself be regime-adaptive: λ≈0 while floor-bound, λ>0 as F→0.**

## Cheaper approximations (if the per-candidate predictor call is too costly in the draft loop)
1. **Static expert-affinity bias:** precompute, from `routing_stats.json`'s co-activation, a token→token expert
   affinity; bias the draft toward tokens that historically co-route — no per-step predictor call.
2. **Self-spec is route-aware *for free*:** a shallow-pass draft (`self-speculation-design.md`) routes through
   the *same* experts as the early layers of the verify, so its tokens are inherently route-correlated with
   the target — the union is smaller by construction. This is why self-spec, despite a non-free draft, may beat
   an external route-agnostic drafter once weight-bound.

## Placement
A **research-bet, weight-bound** lever (`b1-optimization-atlas.md` #2 family, future). Build the λ-knob into
`engine/spec/` drafting and wire `predictor.rs`; validate with `tree_spec_optimizer.py` (the model says when
it pays) once E0b/K5 move the regime. Until then it's correctly dormant — plain big-tree EAGLE3 is the answer.
