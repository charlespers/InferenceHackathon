# The multiplicative stack isn't multiplicative — it's regime-flipping (a check on `fast_decode_research.md` §6)

LOOP-A's `fast_decode_research.md` §6 stacks: `EAGLE3 spec (~3×) × comms-megakernel+NVLS (~2.5×) × expert-dedup
(~1.5–2×) × adaptive-k-drafter` → 300–540 tok/s. The **300–540 band is right** (it matches my cheap+kernel
projection and sits below the ~2000 fp8+spec ceiling, `absolute-ceiling.md`). But the *factors don't multiply*,
for two reasons that change how you tune the stack — both straight out of the floor-bound framework.

## 1. Spec and the megakernel OVERLAP — both attack the floor
- **Spec's ~3×** comes from amortizing the **floor** (one verify forward pays the 188 collectives + overhead
  once for τ tokens) — `why-spec-wins.md`.
- **The megakernel's ~2.5×** comes from **removing that same floor** (no launches, device-side comms, no host).
- So `3× × 2.5×` **double-counts the floor.** Once the megakernel removes the floor, there's less floor for spec
  to amortize → spec's marginal benefit shrinks. The product (7.5×) overstates; the honest combined is smaller.

## 2. Stacking FLIPS THE REGIME — floor-bound → weight-bound — so the optimal config changes
This is the important part. With the floor removed (megakernel + NVLS), the verify forward is no longer
floor-dominated — it's **weight-dominated by the expert UNION** the tree reads. Concretely, on the megakernel
the verify ≈ `device_comms(~0.7ms) + union_weight_read`; for a big tree (union→~52–126 experts) that union read
is **multiple ms** and now *dominates*. So:
- **The big tree that wins while floor-bound (`tree_spec_optimizer.py`) becomes SUBOPTIMAL** once the megakernel
  lands — the union tax it incurs is no longer hidden under the floor. The tree must **shrink** → τ drops →
  spec amortizes less. (This is the F→0 column of `spec_floor_model.py` made real by the megakernel.)
- **expert-dedup and route-aware drafting — moot while floor-bound — now turn ON** (they shrink exactly the
  union term that now dominates). They're not independent multipliers stacked on the floor-bound config; they're
  the levers the *new* regime requires. (Caveat: "expert-dedup ~1.5–2×" is partly already in any verify-cost
  that reads the **union** rather than per-position — don't double-count it against a spec baseline that already
  assumes dedup, e.g. `spec_floor_model.py`'s `0.34+0.66·union/8`.)

## The correct mental model: one cost, divided and reduced
    TPOT_per_token = [ floor(reduced by megakernel/NVLS) + weight(reduced by dedup, union-scaled) ] / τ(tree)
Spec is the **outer ÷τ**; the megakernel reduces the floor term; dedup/route-aware reduce the weight term; the
tree size sets τ **and** the union. You don't multiply independent speedups — you reduce the two terms and pick
the tree that minimizes the whole expression **for the current regime**. As the floor falls, the minimizing tree
shrinks and the weight levers matter more.

## The actionable consequence: ORDER + RE-TUNE, don't just stack
1. **Now (floor-bound, F≈0.86):** big-tree EAGLE3 spec. Dedup is the union assumption; route-aware is off.
2. **After comms/megakernel cut the floor (F→0):** **re-run `tree_spec_optimizer.py`** — it will say *shrink the
   tree*; turn **on** route-aware drafting; the weight levers (fp8→int4, adaptive-k) now pay. The optimal config
   is **not** the floor-bound one scaled up.
3. **adaptive-k-as-drafter** (LOOP-A's idea) is genuinely additive and lossless — it's a *cheaper draft*, hitting
   the draft-cost term (`eagle3-draft-tp.md`), orthogonal to the floor/weight terms. Keep it.

## Net (for LOOP-A)
The 300–540 band is a fair target, but reach it by **co-optimizing**, not multiplying: as each floor lever lands,
the spec tree must shrink and the weight/route levers switch on. `spec_predict.py` + `tree_spec_optimizer.py` +
`backout_floor.py` (measure F each slot) are exactly the loop to re-tune the stack at each regime. The headline
"7.5×" double-counts the floor; the honest combined is ~4–6× **with the config changing as you climb.**
