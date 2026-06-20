# Lookahead (Jacobi) decoding for B=1 — the draft-FREE floor-amortization lever

The team converged on **draft-based** spec (EAGLE3). There's a parallel branch worth having on the table:
**lookahead decoding** (Fu et al.; Jacobi-iteration decoding) amortizes the floor **without any draft model**.
This is the draft-free alternative — model-agnostic, no EAGLE3 head, no vLLM≥0.10.2, no `draft_tp` to tune —
and the floor-bound regime is exactly where its one weakness (extra compute) disappears.

## The mechanism (no draft model)
Sequential decode solves `x_{t+1} = f(x_{≤t})` one token at a time. **Jacobi decoding** instead guesses the
next *n* tokens, feeds the whole guess block through the model in **one forward**, and takes the refined
outputs as the next guess — iterating to a fixed point. **Lookahead decoding** runs this as a 2-D window: each
step's single forward simultaneously (a) advances the Jacobi window of n-gram guesses and (b) **verifies**
n-gram candidates collected in a pool, accepting the longest match. The model verifies its *own* guesses — no
second network.

## Why B=1 floor-bound is the ideal home for it
Lookahead's cost is **extra parallel compute** (the n-gram window makes each forward process W positions
instead of 1). On most engines that compute is the price. **But B=1 decode here is floor-bound, not
compute-bound** (`overhead-attribution.md`: 86% floor, AI≈1, tensor cores idle). So the extra window positions
ride the *same* per-step floor — the GPU was idle anyway — and the step emits the accepted n-gram instead of
1 token. **The floor is amortized for free, with no draft to pay for.** This is the same "one batched forward
pays the floor once" principle as `why-spec-wins.md`, but the batch is the Jacobi window, not a draft tree.

## Lookahead vs EAGLE3 in the floor-bound regime (`spec_predict.py` framing)
    speedup ≈ E[accepted n-gram length] / (verify_cost + draft_cost)
| | EAGLE3 (draft-based) | Lookahead (draft-free) |
|---|---|---|
| draft_cost | ~0.05–0.26 verify-units (draft_tp 8 vs 1, `eagle3-draft-tp.md`) | **0** (no draft) |
| acceptance / τ | high (~3–3.5, trained) | lower (~1.5–2.5, Jacobi n-gram match) |
| verify_cost (MoE union) | union of the draft tree | union of the Jacobi window (same tax) |
| floor-bound speedup | **~2.5–3×** | **~1.5–2.3×** |
| setup | EAGLE3 head + vLLM≥0.10.2 + draft_tp | **none** — any model, any vLLM with the feature |

So **EAGLE3 wins on raw speedup** (trained draft → higher acceptance), but **lookahead wins on
robustness/portability**: zero draft cost, no head dependency, no version floor, model-agnostic. The gap
*narrows* exactly when the draft is expensive (`draft_tp=1` → EAGLE3 ~2.5×, lookahead ~2.0× — close) or the
EAGLE3 head is unavailable/mismatched.

## When lookahead is the right call
- **No usable draft head** (a model without a trained EAGLE3/MTP head) — lookahead is the *only* floor-amortizer.
- **The 0.10.2 venv / head download is a blocker** and you want a win *today* on the system vLLM (it's a config
  flag, not a checkpoint).
- **As a baseline** to attribute EAGLE3's gain: EAGLE3-speedup ÷ lookahead-speedup = the value the *trained
  draft* adds over free self-verification. A cheap, informative control for the 08:45-style runs.
- **Repetitive/structured output** (code, JSON, chat boilerplate): the n-gram pool fills fast → lookahead's τ
  rises toward EAGLE3's, at zero draft cost (overlaps the n-gram-spec sweet spot, `spec-decode-floor-bound.md`).

## The MoE caveat (same tax, no free lunch on the union)
The Jacobi window's W positions read the **expert union** just like a draft tree — so the floor-aware verify
cost (`spec_floor_model.py`) and the regime-adaptive sizing (`tree_spec_optimizer.py`: big window while
floor-bound, shrink as F→0) apply unchanged. Lookahead removes the *draft* cost, not the *verify-union* tax.
On **EP**, the same big-window-balances-the-verify effect holds (`ep-balance-spec-verify.md`).

## Recommendation
Keep EAGLE3 as the headline (highest τ). **Add lookahead as the draft-free control + the portable fallback**:
it's a config flag, costs nothing to try, gives a model-agnostic floor-amortization win, and its speedup vs
EAGLE3's isolates exactly what the trained draft buys. For models with no draft head, it's the *primary* lever.
Folds into the spec experiment queue (E6 family) as a no-setup arm: run lookahead on the system vLLM while the
EAGLE3 venv is the heavier path.
