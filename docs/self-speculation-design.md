# Self-speculation for B=1 MoE decode (design + honest cost model)

A draft-without-a-draft-model approach for the team's `engine/spec/`, motivated by the two problems the
MoE verify-tax analysis surfaced (`spec-decode-moe-tax.md`): (1) the batched verify reads the expert
*union* so trees must be small, and (2) an external draft model has nowhere to run — the **TP8 target uses
all 8 GPUs** at B=1. Self-speculation drafts with the **target's own layers**, so there's no extra model,
no extra GPU, and no training. This doc says exactly when that pays and when it doesn't.

## The mechanism (training-free)
Draft k tokens with a **shallow / layer-skipped pass** of the target (e.g. first `L_d` of 94 layers, or an
adaptively-chosen subset — the SWIFT-style training-free variant), projecting the intermediate hidden state
through the model's own final-norm + `lm_head` (the "logit lens"). Verify all k with the full 94-layer pass
in one batched call. No early-exit head to train; quality of the shallow logits is whatever the pretrained
model gives (measured by E9 below).

## Why it fits *this* problem
- **No extra GPU** — the shallow pass runs on the same TP8 ranks as the target. Resolves the
  drafter-vs-TP8 contention (`spec-decode-moe-tax.md` §placement).
- **No extra model, training-free** — nothing to host or fine-tune (unlike EAGLE/Medusa/MTP).
- **General text** — unlike n-gram/prompt-lookup (which only fires on repetitive/structured prompts),
  a shallow model pass drafts on arbitrary text.

## The honest cost model (why it's a *fallback*, not a free win)
To emit τ accepted tokens in one round of `k` drafts at shallow depth `L_d` (out of 94), in units of "one
full decode step":
```
cost ≈ k · (L_d / 94)      [draft: k sequential shallow passes, B=1 bandwidth-bound]
     + verify_cost(k)       [one batched full pass; MoE union tax, from spec-decode-moe-tax.md]
speedup ≈ τ / cost
```
`verify_cost(k)`: k=2 → 1.6, k=3 → 2.2 (the expert-union tax). So:

| L_d | k | draft cost | verify cost | total | break-even τ |
|---|---|---|---|---|---|
| 12 | 2 | 0.26 | 1.6 | 1.86 | **1.86** |
| 20 | 3 | 0.64 | 2.2 | 2.84 | 2.84 |
| 32 | 3 | 1.02 | 2.2 | 3.22 | 3.22 |

**The draft is NOT free** (it's a real fraction of the 235B), so self-spec's break-even τ is *higher* than:
- **n-gram** (draft ≈ 0 → break-even = verify_cost ≈ 1.6 at k=2) — strictly cheaper *when it fires*.
- a hypothetical tiny external draft (draft ≈ 0) — but that has nowhere to run at B=1 + must be obtained.

So the ranking on this MoE: **n-gram first** (free draft, but repetitive-text only) → **self-spec** (cheap-ish
draft, general text, no model/GPU/training) → **EAGLE/MTP** (best τ, but training + GPU contention). Self-spec
is the **general-text fallback when n-gram acceptance is low and you won't train a head.**

## The deciding question (→ E9): does a shallow pass predict well enough?
Self-spec lives or dies on whether the first `L_d` layers + logit-lens predict the next token with
`τ > break-even`. The logit lens is poor in early layers, usable in late layers — so there's a depth `L_d`
where shallow-agreement clears the bar, but a larger `L_d` raises draft cost. **The sweet spot is empirical.**
`tools/verify_self_speculation.py` measures the agreement-vs-depth curve on a real MoE (logit-lens top-1
match with the full model), which gives the achievable τ per `L_d` → plug into the table above.

## Two refinements specific to the MoE
1. **Computation reuse:** the draft's first `L_d` layers for each position are *already computed* — the
   verify only needs layers `L_d..94` for those positions. A reuse-aware engine pays draft `k·L_d` + verify
   `k·(94−L_d)` *batched* layer-evals (the union tax applies only to the completion layers), shaving the
   verify term. Worth it only if the engine can splice a batched completion onto cached shallow activations.
2. **Self-spec + route-prediction synergy:** the shallow pass *already routed* its experts in layers
   `0..L_d`. Those selections (via `predictor.rs` DirectProxy on the shallow hidden states) predict the
   completion layers' expert union — so the prefetch can warm exactly the verify's union. Self-spec and the
   routing predictor share the same "early signals predict the full pass" structure; build them together.

## How to wire it (`engine/spec/`)
`ModelRunner` already abstracts the model. Add a **shallow mode**: `forward_single(context, tok, max_layers=L_d)`
returning logit-lens logits — that *is* the drafter, so `DrafterPool` becomes "the target in shallow mode"
(N=1, no separate weights). `SpecConfig` gains `self_spec_depth: Option<usize>` and `draft_len: 2..3`. The
acceptance logic (`accept_multi_drafter`) is unchanged — self-spec only changes where the draft logprobs
come from. Gate on **realized speedup** (the table above), and make `L_d` adaptive (SWIFT-style: skip more
layers when the running acceptance is high).

## Verdict
Not a clear win on this MoE — the non-free draft + the verify-tax put break-even at τ≈1.9–2.8, which a
training-free shallow pass may or may not clear. It's the **right fallback** (general text, zero training,
zero extra GPU) and **E9 decides** whether the shallow-agreement curve clears the bar at a small-enough `L_d`.
If E9 shows poor shallow agreement, skip self-spec and rely on n-gram (repetitive) + a trained MTP head
(general) instead.
