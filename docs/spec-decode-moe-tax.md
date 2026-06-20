# Spec-decode on the 235B MoE: size the tree small (a note for `engine/spec/`)

A specific, quantified finding for the team's `engine/spec/` (`SpecConfig`, `accept_multi_drafter`,
`ModelRunner::forward_batch`): the default **`draft_len: 8`** (and multi-drafter `N>1`) is tuned for a
*dense* target and is very likely **net-negative on Qwen3-235B-A22B**. Here's why and what to set instead.

## The MoE verify tax
`forward_batch(N·k)` verifies all `N·k` draft positions in one batched target pass. On a **dense** model
that pass reads the same FFN weights regardless of `N·k`, so the verify cost ≈ 1 decode and big trees are
free — accept more, win more. On an **MoE** each of the `N·k` positions routes to its own top-8 of 128,
so a correct batched expert kernel reads the **union** of experts all positions touch. That union grows:

```
E[union] = 128 · (1 − (120/128)^(N·k))           # 8-of-128 per position, N·k positions
verify_cost ≈ 0.34 + 0.66 · (E[union] / 8)        # in "1 decode" units; 0.66 = routed-expert byte share
```

| N·k (draft positions) | E[union]/128 | verify_cost | break-even τ (accepted/round) |
|---|---|---|---|
| 2 | 15.5 | **1.6×** | 1.6 |
| 3 | 22.5 | 2.2× | 2.2 |
| 4 | 29 | 2.7× | 2.7 |
| 6 | 41 | 3.7× | 3.7 |
| **8 (current default)** | 52 | **4.6×** | **4.6 — you can't accept 4.6 of 8 reliably → LOSS** |
| 32 (N=4 × k=8) | 112 | 9.6× | net-negative |

`speedup ≈ τ / verify_cost`. On a dense model `verify_cost≈1` so `speedup≈τ` (draft_len=8 great). On this
MoE the verify tax dominates, and **draft_len=8 needs τ≥4.6 just to break even.**

## Recommendations for `SpecConfig` on the 235B
1. **`draft_len` 2–3, single drafter (N=1)** as the default for this MoE — not 8. k=2 breaks even at τ=1.6
   (easily achievable with a decent drafter); k=3 at τ=2.2.
2. **Multi-drafter (N>1) is counterproductive here** at B=1: more positions → bigger expert union → higher
   verify tax, and it also contends for the GPUs the TP8 target already uses (see below). Keep N=1 unless a
   drafter is so weak that diversity beats the union cost (measure it).
3. **Gate on *realized wall-clock speedup*, not acceptance.** `RoundStats` tracks `n_accepted/n_proposed`;
   add the verify-pass time so the engine can compute `τ / verify_cost` and auto-shrink the tree when it
   goes < 1.0. High acceptance with a big tree still loses on the MoE — the headline trap.
4. **`forward_batch`'s cost model must include the expert union**, not assume verify ≈ 1 decode. The batched
   expert kernel should read each distinct expert once and apply to all positions routing to it (union cost,
   not `N·k·8` cost) — otherwise the tax is even worse.

## A GPU-placement tension to resolve (B=1 + TP8 target)
`DrafterPool` docs say "each drafter lives on its own GPU." But the **TP8 target uses all 8 GPUs** for one
B=1 stream (bandwidth-bound). A separate-GPU drafter has no free GPU; sharing means the draft steps steal
bandwidth from the target. Options to evaluate: (a) a tiny drafter co-resident on the TP8 ranks (cheap
enough to interleave), (b) **self-speculation** (the target's own early layers as the draft — no separate
model, no extra GPU), or (c) n-gram/prompt-lookup drafting (zero model, zero GPU — the natural first move,
and its draft is free so only the verify tax matters → small k is doubly important).

## Ties to the rest of `charles-work`
- This refines `next-levers-research.md` L1 and **E6** (use `draft_len 2–3`, measure realized speedup).
- The verify tax is the same `c≈0.27/position` from the spec doc, here pinned to their `N·k` tree directly.
- Route prediction (`predictor.rs`) could pre-stage the verify pass's expert union — predicting which
  experts the drafted tokens hit lets the prefetch warm exactly that union (a real synergy: spec + prefetch).
