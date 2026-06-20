# Route-aware tree-shaping — empirical go/no-go (LOOP-A novel lever)

**Status (UPDATED 2026-06-20 ~10:18): LITERATURE-VALIDATED on our exact model — lever upgraded from
"likely marginal" to "build it (as adaptive verification), pending our own V/F confirmation."**

## LITERATURE VALIDATION (deep-research, cited)
- **EVICT** (Pan et al., USTC, arXiv:2605.00342, May 2026 — "Making Every Verified Token Count: Adaptive
  Verification for MoE Speculative Decoding"): training-free, hyperparameter-free, **LOSSLESS**. Measured
  **−32.5% activated experts, −26.6% verify latency, 1.21× avg over EAGLE-3, and 1.25× on Qwen3-235B-A22B
  (our EXACT model)**. So expert-union-aware verification IS a real lossless win here, not marginal.
- **KEY NUANCE:** most of EVICT's win is from verifying ~75% FEWER tokens (a tree-size/COMMS win), NOT the
  direct weight-read (~14%). That's WHY it pays in the floor-bound regime — fewer tokens through the ~188
  collectives. So reframe the lever: "verify fewer, expert-cheaper tokens," a comms win as much as a weight win.
- **My novel delta vs EVICT/XShare(2602.07265)/MoE-Spec(2602.16052):** make expert-union minimization the
  EXPLICIT PRIMARY objective of chain/verify construction (using our measured 44.6% top-8 overlap). EVICT gets
  the union shrink as a side effect of a cost/benefit utility; nobody targets union directly.
- **vLLM CONSTRAINT (important):** vLLM's EAGLE3 path is **CHAIN-ONLY** (no dynamic/tree drafting; issue
  #18327 closed not-planned). So `num_speculative_tokens=K` is a linear chain, not a branching tree. My lever
  in vLLM = **adaptive verification / chain-truncation** (EVICT-style), NOT tree-shaping. True dynamic trees
  need SGLang or a vLLM patch.

**Original status (still the empirical gate): CONTINGENT on our own EAGLE3 V/F measurement. Don't ship a
number we haven't measured. EVICT validates the DIRECTION; our slot confirms the MAGNITUDE on our stack.**

## The lever
EAGLE3's verify pass runs the full 235B MoE over the draft *tree* (N·k positions) in one
forward. On an MoE, that forward reads the **union of experts** the tree positions route to.
A route-aware drafter (tokens biased to overlap in expert space) keeps that union small →
smaller verify weight-read → cheaper verify → wider trees become net-positive.
Model: `tools/spec_moe_model.py` (mine, weight-units).

## Why it might NOT pay — the honest threat (Charles)
`tools/spec_floor_model.py` (charles-work) **already incorporated route-awareness** (overlap
param) into a floor-aware model and found:

```
verify_cost(real) = F + (1-F)·(0.34 + 0.66·union/8)     # F = floor fraction of a decode step
```

At the **measured F≈0.86** (bf16-TP8: 188 all-reduces dominate, weight is only ~14%), the
union tax falls on the 14% weight term, so it **barely bites** — naive big trees already win
~2×, and route-awareness lifts speedup only marginally. Route-awareness becomes first-order
**only as the floor is fixed** (comms tuning + CUDA graphs + FP8 push toward weight-bound).

So: **in today's floor-bound regime my lever is second-order.** I will not claim otherwise,
and I will not implement it on a guess. It is decided by one empirical question.

## The decisive empirical test (captured by the armed 08:45 slot)
The realized EAGLE3 speedup vs its accept-length is a **direct floor probe**:

- Measure **accept-length τ** (vLLM spec metrics in the server log) and **realized speedup**
  `S = tok/s(EAGLE3) / tok/s(baseline)` on the **FP8 + EP** engine (the real target), both
  eager and with CUDA graphs.
- Back out the **effective verify cost** `V = τ / S` (in units of one normal decode step):
  - **V ≈ 1  ⇒ floor-bound verify** (the N·k batch is hidden under the floor). Route-aware
    shaping saves ~nothing. **NO-GO** — report it, move my effort to the empirical EAGLE3
    headline + helping the floor fall.
  - **V noticeably > 1 and growing with tree size ⇒ the verify weight/union term is real.**
    Then the union *is* large enough to tax, and route-aware shaping has headroom.
    **GO** — build the route-biased draft / union-capped tree (the lever in `spec_moe_model.py`).

CUDA graphs + FP8 specifically *cut the floor* (graphs remove launch latency; FP8 halves
weight bytes), so the graphs-on number is where V is most likely to exceed 1 — i.e. where my
lever is most likely to matter. That is exactly why the FP8+graphs measurement (mine, not
Charles's bf16 run) is the one that decides this.

## Decision rule (apply once slot data lands)
1. If FP8+graphs `V = τ/S < 1.3` → **route-aware shaping NO-GO now**; revisit only if later
   comms/kernel work pushes V up. Honestly log the kill.
2. If `V ≥ 1.3` and rises with `num_speculative_tokens` → **GO**: implement union-capped /
   route-biased tree shaping, re-measure, prove it lifts S losslessly.

This keeps my novel work **measurement-gated**, non-duplicative of Charles's floor model
(he modeled it; nobody has measured whether the union tax actually bites on FP8+graphs), and
honest about the regime.
