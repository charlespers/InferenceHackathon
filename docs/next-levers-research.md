# Next Levers — prioritized research for the GPU agent (B=1 max tok/s)

> Planning-agent analysis written while the GPU agent runs `docs/gpu-agent-experiments.md`. Numbers are
> the **honest, adversarially-verified** figures from the research/spec work, re-anchored to the **measured**
> on-box state (K5 kernel `e=0.46`; vLLM `192%128` crash confirmed; **no engine baseline yet**). Gains do
> **not** multiply — most overlap on the same weight-read term and share the fixed comms+sampling floor.

## TL;DR — do this first
**Get the engine baseline (E1: FP8 + `--enable-expert-parallel`) before any optimization.** Everything
below is relative to a number we don't have yet. The baseline also tells us *which term dominates* (weight
vs comms-floor vs sampling), which decides whether the byte levers (int4) or the latency levers
(spec-decode, comms) pay. Then, in order: **n-gram spec-decode** (cheapest real multiplier) → **int4
experts** (biggest byte win, gated on a checkpoint) → **down-proj kernel fix** (cheap, after Nsight) →
route-prefetch (research-bet, low priority).

## Lever priority

### L0 — Engine baseline (prerequisite, not optional)  → maps to E1
- **Why first:** every gain below is "× the baseline." We have a *roofline* (~994 weight-only / ~500–547
  realistic, H100 FP8) but **no measured tok/s**. Real engines often start far below roofline (launch tax,
  EP imbalance, CPU-side overhead), so the first number is likely *well under* 500 — and the gap *is* the
  opportunity surface.
- **Resolves:** real TTFT/TPOT/decode-tok-s, % of roofline, dominant term, per-GPU EP balance (hotspot?).
- **Test:** E1 in the experiment queue.

### L1 — n-gram / prompt-lookup speculative decoding  → new exp E6 (add to queue)
- **Gain (verified):** **~1.1–1.4×** decode tok/s on structured/repetitive/code prompts (τ≈1.2–1.6),
  ~1.0–1.15× on free prose. The **one multiplier orthogonal to every byte lever** — it reduces *memory
  passes per emitted token*. Zero training.
- **Derivation / caveat:** `speedup = τ / (1 + c·draft_positions)`. On this MoE the verify pass reads the
  *union* of experts the drafted positions touch (`c≈0.27/position`, ~5× a dense model), so **keep trees
  narrow/shallow** — bushy trees go *net-negative*. Don't expect the dense-model 1.8–2× headline on FP8 MoE.
- **Test:** relaunch with `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3}'`
  (validate the flag form against vLLM 0.10.1), re-run `bench/measure.py`, read `spec_accept_rate`/τ + tok/s.
  Go/no-go: tok/s up **and** acceptance > ~25% on the test prompt; if acceptance is low, the draft is wasted.
- **Independent.** Stacks on the byte levers. Effort: low. **Recommendation: do-soon (right after E1).**

### L2 — INT4/AWQ expert weights  → new exp E7
- **Gain (verified, NOT 2×):** halving the **66%** routed-expert byte term ≈ **~1.13–1.20× e2e** decode at
  ctx≤8K (the same shape as the FP8→FP4 estimate). The full 2× only hits the expert term; attention,
  lm_head, KV, and the comms/sampling floor don't shrink, and the comms floor's *share* grows as weights
  shrink, compressing the realized gain.
- **Mechanism / risk:** group-wise INT4 (g=128) on experts only; in-register unpack to half/fp8 inside the
  K5 `warp_dot` (compute has slack at B=1, so the unpack should be ~free if it stays in-register — *verify
  it doesn't become issue-bound*). **W4A16 only** — W4A4 is a throughput trap with zero B=1 benefit.
- **Accuracy gate:** expect **<1% MMLU / <1.5% code** for routed-expert-only INT4 on a large MoE; gate on
  bitwise/PPL/task vs FP8 before shipping. **Blocker to resolve first:** does an AWQ/GPTQ-INT4
  Qwen3-235B-A22B checkpoint exist for vLLM, or must we quantize? (That decides effort: hours vs a day.)
- **Overlaps** the weight-read term — do **not** multiply against a TP-layout gain on the same bytes.
  Effort: medium. **Recommendation: do-soon, gated on checkpoint availability + accuracy.**

### L3 — Down-proj kernel fix (k5b `e=0.405 → ~0.48–0.50`)  → maps to E4
- **Gain (honest, small):** lifts the *blended* K5 `e` from 0.46 to **~0.49–0.51** (~+6–10% on the MoE
  kernel only). Real but a micro-win vs L1/L2.
- **Diagnosis (from the A/B split):** the down GEMV's contraction is only 1536 → a 32-lane warp does ~3
  vectorized loads then a 5-step shuffle reduce (**reduce-overhead bound**), and staging all-8-experts `a`
  (48 KB smem) caps occupancy. **Fixes:** (a) per-slot CTA decomposition (stage only `a[slot]`=6 KB →
  higher occupancy); (b) sub-warp split-K (8 lanes/row, 4 rows/warp) to amortize the short reduce.
- **Test (measure before optimizing):** Nsight on `k5b_down_warp` (E4) — confirm **low achieved occupancy**
  + **warp-reduce/issue stalls** + DRAM-throughput < gate/up's. Then I'll push `kernels/k5_experts_warp2.cu`
  with the fix; the agent re-runs `k5_microbench` and reports the new `e`. **Recommendation: do after the
  E4 Nsight** (so the fix is targeted, not speculative).

### L4 — Route-prediction / speculative expert prefetch  → research-bet, low priority
- **Honest verdict:** at B=1 **all weights are HBM-resident**, so there's no PCIe fetch to hide — classic
  "expert prefetch" (EdgeMoE/Fiddler) is **moot here**. The only credible angle is feeding the
  `markov_matrices` route prediction into the **spec-decode draft** (pre-stage which experts drafted tokens
  will hit), i.e. a *better drafter*, not a standalone lever.
- **One experiment that proves/kills it:** from `routing_stats.json`, compute next-token top-8 expert
  **prediction accuracy** of the markov model. If it can't predict the active-expert set well above chance
  (8/128), the prefetch/draft-priming idea is dead. **Recommendation: needs-data; defer until L1–L3 land.**

## Honest cumulative picture (do NOT multiply)
Once E1 gives a baseline `B` tok/s: **+n-gram spec** (~1.1–1.4×, orthogonal) and **+int4 experts**
(~1.13–1.20×, overlaps the weight term) realistically combine to roughly **~1.3–1.7× over `B`** at short
context — *capped by the comms+sampling+host floor*, which neither lever touches. The down-proj fix adds a
few % to the MoE-kernel portion only. The biggest unknown is `B` itself and its dominant term: if the
baseline is comms/launch-floor-bound (likely at first), **CUDA graphs (E3) + the serving fast-path** matter
more than any byte lever until the floor is paid down.

## What the engine baseline (E1) resolves
- Is decode weight-bound (→ int4 pays) or floor-bound (→ graphs/comms/spec pay)?
- How far below the ~500–547 roofline are we, and is EP load-imbalance visible in per-GPU util?
- The denominator for every gain above — turning predictions into measured deltas in the Results Log.
