# N4 — Speculative / Stale Tensor Parallelism (break the serial all-reduce barrier)

**Date:** 2026-06-20 · **Author:** djamoils (LOOP-A) · **Status:** design + experiment spec (pre-GPU)
**Parent:** [`research/fast_decode_research.md`](fast_decode_research.md) §4 (N4), §5 (vs Kog DTP)

> **Why now.** Charles measured custom collectives (NVSHMEM): best primitive is put-barrier
> all-to-all at **17µs × 188 = 3.2 ms/tok → caps ~310 tok/s**. His own conclusion: *a faster
> collective alone tops out ~310 tok/s; 1000 needs **fewer** collectives, not just faster ones.*
> N4 attacks the **count** (188 → 188/K), which is the lever a faster primitive can't reach.
> Nobody on the team is working on it; it is (preliminarily) untried in the literature in this form.

---

## 1. The idea, precisely

Standard TP forces a **serial barrier** at every all-reduce: GPU *i* holds a partial output
`p_i` after a TP block and **must** wait for `y = Σ_i p_i` before the next layer can start.
94 layers × 2 collectives = **188 serial barriers/token ≈ 3 ms**, the dominant wall.

N4 removes the barrier by letting each rank **proceed on an approximation `ŷ` of the reduced
activation** and doing the *true* all-reduce only **every K layers** ("comms refresh"),
reconciling accumulated drift at the refresh. Collective count drops **188 → 188/K**.

Three substitution policies for `ŷ` (the probe in §4 decides which, if any, hold quality):

| Policy | `ŷ` = | Cost | Intuition |
|---|---|---|---|
| **P1 local-only** | `p_i` (this rank's partial, no reduce) | 0 | Crudest; bounds the error. Each rank runs its own shard's view. |
| **P2 temporal-stale** | the *true* reduced value of this layer from the **previous token** | 1 cache read | Adjacent decode tokens have highly correlated activations. |
| **P3 predicted** | `f(p_i, history)` — a tiny learned/low-rank correction from local partial + last reduced value | small GEMV | Best accuracy; "speculate the sum, correct at refresh." |

At each **refresh layer** (every K), do the real all-reduce and **re-inject the exact residual**,
bounding drift to a K-layer window. This is the approximate, **no-retraining** analogue of Kog's
Delayed TP (which is lossless but requires training the model with a √L-compensated DTP architecture
— infeasible for stock Qwen3-235B at a hackathon). See parent §5 for the full comparison.

**Correctness stance:** N4 is *approximate by construction*. It is only viable if the model's
output is robust to K-layer activation staleness. **That is a measurable property** — §4 measures it
before any kernel is written.

---

## 2. Relationship to prior art (VERIFIED by research run `wf_07e8b2cc-0b1`, 23/25 claims confirmed)

**★ Ladder-Residual (arXiv 2501.06589, ICML 2025) — the nearest neighbor and the key paper.**
Reroutes the residual so module *i+1* consumes the **stale output of module *i−1***:
`x_{i+1} = h_{i+1}(x_{i−1}) + x_i`. This decouples `h_{i+1}`'s compute from the all-reduce of `x_i`,
so the collective **overlaps the next layer's compute**. Justification (✅ 3-0): *"activation changes
slowly in Transformer — the norm of each update is small vs the residual."* That is **literally the
N4 hypothesis** (tolerate activation staleness), at fixed **depth-1** staleness.
- **It WORKS at exactly our regime (✅ 3-0):** B=1, TP=8, 8×H100 — 70B model → **23.71% decode-latency,
  30.79% tok/s** (29% e2e; up to ~60% on slow/Ethernet comms). Proves stale-input
  dependency-restructuring pays at B=1 *despite ~0 local compute*, by restructuring dependencies (not
  relying on a fat GEMM — which is why FLUX/FlashOverlap fail here).
- **But it is NOT drop-in lossless (✅ 3-0):** it's an architecture change needing training from
  scratch (1B/3B) or **~3B-token retraining** to convert a checkpoint. Zero-shot conversion of
  Llama-3.1-8B's upper half **collapsed 56.11 → 41.65 avg** before retraining recovered it. Only the
  *upper half* of layers was converted ("touching lower layers destroys knowledge"). 70B variant is
  **dense; MoE untested.**

**Kog Delayed TP — CORRECTED (✅ 3-0; the "lossless deferral" framing was REFUTED 0-3).**
DTP is **NOT** a lossless reorder. It is an **approximate architectural variant that must be
pretrained**: un-reduced local activations are forwarded with a **√L scaling factor to *mimic* the
all-reduce's scale**; *"training… with such no-communication architecture heavily degrades
performance"* and DTP-pretraining only *"claws back"* quality to *"very close to"* vanilla. So Kog's
"lossless" means *quality-preserving after retraining*, not mathematical equivalence. Separately,
their **collective library** is a real, reusable idea: **NaN-sentinel polling** (publish-dependent
buffers init to NaN; consumers poll only the values they need) — **0.80–0.93µs vs 7.59–7.88µs
(~9×)** in a *synthetic* microbench. **This collective trick is separable from the lossy
architecture** — usable with no retraining (see N4b below).

**Compute-comm overlap (FLUX 2406.06858, FlashOverlap 2504.19519):** lossless, but **collapse at
B=1** (✅ 3-0). They hide the collective *behind GEMM compute*; at M=1 there aren't enough tiles/warps
— FLUX even reports **slowdowns at small m** (0.95× at m=64). Confirms overlap-only cannot solve the
B=1 wall; you must restructure dependencies (Ladder/N4), not just overlap.

**Tangential:** Flash Communication (lossy INT4/6/8 payload compression, prefill-focused);
DICE (stale across diffusion *timesteps*, not layers, lossy). Neither applies.

### Novelty verdict (✅ confirmed)
The specific **"sync every K layers, tolerate per-layer TP drift, reconcile periodically"** for B=1
MoE autoregressive decode is **NOT directly covered by any source.** Ladder-Residual is the **K=1**
special case (depth-1 staleness) **with retraining**; Kog DTP is a pretrained √L-deferral. **Genuine
novelty space exists for (a) K>1 and (b) a drop-in *runtime-only* (no-retrain) variant.** ⚠️ **But the
convergent warning across every quality-recovering method (Ladder, Kog) is that they ALL require
training** — strongly predicting a pure no-retrain stale-TP scheme will be **lossy and need
fine-tuning.** That prediction is exactly what Experiment 1 (§4) tests cheaply.

### Three concrete tiers this collapses N4 into
- **N4a (lossless, no retrain, do first): exact deferred-overlap.** Overlap layer L's *exact* NVLS
  all-reduce with L+1…L+δ **weight streaming** (NVLink vs HBM = different HW paths), deferring only as
  far as the true data dependency allows. Free quality; belongs in the N1 megakernel.
- **N4b (no retrain, faster primitive): Kog-style NaN-sentinel collective** issued in-kernel — cuts
  per-collective latency without touching the architecture. Stackable with everything; no quality risk.
- **N4c (the novel bet): runtime-only K-layer stale TP.** The actual N4. High risk per the literature;
  Experiment 1 decides go/no-go. If it needs retraining → fall back to **adopting Ladder-Residual**
  (published, proven at B=1/8×H100) on Qwen3's upper layers as the de-risked version.

---

## 3. Honest caveat absorbed from Charles (`660ef9f`)

The combined stack is **sub-multiplicative and regime-flipping**: spec-decode *amortizes* the comms
floor while a megakernel *removes* it (don't double-count), and as you climb, the regime flips
floor-bound → weight-bound, so the optimal spec-tree shrinks. **Implication for N4:** its win is
largest **while still floor-bound** (i.e. before megakernel+NVLS land). Sequence accordingly — N4
buys the most *now*; its marginal value shrinks once N1 removes the floor. Re-evaluate each slot via
Charles's `tree_spec_optimizer` + `backout_floor`. N4 and N1 are partially **substitutes** on the
comms term, not pure multipliers.

---

## 3.5 Performance ceiling (offline model — `tools/stale_tp_ceiling.py`, no GPU)

Stale-TP's *only* mechanical effect: it hides AR(L) behind the weight-read of L+1 (NVLink vs HBM =
concurrent engines), which `comms_floor.md` §3 proved is impossible *losslessly* at B=1 (serial dep).
Modeling that overlap against the team's own latency numbers:

| per-collective C | comms (serial) | tok/s | comms (+stale) | tok/s (+stale) | gain |
|---|---|---|---|---|---|
| 16µs (baseline) | 3.01 ms | 214 | 1.45 ms | 322 | 1.50× |
| 10µs (CUDA-graph) | 1.88 ms | 282 | 0.32 ms | 505 | 1.79× |
| **7µs (multimem, lever 2)** | 1.32 ms | 336 | **0.00 ms** | **602** | 1.79× |
| 5µs (best case) | 0.94 ms | 385 | 0.00 ms | 602 | 1.57× |

**Key result: stale-TP STACKS with Charles's multimem one-shot AR.** Once C ≤ ~8µs (=weight-read/2)
the *entire* comms term hides behind weight reads → decode hits the ~weight+KV roofline (~600 tok/s
idealized). Stale-TP converts "cheaper comms" (lever 2) into "free comms." Its marginal value *grows*
as C shrinks — the opposite of a substitute. **This is the regime where stale-TP pays.** Everything
here is a performance ceiling **gated on quality** (§4).

## 4. Experiment 1 — Staleness-tolerance probe (OFFLINE, NO GPU SLOT)

**Goal:** the single go/no-go number — *how much K-layer all-reduce staleness does Qwen3-235B
tolerate before output quality breaks?* Pure simulation; no kernels; no GPU-slot queue.

### 4.1 Hook
Monkeypatch the TP all-reduce, identical pattern to the team's existing
`experiments/adaptive_topk/vllm_adaptive_moe.py` (which monkeypatches `FusedMoE`):
- Target `vllm.distributed.communication_op.tensor_model_parallel_all_reduce` (and/or the
  `RowParallelLinear` reduce in attention `o_proj` + MoE down-proj).
- Wrap it with a scheduler that, per `(layer_idx, step)`, either calls the **real** all-reduce
  (refresh layers: `layer_idx % K == 0`) or returns a **substituted** `ŷ` per policy P1/P2/P3.
- Keep a per-layer cache of the last true reduced tensor (for P2/P3) and a small ring buffer for
  drift accounting.

### 4.2 Sweep
- **K ∈ {1, 2, 3, 4, 6, 8}** (K=1 = exact baseline / control).
- **Policy ∈ {P1 local-only, P2 temporal-stale, P3 predicted}** (start P1→P2; P3 only if P2 close).
- **Refresh phase**: align refresh to layer 0 vs interleaved (does *which* layers are exact matter?).
- Optional: only stale the **MLP/MoE** all-reduce (keep attention exact) vs both — attention is
  more sensitive; staling only the cheaper-to-be-wrong collective may preserve quality.

### 4.3 Metrics (vs the K=1 exact baseline)
1. **Greedy token parity** — % of next-token argmax identical to exact. *Primary gate.*
2. **Perplexity / NLL** on a small eval set (WikiText slice + a few task prompts).
3. **KL(exact ‖ stale)** of the output distribution, mean & p95 across positions.
4. **Drift growth** — ‖ŷ − y‖ / ‖y‖ vs layer depth within a refresh window (does error compound or
   stay bounded?).

### 4.4 Success criteria (decide the kernel build)
- **GO (strong):** ≥99% greedy parity at **K≥2** with P2 or P3 → real lossless-enough win; build kernel.
- **GO (conditional):** parity holds only with attention-exact + MLP-stale, or only K=2 → narrower but
  still ~2× on the comms term; build the restricted variant.
- **NO-GO:** parity < ~97% even at K=2 / drift compounds → shelve N4; **fall back to exact
  deferred-overlap (§2) + N5 (halve collective count by replicate-dense/EP-sparse)**, which give
  collective-count reduction *losslessly*.

### 4.5 Deliverables
- `experiments/stale_tp/probe.py` — the monkeypatch + scheduler + metric harness (mirrors
  `vllm_adaptive_moe.py` install pattern).
- `experiments/stale_tp/sweep.sh` — runs the K × policy grid (small ctx, ~200 tokens, greedy).
- `results/stale_tp/` — parity/PPL/KL/drift tables per (K, policy). (gitignored; `git add -f`.)
- Go/no-go writeup appended here as §6.

> Cost note: the probe needs a real Qwen3 forward pass, so it wants the model loaded — but it is
> **read-only on correctness (no perf claim)**, so it can run during any slot or piggyback on an
> EAGLE3 bring-up, and the parity logic itself is GPU-light. If a full-model slot is scarce, first
> validate the harness on a **small Qwen3-MoE** (e.g. 30B-A3B) — staleness tolerance is a structural
> property that should qualitatively transfer.

---

## 5. Experiment 2 — Kernel (only if Experiment 1 = GO)

Folds into the N1 megakernel + Charles's comms:
- Each rank runs ahead on `ŷ` (P2/P3) through a K-layer window; the residual stream is **local** in
  the window.
- At refresh layers, issue the true reduce as an **inline NVLS multimem reduction inside the
  megakernel** (N1) and re-inject the exact residual.
- **Design A (sync-light):** real all-reduce every K layers; downstream uses last-true value between.
- **Design B (speculative + rollback):** run ahead speculatively; at refresh, compare ŷ vs y, and if
  drift exceeds a threshold, **roll back** the window and recompute exactly (bounded worst case).
  Pairs naturally with spec-decode's existing verify/rollback machinery.
- **Parity gate**: every config must pass the token-level parity harness before any tok/s is trusted.

---

## 6. Go/no-go results — ❌ NO-GO (measured 2026-06-20 10:24 UTC, bf16-TP8, 8×H100)

**Verdict: runtime-only stale TP (no retraining) catastrophically destroys quality.** Greedy
parity vs exact, 10 prompts, results in `results/stale_tp/`:

| sweep point | mean_agreement | exact | output |
|---|---|---|---|
| exact (control) | — | — | correct (`def is_palindrome(s): return s == s[::-1]`) |
| **lyr_proxy_k2** (core hypothesis) | **0.000** | 0.00 | gibberish from token 1 |
| lyr_proxy_k4 | 0.032 | 0.00 | gibberish |
| lyr_proxy_k8 | 0.003 | 0.00 | gibberish |
| lyr_local_k2 (control, must degrade) | 0.023 | 0.00 | gibberish ✓ |
| tmp_proxy_k2 (temporal) | 0.046 | 0.00 | gibberish |

**Sanity gates all pass** (this is real, not an artifact): the fork patch reached all 8 TP workers
(`VllmWorker TP0..7 [stale_tp] ctl reload`), `exact` reproduces correct output, and the `local`
control degrades — so the hook genuinely perturbs the all-reduce. Even the **gentlest** setting
(K=2, reuse the all-reduce result from 2 layers back) yields **0% agreement** — output is gibberish
from the first decode token.

**Conclusion (honest, scoped):** the *no-retrain, runtime-substitution* form of stale-TP is **dead**
for Qwen3-235B B=1 decode. This **confirms the literature prior** (Ladder-Residual / Kog DTP both need
training). It does NOT refute stale-TP-with-retraining (Ladder works — 23.7%/30.8% at B=1/8×H100), but
that requires ~3B-token retraining → **out of hackathon scope.** Error-feedback / attention-exact
recovery (the CONDITIONAL branch) is implausible to bridge 0.000→0.99 and is not pursued.

**Why it fails (mechanism):** the substituted value is the all-reduce *output* (a delta added to the
residual). Reusing a 2-layer-old delta as the current layer's delta is a large, wrong perturbation —
unlike Ladder, which reroutes the *residual* (stale by 1) **and is trained to tolerate it**. At
runtime, untrained, the residual stream does not change "slowly enough" at the all-reduce granularity.

**Pivot (per DECISION.md NO-GO branch):** the surviving comms-floor levers are **lossless**:
(1) *exact deferred-overlap* — overlap each layer's exact NVLS all-reduce with the next layer's
weight-stream (no staleness, no quality risk; belongs in the megakernel); (2) Charles's **multimem
one-shot** AR (cut the per-collective constant). Stale-TP is killed as a runtime lever.

## 7. Resolved questions (from research run `wf_07e8b2cc-0b1`)
- ✅ No published K>1 periodic-sync drift-tolerant TP for B=1 decode — **N4c is novel**; Ladder-Residual
  (K=1, retrained) is the frontier.
- ✅ Kog DTP is approximate + pretrained (√L mimics scale); **does NOT transfer without retraining.**
- ✅ Async-TP / FLUX / FlashOverlap give **~no B=1 benefit** (need compute to hide behind; FLUX slower
  at small m). Dependency-restructuring (Ladder/N4), not overlap, is the demonstrated B=1 path.
- ⏳ Open: does Ladder-style staleness pay on a 235B **MoE** (EP all-to-all, not just dense TP
  all-reduce)? Untested anywhere — a second novel angle.
</content>
