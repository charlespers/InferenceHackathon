# Faster B=1 decode for Qwen3-235B-A22B on 8×H100 — research synthesis + novel directions

**Date:** 2026-06-20 · **Author:** djamoils (LOOP-A) · **Status:** research handoff
**Scope:** cutting-edge + novel levers to cut per-token DECODE latency at **batch size 1**
(single-stream, latency-not-throughput) for the comms-bound 235B-A22B / 8×H100 target.

> Provenance: two deep-research runs (fan-out web search → fetch primary sources →
> 3-vote adversarial verification). Run 1 (`wf_539703c7-151`) fetched 18 sources, confirmed
> 9 claims (verifier hit a session limit mid-synthesis → abstentions, NOT refutations).
> Run 2 (`wf_07e8b2cc-0b1`) is the focused follow-up on Kog delayed-TP + speculative/stale-TP
> prior art. Verified claims are marked ✅; real-source-but-not-vote-verified 🟡; my analysis ◆.

---

## 0. The one number that should drive everything

At 85 tok/s a token costs **~11.8 ms**, but the modeled terms only sum to ~6.8 ms
(comms 3.0 + weights 3.3 + KV 0.5). **~5 ms (>40%) is unmodeled overhead** — kernel
launch/sync, sub-roofline bandwidth (K5 hits ~46% of peak, not 80%+), CPU-scheduler bubbles.
That gap **plus** the ~3 ms comms wall is where the wins are — **not** the weight bytes most
of the team is optimizing. At B=1 the regime is **collective-latency-bound + overhead-bound**,
not bandwidth-bound on the comms side (the 8 KB all-reduce is far below the BW-bound regime).

---

## 1. Kog AI — what's real, what's marketing

**Verdict: real engineering; the headline number does NOT transfer to our case.**

- ✅ (3-0) Flagship "3,000 tok/s/request" (8×MI300X) / "2,100" (8×H200), B=1, FP16, no
  spec-decode, is on a **2B *dense* model**. MoE "up to 1.6T" is **discussed, not demonstrated**.
  [blog.kog.ai](https://blog.kog.ai/real-time-llm-inference-on-standard-gpus-3-000-tokens-s-per-request/)
- ❌ (0-2, refuted) The specific "KCCL all-reduce <3µs vs NCCL ~8µs" figure could not be
  substantiated from primary sources — treat the exact number as marketing.
- 🟡 Two real, stealable ideas: **(a) a single-kernel (megakernel) latency-first engine**
  [single-kernel MI300X](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/),
  and **(b) "Delayed Tensor Parallelism" (DTP)** — see §6, this is the one that matters for us.

The takeaway is the **architecture, not the number**: a megakernel that treats TP comms as a
latency-first, kernel-resident, *deferred* primitive — exactly the convergent answer to our wall.

---

## 2. State of the art, rated for the comms-bound B=1 MoE regime

| Lever | Mechanism | Verified | Payoff here |
|---|---|---|---|
| **Megakernel / persistent kernel** (MPK, HazyResearch) | Whole forward step in one launch; SM-level dep graph; in-kernel remote loads/stores so comms overlaps compute & never stalls on CPU scheduler | ✅ MPK 3-0; single-A100 14.5→12.5 ms (~1.16×, approaching 10 ms roofline); up to 1.7× e2e. HazyResearch B=1 Llama-1B ~2.5× vs vLLM 🟡 | **Highest hidden-overhead lever.** Attacks the ~5 ms tax + enables in-kernel collectives. This is our K6 — finish it. [MPK](https://arxiv.org/abs/2512.22219) · [no-bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) · [TP-Llama](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro) |
| **NVLS / multimem all-reduce** (TRT-LLM MultiShot) | In-switch reduction; **2 steps regardless of GPU count** vs Ring's 2N−2; biggest win at small msg / B=1 | 🟡 (NVIDIA, verifier abstained) | **Biggest *direct* comms lever.** Latency-targeted (not BW). ~16µs → ~3–5µs ⇒ comms 3.0 → ~0.6–1.0 ms. [MultiShot](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/) |
| **Fused AllReduce-RMSNorm** (TokenWeave) | Collective on 2–8 SMs via multimem, frees rest for compute | 🟡 8×H100 up to 1.28× | **Low for pure B=1** — overlap needs ≥~1024 tokens; nothing to hide behind single-stream. Useful only with a spec tree. [TokenWeave](https://arxiv.org/abs/2505.11329) |
| **Low-precision collectives** (Flash Communication) | Quantize the communicated tensor | 🟡 | **Low — the honest trap.** 8 KB is latency-bound; halving bytes barely moves it. Wrong lever at B=1. [Flash Comm](https://arxiv.org/abs/2412.04964) |
| **Spec-decode** (EAGLE3 → MTP/EAGLE-3.1) | Verify N tokens per forward pass | ◆ + repo docs | **Only lever that amortizes the *serial collective count*.** Accept ~3–3.5 ⇒ comms/token AND weights/token both ~÷3. Correctly the main bet. |

---

## 3. Honest reprioritization (ranked for OUR regime)

1. **Spec-decode (EAGLE3)** — amortizes everything serial. Already the bet.
2. **NVLS/multimem all-reduce** — direct hit on the 3 ms comms wall; biggest *unclaimed* verified win.
3. **Megakernel (K6)** — the ~5 ms hidden overhead + in-kernel comms; highest ceiling.
4. **FP8 weights** — real, but the term we're *least* bottlenecked on vs comms+overhead.
5. **Adaptive-k / KV-fp8** — secondary; best **repurposed** (see N3).

Byte-reduction levers (FP8, adaptive-k, KV-fp8) barely touch the dominant walls (collective
latency + per-step overhead). Re-aim effort at comms + overhead.

---

## 4. Novel directions — ranked (🟢 build-now / 🟡 research-risk / 🔴 moonshot)

### N1 🟢 Comms-resident megakernel: one launch, in-kernel NVLS reductions
The never-quite-combined thing. Megakernel work is mostly single-GPU/throughput; NVLS is
invoked *via NCCL/TRT-LLM with a launch per collective*. **Fuse them:** build K6 and issue the
all-reduce as an inline **NVLink-multimem reduction from inside the kernel** (TP-Llama remote
load/store + NVSwitch multicast). Kills the 188×2 collective *launches* AND drops each collective
to in-switch latency at once — attacks comms wall + overhead tax in one structure.
**Expected ◆:** comms 3.0 → ~0.7 ms + reclaim ~0.5 ms launch tax ⇒ ~85 → ~150–180 tok/s
*before* spec-decode, then multiplies with EAGLE3. Highest-ceiling concrete build.

### N2 🟡 Expert-deduplicated speculative verification (MoE × spec-decode, done right)
When verifying an EAGLE3 token tree of N candidates, **union expert sets across the tree and
read each unique expert's weights once**, applying to every token that routes to it. Dense
spec-decode amortizes dense weights; MoE gives an *extra* win via **temporal routing coherence**
(adjacent tokens share top-8), so unique experts ≪ 8N. Turns the #1 byte term from O(8N) →
O(unique experts). Pairs with K5's no-padding-floor variable-`nslot` kernel.
**Expected ◆:** ~1.5–2× extra on the weight term on top of spec-decode.

### N3 🟡 Adaptive-k as a *drafter*, not a final-output knob
Sidesteps adaptive-k's quality-gate problem: run the **EAGLE3 draft** at aggressive adaptive-k
+ INT4 (fast, approximate); run **verify** at full FP8 + top-8 (exact). The exact verifier makes
adaptive-k **provably lossless**. Asymmetric precision/sparsity across draft vs verify.

### N4 🔴 Speculative / stale tensor parallelism — break the serial all-reduce barrier
The boldest. Each GPU holds its local 1/8 partial after a TP block; today it must wait for the
all-reduce before the next layer. Instead **predict the reduced activation** (from the previous
token's reduced value — temporal coherence — or a tiny learned correction head), **run ahead
speculatively**, do a true all-reduce only every K layers ("comms refresh"), reconcile drift.
Transformers tolerate small activation perturbations (the whole field quantizes them).
**Mechanism:** "periodic-sync TP" — cuts 188 collectives → 188/K (K=4 ⇒ comms ~0.75 ms).
**Risk:** quality drift; needs a parity gate. **See §6 for how this relates to Kog DTP** — this is
the inference-only, no-retraining, approximate version of what Kog does losslessly via training.

### N5 🟡 Asymmetric parallelism: replicate-dense / EP-sparse (kill the attention all-reduce)
Attention weights are only ~6–7 B params (~7 GB FP8) — **replicate attention on all 8 GPUs
(zero attention all-reduce)**, EP all-to-all only for MoE. Deletes **94 of 188 collectives**.
Prior "attention-replication net loss" finding likely kept TP *and* added replication; do it
fully and pair with hot-expert replication (Jaymin) + adaptive-k to fix EP's 2.6× imbalance.
Re-measure as a **package**. **Expected ◆:** comms → ~1.5 ms if imbalance solved.

### N6 🟡 TP=4 (or 2) × replicate, not TP=8
Collective latency grows with participant count. FP8 235B/4 ≈ 59 GB/GPU fits in 80 GB.
**TP=4 with 2-way replication** = two latency-optimized replicas, cheaper 4-way collectives.
For B=1 *latency* (not throughput) can beat TP=8. Cheap A/B on the existing harness.

---

## 5. DEEP DIVE — Speculative/Stale TP (N4) vs Kog's Delayed TP

This is the comparison worth understanding precisely. Both **defer the per-layer all-reduce and
overlap it with later work** instead of blocking on it. The difference is *how they stay correct*.

**Kog Delayed Tensor Parallelism (DTP)** — from the primary source
([delayed-TP blog](https://blog.kog.ai/delayed-tensor-parallelism-for-faster-transformer-inference/)):
- 🟡 *"at the end of a module we launch the communication of each device local output to all other
  devices, though we do not all-reduce those outputs straight away"* — it **defers + overlaps**,
  does not skip/approximate.
- 🟡 **Lossless** — *"a scaling factor (√L) compensates for delayed aggregation to preserve
  numerical equivalence."* Aggregation happens δ layers later; result equals standard TP.
- 🟡 **Requires the model to be TRAINED with the DTP architecture** — *"training a LLM with the
  DTP architecture gets the best of both worlds."* It is an **architectural change**, not a
  drop-in serving trick.
- 🟡 Hides the collective **behind weight streaming**, explicitly targets *"batch-size-one token
  generation speed…the metric that matters."* No custom comms library is described in this post
  (the overlap comes from the architecture, using standard collectives).
- 🟡 Demonstrated on a **2B model** (MI300X / H200).

**The key mechanistic insight DTP exploits (and we can too):** at B=1 there is ~0 compute to hide
comms behind — **but there is ~3.3 ms of weight streaming per token.** NVLink collectives and HBM
weight reads use **different hardware paths**, so an (exact) all-reduce of layer L can run
*concurrently* with the weight-load of layers L+1…L+δ. That overlap is the whole game.

**How N4 (speculative/stale TP) relates:**

| | Kog DTP | N4 Speculative/Stale TP |
|---|---|---|
| Defers the all-reduce? | Yes, δ layers | Yes, K layers |
| Correctness | **Lossless** (√L-compensated) | **Approximate** (tolerate drift) or **speculative** (predict + rollback) |
| Needs retraining? | **Yes** — model trained in DTP form | **No** — runs on stock Qwen3-235B |
| What fills the gap | Exact deferred value, overlapped w/ weight streaming | Predicted/stale value (temporal coherence or learned correction) |
| Risk | Low quality risk, high *adoption* cost (can't retrain 235B at a hackathon) | Low adoption cost, **quality risk** (needs parity gate) |
| Relationship | The lossless, training-time ideal | The inference-only, no-retraining approximation of the same idea |

◆ **Bottom line:** N4 is essentially *"DTP without the retraining"* — we trade Kog's √L exactness
for an approximation we can validate with a parity gate. **There is a third, safer middle option
that may dominate both:** **exact deferred-overlap on the stock model** — overlap layer L's *exact*
NVLS all-reduce with layers L+1…L+δ weight streaming, deferring only as far as the data dependency
*actually* allows (the residual needs L's reduced output before L+1's attention, but the MLP
all-reduce and the next attention's QKV weight-load can overlap). This is lossless, needs no
retraining, and is the natural thing to bake into the N1 megakernel. **Recommended sequencing:**
do exact deferred-overlap first (free, lossless), then push to stale/speculative (N4) only if the
exposed-comms residue is still material. Whether N4 (true cross-layer staleness on a stock model)
is novel for B=1 MoE decode is the open question Run 2 (`wf_07e8b2cc-0b1`) is checking; preliminary
read is that **exact** overlap (Flux/async-TP/DTP) is well-explored but **approximate cross-layer
staleness on a non-retrained model for single-stream decode is largely untried.**

---

## 6. The multiplicative stack (the headline)

Not alternatives — they multiply, because they hit *different* terms:

```
EAGLE3 spec-decode            (~3× on comms+weights via amortization)
  × comms-resident megakernel + NVLS   (comms 3.0→~0.7 ms, reclaim ~0.5 ms overhead)
  × expert-dedup verification          (~1.5–2× on the weight term)
  × adaptive-k-as-drafter              (cheaper draft, lossless)
  [+ exact deferred-overlap / N4 if comms residue remains]
```
Conservatively this is the path from **85 tok/s toward the 300–540 tok/s roofline band.**

---

## 7. Next directions / action plan

1. **Finish Run 2** (`wf_07e8b2cc-0b1`) to confirm whether cross-layer stale TP is genuinely
   untried, and to pin Kog DTP's δ/√L details. (Session quota now clear.)
2. **Prototype N1** — inline NVLS multimem reduction inside the K6 megakernel. Highest ceiling;
   it's finishing the megakernel we started + one in-kernel collective primitive.
3. **Implement exact deferred-overlap (§6)** — lossless, no retraining, folds into N1.
4. **A/B N5 + N6** on the existing harness this week — cheap, each is a candidate 1.5–2× on comms.
5. **Repurpose adaptive-k as a drafter (N3)** — makes our existing work lossless and useful.
6. Validate everything with the token-level **parity gate** before trusting any throughput number
   (esp. N4, which is approximate).

## Sources
- Kog: [3000 tok/s](https://blog.kog.ai/real-time-llm-inference-on-standard-gpus-3-000-tokens-s-per-request/) ·
  [Delayed TP](https://blog.kog.ai/delayed-tensor-parallelism-for-faster-transformer-inference/) ·
  [single-kernel MI300X](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/) ·
  [AMD 3.5×](https://www.amd.com/en/blogs/2025/kog-reaches-3-5x-breakthrough-inference-speed-on-amd-instinct-mi.html)
- Megakernels: [MPK arXiv 2512.22219](https://arxiv.org/abs/2512.22219) ·
  [HazyResearch no-bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) ·
  [TP-Llama megakernel](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-intro) ·
  [Compiling LLMs into a megakernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17)
- Collectives: [TRT-LLM MultiShot](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/) ·
  [TokenWeave 2505.11329](https://arxiv.org/abs/2505.11329) ·
  [Flash Communication 2412.04964](https://arxiv.org/abs/2412.04964)
- MoE/spec: [DeepSeek-V3 (MTP) 2412.19437](https://arxiv.org/pdf/2412.19437)
</content>
</invoke>
