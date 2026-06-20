# Confidence-Adaptive Top-k Expert Selection — SOTA Research Brief

**Target model:** Qwen3-235B-A22B — fine-grained MoE, **top-8 of 128** routed experts, no shared experts, 94 layers, 22B active params ([HF model card](https://huggingface.co/Qwen/Qwen3-235B-A22B), [apxml specs](https://apxml.com/models/qwen3-235b-a22b)).
**Goal:** cut **batch-size-1** decode latency by routing confident tokens to fewer experts (smaller k) at inference, ideally **without retraining**, with **greedy decoding**.
**Date:** 2026-06-20.

---

## TL;DR — Recommended starting policy

> **Per-token cumulative-softmax-mass gating with a floor:**
> Sort router softmax probs descending; pick the smallest k whose cumulative mass ≥ **p = 0.90**, clamped to **k ∈ [2, 8]**.
> Equivalently as a schedule on the existing top-8 weights:
> **k = 2 if top-2 mass > 0.90; elif k = 4 if top-4 mass > 0.90; elif k = 6 if top-6 mass > 0.90; else k = 8.**

Rationale (detail below): top-p / cumulative-mass routing is the published mechanism that "spends" experts only where the router is uncertain ([Huang et al., ACL 2024](https://aclanthology.org/2024.acl-long.696.pdf)); a hard **k_min ≥ 2** protects the head experts, which in **fine-grained** models are *less* redundant and most damaging to drop ([DeepSeekMoE, 2401.06066](https://arxiv.org/html/2401.06066v1)); greedy decoding is the regime most tolerant of dropping low-mass tail experts ([Certain Head, Uncertain Tail, 2602.02443](https://arxiv.org/pdf/2602.02443)). Start at p=0.90, sweep p ∈ {0.85, 0.90, 0.95} and report avg-k vs MMLU/HumanEval/PPL.

---

## 1. Methods for dynamic / adaptive number-of-experts in MoE inference

| Method | How it decides k per token | Threshold / criterion | Retrain-free? | Headline result |
|---|---|---|---|---|
| **Dynamic Routing / "Harder Tasks Need More Experts"** (Huang et al., ACL 2024) — the canonical **top-p / cumulative-mass** method | Add experts in descending prob order until cumulative router softmax mass ≥ p (Eq. 4: smallest k s.t. Σ pⱼ ≥ p) | **p = 0.4** in their (from-scratch, normalized-prob) setup; sensitivity swept p∈[0.1,0.7] | Trained with it (not pure post-hoc) | Avg **1.76 experts/token** at inference, **+0.7%** avg / **+2.0%** BBH over fixed Top-2, with <90% of Top-2's active params. ([paper](https://arxiv.org/html/2403.07652v1), [ACL pdf](https://aclanthology.org/2024.acl-long.696.pdf)) |
| **AdaMoE** (Findings of EMNLP 2024) | Adds non-computational **"null experts"** to the pool and raises k; a token's real-expert count = (k − #nulls selected), so it varies per token | Load-balancing loss targets average null-expert usage (sets the *average* k, not a per-token hard threshold) | Needs light fine-tuning | Mixtral-8x7B: **−14.5% FLOPs and +1.69% acc on ARC-C** simultaneously. ([abs](https://arxiv.org/abs/2406.13233), [pdf](https://aclanthology.org/2024.findings-emnlp.361.pdf)) |
| **Ada-K Routing** (arXiv 2410.10456) | Lightweight learnable **allocator** module per layer predicts how many experts a token gets; trained with **PPO** (non-differentiable decision) | Learned per-token allocation; no fixed prob threshold | Adds + trains allocators | **>25% FLOPs cut, >20% speedup** with improved benchmarks on 4 MoE LLMs incl. Mixtral-8x22B. **⚠️ WITHDRAWN by authors** (coauthor disagreement) — treat as directional only. ([abs](https://arxiv.org/abs/2410.10456)) |
| **DTop-p / Sparsity-Controllable Dynamic Top-p** (arXiv 2512.13996) | top-p routing + a **PI controller** that adjusts the global prob threshold to hit a target average k | Controller drives a target avg active-expert count | Pre-training method | Confirms top-p gives confident tokens fewer experts, uncertain tokens more. ([pdf](https://arxiv.org/pdf/2512.13996)) |
| **LExI** (arXiv 2509.02753) — *layer-adaptive*, post-hoc | Per-**layer** (not per-token) active-expert count via sensitivity analysis | Keeps more experts in later/sensitive layers, fewer early | **Yes, no retrain** | On Mixtral / Qwen-MoE / DeepSeek-MoE, layer-adaptive reduction beats uniform reduction at equal compute on MMLU/PPL. ([pdf](https://www.arxiv.org/pdf/2509.02753)) |

**Takeaway on mechanism:** the cleanest knob for us is **cumulative softmax mass (top-p)** — it is exactly "route to fewer experts when the router is confident." Ada-K/AdaMoE achieve similar adaptivity but require training; LExI is retrain-free but only layer-granular (a good *complementary* coarse prior).

---

## 2. Quality impact of REDUCING top-k at inference, no retraining

**General law (all sources agree):** the **#1 (argmax) expert is critical**; the **2nd is often swappable**; **lower-ranked / tail experts drop cheaply**.

- **Plug-and-play routing study** (Shahout et al., 2510.03293): "the top-1 expert is critical for all MoEs… the top-2 expert in Mixtral / Phi-MoE can be swapped without major PPL loss"; **dynamic routing keeps MMLU within ≈0.02 absolute (≤2%)** of baseline top-k, no retraining. ([pdf](https://arxiv.org/pdf/2510.03293))
- **Mixtral 8x7B, halving experts (coarse, top-2→top-1):** noticeable but not catastrophic. Illustrative figures reported for static expert reduction (GLUE acc / WikiText-103 normalized PPL): full → top-4 → top-2 → top-1 ≈ **86.1 / 83.7 / 78.9 / 71.4** acc and **100 / 88.7 / 79.4 / 65.3** norm-PPL — i.e. coarse MoE pays a real price below its trained k=2 ([EAC-MoE, 2508.01625](https://arxiv.org/pdf/2508.01625); numbers from search summary, verify against the paper's tables before quoting in a paper).
- **Fine-grained models tolerate aggressive cuts:** Qwen2-57B-A14B and Qwen3-30B-A3B tolerate **~24–25% expert removal**, and **Qwen3-Coder-480B near-lossless at 50% experts pruned** under router-weighted pruning ([REAP, 2510.13999](https://arxiv.org/html/2510.13999)); retraining-free neuron-recombination pruning drops **25–50% of experts at near-baseline PPL** ([2509.10377](https://arxiv.org/pdf/2509.10377)).
- **Greedy vs sampling:** greedy is **more tolerant**. "Greedy-decoding accuracy stays stable even when active experts are cut to half the default top-k… top-half experts suffice for deterministic generation," whereas temperature **sampling (pass@n) is more sensitive** to expert reduction ([Certain Head, Uncertain Tail, 2602.02443](https://arxiv.org/pdf/2602.02443)). **This favors our greedy use case.**

---

## 3. Confidence-adaptive k vs static k at equal average compute

**Yes — adaptive beats static at equal average compute, consistently:**
- **Dynamic Routing (top-p)** activated **avg 1.76 experts** yet **beat fixed Top-2** by +0.7% avg / +2.0% BBH — i.e. *fewer* avg experts than static-2 *and higher* quality ([2403.07652](https://arxiv.org/html/2403.07652v1)). It spends experts on hard tokens (BBH 1.87 avg) and saves on easy ones.
- **LExI:** adaptive (per-layer) reduction strictly dominates uniform reduction on the accuracy-vs-compute frontier ([2509.02753](https://www.arxiv.org/pdf/2509.02753)).
- **Mechanistic reason:** routing confidence varies by token, layer (middle layers want more), and word type (content words want more) — so a fixed k is wasteful on easy tokens and starves hard ones ([Ada-K analysis, 2410.10456](https://arxiv.org/abs/2410.10456)).

**Typical mass thresholds:** post-hoc / inference-time top-p work commonly cites cumulative softmax mass in **~0.9–0.95**; the original from-scratch Dynamic-Routing used a lower p=0.4 because it renormalized over the full softmax. For **post-hoc gating over an already-sharp router (top-8 of 128)**, **p ≈ 0.90** is the right starting region — sweep 0.85 / 0.90 / 0.95.

---

## 4. Does FINE granularity make tail experts MORE or LESS droppable?

**Split the question by expert rank — this is the key nuance and it directly motivates a mass-based (not fixed-k) policy:**

- **TAIL (low-mass) experts → MORE droppable when fine-grained.** With 128 small experts the routing mass spreads across many experts, so the lowest-ranked of the top-8 carry little unique signal and prune cheaply ([2509.10377](https://arxiv.org/pdf/2509.10377); REAP near-lossless @50% on Qwen3-Coder, [2510.13999](https://arxiv.org/html/2510.13999)).
- **HEAD (high-mass) experts → LESS redundant when fine-grained.** DeepSeekMoE's own ablation shows fine-grained models are **MORE sensitive to disabling top routed experts** than coarse GShard×1.5 — fine-grained = "ultimate specialization," **lower redundancy among the head** (Pile loss degrades faster as top experts are masked) ([2401.06066 §4.5](https://arxiv.org/html/2401.06066v1)). Coarse Mixtral, by contrast, has redundant overlapping experts.

**Implication for Qwen3 (fine, top-8/128):** the **tail of the top-8 is a fat, cheap budget to cut**, *but you must keep the head intact*. That is exactly what a **cumulative-mass threshold with a k_min floor** does: it trims tail experts only when the head already owns most of the mass, and never drops below k_min head experts. A blind static "always top-4" risks clipping a genuine head expert on the hard tokens where fine-grained specialization matters most.

---

## Recommended starting policy (justified)

```python
# Per token, per MoE layer. Router gives softmax probs over 128 experts.
P_SORTED = sort(router_probs, descending=True)      # already have top-8 weights
P_MASS   = 0.90                                       # cumulative-mass threshold (sweep 0.85/0.90/0.95)
K_MIN, K_MAX = 2, 8                                   # floor protects head; cap = native k

k = smallest k in [1..8] s.t. cumsum(P_SORTED)[k-1] >= P_MASS
k = clamp(k, K_MIN, K_MAX)
# route token to top-k experts, renormalize their gates to sum to 1
```

Equivalent stepwise schedule on existing top-8 weights:
**k = 2 if mass(top-2) > 0.90  ·  elif k = 4 if mass(top-4) > 0.90  ·  elif k = 6 if mass(top-6) > 0.90  ·  else k = 8.**

Why these choices:
1. **Mass threshold, not fixed k** — only cuts when the router is confident; preserves quality better at equal avg compute (§3). p=0.90 is the standard inference-time region; sweep up to 0.95 if quality dips.
2. **k_min = 2 floor** — fine-grained head experts are low-redundancy and the argmax is critical (§2, §4); never run k=1 in the latency-critical path. (Optional: lower to k_min=1 only if a sweep shows it's safe.)
3. **Renormalize selected gates** — standard when dropping experts so the layer output magnitude is preserved.
4. **Greedy decoding** is our regime and the most tolerant of tail-expert dropping (§2) — so the average-k savings should come at minimal MMLU/HumanEval cost.
5. **Stack with a layer prior later** — LExI shows early layers tolerate deeper cuts; once the per-token policy works, allow a lower p (or lower k_min) in early layers, higher in late layers.

**Validation plan:** sweep p∈{0.85,0.90,0.95}; report **avg active experts/token** and **decode latency** vs **MMLU (5-shot, greedy)**, **HumanEval pass@1 (greedy)**, and **WikiText PPL**. Accept the most aggressive p that holds quality within ~1% absolute of the top-8 baseline.

---

## Most relevant references

1. **Huang et al., "Harder Tasks Need More Experts: Dynamic Routing in MoE" (ACL 2024)** — the canonical cumulative-softmax-mass (top-p) adaptive-k method; avg 1.76 experts beats fixed Top-2. → core mechanism. [https://aclanthology.org/2024.acl-long.696.pdf](https://aclanthology.org/2024.acl-long.696.pdf) / [https://arxiv.org/html/2403.07652v1](https://arxiv.org/html/2403.07652v1)
2. **"Certain Head, Uncertain Tail: Expert-Sample for Test-Time Scaling in Fine-Grained MoE" (2602.02443)** — fine-grained-specific; greedy tolerates cutting to half-k, sampling does not; head certain / tail uncertain. → justifies greedy + mass gating on Qwen3. [https://arxiv.org/pdf/2602.02443](https://arxiv.org/pdf/2602.02443)
3. **DeepSeekMoE (2401.06066, §4.5)** — fine-grained head experts are *less* redundant (more sensitive to disabling top experts) than coarse GShard. → justifies the k_min floor that protects head experts. [https://arxiv.org/html/2401.06066v1](https://arxiv.org/html/2401.06066v1)

Supporting: AdaMoE [2406.13233](https://arxiv.org/abs/2406.13233), Ada-K [2410.10456](https://arxiv.org/abs/2410.10456) *(withdrawn)*, LExI [2509.02753](https://www.arxiv.org/pdf/2509.02753), retraining-free expert pruning [2509.10377](https://arxiv.org/pdf/2509.10377), REAP [2510.13999](https://arxiv.org/html/2510.13999), plug-and-play routing [2510.03293](https://arxiv.org/pdf/2510.03293).
