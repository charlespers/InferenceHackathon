# Depth Reduction for B=1 Decode of Qwen3-235B-A22B on 8×H100 (vLLM)

**Question:** The decode is comms-bound (~188 serial all-reduces ≈ 3 ms/token). The most direct
structural attack is to **execute fewer layers**. Does depth reduction (early-exit, self-speculative
layer-skip draft, or static layer pruning) actually beat the comms floor at B=1, and what is the
minimal vLLM path?

**TL;DR / Bottom line (read this first):**

- The cleanest, lossless, highest-ceiling depth-reduction win is **self-speculative decoding where the
  DRAFT = the target model run only to an early-exit layer `L_d`** (logit-lens over the shared
  norm+LM head), and the **full 94-layer model verifies (lossless)**. Speedup =
  `τ / (γ·r + 1)` with `r = L_d/94` and `τ = (1−α^(γ+1))/(1−α)`.
- **But on Qwen3 specifically it probably does NOT beat the already-available EAGLE3 draft head**, and
  it carries real acceptance risk. Qwen3 is *pruning-fragile* (random-accuracy transition at ~20% depth
  vs ~50% for Llama-2), so the shallow exits that make the draft cheap (`r` small) are exactly where
  acceptance `α` collapses. The two requirements (small `r`, high `α`) fight each other on this model.
- **The pragmatic recommendation:** the off-the-shelf **`lmsys/Qwen3-235B-A22B-EAGLE3`** draft head
  already delivers ~3.0–3.5 accepted tokens/cycle, lossless, with a near-free draft — that is the
  depth-reduction-flavored win without the research risk. The novel self-speculative depth-draft is
  worth a **bounded experiment** only as an attempt to beat EAGLE3, and the realistic outcome is that
  it ties or loses. **Static layer pruning is the weakest option on Qwen3** and is not recommended
  beyond ~1 layer without healing.

---

## 0. Comms floor, honestly reframed

| Quantity | Value |
|---|---|
| All-reduces per token | 188 (2/layer × 94: attn out-proj + MoE down-proj, both `RowParallelLinear`) |
| All-reduce latency | ~16 µs |
| **Comms floor** | **~3.0 ms/token** |
| Observed decode | ~85 tok/s → **~11.8 ms/token** |
| **Comms as % of token budget** | **~26%** |

**Honesty correction up front:** comms is ~3 ms of an ~11.8 ms token, i.e. **~26%**, not the whole
budget. The other ~74% is weight reads (235B params, but only ~22B active + all 128 experts' router /
the 8 selected experts' weights streamed per layer), KV reads, kernel launch, and sampling. This
matters: **skipping a layer removes BOTH its 2 all-reduces AND its weight reads from the critical
path**, so the per-layer saving is larger than the comms-only share — skipping a fraction `f` of layers
saves ≈ `f` of *both* the comms floor and the per-layer weight/compute cost. So depth reduction attacks
the dominant cost, not just the 26% comms slice. The "comms-bound" framing understates the prize
slightly (good), but it also means the absolute ceiling is bounded by the non-layer-scaling overhead
(sampling, launch, embedding/final-head) which depth reduction cannot remove.

**Key comms-bound subtlety for verification (this is what makes spec decoding attractive here):** a
verify pass over `γ+1` tokens at B=1 pays the *same 188 all-reduces* as a single-token decode — the
collective **count is per-layer, independent of token count**; only the message size grows by ~(γ+1)×,
which is negligible at decode payload sizes. So **verify ≈ one full-token decode**, and the draft's
layer savings translate ~1:1 into avoided collectives. The cost model below uses "1 full 94-layer
single-token decode" as the unit.

---

## 1. The KEY IDEA worked out: self-speculative depth-draft

**Mechanism:** Draft autoregressively for `γ` tokens using only the first `L_d` of 94 layers, emitting
draft logits via the model's **shared final RMSNorm + LM head (logit lens)**. Then the full 94-layer
model verifies all `γ+1` positions in one pass; vLLM's rejection sampler accepts the longest correct
prefix. Lossless by construction (verification preserves the target distribution exactly, greedy or
sampled).

**Cost model (units = one full 94-layer single-token decode):**

```
r   = L_d / 94                          # draft depth fraction (= comms + weight saving per draft step)
τ   = (1 − α^(γ+1)) / (1 − α)           # expected accepted tokens per cycle (incl. bonus token)
draft cost  = γ · r                     # γ shallow autoregressive steps
verify cost = 1                         # one full pass, ~free beyond 1 token's comms (see §0)
speedup ≈ τ / (γ·r + 1)
```

**Break-even (γ=1):** need `1+α > r+1` ⟹ **`α > r`.** Acceptance must exceed the draft's depth
fraction. Necessary but far from sufficient for a *useful* win.

**Speedup table (best γ chosen per cell):**

| `L_d` (r) | α=0.5 | α=0.7 | α=0.8 | α=0.9 | α=0.95 | α=0.98 |
|---|---|---|---|---|---|---|
| 8  (0.09) | 1.50× | 2.07× | 2.62× | **3.71×** | 4.75× | 5.56× |
| 15 (0.16) | 1.33× | 1.71× | 2.05× | 2.69× | 3.34× | 3.91× |
| 24 (0.26) | 1.19× | 1.45× | 1.67× | 2.06× | 2.43× | 2.83× |
| 47 (0.50) | 1.00× | 1.13× | 1.22× | 1.38× | 1.51× | 1.66× |
| 60 (0.64) | 0.92× | 1.04× | 1.10× | 1.19× | 1.27× | 1.37× |

**Reading the table — the central tension:**

- The big wins (≥2.5×) live in the **top-left-to-right**: **shallow draft (`L_d`≤15) AND high
  acceptance (α≥0.8)**.
- A *shallow* draft on Qwen3 will have **low** α (the model is pruning-fragile; "robustness disappears
  once two layers are removed" for Qwen3-8B; transition-to-random at ~20% depth). So you are pushed to
  **deep** exits (`L_d`≈47–60) to keep α high — but there the speedup collapses to 1.2–1.7× **even at
  α=0.95–0.98**, because the draft is no longer cheap.
- **The two levers are anti-correlated on this model.** Shallow ⇒ cheap draft but α tanks; deep ⇒ α
  recovers but draft isn't cheap. This is the fundamental reason depth-draft self-speculation is a hard
  win on Qwen3 specifically.

**Does sharing weights / KV make the draft ~free?** Partially. **Weights:** yes — the draft reuses the
target's layer-`0..L_d` tensors, no extra memory, and the verify pass can in principle reuse the
draft's first `L_d` layers' activations for the first token (LayerSkip does this). **KV cache: NO, not
in idiomatic vLLM.** vLLM's block manager is per-model; EAGLE/MTP drafts keep a **separate KV cache**.
Even a weight-shared early-exit draft would need the target's layers `>L_d` to find populated KV at
verify time, but the draft only filled positions `0..L_d` — so draft and target maintain distinct KV
state. The draft is *cheap* (`r×` the comms+weights) but **not free**; the `γ·r` term is real.

---

## 2. SOTA evidence base for the self-speculative depth-draft

These give the empirical `α` and layer-skip fraction `c≈r` to plug into the model. **No method below
reports MoE results** — this is an unfilled gap and a source of risk.

| Method | Train? | Lossless | α (acceptance) | skip frac `c≈r` | Speedup | Largest model |
|---|---|---|---|---|---|---|
| **Draft & Verify** (ACL'24, 2309.08168) | No (offline Bayesian-opt of skip set) | Yes | **0.87–0.93** (CNN/DM 13B) | **≈0.5** | 1.4–2.0× | **70B** (1.73×) |
| **SWIFT** (ICLR'25, 2410.06916) | **None (on-the-fly skip search)** | Yes | **0.98–1.00** greedy; 0.88 (34B code) | **≈0.45–0.50** | 1.3–1.6× | **70B** (1.48×) |
| **LayerSkip** (Meta, 2404.16710) | **Yes (recipe: layer-dropout+EE loss)** | Yes | 0.55–0.69 hard tasks; ≤0.97 structured | exit ratio ≈0.2–0.37 | 1.8–2.16× | 13B |
| **Kangaroo** (NeurIPS'24, 2404.18911) | Adapter only (67M) | Yes | not tabulated (compression 1.4–2.2) | ℓ/L≈0.06–0.1 + adapter | 1.66–1.68× | 13B |

**Crucial cross-check against the table in §1:** the *training-free, 70B-tested* methods (Draft&Verify,
SWIFT) operate at **`c≈0.5`** (they skip ~half the layers) and even there only hit **1.4–1.7×** — which
matches the `r=0.5` row of my model almost exactly (1.38–1.66× at α=0.9–0.98). They do **not** operate
at the cheap `L_d≤15` corner, because skipping that many layers training-free destroys acceptance.
**This is independent confirmation that the realistic operating point is `r≈0.5, ~1.5×`, not the
fantasy `r=0.09, 3.7×` corner.** LayerSkip reaches the cheap corner only by *retraining the model* with
its recipe — which is off the table for a 235B hackathon model.

**Adaptive-depth early-exit (CALM, SkipDecode, EE-LLM):** all **lossy**, all require early-exit
training, none tested above ~6.7B (SkipDecode/OPT) or beyond T5-XXL (CALM/EE-LLM). **Not viable** for a
lossless, no-retrain, 235B target. Discard.

---

## 3. Static depth pruning (skip K layers, no verification)

Simpler, no verifier, immediate comms+weight cut — but **lossy**, needs a quality gate, and is the
**weakest option on Qwen3.**

| Skip K | Layers left | Comms saved | Comms-bound speedup (≈ full speedup, since weights also scale) |
|---|---|---|---|
| 9  (~10%) | 85 | 10% | ~1.11× |
| 14 (~15%) | 80 | 15% | ~1.18× |
| 19 (~20%) | 75 | 20% | ~1.25× |
| 24 (~26%) | 70 | 26% | ~1.34× |

**Why it's weak on Qwen3:**

- **Qwen3 transition-to-random is at ~20% depth** (Gromov et al.), vs ~45–55% for Llama-2. So the
  ~1.25× point (skip 19) is essentially at the cliff edge.
- Worse: the famous "drop ~half the layers" numbers are **QA-benchmark (MMLU/BoolQ) robustness, NOT
  perplexity.** C4 next-token loss degrades **smoothly from the very first layer dropped.** For *open
  generation* (your actual workload), quality erodes immediately — the multiple-choice robustness is a
  mirage for a generation latency/quality tradeoff.
- **MoE makes it worse, not better.** Your hypothesis was "skip K redundant *middle* layers." The MoE
  literature finds the **opposite**: in MoE the *middle* layers carry the most diverse experts and are
  the **least** redundant; redundancy clusters at the **first and last** layers. So "skip the middle of
  a deep MoE" is contradicted by the evidence.
- Best-quality-per-layer is **attention-only** dropping (attention is far more redundant than FFN:
  Llama-2-70B tolerates 50% attention-layer removal at −2.4%), but attention is the *cheaper* sublayer —
  dropping it saves only **1 of the 2** all-reduces/layer and little weight, so the comms win is halved.

**Verdict on static:** realistic safe budget without healing is **~1 layer** (Qwen3-8B breaks at 2).
Even an optimistic ~10% gate buys only ~1.11×, is lossy, and needs a per-deployment quality gate. Not
worth it versus the lossless spec-decode route. Healing (QLoRA / continual pretrain) recovers more but
requires training a 235B model — out of scope.

---

## 4. The honest competitive baseline: EAGLE3 already exists for this exact model

This is the result that reframes the whole question. **`lmsys/Qwen3-235B-A22B-EAGLE3`** is a real,
published ~1B-param EAGLE3 draft head for the exact target:

- Reported **acceptance length τ ≈ 3.02 (MT-Bench) to 3.54 (GSM8K)** — i.e. ~3 accepted tokens/cycle.
- Draft head is tiny (TP=1, runs alongside TP=8 target), so draft cost ≈ negligible vs the 94-layer
  verify ⟹ **speedup ≈ τ/(1+small) ≈ 1.8–2.4× realistic.**
- **Lossless**, first-class vLLM method (`speculative_config={"method":"eagle3", ...,
  "draft_tensor_parallel_size":1}`).
- Caveat: the card documents **SGLang**; expect minor tensor-key/config adaptation to load in vLLM.
- A second checkpoint exists: `nvidia/Qwen3-235B-A22B-Eagle3` (TensorRT-oriented).

**Compare to the self-spec depth-draft:** to *beat* EAGLE3 (~2.4×), the depth-draft needs to land in
the `≥2.4×` region of the §1 table — i.e. `L_d≤24` with `α≥0.95`, **or** `L_d≤15` with `α≥0.9`. Given
Qwen3's fragility, **achieving α≥0.9 at L_d≤24 is unlikely** (training-free 70B methods need `c≈0.5`
just to hold α≥0.9). So the most probable outcome is **depth-draft ≈ 1.4–1.7×, i.e. it LOSES to the
existing EAGLE3 head.** EAGLE3 wins because its tiny trained head achieves high τ at near-zero draft
cost — beating an untrained logit-lens draft that must run dozens of real (comms-paying) layers.

---

## 5. Minimal vLLM implementation paths

**Path A — EAGLE3 (recommended, lowest risk, ~1–2 days):**
```python
LLM(model="Qwen/Qwen3-235B-A22B", tensor_parallel_size=8,
    speculative_config={"method":"eagle3",
                        "model":"lmsys/Qwen3-235B-A22B-EAGLE3",
                        "draft_tensor_parallel_size":1,
                        "num_speculative_tokens":4})
```
Validate the checkpoint loads in vLLM (SGLang-trained — may need key remap). Validate on the small
sibling `Tengyunw/qwen3_8b_eagle3` first to de-risk the pipeline cheaply.

**Path B — novel self-speculative depth-draft (the research bet, ~1–2 weeks, likely ties/loses):**
1. **Custom model** registered via the `vllm.general_plugins` entrypoint
   (`ModelRegistry.register_model("Qwen3EarlyExitForCausalLM", "...:cls")`, lazy string form). It
   references the target's layers `0..L_d`, then applies the **shared `model.norm` + `lm_head`** to
   produce draft logits (logit lens).
2. **Custom proposer** in `vllm/v1/spec_decode/` mirroring `EagleProposer`'s
   `load_model`/`prepare_inputs`/`propose(k)` interface; hand `draft_probs` + `target_logits` to the
   existing `RejectionSampler` (lossless, already implemented).
3. **KV cache:** accept **separate draft KV cache** (vLLM idiom). Weight-sharing is feasible; KV-sharing
   fights the per-model block manager — don't.
4. **Choose `L_d`** by a cheap offline sweep using angular-distance / Block-Influence between residual
   states (the standard training-free skip-selection signals) to find the shallowest `L_d` that holds
   α; expect to land near `L_d≈47` (`r≈0.5`).
5. Gate the experiment: **if measured α at the chosen `L_d` doesn't clear the §1 break-even for the
   target γ, stop** — it will not beat EAGLE3.

**Path C — static prune (not recommended on Qwen3):** drop ~1 attention layer behind a perplexity +
benchmark quality gate; ~1.0–1.1×, lossy. Only if A and B are both blocked.

**Qwen3-specific robustness notes:** 94 layers, hidden 4096, GQA 64Q/4KV, head_dim 128, 128 experts
top-8, `decoder_sparse_step=1` (every layer is MoE), **no native MTP head** (unlike DeepSeek-V3 — so
the `"mtp"` method is unavailable; EAGLE3 is the off-the-shelf route). Pruning-fragile (~20% transition,
breaks at 2 layers for the 8B). Under vLLM's modern large-MoE path (DP-attention + EP-MoE) the per-layer
collective is an **all-to-all** rather than all-reduce, but it is *still per-MoE-layer*, so skipping a
layer still removes it from the critical path — the depth-reduction lever survives the EP switch.

---

## 6. Final answer to the brief

- **Single most promising depth-reduction approach for B=1 235B/8×H100:** self-speculative decoding with
  a **shallow early-exit logit-lens draft + full-model lossless verify** (the KEY IDEA). It is the only
  *lossless, no-retrain, high-ceiling* member of the family. **However**, on Qwen3 its realistic
  operating point is `L_d≈47 (r≈0.5)`, α≈0.9 ⟹ **~1.4–1.7×**, which **loses to the already-published
  `lmsys/Qwen3-235B-A22B-EAGLE3` head (~1.8–2.4×, also lossless, near-free draft).**
- **Quantified comms+weight saving:** each skipped layer removes 2 collectives (~32 µs) + that layer's
  weight reads. A draft at `r=0.5` pays ~half the 188 collectives per draft step; the *net* token-rate
  gain is governed by `τ/(γ·r+1)`, not the raw per-step saving.
- **Acceptance/quality bar to win (vs EAGLE3 ~2.4×):** `α≥0.95 at L_d≤24`, or `α≥0.9 at L_d≤15`.
  Both are **above** what training-free layer-skip achieves on 70B dense models (which need `c≈0.5` to
  hold α≥0.9), and Qwen3 is *more* fragile than Llama. **Bar is probably not clearable without
  retraining (LayerSkip recipe), which is out of scope for 235B.**
- **Risk:** (1) acceptance collapses at shallow exits on a pruning-fragile MoE; (2) no MoE early-exit
  result exists in the literature — uncharted; (3) verify/draft separate-KV plumbing in vLLM is custom
  work; (4) even at best it likely only ties EAGLE3.
- **Minimal vLLM path:** ship **EAGLE3 (Path A)** as the real win; run the **self-spec depth-draft
  (Path B)** as a *gated* research experiment that stops the moment measured α fails the §1 break-even.
- **Does it beat the comms floor?** Depth reduction *does* attack the dominant cost (it cuts comms AND
  weights, ~26% of the budget is comms but per-layer weight reads are the larger share). But the
  *lossless* way to exploit it (spec decoding) is **already better served by EAGLE3** than by an
  untrained depth-draft on this fragile MoE. **Honest conclusion: the depth-draft is a legitimate novel
  idea but, un-retrained on Qwen3-235B, it is unlikely to beat the existing EAGLE3 baseline. The genuine
  comms-reducing win here is "run a tiny trained draft head," which EAGLE3 already is — and the most
  defensible novel contribution would be combining EAGLE3's trained draft with depth-truncated
  verification or measuring whether a depth-draft can exceed EAGLE3's τ, not assuming it will.**
```
```

## Sources

- LayerSkip — https://arxiv.org/abs/2404.16710
- CALM — https://arxiv.org/abs/2207.07061
- SkipDecode — https://arxiv.org/abs/2307.02628
- Draft & Verify — https://arxiv.org/abs/2309.08168 / https://aclanthology.org/2024.acl-long.607/
- Kangaroo — https://arxiv.org/abs/2404.18911
- EE-LLM — https://arxiv.org/abs/2312.04916
- SWIFT — https://arxiv.org/abs/2410.06916
- Unreasonable Ineffectiveness of Deeper Layers (Gromov) — https://arxiv.org/abs/2403.17887
- ShortGPT — https://arxiv.org/abs/2403.03853
- What Matters in Transformers / Not All Attention — https://arxiv.org/abs/2406.15786
- Layer Pruning Harms Test-Time Scaling (Qwen3) — https://arxiv.org/pdf/2510.22228
- Qwen3-235B-A22B config — https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
- lmsys EAGLE3 head — https://huggingface.co/lmsys/Qwen3-235B-A22B-EAGLE3
- nvidia EAGLE3 head — https://huggingface.co/nvidia/Qwen3-235B-A22B-Eagle3
- vLLM speculative decoding — https://docs.vllm.ai/en/latest/features/speculative_decoding/
- vLLM EAGLE — https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/
