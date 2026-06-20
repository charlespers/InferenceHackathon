# Breaking the serial collective chain for B=1 decode
### Qwen3-235B-A22B on 8×H100, 94 MoE layers

> Goal: cut the *number* of serial all-reduces on the critical path of a single
> request, not just the per-collective latency. The 94 serial layers force
> ~188 serial collectives; at B=1 these are latency-bound (payload is tiny), so
> they cannot be hidden by faster links alone — only by **re-ordering**,
> **amortizing**, or **removing** synchronization points.

---

## 0. The wall, quantified (from this repo's own model)

Using `src/inferutil/latency.py` + `hardware.py` (H100: `collective_latency_s = 5e-6`,
`hbm_bw = 3.35 TB/s`, 8 GPUs) and `model.py` (94 layers, 22B active):

| Term | bf16 | fp8 weights |
|---|---|---|
| Weight read (22B active / 8·3.35TB/s) | 1.64 ms | 0.82 ms |
| KV read (long ctx) | varies | varies |
| **Comms, TP (2 AR/layer × 94 × 5µs)** | **0.94 ms** | **0.94 ms** |
| Comms, hybrid (3 coll/layer × 94 × 5µs) | 1.41 ms | 1.41 ms |
| Compute | ~negligible | ~negligible |

The crucial observation the team has already made: **once weights are fp8
(0.82 ms), comms (0.94 ms) becomes the single largest term.** Comms does not
shrink with quantization, placement, or kernels — it is `count × fixed_latency`.
So the only way through is to attack `count`. That reframes every idea below as:
*how low can the per-layer serial-collective count go, and at what risk?*

Collective-count targets (×94 layers × 5µs):
- 2.0/layer (today, TP) = 0.94 ms
- 1.0/layer = 0.47 ms
- 0.5/layer = 0.235 ms
- 0.0/layer on critical path (deferred/hidden) → comms leaves the critical path entirely.

---

## 1. Comms amortization via wide / tree speculation  — **the highest-confidence win**

### Mechanism (the part the team may be under-framing)
At B=1, verifying `k` draft tokens in **one** forward pass pays **one** set of
collectives for all `k` tokens. The comms term divides by the *accepted* tokens
per verify, not the drafted tokens:

```
comms_per_accepted_token = (collectives_per_layer × 94 × 5µs) / τ
```

where `τ` = mean accepted length per verification cycle. This is the *real*
reason spec-decode wins in a comms-bound regime — it is a **collective-count
divider**, not a compute trick. The team has n-gram spec; the comms-optimal
framing is to **maximize τ per collective**, which is a different objective from
maximizing raw draft length.

### The numbers
- n-gram / prompt-lookup drafting: τ ≈ 1.5–2.0 on general text (high variance,
  great on repetitive/code, poor on reasoning).
- EAGLE-3 tree drafting: τ ≈ **2.4 (40% accept) up to 2.5–5.7** on long
  requests; +6–12 pp acceptance over EAGLE/Medusa via cross-layer feature fusion.
- A **tree** (vs linear draft) is strictly better here because tree attention
  verifies many candidate continuations in the *same* forward pass = same single
  set of collectives, and you accept the best path. So the right metric is
  **accepted-tokens-per-collective-round**, and the tree shape should be chosen
  to maximize *that*, not flops.

### Comms-amortized ceiling
With τ tokens per verify, effective comms/token:

| τ | comms/token (TP, 0.94 ms base) |
|---|---|
| 1 (no spec) | 0.94 ms |
| 2 | 0.47 ms |
| 3 | 0.31 ms |
| 4 | 0.235 ms |
| 5.7 (EAGLE-3 best) | 0.165 ms |

So wide tree spec can plausibly push the comms contribution from **0.94 ms →
~0.2–0.3 ms** — an ~3–4× cut on the dominant term, with **zero accuracy loss**
(verification is exact / distribution-preserving).

### The MoE "expert-union verify tax" (the catch the team must model)
This is the part that bounds `k`. Verifying `k` tokens in one pass means the
layer must activate the **union** of all experts those k tokens route to, not 8.
- 1 token → 8 experts active.
- k tokens (tree of width w) → up to `min(128, ~8·k_eff)` experts, where
  `k_eff` = number of distinct tree positions verified.
- This **inflates the weight-read term** (the other dominant cost): reading the
  expert union instead of 8 experts. At k where the union saturates toward all
  128 experts, weight-read for the MoE block approaches the *dense-equivalent*
  read and the per-token weight cost stops improving.

**Comms-optimal objective:** choose tree width/depth to
`maximize τ / (1 + extra_weight_read_from_expert_union)`. There is an interior
optimum: small trees (τ≈2–3) barely grow the expert union (the repo's
`routing_predict.py` persistence + DirectProxy data shows high token-to-token
expert overlap, so consecutive draft tokens reuse experts → union grows slowly),
but very wide trees blow up the union and pay it back in HBM. The team already
has the data to *measure* this curve: feed the collected `routes` through an
expert-union model per tree shape.

**Rating: feasibility 5/5, risk 1/5 (exact, already partly built).** This is the
one that realistically beats the standard levers because it (a) attacks the now-
dominant comms term directly, (b) is loss-free, (c) composes with fp8/placement,
and (d) the repo's routing data already lets them tune the MoE tax. Net: comms
0.94→~0.25 ms.

---

## 2. Async / bounded-stale tensor parallelism — **the boldest structural break**

### The idea
Remove the all-reduce from the **critical path** by letting layer `L+1` start on
*local* (un-all-reduced) activations and reconciling the global sum δ layers
later. The collective still happens, but **off the critical path**, overlapped
with the δ following layers' weight-streaming. On the critical path the
collective count → effectively **0**.

### Prior art (this is real, not hand-waving)
- **Ladder Residual** (ICLR'25, `openreview 6R4TGPd74N`): architectural change
  that routes the residual so the all-reduce of block `L` overlaps the compute
  of block `L+1` — "decouple communication from computation." **29% end-to-end
  inference speedup, 8B, TP=8.** Requires training from scratch, or adapting an
  existing model with ~3B tokens of retraining at "minimal accuracy degradation."
- **Delayed Tensor Parallelism (DTP)** (kog.ai): explicitly targets
  **batch-size-1** inference. Defers all-reduce by δ layers; computation proceeds
  on local outputs; "hides communication behind computation and weight
  streaming." Reports it "claws back most of the performance loss" vs naively
  dropping comms, beating Ladder Residual on final loss at slightly less exposed
  wait-time. Requires retraining from scratch (naive comms-drop degrades badly;
  the delayed design recovers it).
- **Partially Synchronized Activations (arXiv 2506.19645)**: forward passes use
  stale activations from step *t-k*; 50–75% reduction in blocking collectives per
  layer. (Caveat: framed for *training* throughput; the "no retraining" claim is
  about *its own* training run, not about retrofitting an inference model — do
  not over-read it.)

### Why it's promising and why it's risky
- Upside: this is the **only** family that takes collectives to ~0 on the
  critical path. For a comms-bound B=1 regime that is the theoretical jackpot —
  comms 0.94 ms → ~0.1 ms exposed.
- Risk: **all the inference-grade variants (Ladder, DTP) require retraining.**
  You cannot bolt this onto stock Qwen3-235B and keep accuracy — dropping/delaying
  the AR without the matching residual restructuring "causes significant
  degradation." For a hackathon on a *fixed pretrained* model, retraining 235B is
  out of scope.
- A *training-free* δ=1 stale-TP variant is conceivable (use last layer's local
  shard, correct next layer) but with **no convergence guarantee** — at B=1
  decode a single wrong top-8 route or wrong argmax flips the token, and errors
  compound autoregressively. Honest read: untested, likely accuracy cliff on
  reasoning prompts, would need empirical bounding on the repo's prompt set.

**Rating: feasibility 2/5 (needs retrain or risky training-free hack),
risk 4/5.** Highest ceiling, lowest deployability on a fixed model. Strong as a
"what an architecture co-designed for 8×H100 B=1 *should* look like" argument;
weak as a same-week patch.

---

## 3. Collapse collectives by re-sharding — **the safe structural win, <2/layer**

### What's collapsible
A standard TP transformer block does **2 all-reduces/layer** (post-attention,
post-MLP). The repo's `hybrid` plan does 3 (all2all dispatch + combine + attn
AR). Targets:

**(a) Sequence/parallel-residual fusion → toward 1 AR/layer.**
Models with a *parallel attention+MLP* block (attn and MLP read the same
layernorm'd input, run concurrently, sum outputs) need only **one** all-reduce
for the combined `attn_out + mlp_out` — **2→1 per layer**, exactly halving the
comms count to **0.47 ms**. This is an architecture property (GPT-J/PaLM/Falcon
use it); Qwen3 is serial attn→MLP, so this is a *redesign*, not a config flip.
For a fixed model it's not free, but it sets the realistic floor at 1/layer.

**(b) Reduce-scatter + all-gather restructuring (sequence parallel).**
Replace `all-reduce` with `reduce-scatter` (end of block) + `all-gather`
(start of next block). Same 2 collectives, but each moves *half* the bytes and,
more importantly, the all-gather of block `L+1`'s input can be **fused/overlapped
with** the reduce-scatter of block `L`. This is latency-neutral at B=1 (payloads
already tiny) but is the enabling substrate for overlap (idea 2). On its own:
does **not** cut the count, only the bytes — so at B=1 (latency-bound) it's a
near-no-op. Honest: low value alone here.

**(c) Cross-layer linear fusion.** Process two consecutive layers' QKV/MLP
projections before syncing, so one collective covers two layers' worth of
partial sums. Only valid where the second layer's input doesn't depend on the
first layer's *reduced* output — which in a serial residual stream it **does**.
So true 2-layer fusion = idea 2 (staleness) in disguise. Without staleness, (c)
is not achievable.

### Minimal achievable collective count per layer
- Stock Qwen3 serial block, exact: **2/layer** is the hard floor (you must reduce
  attn before MLP reads it, and reduce MLP before the next layer).
- Parallel-block redesign, exact: **1/layer** (0.47 ms).
- Stale/delayed (idea 2), exact-math-but-deferred: **~0/layer on critical path**.

**Rating: feasibility 4/5, risk 2/5 — but only the parallel-block 2→1 actually
cuts count, and that needs a model that's built that way.** For *stock* Qwen3 the
exact, training-free floor is 2/layer; you cannot honestly go below it without
either staleness (idea 2) or a different architecture. Reduce-scatter/SP is worth
doing as plumbing for overlap, not as a standalone B=1 win.

---

## 4. Genuinely novel angles for single-request 8-GPU 94-layer MoE

Ranked by how much they cut the serial-collective **count**:

### 4a. Speculative *parallelism* — run the route prediction to skip the dispatch collective (novel, medium risk)
The repo's `routing_predict.py` already shows **DirectProxy (L→L+1 cross-layer
route prediction, zero-training)** and high **persistence**. Exploit it to
**pre-place** each token's experts so the expert-parallel **all-to-all dispatch
collective is replaced by a local read** when the prediction hits. On a hit
(measured hit-rates are high for the union of prev-token-8 + top-R hot), the
dispatch+combine collective for that layer **disappears** — the token's experts
are already resident on the local GPU. On a miss, fall back to all-to-all.
Expected collectives/layer = `2 × (1 - hit_rate)` for the MoE part. If hit-rate
≈ 0.8, that's effectively **2→0.4** MoE collectives/layer. This is **comms-count
reduction driven by predictability**, uniquely enabled by the data this repo
already collected. Loss-free (fallback on miss). **Feasibility 3/5, risk 3/5** —
the hard part is replicating enough experts to make local-serve common without
blowing the 57 GB/GPU weight budget; the repo's coverage curves bound this.

### 4b. Layer-group "collective batching" via micro-speculation (novel framing)
Combine 1 + 2: draft a *short* chain locally with no cross-GPU sync at all
(staleness δ = draft length), then do **one** synchronizing verify pass for the
whole chain. You pay full collectives only on the verify, and the draft steps run
at **0 collectives**. This is the comms-amortization of idea 1 generalized to
*also* amortize the draft side. Ceiling: comms/token → `0.94 / τ` with the draft
steps contributing **zero** collectives. **Feasibility 4/5, risk 2/5** if the
drafter is a separate small model (no staleness in the *target*); this is
basically EAGLE-style self-speculation, reframed as a collective-count argument.

### 4c. Pipeline the layers across GPUs instead of tensor-splitting them (rejected, but instructive)
Pure pipeline parallelism (each GPU owns ~12 contiguous layers) has **0
intra-layer collectives** — only point-to-point handoffs at stage boundaries
(7 sends for 8 stages). That's **7 P2P vs 188 all-reduces**. The catch: at B=1
there's no pipeline to fill, so 7/8 of the GPUs idle and you lose the HBM-
bandwidth aggregation — weight read goes from 0.82 ms (8-way) to ~6.5 ms
(1-way per stage active). Net **much worse** for B=1. Documented here because it's
the obvious "kill the collectives" idea and it's a trap: it trades 0.94 ms of
comms for ~6 ms of weight read. **Feasibility 5/5, value negative.**

---

## Bottom line

**Most promising way to cut the serial-collective count for B=1 235B/8×H100:**

> **Wide *tree* speculative decoding tuned for comms-amortization** (idea 1),
> ideally combined with **predictive expert pre-placement** (idea 4a) to also
> knock out the MoE dispatch collective on prediction hits.

Reasoning: comms is now the co-dominant term (0.94 ms vs 0.82 ms fp8 weights) and
it is `count × 5µs` — immune to every per-collective lever. Tree spec is the only
**loss-free** mechanism that divides the collective count by τ (2.4–5.7), pulling
comms to ~0.2–0.3 ms, and it **composes** with fp8, placement, and the team's
existing routing/prefetch work. The repo *already* has the two assets needed to
push it past textbook spec-decode: (1) n-gram drafting to upgrade to a tree, and
(2) `routing_predict.py` data to model the MoE expert-union verify tax and to
drive 4a. The binding constraint is the expert-union weight-read tax, which is
*measurable from data the team already has* — so the optimization is concrete,
not speculative.

| Idea | What it cuts | Coll/layer | Comms result | Feasibility | Risk | Beats standard levers? |
|---|---|---|---|---|---|---|
| 1. Wide tree spec (comms-amortized) | count ÷ τ | 2/τ | 0.94→~0.25 ms | **5/5** | **1/5** | **Yes — loss-free, composable** |
| 2. Async/stale/delayed TP (Ladder/DTP) | AR off critical path | ~0 | 0.94→~0.1 ms | 2/5 | 4/5 | Highest ceiling, but needs retrain |
| 3. Re-shard / collapse (parallel block) | 2→1 (redesign only) | 1 | 0.94→0.47 ms | 4/5* | 2/5 | Only with parallel-block arch |
| 4a. Predictive expert pre-placement | MoE dispatch on hit | 2·(1−hit) | MoE coll ÷ ~5 | 3/5 | 3/5 | Composes; repo-data-enabled |
| 4b. Micro-spec layer-group (draft@0 coll) | draft coll → 0 | 0 on draft | comms ÷ τ | 4/5 | 2/5 | Variant of 1 |
| 4c. Pipeline parallel | all-reduces → P2P | 0 | comms→~0 but +6 ms weight | 5/5 | — | **No — net regression at B=1** |

\* 4/5 *if* targeting a parallel-attention/MLP architecture; for stock serial
Qwen3 the exact training-free floor is 2/layer.

**The honest verdict:** Idea 2 is the boldest and has the highest theoretical
ceiling (collectives → 0 on the critical path), and the literature
(Ladder Residual, Delayed TP) proves it's real — but it requires retraining
235B, so it's a *co-design argument*, not a hackathon patch. **Idea 1
(+4a) is the one that realistically beats the standard levers this week:** it is
exact, it directly divides the now-dominant comms term, and the team is already
75% of the way there.

---

### Sources
- Speculative decoding in vLLM (tree verify, comms amortization): https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference
- EAGLE-3 (acceptance length, tree drafting): https://openreview.net/pdf?id=4exx1hUffq , https://huggingface.co/blog/lujangusface/tw-eagle3-gpu
- Ladder Residual (overlap AR with compute, 29% inference speedup, TP=8): https://openreview.net/forum?id=6R4TGPd74N
- Delayed Tensor Parallelism (B=1 inference, defer AR by δ layers): https://blog.kog.ai/delayed-tensor-parallelism-for-faster-transformer-inference/
- Partially Synchronized Activations (stale activations, 50–75% fewer blocking collectives): https://arxiv.org/pdf/2506.19645
- Flash Communication (quantized AR — per-collective cost, NOT count): https://arxiv.org/html/2412.04964v1
- Sequence parallelism / reduce-scatter+all-gather restructuring (Megatron): https://arxiv.org/pdf/2105.13120 , https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
- Speculative decoding survey: https://arxiv.org/pdf/2502.19732
- MoE-Spec (expert budgeting under speculative decoding): https://arxiv.org/pdf/2602.16052
