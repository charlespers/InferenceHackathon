# The B=1 decode comms floor for Qwen3-235B-A22B on 8×H100 (vLLM, NVSwitch)

**Scope.** Decode at batch=1 is memory-bandwidth + collective-latency bound. With TP=8 the
collectives are ~188 serial all-reduces/step (2/layer × 94 layers) on tiny ~8 KB payloads
(hidden 4096 × 2 B). At ~16 µs/collective that is **~3.0 ms/token of pure comms** — the dominant
term behind the ~85 tok/s. Charles owns generic `nccl-tests` + a one-shot all-reduce. This doc
finds the **structural** comms levers that sit *outside* generic NCCL tuning, with quantified
µs/collective and e2e estimates.

Model facts (verified from HF `config.json`, `Qwen/Qwen3-235B-A22B`, `model_type: qwen3_moe`):
94 layers, hidden 4096, 64 Q heads × 128 head_dim (Q dim 8192), 4 KV heads (GQA-16), 128 experts,
top-8, expert intermediate 1536, no attn bias, bf16. Active params/token ≈ 20.9 B (attn 71.3 M/layer
+ 8 active experts 151 M/layer + router).

---

## TL;DR — the two top UNCLAIMED comms-floor levers

Charles already owns: generic NCCL algo/proto sweeps and a one-shot all-reduce kernel. The two
structural levers he is **not** doing, ranked by expected e2e payoff:

1. **NVLink-SHARP / multimem one-shot all-reduce captured in the CUDA graph (cut the per-collective
   constant, not the count).** This is the single highest-confidence win. The 16 µs is almost
   entirely *fixed overhead* (kernel launch + 8-way grid sync + flag handshake); the 8 KB transfer
   itself is <1 µs. A multimem/NVLS one-shot kernel that runs on 2–8 SMs inside the decode CUDA graph
   (TokenWeave-style) plus LL protocol drives the floor to **~7 µs** (realistic), even **~5 µs**
   best case. That alone takes comms from 3.01 ms → ~1.32 ms and lifts e2e from ~216 → ~340 tok/s
   in our model. Charles's "one-shot" kernel must specifically be the **NVLS/multimem in-switch-reduce**
   variant captured in-graph — generic one-shot over NVLink P2P does *not* get there.
   *Caveat:* vLLM already defaults to a custom one-shot AR at 8 KB (≪ its 256 KB cutoff on 8×H100),
   and CUDA-graph decode already removes launch overhead — so part of this may already be in effect.
   The unclaimed delta is forcing the **NVLS/SHARP multimem** path and LL protocol specifically and
   verifying it is the kernel actually selected.

2. **Reduce the collective COUNT by changing the parallelism layout — but NOT via naive attention
   replication (see below). The real count-reducing lever is DP-attention + EP, which removes the
   dense post-attention all-reduce and replaces the dense post-MoE all-reduce with a *sparse* top-k
   dispatch/combine all-to-all.** This is structural (layout change), not NCCL tuning, and is the
   only lever that attacks the 188-collective count itself. The honest verdict (below) is that at
   **B=1 single-stream** this is a wash-to-loss on latency, so its value is conditional — but it is
   the headline structural idea and the analysis is the deliverable.

Lever 3 (overlap comms with the next layer's weight read) is **not feasible at B=1** as a config
change and only marginally feasible as a high-effort, fragile code change — details in §3.

---

## 1. Reduce collective count via attention replication — full analysis (HEADLINE)

**The idea.** Standard TP=8 does 2 all-reduces/layer: post-attention (reduce the O-projection
partials) and post-MLP/MoE. If attention is **replicated** (TP=1 for attention — every GPU holds
full attention weights and computes the full attention locally) and only the MoE is
expert-parallelized, there is **no post-attention all-reduce**. Naively: 188 → 94 collectives.

**Does vLLM support it? Yes — natively, as "DP attention + EP MoE."**
This is exactly vLLM's recommended large-MoE serving mode (the DeepSeek-V3 / "Wide-EP" path), driven
by `--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel`. Per the vLLM Expert
Parallel docs: with TP=1, attention weights are **"replicated across all DP ranks"** (each GPU
computes attention locally, no cross-rank all-reduce); MoE experts are **"sharded across all EP
ranks"** with EP_SIZE = TP_SIZE × DP_SIZE. RFC #16037 is the design; it is implemented and in
production. No code change needed to *enable* it. Qwen3-235B-A22B (`qwen3_moe`) is supported.

**But the comms pattern is not a clean halving.** In DP-attn+EP the per-MoE-layer communication is a
**dispatch + combine all-to-all pair** (route each token to the GPUs holding its top-8 experts, then
gather results back), not a single all-reduce. So per layer you get **0 attention collectives + 2
MoE all-to-alls** ≈ still ~188 ops/step. The win is *character*, not count: the dense bandwidth-heavy
post-attention all-reduce is eliminated, and the dense post-MoE all-reduce becomes a **sparse**
top-k all-to-all (each token touches only the GPUs for its 8 experts, not all 8). Backends:
`allgather_reducescatter` (NCCL), `deepep_low_latency` (CUDA-graph-capable, the right choice for
B=1 decode), `pplx`. Select via `VLLM_ALL2ALL_BACKEND`.

**The B=1 latency trap (the key quantified finding).** Replicating attention is **not free at B=1**,
because it moves the attention weights from a 1/8 shard to a full per-GPU read. Decode is
memory-bound, so what matters per layer is *weight-read time*, and the apples-to-apples
attention-side comparison is:

| Attention-side cost / layer | Weight read | Post-attn collective | Total |
|---|---|---|---|
| **TP=8** (attn sharded /8) | 5.3 µs | 16 µs AR | **21.3 µs** |
| **Replicated** (attn full, no AR) | 42.6 µs | 0 | **42.6 µs** |

Removing the 16 µs all-reduce saves **16 µs/layer**, but replication adds **+37.3 µs/layer** of
attention weight-read (42.6 − 5.3). **Net −21.3 µs/layer ≈ −2.0 ms/token → replication LOSES at B=1.**

**Break-even:** replication only wins if the per-collective latency exceeds **~37.3 µs**. On 8×H100
NVSwitch the collective is ~16 µs (and we are trying to push it to ~7 µs), so replication is
strictly worse on this hardware. It would only pay off on slower interconnects (PCIe/multi-node,
where an all-reduce is tens of µs) or for a model whose attention weights are tiny relative to the
saved collective.

**Memory cost of replication** (for completeness, it is *not* the binding constraint):
per-layer attention weights = 71.3 M params × 2 B = **142.6 MB**; ×94 = **~13.4 GB/GPU** replicated
vs **~1.68 GB/GPU** sharded → **+11.7 GB/GPU**. On 80 GB H100 this is affordable (experts dominate:
~228 B params ≈ 57 GB/GPU sharded 8-way). So memory does **not** kill the idea — **the B=1
memory-bandwidth penalty does.** Note: the usual DP-attention motivation is avoiding **KV-cache**
duplication, an MLA (DeepSeek) argument; Qwen3 is plain GQA with 4 KV heads, so that benefit is muted.

**Verdict on the headline idea.** Correct in mechanism (eliminates the post-attn all-reduce, vLLM
supports it natively), but **at B=1 single-stream it is a net latency loss of ~2 ms/token** because
the attention weight-read penalty (37 µs/layer) exceeds the collective saving (16 µs/layer). It is a
**throughput** mode (built for high concurrency where each DP rank fills its own batch; at B=1 only
1 of 8 GPUs does real attention work and the other 7 run dummy forward steps in MoE lockstep). For
this B=1 latency objective, **keep TP=8 attention and attack the collective constant instead** (§2).

---

## 2. Lower the per-collective latency — best small-message config on 8×H100

For the ~8 KB message, the ranking (cited: "Demystifying NCCL" arXiv 2507.04786; NCCL docs; vLLM source):

- **PROTO:** `LL` is lowest-latency for tiny messages (~1 µs/hop; 8-byte atomic flag-poll sync).
  `LL128` ~2 µs/hop (preserves BW, NCCL's usual NVLink auto-pick). `Simple` ~6 µs/hop — wrong here.
  → **force `NCCL_PROTO=LL`** when on the NCCL path.
- **ALGO:** at 8 GPUs, `Tree`/`Ring` are both latency-competitive at 8 KB; `Tree` is typically picked.
- **NVLS (NVLink SHARP) does NOT help at 8 KB.** In-switch reduction has registration/multimem setup
  overhead and only wins at **≥256 MiB** (it *loses* to Ring by 5–27% at 4–128 MiB; people set
  `NCCL_NVLS_ENABLE=0` for small-message paths). So **NVLS-as-NCCL-algo is the wrong knob** — but
  NVLS **multimem as a one-shot custom-kernel primitive** (not the NCCL NVLS algo) is exactly what
  cuts the constant (see below). These are two different uses of the same hardware.
- **vLLM custom all-reduce is ON by default** and is what actually runs at 8 KB: `should_custom_ar`
  allows world ∈ {2,4,6,8} + full NVLink; the 8×H100 cutoff is **256 KB** (`CUSTOM_ALL_REDUCE_MAX_SIZES["9.0"][8] = 256 KB`),
  so 8 KB ≪ cutoff → **custom one-shot kernel, not NCCL**, by default. Recent vLLM also has a
  symmetric-memory **multimem one-shot** NCCL path tuned to win at very small sizes on 8 GPUs.
- **one-shot vs two-shot:** one-shot (broadcast + local reduce, or NVSwitch multimem) is lowest
  latency at small message / small world — the right choice here. Two-shot can be *slower* at small
  sizes (vLLM #36481: 16 KB → custom 2-shot 11.38 µs vs NCCL 10.66 µs).

**The 16 µs breakdown** (all overhead; 8 KB / 900 GB/s ≈ 0.009 µs of actual transfer):
kernel launch ~5–10 µs (eliminated by CUDA graphs), 8-way grid sync + flag handshake ~3–6 µs,
per-hop protocol ~2–5 µs. **Practical floor ≈ 5–10 µs** with CUDA graphs + LL + multimem one-shot;
**single-digit µs (~5–7 µs)** is the realistic best case (sub-µs is link-hop latency, not a full
8-GPU collective). The measured 16 µs is consistent with an **eager / un-tuned-protocol** path —
there is real room to ~7 µs.

**Quantified effect of cutting the constant (TP=8, 2 collectives/layer × 94):**

| Per-collective | Comms/token | (vs 16 µs) |
|---|---|---|
| 16 µs | 3.01 ms | baseline |
| 10 µs | 1.88 ms | −1.13 ms |
| **7 µs** (custom multimem one-shot + CUDA graph + LL) | **1.32 ms** | **−1.69 ms** |
| 5 µs (best case) | 0.94 ms | −2.07 ms |

**Best small-message config (single line):** keep custom all-reduce enabled (do **not**
`--disable-custom-all-reduce`), run decode under CUDA graphs (avoid `--enforce-eager`), ensure the
selected kernel is the **NVLS/multimem one-shot**, and on any NCCL fallback set
`NCCL_PROTO=LL NCCL_ALGO=Tree NCCL_NVLS_ENABLE=0`.

**MSCCL++ caveat (honest sizing):** measured custom small-message AR advantage over NCCL is modest
— ~**1.0–1.11× e2e** decode (MSCCL++, ASPLOS '26, 8×H100 TP8 Llama-70B). So the move from 16→7 µs
here assumes you are currently on a *non-graph / non-multimem* path; if vLLM defaults already give
you ~10 µs, the remaining headroom is a few µs, not 9.

---

## 3. Overlap comms with the next layer's weight read — feasibility

**Idea.** Per-layer weight read (~16.6 µs for the whole layer's active weights /8; attention QKV
portion ~5.3 µs) is comparable to the all-reduce (16 µs), so hide the AR of layer N behind the
weight read of layer N+1.

**Verdict: NOT feasible at B=1 as a config; high-effort, fragile, low-payoff as code.**

- **No vLLM mechanism does this today.** vLLM's async-TP / sequence-parallel overlap is **off by
  default** (#25277), requires **static compile shapes** (decode is dynamic), and overlaps *chunked
  GEMMs* — it needs M≫1 to have tiles to pipeline. At B=1 every matmul is an M=1 GEMV; there are no
  tiles. vLLM Dual-Batch-Overlap (`--enable-dbo`) overlaps the **MoE all-to-all** (not the TP AR),
  needs DP>1 + EP + DeepEP, and is **token-threshold gated** → inactive at B=1.
- **The dependency is serial.** The post-attn AR produces the activation input to the *same* layer's
  MLP; the MLP cannot start until it lands. You can in principle start *reading* layer N+1's weights
  (HBM read) concurrently with the AR (NVLink/few-SM op) — different engines — but the GEMM *compute*
  still waits on the activation, and at B=1 the GEMM **is** the weight read, so you only hide the
  weight-read of N+1's first GEMM, bounded by min(AR, that read) ≈ a few µs/layer.
- **No prefetch primitive.** GPUs have no "stage a layer's weights to on-chip memory ahead of time"
  call; H100 L2 (50 MB) cannot hold an A22B layer, so L2 warming buys ~nothing.
- **What it would take (code, not config):** AR on its own stream as a 2–8-SM multimem kernel
  (TokenWeave-style) + a separately scheduled kernel streaming N+1's first weight matrix + cross-stream
  CUDA-graph events for stable capture. Published decode-overlap work (TokenWeave ~18% at **1024
  tok/batch**, NanoFlow, DeepSeek TBO) **all requires ≥2 token subsets to split** and shows
  regressions at small M on Hopper — none demonstrate a B=1 benefit.

So lever 3 is the weakest: pursue lever 1's *constant* (§2) instead of overlap at B=1.

---

## Expected e2e tok/s impact (TP=8, our latency model; weight-read 1.56 ms/token, KV negligible @ short seq)

| Configuration | Comms/token | Total/token | tok/s |
|---|---|---|---|
| Baseline TP=8, 16 µs collective | 3.01 ms | ~4.6 ms | **~216** |
| **TP=8 + multimem one-shot @7 µs (lever 2)** | 1.32 ms | ~2.9 ms | **~340** |
| TP=8 + 5 µs best case | 0.94 ms | ~2.5 ms | ~390 |
| Attn-replicated DP+EP (lever 1) @16 µs | 3.01 ms a2a | ~8.2 ms | ~123 (worse — read penalty) |

(The prompt's ~85 tok/s reflects a heavier comms constant / longer-context KV / eager path than this
idealized model; the *relative* ranking is the takeaway: **lever 2 is the win; lever 1 loses at B=1;
lever 3 is impractical.** These align with `src/inferutil/latency.py`, which already models TP as
`2 × n_layers × collective_latency_s` and EP as `(2+1) × n_layers × collective_latency_s` — i.e. the
attention-replication count reduction is *not* currently credited there, consistent with this finding.)

---

## Config-vs-code implementation paths

| Lever | Path | Action |
|---|---|---|
| **2 — multimem one-shot AR, lowest constant** | **Config (mostly)** | Keep custom AR on; no `--enforce-eager` (CUDA graphs); force/verify NVLS multimem one-shot kernel; NCCL fallback `NCCL_PROTO=LL NCCL_ALGO=Tree NCCL_NVLS_ENABLE=0`. Code only if forcing a specific kernel selection vLLM doesn't auto-pick. **Do this first.** |
| **1 — attn-replication / DP+EP** | **Config** to enable (`-tp 1 -dp 8 --enable-expert-parallel`, `VLLM_ALL2ALL_BACKEND=deepep_low_latency`) | Enabling is free, but **measured loss at B=1**; only revisit for high-concurrency throughput, not single-stream latency. |
| **3 — AR/weight-read overlap** | **Code**, high effort | Multi-stream multimem AR kernel + weight-stream + cross-stream graph events. Not worth it at B=1. |

## Sources
- vLLM Expert Parallel docs: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- vLLM Data Parallel docs: https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/
- RFC #16037 (DP attention + EP MoE): https://github.com/vllm-project/vllm/issues/16037
- vLLM Wide-EP blog: https://vllm.ai/blog/2025-12-17-large-scale-serving ; Red Hat Wide-EP: https://developers.redhat.com/articles/2025/09/08/scaling-deepseek-style-moes-vllm-and-llm-d-using-wide-ep
- Qwen3-235B-A22B config: https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
- Demystifying NCCL (LL/LL128/Simple per-hop): https://arxiv.org/html/2507.04786v1
- vLLM custom_all_reduce.py: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py ; all_reduce_utils.py (256 KB cutoff): https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/device_communicators/all_reduce_utils.py
- vLLM #36481 (2-shot vs NCCL latency), #9699 (NVLS empirical): https://github.com/vllm-project/vllm/issues/36481 , https://github.com/vllm-project/vllm/issues/9699
- TokenWeave (multimem one-shot, 2–8 SMs): https://arxiv.org/pdf/2505.11329 ; MSCCL++ (1.11× decode): https://arxiv.org/pdf/2504.09014
- TensorRT-LLM one-shot AR (broadcast+local reduce): https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html
- vLLM async-TP off by default (#25277): https://github.com/vllm-project/vllm/issues/25277 ; vLLM DBO: https://docs.vllm.ai/en/latest/design/dbo/
- NanoFlow: https://arxiv.org/abs/2408.12757 ; FLUX: https://arxiv.org/pdf/2406.06858 ; LMSYS large-scale EP/TBO: https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
