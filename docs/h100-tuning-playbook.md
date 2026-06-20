# 8×H100 Latency Tuning Playbook — Qwen3-235B-A22B @ B=1

**Objective:** minimize TTFT and inter-token latency (TPOT) for a *single* request
(batch size 1) of Qwen3-235B-A22B on 8×H100 (80GB HBM3, NVLink/NVSwitch full mesh).

> All throughput/latency numbers below are **estimates to validate on-box**, not
> guarantees. Decode at B=1 is **memory-bandwidth bound**: each token re-reads the
> active weights + KV cache once, so bytes-moved-per-token is the quantity to drive down.

Model facts: 235B total / 22B active params · 94 layers · 128 experts/layer · top-8 routing.

---

## A. Parallelism layout

Recommended: **hybrid TP=2 × EP=8** (16 experts/GPU).

- **Expert-parallel (EP=8):** shard the 128 experts across 8 GPUs (~16 each). Top-8 routing
  means each token's experts are scattered across GPUs → an **all-to-all** dispatch/combine
  per MoE layer. On NVLink this is latency-cheap (~tens of µs) but it's the comms term to watch.
- **Tensor-parallel (TP=2):** split attention + dense projections within each pair of ranks.
  Halves per-rank activation/weight footprint and per-layer GEMV latency, at the cost of a
  small intra-group all-reduce each layer.
- Avoid pipeline parallelism for B=1 — pipeline bubbles dominate when there's one sequence.

Verify the box actually has full NVSwitch mesh (`nvidia-smi topo -m` → expect `NV#`/`NVLink`
between all pairs, not `PHB`/`SYS`). All-to-all latency collapses without it.

---

## B. Precision & memory

- **Weights: FP8 (e4m3)** with per-channel/row scales. ~4× smaller than FP16 → ~4× less
  bandwidth per token → the single biggest decode-latency win. ~235B × 1B / 8 ≈ **29 GB/GPU**.
- **KV cache: INT8/FP8** (per-layer symmetric). Decode re-reads the whole KV each step; halving
  it directly cuts TPOT. Accuracy impact on attention is typically <1–2%.
- **Budget per 80GB GPU:** ~29 GB weights + a few hundred MB KV (16K ctx) + ~5–10 GB prefill
  activations ⇒ **~40+ GB headroom**. Plenty for longer prefill windows or higher-precision
  hot paths if a layer proves sensitive.
- Calibrate quantization on a small set (~10 prompts × 1K tokens); dequant in-kernel, no host hop.

---

## C. Decode-latency kernels

1. **CUDA graphs** — capture the full B=1 decode step once at warmup and replay. Removes
   per-kernel launch overhead, which is a large fraction of TPOT at B=1. (~1.1–1.3×.)
2. **Single-query attention** — FlashInfer / FlashAttention decode path: KV already cached,
   compute the 1×D query against cached K/V in registers, no shared-memory re-reads.
3. **Kernel fusion** — fuse (a) dequant+GEMV, (b) gated FFN `down(silu(gate(x))⊙up(x))` into
   one kernel, (c) attention-output + RMSNorm. Each fusion removes an activation round-trip to HBM.
4. **Contiguous KV for decode** — paged KV helps prefill batching but adds indexing per step;
   for pure B=1 sequential decode, contiguous per-sequence KV is faster. Only page if prefill
   becomes the bottleneck.

---

## D. Speculative decoding ("prediction")

Decode is bandwidth-bound, so drafting cheap tokens and verifying them in one target pass is a
direct latency win. Start cheap, escalate if needed:

1. **Prompt-lookup / n-gram (zero training):** draft from n-gram matches in the context
   (n=2–3, draft length 4–6). Acceptance ~20–40% on repetitive/structured prompts. Good baseline,
   greedy-only.
2. **MTP / EAGLE / Medusa (learned draft):** a small auxiliary head predicts K next tokens;
   verify against the target in one pass. Acceptance ~60–80% → ~1.3–1.8× decode speedup. Qwen3
   has MTP support paths in SGLang.

**Surface acceptance rate live** — it's the headline knob: low acceptance means the draft is
wasted compute. (The console already renders `spec.accepted/proposed` per token and an
aggregate accept-rate.)

---

## E. Serving engine

**Recommended: SGLang.** Native MoE expert-parallelism, speculative decoding (incl. MTP),
CUDA-graph decode, FP8, and exposed NCCL tuning — lowest B=1 latency of the mainstream stacks.

- **TensorRT-LLM:** marginally faster kernels on some ops, but heavy engine-compile / `.plan`
  workflow; consider only if SGLang leaves latency on the table.
- **vLLM:** excellent throughput, but continuous-batching scheduler adds per-request latency —
  not ideal for a pure B=1 latency target.

Whatever you pick, it just needs to speak the console's contract (OpenAI SSE). To light up the
GPU/expert viz, emit per-token routing into `x_telemetry` via `server/backend.py:RealEngineBackend`.

Example SGLang launch (validate flags against installed version):
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-235B-A22B-FP8 \
  --tp 2 --ep 8 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE --speculative-num-steps 4 \
  --enable-cuda-graph \
  --host 0.0.0.0 --port 8000
```

---

## F. NCCL / comms tuning (expert all-to-all)

Tune for small-message latency, not bulk bandwidth. Start here and measure:

| Env var | Suggested | Why |
|---|---|---|
| `NCCL_P2P_LEVEL` | `NVL` | Force NVLink peer paths |
| `NCCL_ALGO` | `RING,TREE` | Tree for small collectives, ring for large |
| `NCCL_PROTO` | `LL128` | Low-latency protocol for small payloads |
| `NCCL_MAX_NCHANNELS` | `32` | Saturate the mesh (default may under-use links) |
| `NCCL_DEBUG` | `INFO` | Confirm it actually chose NVLink/NVLS |

Sanity-check all-to-all latency with `nccl-tests` (`alltoall_perf`) before blaming the model.

---

## G. Measurement loop (tune in this order)

Fixed window for comparability: **512 prompt / 128 decode / greedy / seed 0 / B=1.**
Watch **TTFT**, **decode tok/s** (= 1000/TPOT_ms), and **bytes-moved/token** (the bandwidth
roofline). Change ONE thing at a time, re-measure, keep what wins:

1. **Baseline** — single GPU, FP16, no tricks. Record TTFT + tok/s.
2. **FP8 weights + INT8/FP8 KV** — expect the biggest jump (~1.5–2×).
3. **CUDA graphs** — remove launch overhead (~1.1–1.3×).
4. **Kernel fusion** (GEMV/dequant, gated FFN, attn+norm) (~1.2–1.4×).
5. **EP=8** across the 8 GPUs — near-linear *if* all-to-all stays cheap; watch comms.
6. **TP=2** within groups — ~10% all-reduce overhead, but unlocks the per-layer latency cut.
7. **Speculative decoding** — n-gram first, then MTP/EAGLE; track acceptance rate.
8. **NCCL tuning** — last 5–10% once comms is the residual bottleneck.

**Stopping rule:** stop when TPOT approaches the bandwidth roofline
(`bytes_per_token / usable_HBM_BW`) — at that point you're physics-limited and further kernel
work yields little; remaining wins come only from moving fewer bytes (more quant, more
speculation) or more GPUs.
