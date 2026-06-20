# Latency-oriented Qwen3-235B-A22B inference on 8×H100

Design notes for the hackathon. Target: **minimize per-token decode latency at
batch size 1** (not throughput). Numbers below come from `inferutil` — a
pure-stdlib analytical model (`python -m inferutil`). They are roofline lower
bounds; real numbers slot in beside them once conifer + hardware land.

## The model (validated against published config.json)

| | |
|---|---|
| layers | 94 (every layer is MoE, `decoder_sparse_step=1`) |
| hidden | 4096 |
| attention | 64 Q heads / 4 KV heads (GQA), head_dim 128 |
| experts | 128 routed, top-8, **no shared expert** |
| expert FFN | SwiGLU, inner dim 1536 |
| total / active | **235.1B / 21.6B** (9.2%) |
| KV / token | 192.5 KB (2·94·512·2B) — small, thanks to GQA |

## The single most important fact

**B=1 decode is memory-bandwidth bound, not compute bound.** A matmul at B=1 is
a GEMV: ~1 FLOP per weight byte read. Compute is ~0.1% of the latency budget.
*Tuning FLOPs is wasted effort.* Every optimization that matters reduces **bytes
read from HBM** or **collective latency**, not FLOPs.

Floor (perfect balance, no comms): active 21.6B × 2B / (8 × 3.35 TB/s)
≈ **1.85 ms/token ≈ 540 tok/s**. That is the wall. Everything below is about how
close we get and where we leak.

## Where the latency goes (hybrid TP-attn + EP-experts, bf16, 32k ctx)

| term | ms | what it is | lever |
|---|---|---|---|
| weight reads | 3.33 | active expert + attn weights from HBM | **FP8** (~halves it) |
| comms | 1.41 | 94 layers × small all-to-all/all-reduce (latency-bound) | fuse / overlap |
| KV reads | 0.47 | read whole KV cache each token (grows with ctx) | KV compression |
| compute | 0.006 | the GEMVs themselves | — (ignore) |

## Plan choice: TP vs EP at B=1 (counterintuitive)

- **Plain TP=8**: ~3.03 ms. Every GPU reads an even 1/8 of all active weights.
  Perfectly balanced; pays 2 all-reduces/layer.
- **Naive EP=8**: ~5.21 ms — *slower*. Only 8 active experts over 8 GPUs, so the
  busiest GPU runs **2.6×** its fair share (E[max] of 8 balls in 8 bins) while
  others idle. Imbalance, not comms, is the killer.

So "distribute experts across GPUs" is a *throughput* win, not an automatic
*latency* win. EP beats TP only once the imbalance is removed — by static
placement of hot experts, replication, or **predicting the route and prefetching
the right experts onto the right GPU ahead of time.** That is precisely the
high-value target for the prediction workstream.

## Optimization priority (B=1 latency)

1. **FP8 weights** — halves the dominant term. 192 → 282 tok/s. Highest ROI.
2. **Expert placement / prediction** — kill the 2.6× imbalance and hide
   expert-transfer latency behind attention. Unlocks EP and speculative prefetch.
3. **Comms fusion / overlap** — 94 tiny collectives = 1.4 ms of pure latency
   tax. Overlap dispatch with the previous layer's compute; fuse where possible.
4. **KV-cache compression** — caps the attention term at long context (25 GB at
   128k); matters once we push context.
5. **B=1 GEMV + decode-attention kernels** — make the matmuls actually hit the
   BW roofline (the analytical floor assumes 100% BW utilization; real kernels
   don't). This is where conifer's kernels + custom Triton come in.

## Suggested workstreams (maps to teammates)

- **Runtime / sharding**: weight layout, TP+EP placement, the decode loop.
- **Prediction / prefetch**: route prediction, hot-expert replication, overlap.
- **Kernels**: FP8 GEMV, fused MoE dispatch, decode attention for GQA at B=1.
- **Comms**: collective fusion + compute/comm overlap across the 94 layers.

## Open questions (revisit when conifer lands)

- What does conifer give us — weight loading, a decode loop, kernels, comms?
- FP8 from the start, or bf16 first then quantize?
- Measured `collective_latency_s` on the real NVSwitch fabric (model uses 5 µs).
- Are we H100 (3.35 TB/s) or H200 (4.8 TB/s)? H200 moves the floor to ~1.3 ms.
