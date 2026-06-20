# charles-work — B=1 latency hypertuning for Qwen3-235B-A22B (8×H100)

A session's work on **maximizing single-user (batch-1) tok/s** for Qwen3-235B-A22B: first-principles
research → an implementation spec → **measured kernel hypertuning on the real H100** → engine validation.
Hardware confirmed on-box: **8×H100 80GB HBM3 (~3.35 TB/s)** (an earlier "H200" was not this box).

## The one idea
B=1 decode is **memory-bandwidth bound** (GEMV, arithmetic intensity ≈1; tensor cores ~99% idle).
`TPOT = bytes-moved-per-token / usable-HBM-BW + exposed latency`. Everything here is "move fewer bytes,
expose less latency, accept more tokens per memory pass." Throughput tricks (continuous batching, paged
KV, big-GEMM tiling) are neutral-to-harmful at B=1.

## Deliverables (in this branch)

| Artifact | What |
|---|---|
| `docs/b1-latency-architecture.md` | 15 B=1-latency avenues, each adversarially verified vs the roofline. Two playbook-inverting findings: **EP→TP8** (EP is busiest-rank bound at B=1) and **quantize the *big* thing hardest** (routed experts = most bandwidth, least sensitive). |
| `docs/b1-tp8-moe-rearchitecture-h200.md` | Implementation spec for the TP8 column-sharded MoE. (Filename says h200; numbers scale ÷1.433 to this H100; break-evens/contexts unchanged.) **Predicted the exact vLLM launch crash we then hit.** |
| `kernels/k5_experts_warp.cu` | The **measured-best K5 MoE-expert kernel** — warp-per-row + split-K (coalesced) + fp8x2→half2 dequant + full occupancy. |
| `kernels/k5_experts_tuned.cu` | Intermediate fused single-kernel variant (128-bit loads + smem-y + hoisted scale), 8.5×. |
| `kernels/k5_microbench.cu` | Reproducible correctness + HBM-bandwidth harness (reference vs winner, A/B split). |
| `docs/k5-kernel-results-h100.md` | The measured optimization journey + remaining headroom. |

## Headline measured result (real H100, K5 = the ~14.2B/token bottleneck)

| K5 version | e (frac of 3.35 TB/s) | GB/s | vs scalar |
|---|---|---|---|
| scalar skeleton (`k5_experts.cu`) | 0.005 | 15 | 1× |
| +128-bit loads, smem-y, hoisted scale | 0.039 | 131 | 8.5× |
| +tile across CTAs (fill 132 SMs) | 0.257 | 862 | 56× |
| +warp-per-row, split-K (coalesced) | 0.431 | 1444 | 94× |
| +fp8x2→half2 dequant, full occupancy | **0.459** | **1538** | **~100×** |

Correctness clean throughout (max relative error **3.2e-5** vs the scalar reference). Each step chosen by
*measuring* where the previous was bound: load-width → SM occupancy → memory coalescing → dequant.
Reproduce: `nvcc -arch=sm_90a -O3 --use_fast_math kernels/k5_microbench.cu -I kernels -o k5bench && ./k5bench 264 1024 3350`.

## Engine validation (the spec paid off)
The FP8 vLLM launch (`--tensor-parallel-size 8`) crashed with exactly the spec's §6.1 prediction:
`ValueError: output_size of gate's and up's weight = 192 is not divisible by block_n = 128` — i.e.
`MOE_INTER 1536 / TP8 = 192`, `192 % 128 ≠ 0`. **Fix = expert-parallel** (keep experts whole) or TP4×EP2
(384, divisible) or BF16. End-to-end benchmark (`bench/measure.py` → `bench/roofline.py`) is validated
against the mock and ready; running it against the real engine is pending GPU availability.

## Open / next
- **Run the engine vs the benchmark** for real TTFT/TPOT/tok-s (blocked only on GPU availability; harness ready).
- **K5 headroom**: down-proj kernel (e=0.405, short contraction) → block-level reduce; int4 expert weights (next ~2× byte win); persistent kernel + K6 CUDA-graph capture.
- **Route prediction**: `routing_stats.json` `markov_matrices` are ideal for speculative expert-prefetch.
