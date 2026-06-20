# kernels/ — fused CUDA decode kernels for Qwen3-235B-A22B (B=1, sm_90a)

**Best-first-guess skeletons** implementing the fusion map. Structure + signatures + the fused
math are real; every tuning degree-of-freedom is marked `TODO(on-box)`. These are the
*second-half* play — ship wins from quant + parallelism + CUDA-graph + speculation first
(`docs/kernel-design/trajectory.md`), then hand-tune kernels where the profiler says it pays.

Public model facts + standard CUDA only — no proprietary engine internals.

## Files ↔ fusion map (one decode layer)

| File | Kernel | Fuses | Shapes |
|---|---|---|---|
| `common.cuh` | shared | Qwen3 shapes as constants + fp8 dequant/rmsnorm/silu helpers | — |
| `k1_attn_prologue.cu` | K1 | input-RMSNorm → fused QKV GEMV → per-head QK-norm → RoPE → KV write | W 4096×9216 |
| `k2_flash_decode.cu` | K2 | split-KV single-query GQA online-softmax (2-pass) | 64Q/4KV, hd128 |
| `k3_attn_epilogue.cu` | K3 | O-proj GEMV **+ fused residual** | W 8192×4096 |
| `k4_router.cu` | K4 | post-RMSNorm → gate GEMV → fp32 softmax → top-8 → renorm (on-device) | W 128×4096 |
| `k5_experts.cu` | K5 | gate+up+silu fused, down×gate+accumulate; grouped/persistent over 8; EP hooks | W 3072×4096, 4096×1536 |
| `k6_graph_capture.cu` | K6 | whole-step CUDA-graph capture/replay + on-device sampling | — |
| `nvshmem_comms.cu` | sync | device-initiated (NVSHMEM) collectives standalone microbench: recursive-doubling AR + put-based A2A vs NCCL floor | — |
| `overlap_decode.cu` | sync | NCCL collective hidden behind independent compute (chunked + layer-pipeline schemes) | — |
| `nvshmem_overlap_decode.cu` | sync | **the combination**: NVSHMEM AR (cheap per-call) *and* overlapped with the next layer's GEMV (double-buffered, event-gated) — neither `nvshmem_comms.cu` nor `overlap_decode.cu` alone does both | — |

## Build (on the box)

```bash
nvcc -arch=sm_90a -O3 --use_fast_math -c kernels/k*.cu -I kernels/
# k6 is host-side (CUDA graph) + links the device kernels into one captured step.
```

## Priority of hand-tuning (highest payoff first)
1. **K5 experts** — ~14.2B/token, the bottleneck. fp8→int4 weights, 128-bit loads, persistent kernel.
2. **K1 prologue** — restructure to **warp-per-head** so the QK-norm 128-dim reduce + RoPE are warp-local.
3. **K2 flash-decode** — tune #KV-splits to fill the 132 SMs at the target context length.
4. **K3 / K4** — smaller; the fused residual (K3) and on-device top-8 (K4) are the wins, already sketched.

## Known skeleton gaps (intentional `TODO(on-box)`)
- Reductions are written as sketches — replace with `cub::BlockReduce` / warp shuffles.
- K1 needs the warp-per-head restructure for the per-head QK-norm reduce.
- fp8 loads are scalar — vectorize to 128-bit (16×fp8) with ILP.
- K5 uses `atomicAdd` for the residual accumulate — switch to per-CTA partial + tree reduce.
- All kernels must be validated for **CUDA-graph capture** and wired into `k6`.
- Validate numerics against a reference (HF transformers) before trusting any speed number.
