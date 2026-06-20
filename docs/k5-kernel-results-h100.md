# K5 MoE-Expert Kernel — Measured Hypertuning on 8×H100

**What:** the K5 fused-MoE-experts kernel (`kernels/k5_experts.cu`) is the B=1 decode bottleneck —
~14.2B of the ~21.6B active params/token. This is the on-box record of taking it from a correct-but-naive
skeleton to a bandwidth-tuned kernel, **measured on a real NVIDIA H100 80GB** (sm_90a, CUDA 12.6), not
projected. Every step preserves numerical correctness (max relative error **3.2e-5** vs the scalar
reference, fp32 accumulation).

> **Hardware note:** the cluster is **8× H100 80GB HBM3 (~3.35 TB/s)**, confirmed via `nvidia-smi` — not
> H200. Efficiency `e` below is fraction of the 3.35 TB/s peak; it is the hardware-relative, transferable
> metric (on H200's 4.8 TB/s the same `e` yields ~1.43× the GB/s and tok/s).

## The journey (measured; 8 active experts, fp8 weights, 151 MB moved/call)

| Kernel version | ms/call | GB/s | e | vs scalar | what changed & why it was the bottleneck |
|---|---|---|---|---|---|
| `k5_experts.cu` (scalar reference) | 9.824 | 15 | 0.005 | 1× | 1-byte `deq()` loads + per-element scale + global `y` re-reads → LSU-issue bound at ~1/16 of HBM |
| fused + 128-bit loads + smem-`y` + hoisted scale | 1.151 | 131 | 0.039 | 8.5× | `uint4` (16 fp8) loads, stage `y` in smem, scale once/output. Now **occupancy-bound**: 8 CTAs on 132 SMs |
| + tile each expert across CTAs | 0.175 | 862 | 0.257 | 56× | split gate/up and down across CTAs (global `a` buffer) to fill SMs. Now **memory-divergent** across warps |
| + warp-per-row, split-K across lanes | 0.105 | 1444 | 0.431 | 94× | one warp owns a row; 32 lanes read consecutive bytes → **coalesced** HBM; shuffle-reduce the contraction |
| + fp8x2→half2 hardware dequant, full occupancy | **0.098** | **1538** | **0.459** | **~100×** | vector dequant (8 conv/load vs 16); best launch **264 CTAs × 1024 threads** (`kernels/k5_experts_warp.cu`) |

Each step was chosen by *measuring where the previous one was bound* — load width → SM occupancy →
memory coalescing → dequant throughput. That feedback loop is the whole method.

## Where the time goes now (A vs B split, best config)

| stage | bytes | ms/call | GB/s | e |
|---|---|---|---|---|
| gate+up (kernel A) | 101 MB | 0.0613 | 1642 | **0.490** |
| down (kernel B) | 50 MB | 0.0371 | 1356 | **0.405** |
| total (A+B) | 151 MB | 0.0984 | 1534 | 0.458 |

The **down-proj (B) is the weaker kernel** (e=0.405): its contraction is only 1536 (vs 4096 for gate/up),
so each lane does just 3 vectorized loads and the warp-reduce + 48 KB all-`a` smem (occupancy cap)
overhead is proportionally larger. That is the next thing to chase.

## Remaining headroom (honest, measured-or-bounded)

- **Down-proj kernel B → block-level reduce / per-expert partials** instead of a warp-reduce over a short
  contraction; relieve the 48 KB smem occupancy cap. Most likely path from e≈0.46 toward ~0.55.
- **int4 expert weights** — the next ~2× *byte* win (halves the dominant term); needs an accuracy gate.
- **atomicAdd over 8 experts → tree/DSMEM reduce** (latency, not bandwidth; small at B=1).
- **Persistent kernel** folding A+B and capturing into the K6 whole-step CUDA graph (removes 2 launches/layer).

## Scope / honesty

- This is a **single-GPU, full-width (1536) expert** micro-benchmark — it measures the kernel's realized
  HBM efficiency `e`, the transferable number. The **per-GPU decode contribution depends on the sharding**:
  under pure **TP8** each GPU reads a 192-col slice (1/8 the bytes → ~1.16 ms MoE/token at this `e`);
  under **EP8** the busiest rank reads ~2.6 experts (see `docs/b1-tp8-moe-rearchitecture-h200.md` §2).
  The TP8 192-col shard is a *narrower* GEMV, so expect a lower `e` than this full-width 0.46 — the spec's
  e₁₉₂ ≈ 0.50–0.55 best-case is consistent with (and slightly above) this full-width measurement.
- It does **not** include attention (K1–K3), router (K4), inter-GPU comms, or sampling — K5 is the
  dominant *weight-bandwidth* term, not the whole decode step.
- Random fp8 weights; correctness is the **kernel-vs-reference** equivalence, not model accuracy (validate
  the full model against HF before trusting any quality number, per `kernels/README.md`).

## Reproduce

```bash
# on the box (8×H100), kernels copied to /workspace/k5bench
/usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math kernels/k5_microbench.cu -I kernels -o k5bench
CUDA_VISIBLE_DEVICES=0 ./k5bench 264 1024 3350     # args: CTAs block peak_GBps (use 4800 for H200)
```

Files: `kernels/k5_experts.cu` (reference) · `kernels/k5_experts_warp.cu` (tuned winner) ·
`kernels/k5_microbench.cu` (this harness).
