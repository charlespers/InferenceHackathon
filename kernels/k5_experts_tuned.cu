// K5-tuned — bandwidth-optimized drop-in candidate for k5_experts.cu (the ~14.2B/token bottleneck).
//
// Numerically EQUIVALENT to k5_experts_fused (same fused math, same layouts, same residual accum).
// This file implements the three highest-payoff, correctness-preserving B=1 wins from the K5 TODO
// list and docs/b1-tp8-moe-rearchitecture-h200.md §4 — the ones that lift realized HBM-bandwidth
// efficiency (e) without changing the host call signature:
//
//   1. 128-BIT VECTORIZED fp8 LOADS (16 fp8 / 128-bit transaction). The skeleton's scalar `deq()`
//      issues one 1-byte LDG per weight -> caps on LSU issue at ~1/16 of HBM. This is THE e-killer.
//      Decode at B=1 is pure bandwidth (~1 FLOP/byte): the only metric that moves is bytes/sec, and
//      coalesced 128-bit loads are how you get it. (common.cuh:39 TODO, kernels/README.md gap #3.)
//   2. STAGE y[] IN SHARED MEMORY. The gate/up contraction re-reads y[k] from *global* MOE_INTER
//      times per expert (k5_experts.cu:31). y is 4096 f32 = 16 KB; stage once, reuse from smem.
//   3. HOIST THE PER-OUTPUT-CHANNEL SCALE out of the contraction. The fp8 scale is constant along
//      the contraction (per-out-channel), so  sum_k y_k * (w_k * s)  ==  s * sum_k y_k * w_k.
//      The skeleton multiplies by `s` every element (k5_experts.cu:32-33,41); do it once per output.
//      Algebraically identical, removes MOE_INTER*HIDDEN+HIDDEN*MOE_INTER FMAs of scale.
//
// STILL TODO (documented in §4.2 / §7 of the spec; higher effort, need on-box profiling):
//   - atomicAdd residual accumulate (line below) -> per-CTA partial + DSMEM/cluster tree reduce.
//   - persistent/grouped kernel over the TOP_K_local experts (amortize launch, keep y resident).
//   - int4 expert weights (the next ~2x byte win after fp8) — biggest remaining lever.
//   - TP8 column-shard (192-of-1536) needs the down-proj re-laid-out N-major or its read goes strided
//     and latency-bound (spec §4.1, the e_down ~0.36 trap). This single-GPU full-width kernel reads
//     the down-proj contiguously already; the re-layout matters only when you column-shard for TP8.
//
// NOT COMPILED ON THIS BOX (darwin, no nvcc/GPU). Build + validate on the H200 box before trusting:
//   nvcc -arch=sm_90a -O3 --use_fast_math -c kernels/k5_experts_tuned.cu -I kernels/
//   and diff outputs against k5_experts_fused on a reference activation (must match to fp tolerance).
#include "common.cuh"
using namespace q3;

// Raw (un-scaled) dot product of a CONTIGUOUS fp8 weight row w[0..n) with f32 activations ys[0..n),
// using 128-bit loads (uint4 = 16 fp8) so the weight stream hits HBM at full width. Requires n % 16 == 0
// and 16-byte-aligned w (guaranteed here: rows start at j*HIDDEN / o*MOE_INTER, both multiples of 16).
__device__ __forceinline__ float dot_fp8_vec(const fp8* __restrict__ w,
                                             const float* __restrict__ ys, int n) {
  float acc = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;                                  // 16 fp8 per 128-bit load
  #pragma unroll 4
  for (int v = 0; v < nv; ++v) {
    uint4 packed = wv[v];                                 // one coalesced 128-bit transaction
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&packed);
    const float* yy = ys + (v << 4);
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
      fp8 f; f.__x = b[t];                                // reinterpret byte as e4m3
      acc += yy[t] * float(f);                            // convert in-register (compute has slack)
    }
  }
  return acc;
}

// Same signature + semantics as k5_experts_fused. One CTA per active expert slot (grid.x = TOP_K_local).
extern "C" __global__ void k5_experts_fused_tuned(
    const float* __restrict__ y,
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    const fp8* const* __restrict__ Wd,  const float* const* __restrict__ Wd_scale,
    float* __restrict__ h_io) {
  const int slot = blockIdx.x;
  const int e  = sel_idx[slot];
  const float gw = sel_w[slot];

  extern __shared__ float smem[];          // [HIDDEN] staged y  ++  [MOE_INTER] silu(gate)*up
  float* ys = smem;                        // 4096 f32 = 16 KB
  float* a  = smem + HIDDEN;               // 1536 f32 =  6 KB   (launch with 22 KB dynamic smem)

  // (2) stage y into shared memory once; every gate/up dot below reads it from smem, not global.
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) ys[k] = y[k];
  __syncthreads();

  // gate+up fused, per output channel j:  a[j] = silu(s_g * <y,grow>) * (s_u * <y,urow>)
  const fp8* Wgu_e = Wgu[e];
  const float* gsc = Wgu_scale[e];
  for (int j = threadIdx.x; j < MOE_INTER; j += blockDim.x) {
    const fp8* grow = Wgu_e + (size_t)j * HIDDEN;                 // gate row j   (contiguous over k)
    const fp8* urow = Wgu_e + (size_t)(MOE_INTER + j) * HIDDEN;   // up   row j
    const float g = dot_fp8_vec(grow, ys, HIDDEN) * gsc[j];               // (3) scale hoisted
    const float u = dot_fp8_vec(urow, ys, HIDDEN) * gsc[MOE_INTER + j];
    a[j] = silu(g) * u;
  }
  __syncthreads();

  // down GEMV (contiguous over j) with routing weight folded in + accumulate into the residual.
  const fp8* Wd_e = Wd[e];
  const float* dsc = Wd_scale[e];
  for (int o = threadIdx.x; o < HIDDEN; o += blockDim.x) {
    const fp8* drow = Wd_e + (size_t)o * MOE_INTER;                       // contiguous over j
    const float acc = dot_fp8_vec(drow, a, MOE_INTER) * dsc[o];          // (3) scale hoisted
    atomicAdd(&h_io[o], gw * acc);   // TODO(on-box): per-CTA partial + tree/DSMEM reduce (spec §4.2)
  }
  // EP NOTE (unchanged): under expert-parallel, y arrives via dispatch all-to-all and gw*acc partials
  // return via combine all-to-all. Under pure TP8 (spec §2.3) the experts are column-sharded instead
  // and this kernel's down-proj output is one [HIDDEN] partial reduced by the layer all-reduce.
}

// --- Suggested launch (host side, inside the k6 graph capture) ---
//   int threads = 256;
//   size_t smem = (HIDDEN + MOE_INTER) * sizeof(float);              // 22 KB
//   cudaFuncSetAttribute(k5_experts_fused_tuned,
//                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
//   k5_experts_fused_tuned<<<TOP_K_local, threads, smem, stream>>>(...);
