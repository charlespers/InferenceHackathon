// k5_experts.cu — fused MoE-expert computation for Qwen3-235B-A22B, B=1 DECODE.
//
// This is the single-token decode latency bottleneck: ~14.2B of the ~21.6B active
// params/token live in the 8 active experts. At batch size 1 each expert projection is a
// GEMV (M=1), so the kernel is HBM-bandwidth-bound, NOT compute-bound. The whole game is to
// read the fp8 expert weights at peak HBM bandwidth (H100: 3.35 TB/s) and never touch HBM
// more than once.
//
// Qwen3-235B-A22B expert (verified shapes, see common.cuh):
//   hidden = 4096, 128 experts, top-8 active, no shared expert.
//   SwiGLU expert: gate_proj [4096->1536], up_proj [4096->1536], down_proj [1536->4096].
//   a = silu(gate(x)) * up(x)  (elementwise, 1536), then h += sel_w * down(a) (4096).
//   gate|up are stacked into Wgu [3072,4096]; down is Wd [4096,1536].
//   Weights are fp8 e4m3 with per-output-channel fp32 scales; dequantized in-kernel.
//
// FUSION (the point — beat a generic fp8 grouped-GEMM that falls back to a slow path at M=1):
//   Kernel A (gate+up): one fused pass produces a[slot][1536] = silu(s_g*<y,gate_j>)*(s_u*<y,up_j>)
//                       so x is read once and the gate/up halves share the staged activation.
//   Kernel B (down):    h_io[o] += sel_w * s_d * <a[slot], down_o>, routing weight folded into the
//                       epilogue and accumulated straight into the residual (no extra HBM round-trip).
// The two kernels share a small global `a` buffer (8*1536 floats); both tile across all 132 SMs.
//
// BANDWIDTH STRATEGY at B=1:
//   * warp-per-output-row with split-K across the warp's 32 lanes: consecutive lanes read
//     consecutive 16-byte (uint4 = 16 fp8) chunks of the SAME weight row -> fully coalesced HBM.
//     (thread-per-row instead would have 32 threads reading rows HIDDEN apart -> memory-divergent.)
//   * 128-bit vectorized fp8 loads, hardware fp8x2->half2 dequant, 2 FP accumulators for ILP.
//   * grid-stride over (slot, row) with thousands of warps to fill the machine and hide latency.
//   * per-output-channel scale folded once onto the reduced dot (scale is per row, not per element).
//
// Build + self-test (compiles cleanly, validates vs CPU fp32 reference, prints HBM bandwidth):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k5_experts.cu -o /tmp/k5 && /tmp/k5
//
// Standard CUDA only. Uses common.cuh for shapes; all extra helpers are defined locally here so
// this file never edits common.cuh.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

// ---------------------------------------------------------------------------------------------
// Device dot-product primitive: warp dots a contiguous fp8 weight row against a staged f32 vector.
// ---------------------------------------------------------------------------------------------
// Computes sum over k in [0,n) of w[k]*ys[k], where w is a K-major (contiguous) fp8 row and ys is
// the staged activation in shared memory. The contraction is split across the warp's 32 lanes so
// consecutive lanes load consecutive uint4 (16 fp8) chunks of the row -> coalesced 128-bit HBM
// loads. Dequant uses the hardware fp8x2->half2 conversion (8 vector converts per 128-bit load vs
// 16 scalar casts) and two accumulators for instruction-level parallelism. n must be a multiple of
// 16 (HIDDEN=4096 and MOE_INTER=1536 both are). Result is valid on lane 0.
static __device__ __forceinline__ float warp_dot_fp8(const fp8* __restrict__ w,
                                                     const float* __restrict__ ys,
                                                     int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;                                  // 16 fp8 per uint4
  for (int v = lane; v < nv; v += 32) {                   // lanes 0..31 -> consecutive uint4
    uint4 p = wv[v];
    const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
    const float* yy = ys + (v << 4);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {                         // 4 x 32-bit words = 4 x (2 fp8 pairs)
      unsigned wq = wu[q];
      __nv_fp8x2_e4m3 lo, hi;
      lo.__x = (unsigned short)(wq & 0xffffu);
      hi.__x = (unsigned short)(wq >> 16);
      float2 fl = __half22float2((__half2)lo);
      float2 fh = __half22float2((__half2)hi);
      const float* yq = yy + (q << 2);
      a0 += yq[0] * fl.x;  a1 += yq[1] * fl.y;
      a0 += yq[2] * fh.x;  a1 += yq[3] * fh.y;
    }
  }
  float acc = a0 + a1;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                             // valid on lane 0
}

// ---------------------------------------------------------------------------------------------
// Kernel A — fused gate+up:  a_glb[slot][j] = silu(s_g * <y, gate_j>) * (s_u * <y, up_j>)
// ---------------------------------------------------------------------------------------------
// One warp per output channel j of one slot; grid-stride over the (slot, j) work items so the whole
// grid stays busy across all 132 SMs. Wgu[e] is the stacked [2*MOE_INTER, HIDDEN] gate|up matrix:
// rows [0, MOE_INTER) are gate, rows [MOE_INTER, 2*MOE_INTER) are up. y is staged once into shared
// memory per CTA. Launch with dynamic smem = HIDDEN*sizeof(float).
extern "C" __global__ void k5a_gateup(
    const float* __restrict__ y,
    const int* __restrict__ sel_idx,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    float* __restrict__ a_glb, int nslot) {
  extern __shared__ float ys[];                           // [HIDDEN]
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) ys[k] = y[k];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int total = nslot * MOE_INTER;
  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / MOE_INTER;
    const int j    = item - slot * MOE_INTER;
    const int e    = sel_idx[slot];
    const fp8*   W = Wgu[e];
    const float* S = Wgu_scale[e];
    const float g = warp_dot_fp8(W + (size_t)j * HIDDEN,                ys, HIDDEN, lane);
    const float u = warp_dot_fp8(W + (size_t)(MOE_INTER + j) * HIDDEN,  ys, HIDDEN, lane);
    if (lane == 0)
      a_glb[(size_t)slot * MOE_INTER + j] = silu(g * S[j]) * (u * S[MOE_INTER + j]);
  }
}

// ---------------------------------------------------------------------------------------------
// Kernel B — down projection + routed accumulate:  h_io[o] += sel_w * s_d * <a[slot], down_o>
// ---------------------------------------------------------------------------------------------
// One warp per output channel o of one slot; grid-stride over (slot, o). The full a buffer
// (nslot*MOE_INTER floats) is staged into shared memory once per CTA. The routing weight sel_w and
// the per-output-channel down scale are folded into the epilogue, and the partial is accumulated
// straight into the residual stream h_io via atomicAdd (the 8 experts race on the same 4096 rows).
// Launch with dynamic smem = nslot*MOE_INTER*sizeof(float).
extern "C" __global__ void k5b_down(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot) {
  extern __shared__ float as[];                           // [nslot*MOE_INTER]
  const int na = nslot * MOE_INTER;
  for (int i = threadIdx.x; i < na; i += blockDim.x) as[i] = a_glb[i];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int total = nslot * HIDDEN;
  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / HIDDEN;
    const int o    = item - slot * HIDDEN;
    const int e    = sel_idx[slot];
    const float gw = sel_w[slot];
    const fp8*   W = Wd[e];
    const float* S = Wd_scale[e];
    const float d = warp_dot_fp8(W + (size_t)o * MOE_INTER, as + (size_t)slot * MOE_INTER,
                                 MOE_INTER, lane);
    if (lane == 0) atomicAdd(&h_io[o], gw * d * S[o]);
  }
}

// ---------------------------------------------------------------------------------------------
// k5_experts_fused — single-kernel baseline (one CTA per active expert, thread-per-row).
// ---------------------------------------------------------------------------------------------
// Kept as a correctness baseline with the original signature and launch convention (<<<TOP_K,
// threads>>>, no dynamic smem): same fused SwiGLU + routed down-accumulate in one launch. It is
// bandwidth-suboptimal at B=1 (thread-per-row loads are memory-divergent) — the two warp-per-row
// kernels above are the production path — but it is simple to reason about and validate against.
extern "C" __global__ void k5_experts_fused(
    const float* __restrict__ y,
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    const fp8* const* __restrict__ Wd,  const float* const* __restrict__ Wd_scale,
    float* __restrict__ h_io) {
  const int slot = blockIdx.x;
  const int e    = sel_idx[slot];
  const float gw = sel_w[slot];

  __shared__ float a[MOE_INTER];                          // silu(gate)*up, shared across the down pass
  const fp8*   Wg = Wgu[e];
  const float* Sg = Wgu_scale[e];
  for (int j = threadIdx.x; j < MOE_INTER; j += blockDim.x) {
    const fp8* grow = Wg + (size_t)j * HIDDEN;
    const fp8* urow = Wg + (size_t)(MOE_INTER + j) * HIDDEN;
    float g = 0.f, u = 0.f;
    for (int k = 0; k < HIDDEN; ++k) { float yk = y[k]; g += yk * (float)grow[k]; u += yk * (float)urow[k]; }
    a[j] = silu(g * Sg[j]) * (u * Sg[MOE_INTER + j]);
  }
  __syncthreads();

  const fp8*   Wdn = Wd[e];
  const float* Sd  = Wd_scale[e];
  for (int o = threadIdx.x; o < HIDDEN; o += blockDim.x) {
    const fp8* drow = Wdn + (size_t)o * MOE_INTER;
    float acc = 0.f;
    for (int j = 0; j < MOE_INTER; ++j) acc += a[j] * (float)drow[j];
    atomicAdd(&h_io[o], gw * acc * Sd[o]);
  }
}

// ---------------------------------------------------------------------------------------------
// Launch-config helper — pick a grid that fills the H100 (132 SMs) with resident warps.
// ---------------------------------------------------------------------------------------------
struct K5Launch {
  int ctasA, ctasB, block;
  size_t smemA, smemB;
};

static inline K5Launch k5_plan(int nslot, int block = 1024) {
  K5Launch L;
  L.block = block;
  // Enough warps to cover the work with several resident waves over the SMs. gate+up has
  // nslot*MOE_INTER rows, down has nslot*HIDDEN rows; one warp per row. ~264 CTAs * 1024 threads
  // = 8448 warps comfortably oversubscribes 132 SMs (which is what we want to hide HBM latency).
  const int warps_per_cta = block >> 5;
  auto ctas_for = [&](int rows) {
    int need = (rows + warps_per_cta - 1) / warps_per_cta;
    int cap  = 264;                          // ~2 CTAs/SM at 1024 threads; oversubscribe lightly
    return std::min(std::max(need, 132), cap);
  };
  L.ctasA = ctas_for(nslot * MOE_INTER);
  L.ctasB = ctas_for(nslot * HIDDEN);
  L.smemA = (size_t)HIDDEN * sizeof(float);
  L.smemB = (size_t)nslot * MOE_INTER * sizeof(float);
  return L;
}

// Convenience launcher (sets the >48KB dynamic-smem opt-in for kernel B and zeroes the residual
// is the caller's job — h_io must already hold the residual you want to accumulate into).
static inline void k5_launch(const float* y, const int* sel_idx, const float* sel_w,
                             const fp8* const* Wgu, const float* const* Wgu_scale,
                             const fp8* const* Wd, const float* const* Wd_scale,
                             float* a_glb, float* h_io, int nslot, cudaStream_t s = 0) {
  K5Launch L = k5_plan(nslot);
  cudaFuncSetAttribute(k5a_gateup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemA);
  cudaFuncSetAttribute(k5b_down,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemB);
  k5a_gateup<<<L.ctasA, L.block, L.smemA, s>>>(y, sel_idx, Wgu, Wgu_scale, a_glb, nslot);
  k5b_down  <<<L.ctasB, L.block, L.smemB, s>>>(sel_idx, sel_w, Wd, Wd_scale, a_glb, h_io, nslot);
}

// =============================================================================================
// Host-side: CPU fp32 reference (always compiled, reusable), then deterministic input generation
// and the microbenchmark main() (guarded so this file can be #included as a kernel library).
// =============================================================================================

// NOTE on fp8 round-trip: weights are built once on the host as __nv_fp8_e4m3 (the cast applies the
// e4m3 rounding), and the SAME bytes are uploaded to the GPU. Both the CPU reference and the GPU
// dequant read them back via float(fp8), so the round-trip is identical on both sides; the only
// kernel-vs-reference delta is fp32 accumulation order, well under the 1e-2 tolerance.

// ---- CPU fp32 reference --------------------------------------------------------------------
// Mirrors the fused kernel exactly (after fp8 round-trip on the weights): for each active expert
//   a_j = silu( s_g[j] * sum_k y_k * deq(Wgu[gate_j,k]) ) * ( s_u[j] * sum_k y_k * deq(Wgu[up_j,k]) )
//   h_o += sel_w * s_d[o] * sum_j a_j * deq(Wd[o,j])
// Weights are passed as the fp8 arrays actually uploaded to the GPU (so the round-trip matches).
void k5_reference(const float* y, const int* sel_idx, const float* sel_w,
                  const fp8* const* Wgu, const float* const* Wgu_scale,
                  const fp8* const* Wd,  const float* const* Wd_scale,
                  float* h_io, int nslot) {
  std::vector<float> a(MOE_INTER);
  for (int slot = 0; slot < nslot; ++slot) {
    const int e = sel_idx[slot];
    const fp8*   W  = Wgu[e];
    const float* Sg = Wgu_scale[e];
    for (int j = 0; j < MOE_INTER; ++j) {
      const fp8* grow = W + (size_t)j * HIDDEN;
      const fp8* urow = W + (size_t)(MOE_INTER + j) * HIDDEN;
      double g = 0.0, u = 0.0;
      for (int k = 0; k < HIDDEN; ++k) {
        g += (double)y[k] * (double)(float)grow[k];
        u += (double)y[k] * (double)(float)urow[k];
      }
      float gs = (float)g * Sg[j];
      float us = (float)u * Sg[MOE_INTER + j];
      a[j] = (gs / (1.0f + expf(-gs))) * us;             // silu(gs) * us
    }
    const fp8*   Wdn = Wd[e];
    const float* Sd  = Wd_scale[e];
    const float  gw  = sel_w[slot];
    for (int o = 0; o < HIDDEN; ++o) {
      const fp8* drow = Wdn + (size_t)o * MOE_INTER;
      double acc = 0.0;
      for (int j = 0; j < MOE_INTER; ++j) acc += (double)a[j] * (double)(float)drow[j];
      h_io[o] += gw * (float)acc * Sd[o];
    }
  }
}

#ifndef K5_NO_MAIN

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                     \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));         \
  exit(1); } } while (0)

// ---- deterministic, seeded host-side input generation -------------------------------------
// A tiny splitmix-style hash so the CPU reference and the GPU see byte-identical inputs.
static inline unsigned hash_u(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static inline float rnd(unsigned seed, size_t i, float scale, bool positive) {
  unsigned h = hash_u((unsigned)(i * 2654435761u) ^ (seed * 40503u));
  float v = (((h % 2001) / 1000.0f) - 1.0f) * scale;     // in [-scale, scale]
  return positive ? (fabsf(v) + 1e-3f) : v;
}

int main(int argc, char** argv) {
  const int E = 8;                                        // TOP_K active experts
  const int BLK  = (argc > 1) ? atoi(argv[1]) : 1024;
  const double PEAK = (argc > 2) ? atof(argv[2]) : 3350.0;  // GB/s; H100 HBM3 = 3.35 TB/s

  int ndev = 0, dev = 0; cudaDeviceProp prop;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    printf("No CUDA device found.\n"); return 1;
  }
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s  SMs=%d  assumed HBM peak=%.0f GB/s\n", prop.name, prop.multiProcessorCount, PEAK);

  const size_t gu_n = (size_t)2 * MOE_INTER * HIDDEN;     // 3072*4096 fp8 per expert
  const size_t d_n  = (size_t)HIDDEN * MOE_INTER;         // 4096*1536 fp8 per expert

  // ---- build inputs on the host (so the CPU reference uses the exact fp8 bytes) --------------
  std::vector<std::vector<fp8>>   Wgu_host(E), Wd_host(E);
  std::vector<std::vector<float>> Sgu_host(E), Sd_host(E);
  for (int e = 0; e < E; ++e) {
    Wgu_host[e].resize(gu_n);  Wd_host[e].resize(d_n);
    Sgu_host[e].resize(2 * MOE_INTER);  Sd_host[e].resize(HIDDEN);
    for (size_t i = 0; i < gu_n; ++i) Wgu_host[e][i] = (fp8)rnd(1u + e, i, 0.25f, false);
    for (size_t i = 0; i < d_n;  ++i) Wd_host[e][i]  = (fp8)rnd(100u + e, i, 0.25f, false);
    for (int i = 0; i < 2 * MOE_INTER; ++i) Sgu_host[e][i] = rnd(7u + e, i, 0.02f, true);
    for (int i = 0; i < HIDDEN; ++i)        Sd_host[e][i]  = rnd(13u + e, i, 0.02f, true);
  }
  std::vector<float> y_host(HIDDEN);
  for (int k = 0; k < HIDDEN; ++k) y_host[k] = rnd(99u, k, 1.0f, false);
  std::vector<int>   sel_host(E);
  std::vector<float> selw_host(E);
  for (int e = 0; e < E; ++e) { sel_host[e] = e; selw_host[e] = 0.1f + 0.01f * e; }

  // ---- upload ---------------------------------------------------------------------------------
  std::vector<fp8*>   Wgu_dp(E), Wd_dp(E);
  std::vector<float*> Sgu_dp(E), Sd_dp(E);
  for (int e = 0; e < E; ++e) {
    CK(cudaMalloc(&Wgu_dp[e], gu_n * sizeof(fp8)));
    CK(cudaMalloc(&Wd_dp[e],  d_n  * sizeof(fp8)));
    CK(cudaMalloc(&Sgu_dp[e], 2 * MOE_INTER * sizeof(float)));
    CK(cudaMalloc(&Sd_dp[e],  HIDDEN * sizeof(float)));
    CK(cudaMemcpy(Wgu_dp[e], Wgu_host[e].data(), gu_n * sizeof(fp8), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Wd_dp[e],  Wd_host[e].data(),  d_n  * sizeof(fp8), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sgu_dp[e], Sgu_host[e].data(), 2 * MOE_INTER * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd_dp[e],  Sd_host[e].data(),  HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
  }
  const fp8 **Wgu_d, **Wd_d; const float **Sgu_d, **Sd_d;
  CK(cudaMalloc(&Wgu_d, E * sizeof(fp8*)));  CK(cudaMemcpy(Wgu_d, Wgu_dp.data(), E * sizeof(fp8*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd_d,  E * sizeof(fp8*)));  CK(cudaMemcpy(Wd_d,  Wd_dp.data(),  E * sizeof(fp8*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sgu_d, E * sizeof(float*))); CK(cudaMemcpy(Sgu_d, Sgu_dp.data(), E * sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd_d,  E * sizeof(float*))); CK(cudaMemcpy(Sd_d,  Sd_dp.data(),  E * sizeof(float*), cudaMemcpyHostToDevice));

  int   *sel_d; float *selw_d, *y_d, *h_d, *a_d;
  CK(cudaMalloc(&sel_d,  E * sizeof(int)));    CK(cudaMemcpy(sel_d,  sel_host.data(),  E * sizeof(int),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&selw_d, E * sizeof(float)));  CK(cudaMemcpy(selw_d, selw_host.data(), E * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&y_d, HIDDEN * sizeof(float))); CK(cudaMemcpy(y_d, y_host.data(), HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&h_d, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&a_d, (size_t)E * MOE_INTER * sizeof(float)));
  CK(cudaDeviceSynchronize());

  // ---- correctness: GPU fused kernels vs CPU fp32 reference (residual starts at 0) -----------
  std::vector<float> ref(HIDDEN, 0.0f), got(HIDDEN, 0.0f);
  // host pointer arrays of the fp8 buffers for the reference (same bytes uploaded to the GPU)
  std::vector<const fp8*> Wgu_hp(E), Wd_hp(E); std::vector<const float*> Sgu_hp(E), Sd_hp(E);
  for (int e = 0; e < E; ++e) { Wgu_hp[e] = Wgu_host[e].data(); Wd_hp[e] = Wd_host[e].data();
                                Sgu_hp[e] = Sgu_host[e].data(); Sd_hp[e] = Sd_host[e].data(); }
  k5_reference(y_host.data(), sel_host.data(), selw_host.data(),
               Wgu_hp.data(), Sgu_hp.data(), Wd_hp.data(), Sd_hp.data(), ref.data(), E);

  CK(cudaMemset(h_d, 0, HIDDEN * sizeof(float)));
  k5_launch(y_d, sel_d, selw_d, Wgu_d, Sgu_d, Wd_d, Sd_d, a_d, h_d, E);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(got.data(), h_d, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));

  double max_abs = 0.0, max_rel = 0.0;
  for (int i = 0; i < HIDDEN; ++i) {
    double ad = fabs((double)ref[i] - (double)got[i]);
    max_abs = std::max(max_abs, ad);
    max_rel = std::max(max_rel, ad / (fabs((double)ref[i]) + 1e-6));
  }
  printf("correctness vs CPU fp32 reference:  max_abs=%.3e  max_rel=%.3e  -> %s (<1e-2)\n",
         max_abs, max_rel, (max_abs < 1e-2 ? "PASS" : "FAIL"));

  // ---- microbenchmark: cudaEvent timing over many iters --------------------------------------
  K5Launch L = k5_plan(E, BLK);
  CK(cudaFuncSetAttribute(k5a_gateup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemA));
  CK(cudaFuncSetAttribute(k5b_down,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemB));

  cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
  const int WARM = 30, IT = 300;

  auto bench = [&](auto launch) -> float {
    for (int i = 0; i < WARM; ++i) launch();
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
    for (int i = 0; i < IT; ++i) launch();
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms; CK(cudaEventElapsedTime(&ms, s, e)); return ms / IT;
  };

  auto runA = [&]() {
    k5a_gateup<<<L.ctasA, L.block, L.smemA>>>(y_d, sel_d, Wgu_d, Sgu_d, a_d, E);
  };
  auto runB = [&]() {
    k5b_down<<<L.ctasB, L.block, L.smemB>>>(sel_d, selw_d, Wd_d, Sd_d, a_d, h_d, E);
  };
  auto runAB = [&]() { runA(); runB(); };

  float msA  = bench(runA);
  float msB  = bench(runB);
  float msAB = bench(runAB);
  CK(cudaGetLastError());

  // Bytes that MUST be read from HBM per token: the fp8 expert weights (the bottleneck). The y, a,
  // scales and h are negligible (<1 MB total) next to the ~151 MB of fp8 weights.
  const double bytesA = (double)E * gu_n;                 // gate+up weights
  const double bytesB = (double)E * d_n;                  // down weights
  const double bytesT = bytesA + bytesB;
  auto gbps = [](double bytes, float ms) { return bytes / 1e6 / ms; };   // bytes/ms = GB/s

  printf("\nper-token expert weight read: %.1f MB  (gate+up %.1f MB + down %.1f MB)\n",
         bytesT / 1e6, bytesA / 1e6, bytesB / 1e6);
  printf("launch: block=%d  CTAs(A)=%d  CTAs(B)=%d\n", L.block, L.ctasA, L.ctasB);
  printf("  %-22s %10s %10s %10s\n", "stage", "us/tok", "GB/s", "%HBMpeak");
  printf("  %-22s %10.2f %10.1f %9.1f%%\n", "gate+up (A)", msA  * 1e3, gbps(bytesA, msA),  100.0 * gbps(bytesA, msA)  / PEAK);
  printf("  %-22s %10.2f %10.1f %9.1f%%\n", "down    (B)", msB  * 1e3, gbps(bytesB, msB),  100.0 * gbps(bytesB, msB)  / PEAK);
  printf("  %-22s %10.2f %10.1f %9.1f%%\n", "fused   (A+B)", msAB * 1e3, gbps(bytesT, msAB), 100.0 * gbps(bytesT, msAB) / PEAK);
  printf("\nMoE-expert decode over %d layers: %.2f ms/token\n", N_LAYERS, msAB * N_LAYERS);

  // ---- cleanup --------------------------------------------------------------------------------
  for (int e2 = 0; e2 < E; ++e2) {
    cudaFree(Wgu_dp[e2]); cudaFree(Wd_dp[e2]); cudaFree(Sgu_dp[e2]); cudaFree(Sd_dp[e2]);
  }
  cudaFree(Wgu_d); cudaFree(Wd_d); cudaFree(Sgu_d); cudaFree(Sd_d);
  cudaFree(sel_d); cudaFree(selw_d); cudaFree(y_d); cudaFree(h_d); cudaFree(a_d);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return 0;
}
#endif // K5_NO_MAIN
