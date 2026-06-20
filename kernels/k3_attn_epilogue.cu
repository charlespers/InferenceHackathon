// K3 — attention epilogue: O-proj GEMV (Q_DIM=8192 -> HIDDEN=4096) with FUSED residual add.
//   h_out = h_in + Wo @ attn_out.   One fewer dispatch than proj-then-add.
//
// Fleshed out to the repo's warp-per-output-row, coalesced fp8-GEMV idiom (mirrors k5_experts.cu
// warp_dot_fp8 and k1_attn_prologue.cu k1_warp_dot):
//   * WARP-PER-OUTPUT-ROW: one warp owns one of the HIDDEN=4096 output channels o; it dots the
//     fp8 weight row Wo[o, 0..Q_DIM) against the staged attn_out[Q_DIM] activation.
//   * SPLIT-K across the 32 lanes: consecutive lanes read consecutive 16-byte (uint4 = 16 fp8)
//     chunks of the SAME weight row -> fully coalesced 128-bit HBM loads. Hardware fp8x2->half2
//     dequant (8 vector converts per 128-bit load) + 2 FP accumulators for ILP.
//   * attn_out[Q_DIM] is staged once into shared memory per CTA so the GEMV reads it from smem.
//   * per-output-channel scale folded once onto the reduced dot; residual h_in[o] added in the
//     epilogue and written straight to h_out[o] -> no extra HBM round-trip.
//
// Q_DIM=8192 is a multiple of 16, so the uint4 (16-fp8) vectorization is exact.
//
// Build:  nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k3_attn_epilogue.cu -o /tmp/k3
//         (also #included as a kernel library by decode_step.cu)
#include "common.cuh"
using namespace q3;

#ifndef Q3_K3_DEFS
#define Q3_K3_DEFS
// Coalesced split-K dot of one fp8 weight row w[0..n) with the staged activation xs[0..n) (smem),
// collaborating across a 32-lane warp.  n must be a multiple of 16.  Result valid on lane 0.
static __device__ __forceinline__ float k3_warp_dot(const fp8* __restrict__ w,
                                                     const float* __restrict__ xs,
                                                     int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;                                   // 16 fp8 per uint4
  for (int v = lane; v < nv; v += 32) {                    // consecutive lanes -> consecutive uint4
    uint4 p = wv[v];
    const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
    const float* xx = xs + (v << 4);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      unsigned wq = wu[q];
      __nv_fp8x2_e4m3 lo, hi;
      lo.__x = (unsigned short)(wq & 0xffffu);
      hi.__x = (unsigned short)(wq >> 16);
      float2 fl = __half22float2((__half2)lo);
      float2 fh = __half22float2((__half2)hi);
      const float* xq = xx + (q << 2);
      a0 += xq[0]*fl.x;  a1 += xq[1]*fl.y;
      a0 += xq[2]*fh.x;  a1 += xq[3]*fh.y;
    }
  }
  float acc = a0 + a1;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                              // valid on lane 0
}
#endif // Q3_K3_DEFS

// attn_out: [Q_DIM=8192]            attention output (from K2 reduce)
// Wo:       fp8 [HIDDEN, Q_DIM]     O-proj, K-major (in=Q_DIM contiguous), scale [HIDDEN] per-out-channel
// h_in:     [HIDDEN]                residual stream coming in
// h_out:    [HIDDEN]                residual stream out = h_in + Wo @ attn_out  (may alias h_in)
// Launch:   warp-per-row, grid-stride over HIDDEN rows; dynamic smem = Q_DIM*sizeof(float).
extern "C" __global__ void k3_attn_epilogue(
    const float* __restrict__ attn_out,
    const fp8*  __restrict__ Wo, const float* __restrict__ Wo_scale,
    const float* __restrict__ h_in,
    float* __restrict__ h_out) {
  extern __shared__ float xs[];                            // [Q_DIM]
  for (int k = threadIdx.x; k < Q_DIM; k += blockDim.x) xs[k] = attn_out[k];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int o = gwarp; o < HIDDEN; o += nwarp) {
    float acc = k3_warp_dot(Wo + (size_t)o * Q_DIM, xs, Q_DIM, lane);
    if (lane == 0) h_out[o] = h_in[o] + acc * Wo_scale[o]; // fused residual add
  }
}

// Launch helper: stage attn_out[Q_DIM] in dynamic smem; pick a grid that covers HIDDEN output rows
// with enough resident warps to fill the H100 (132 SMs) and hide HBM latency.
#ifdef Q3_K3_LAUNCH_HELPER
static inline void k3_launch(const float* attn_out, const fp8* Wo, const float* Wo_scale,
                             const float* h_in, float* h_out, cudaStream_t stream = 0) {
  const int block = 256;                                   // 8 warps/CTA
  const int warps_per_cta = block >> 5;
  // HIDDEN=4096 rows / 8 warps = 512 CTAs needed for one warp/row; cap to a light oversubscribe.
  int ctas = (HIDDEN + warps_per_cta - 1) / warps_per_cta; // 512
  if (ctas > 264) ctas = 264;                              // grid-stride handles the remainder
  const size_t smem = (size_t)Q_DIM * sizeof(float);
  cudaFuncSetAttribute(k3_attn_epilogue, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  k3_attn_epilogue<<<ctas, block, smem, stream>>>(attn_out, Wo, Wo_scale, h_in, h_out);
}
#endif
