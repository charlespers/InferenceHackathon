// K4 — MoE router, fully on-device (no host sync).
//   post-RMSNorm(h, w_post_norm) -> gate GEMV (Wgate [N_EXPERTS=128, HIDDEN] fp8) -> fp32 softmax
//   over 128 logits -> top-8 -> renormalize the 8 selected weights to sum 1 (norm_topk_prob=true).
//   Writes sel_idx[8] + sel_w[8] in device memory so K5 reads them with no host round-trip.
//   No shared expert.
//
// Fleshed out to the repo's warp-per-output-row, coalesced fp8-GEMV idiom (mirrors k5_experts.cu
// warp_dot_fp8 / k1_attn_prologue.cu):
//   * stage the normed activation y[HIDDEN] once into shared memory (block-wide RMSNorm, no HBM
//     round-trip for y).
//   * WARP-PER-EXPERT for the gate GEMV: warp e dots the fp8 weight row Wgate[e, 0..HIDDEN) against
//     y, split-K coalesced across the 32 lanes (consecutive lanes -> consecutive uint4 = 16 fp8).
//     With 128 experts this is 128 warps of work; a single CTA (or a few) covers it grid-strided.
//   * softmax over 128 (fp32) + top-8 selection + renormalize on lane 0 (the selection is tiny,
//     O(128*8); keeping it single-threaded avoids a fragile parallel top-k and is well off the
//     HBM-bound critical path — the weight read dominates).
//
// HIDDEN=4096 is a multiple of 16 so the uint4 (16-fp8) vectorization is exact.
//
// Build:  nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k4_router.cu -o /tmp/k4
//         (also #included as a kernel library by decode_step.cu)
#include "common.cuh"
#include <cfloat>
using namespace q3;

#ifndef Q3_K4_DEFS
#define Q3_K4_DEFS
// Coalesced split-K dot of one fp8 weight row w[0..n) with the staged activation ys[0..n) (smem),
// collaborating across a 32-lane warp.  n must be a multiple of 16.  Result valid on lane 0.
static __device__ __forceinline__ float k4_warp_dot(const fp8* __restrict__ w,
                                                     const float* __restrict__ ys,
                                                     int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;
  for (int v = lane; v < nv; v += 32) {
    uint4 p = wv[v];
    const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
    const float* yy = ys + (v << 4);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      unsigned wq = wu[q];
      __nv_fp8x2_e4m3 lo, hi;
      lo.__x = (unsigned short)(wq & 0xffffu);
      hi.__x = (unsigned short)(wq >> 16);
      float2 fl = __half22float2((__half2)lo);
      float2 fh = __half22float2((__half2)hi);
      const float* yq = yy + (q << 2);
      a0 += yq[0]*fl.x;  a1 += yq[1]*fl.y;
      a0 += yq[2]*fh.x;  a1 += yq[3]*fh.y;
    }
  }
  float acc = a0 + a1;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                              // valid on lane 0
}
#endif // Q3_K4_DEFS

// h:           [HIDDEN]                 residual stream (post-attn) input
// w_post_norm: [HIDDEN]                 post-attention RMSNorm weights
// Wgate:       fp8 [N_EXPERTS, HIDDEN]  router gate, K-major; scale [N_EXPERTS] per-out-channel
// sel_idx:     [TOP_K]                  (out) selected expert ids
// sel_w:       [TOP_K]                  (out) renormalized gate weights (sum to 1)
// Launch:  ONE CTA (e.g. 256 threads = 8 warps), dynamic smem = HIDDEN*sizeof(float).  A single CTA
//          keeps logits[] + selection in one block; warps grid-stride over the 128 experts.
extern "C" __global__ void k4_router(
    const float* __restrict__ h,
    const float* __restrict__ w_post_norm,
    const fp8*  __restrict__ Wgate, const float* __restrict__ Wgate_scale,
    int* __restrict__ sel_idx,
    float* __restrict__ sel_w) {
  extern __shared__ float smem[];                          // [HIDDEN] staged y
  float* ys = smem;
  __shared__ float logits[N_EXPERTS];

  // ---- post-RMSNorm -> y[HIDDEN] staged in shared memory (block-wide, no HBM round-trip) ----
  float part = 0.f;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) { float v = h[i]; part += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
  __shared__ float warp_ss[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) warp_ss[wid] = part;
  __syncthreads();
  __shared__ float rinv_sh;
  if (threadIdx.x == 0) {
    float ss = 0.f; int nw = (blockDim.x + 31) >> 5;
    for (int i = 0; i < nw; i++) ss += warp_ss[i];
    rinv_sh = rsqrtf(ss / HIDDEN + RMS_EPS);
  }
  __syncthreads();
  const float rinv = rinv_sh;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) ys[i] = h[i] * rinv * w_post_norm[i];
  __syncthreads();

  // ---- gate GEMV: warp-per-expert, split-K coalesced over HIDDEN ----
  const int gwarp = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;
  for (int e = gwarp; e < N_EXPERTS; e += nwarp) {
    float acc = k4_warp_dot(Wgate + (size_t)e * HIDDEN, ys, HIDDEN, lane);
    if (lane == 0) logits[e] = acc * Wgate_scale[e];
  }
  __syncthreads();

  // ---- fp32 softmax over 128 -> top-8 -> renormalize to sum 1 (single-threaded; off-critical-path) ----
  if (threadIdx.x == 0) {
    float mx = -FLT_MAX;
    for (int e = 0; e < N_EXPERTS; ++e) mx = fmaxf(mx, logits[e]);
    float sum = 0.f;
    for (int e = 0; e < N_EXPERTS; ++e) sum += __expf(logits[e] - mx);
    const float inv_sum = 1.f / sum;
    // top-8 by probability, masking out already-picked experts; renormalize the chosen 8 to sum 1.
    float chosen = 0.f;
    for (int s = 0; s < TOP_K; ++s) {
      int   bi = -1;
      float bv = -1.f;
      for (int e = 0; e < N_EXPERTS; ++e) {
        bool taken = false;
        for (int j = 0; j < s; ++j) if (sel_idx[j] == e) { taken = true; break; }
        if (taken) continue;
        float p = __expf(logits[e] - mx) * inv_sum;
        if (p > bv) { bv = p; bi = e; }
      }
      sel_idx[s] = (bi >= 0 ? bi : s);          // robust: nan logits can leave bi=-1 -> clamp valid
      sel_w[s]   = (bv >= 0.f ? bv : 0.f);
      chosen    += sel_w[s];
    }
    const float inv_chosen = 1.f / chosen;
    for (int s = 0; s < TOP_K; ++s) sel_w[s] *= inv_chosen;   // renormalize to sum 1
  }
}

// Launch helper: ONE CTA, dynamic smem = HIDDEN floats for the staged y.
#ifdef Q3_K4_LAUNCH_HELPER
static inline void k4_launch(const float* h, const float* w_post_norm,
                             const fp8* Wgate, const float* Wgate_scale,
                             int* sel_idx, float* sel_w, cudaStream_t stream = 0) {
  const int block = 256;                                   // 8 warps -> covers 128 experts grid-strided
  const size_t smem = (size_t)HIDDEN * sizeof(float);
  cudaFuncSetAttribute(k4_router, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  k4_router<<<1, block, smem, stream>>>(h, w_post_norm, Wgate, Wgate_scale, sel_idx, sel_w);
}
#endif
