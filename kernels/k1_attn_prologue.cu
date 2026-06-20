// K1 — attention prologue for Qwen3-235B-A22B, B=1 decode (sm_90a / H100).
//
// THE BOTTLENECK (and the fix):
//   The QKV projection at B=1 is a GEMV against W[QKV_OUT=9216, HIDDEN=4096] fp8 e4m3 (~38 MB).
//   It is purely HBM-bandwidth-bound: the whole game is to read those 38 MB at near-peak HBM
//   bandwidth and never touch HBM more than once.  The previous version used a "warp-per-HEAD"
//   layout that ran 128 *sequential* warp-collaborative dots per head (each dot only had the warp's
//   32 lanes striding HIDDEN), which left the machine massively under-occupied -> 77 GB/s (2.3%).
//
//   This rewrite splits the prologue into two kernels and uses the EXACT fast idiom from
//   k5_experts.cu's `warp_dot_fp8` for the GEMV:
//
//   Kernel A  (k1_qkv_gemv): input-RMSNorm(h, w_in_norm) -> stage normed x[HIDDEN] in shared mem ->
//             fused QKV GEMV.  ONE WARP PER OUTPUT ROW o in [0, 9216): the warp's 32 lanes read
//             consecutive uint4 (16xfp8) chunks of the SAME weight row -> fully coalesced 128-bit
//             HBM loads; hardware fp8x2->half2 dequant; 2 FP accumulators for ILP.  Grid-strides over
//             all 9216 rows with thousands of resident warps to fill the 132 SMs and hide latency.
//             The per-out-channel scale is folded once onto the reduced dot.  Writes the raw
//             projection proj[9216] (q | k | v) to a small scratch buffer in HBM.
//
//   Kernel B  (k1_epilogue): the cheap part (~8704 elems, basically free).  Per-head QK-norm
//             (RMSNorm over HEAD_DIM=128, fp32) + RoPE (theta=1e6, GPT-NeoX "rotate-half") on the q
//             and k heads; v heads pass straight through.  Writes out_q[Q_DIM], and the quantized
//             fp8 k/v cache slots.  WARP-PER-HEAD here is cheap and keeps each head's 128 values
//             warp-local (the per-head reduction is a single warp shuffle).  This fusion is what
//             killed bandwidth before, so it is now decoupled from the big GEMV.
//
// The public entry point (`k1_attn_prologue`) and launch helper (`k1_launch`) keep their original
// names and argument order so decode_step.cu and k12_bench.cu call them unchanged.  Numerics match
// the CPU fp32 reference in k12_bench.cu to < 1e-2 (the only kernel-vs-reference delta is fp32
// accumulation order on the dot; the fp8 weight/cache round-trip is byte-identical on both sides).
//
// Build:  nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k1_attn_prologue.cu -o /tmp/k1
//         (this file is also #included by k12_bench.cu / decode_step.cu as a kernel library)
#include "common.cuh"
using namespace q3;

#ifndef Q3_K1_DEFS
#define Q3_K1_DEFS

// 72 "head rows" for the epilogue = 64 Q + 4 K + 4 V, each owning HEAD_DIM=128 contiguous channels.
//   row  0..63 : Q head r,     proj base = r*HEAD_DIM                       (-> out_q[r*128..])
//   row 64..67 : K head r-64,  proj base = Q_DIM + (r-64)*HEAD_DIM          (-> kv_k slot)
//   row 68..71 : V head r-68,  proj base = Q_DIM + KV_DIM + (r-68)*HEAD_DIM (-> kv_v slot)
constexpr int Q3_HEAD_ROWS = N_Q_HEADS + 2 * N_KV_HEADS;   // 72

// ---------------------------------------------------------------------------------------------
// Coalesced split-K dot of one fp8 weight row w[0..n) against the staged x[0..n) (shared mem),
// collaborating across a whole 32-lane warp.  Identical fast idiom to k5_experts.cu warp_dot_fp8:
// consecutive lanes load consecutive uint4 (16 fp8) chunks of the SAME row -> coalesced 128-bit HBM
// loads; hardware fp8x2->half2 dequant (8 vector converts per 128-bit load); two FP accumulators for
// ILP.  n must be a multiple of 16 (HIDDEN=4096 is).  Returns the *unscaled* sum, valid on lane 0.
// (Uniquely named so this file composes into the single-TU decode_step.cu alongside k5's copy.)
static __device__ __forceinline__ float k1_warp_dot(const fp8* __restrict__ w,
                                                     const float* __restrict__ xs,
                                                     int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;                                  // 16 fp8 per uint4
  for (int v = lane; v < nv; v += 32) {                   // lanes 0..31 -> consecutive uint4
    uint4 p = wv[v];
    const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
    const float* xx = xs + (v << 4);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {                         // 4 x 32-bit words = 4 x (2 fp8 pairs)
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
  return acc;                                             // valid on lane 0
}

#endif // Q3_K1_DEFS

// ---------------------------------------------------------------------------------------------
// Kernel A — input RMSNorm + fused QKV GEMV (the HBM-bandwidth-bound part).
// ---------------------------------------------------------------------------------------------
// 1) Block-reduce sum-of-squares of h[HIDDEN], compute rms_inv, and stage the normed input
//      x[i] = h[i] * rms_inv * w_in_norm[i]
//    into shared memory once per CTA (so the 9216-row GEMV reads x from smem, not HBM).
// 2) One warp per output channel o in [0, QKV_OUT); grid-stride over all 9216 rows so a small grid
//    of many resident warps fills all 132 SMs.  Each warp coalesced-dots W[o, :] against x and folds
//    the per-channel scale, writing the raw projection proj[o].
// Launch with dynamic smem = HIDDEN*sizeof(float).
extern "C" __global__ void k1_qkv_gemv(
    const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale,
    float* __restrict__ proj) {
  extern __shared__ float k1_xs[];                        // [HIDDEN] staged normed input

  // ---- input RMSNorm (block reduction of sum-of-squares) ----
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
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) k1_xs[i] = h[i] * rinv * w_in_norm[i];
  __syncthreads();

  // ---- fused QKV GEMV: warp-per-output-row, grid-stride over all 9216 rows ----
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int o = gwarp; o < QKV_OUT; o += nwarp) {
    float r = k1_warp_dot(Wqkv + (size_t)o * HIDDEN, k1_xs, HIDDEN, lane);
    if (lane == 0) proj[o] = r * Wqkv_scale[o];
  }
}

// ---------------------------------------------------------------------------------------------
// Kernel B — cheap epilogue: per-head QK-norm + RoPE -> out_q, and KV-cache write.
// ---------------------------------------------------------------------------------------------
// One warp per "head row" (72 total); a small launch covers them all.  The warp loads its head's 128
// projection values from proj (lane L owns d in {L, L+32, L+64, L+96}), does:
//   * Q / K heads: RMSNorm over HEAD_DIM (warp-shuffle reduce) * per-head norm weight, then RoPE.
//     Q -> out_q[Q_DIM];  K -> quantized into kv_k slot.
//   * V heads: straight through, quantized into kv_v slot (no norm / no rope).
// This touches only ~8704 elements -> effectively free next to the 38 MB GEMV.
extern "C" __global__ void k1_epilogue(
    const float* __restrict__ proj,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ out_q, fp8* __restrict__ kv_k, fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale) {
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;

  for (int row = gwarp; row < Q3_HEAD_ROWS; row += nwarp) {
    const int is_q = (row < N_Q_HEADS);
    const int is_k = (!is_q && row < N_Q_HEADS + N_KV_HEADS);
    int proj_base, head_local;
    if (is_q)      { head_local = row;                              proj_base = head_local * HEAD_DIM; }
    else if (is_k) { head_local = row - N_Q_HEADS;                  proj_base = Q_DIM + head_local*HEAD_DIM; }
    else           { head_local = row - N_Q_HEADS - N_KV_HEADS;     proj_base = Q_DIM + KV_DIM + head_local*HEAD_DIM; }

    // Load this head's 128 channels: lane L owns chan[c] = proj[base + c*32 + L], c in [0,4).
    float chan[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) chan[c] = proj[proj_base + c * 32 + lane];

    if (!is_q && !is_k) {
      // ---- V head: no norm / no rope; quantize straight into the cache slot ----
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane;
        int slot = head_local * HEAD_DIM + d;             // index into [KV_DIM]
        float s = kv_v_scale ? kv_v_scale[slot] : 1.f;
        kv_v[slot] = fp8(chan[c] / s);                    // quantize: stored = val/scale
      }
      continue;
    }

    // ---- Q or K head: per-head RMSNorm over HEAD_DIM (fp32, warp-local reduce) ----
    float ss = 0.f;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) ss += chan[c] * chan[c];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    ss = __shfl_sync(0xffffffffu, ss, 0);                 // broadcast lane0 -> all lanes
    float hn = rsqrtf(ss / HEAD_DIM + RMS_EPS);
    const float* wn = is_q ? q_norm : k_norm;
    float normed[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) normed[c] = chan[c] * hn * wn[c * 32 + lane];

    // ---- RoPE (theta=1e6, GPT-NeoX "rotate-half": pairs (i, i+HEAD_DIM/2)) ----
    //   out[i]      = x[i]*cos[i]     - x[i+H/2]*sin[i]
    //   out[i+H/2]  = x[i+H/2]*cos[i] + x[i]*sin[i]
    // lane L owns d in {L, L+32, L+64, L+96}; partner of d is d^64, which lives on the SAME lane
    // (64 = 2*32 flips bit 6, keeps d&31).  So pairs are register-local: (slot 0 <-> 2), (1 <-> 3).
    //   pair (0,2): d0=lane (<64 lower half), d2=lane+64 (upper) -> cos/sin index = lane.
    //   pair (1,3): d1=lane+32 (<64 lower),   d3=lane+96 (upper) -> cos/sin index = lane+32.
    float roped[HEAD_DIM / 32];
    {
      float c0 = rope_cos[lane],      s0 = rope_sin[lane];
      float c1 = rope_cos[lane + 32], s1 = rope_sin[lane + 32];
      roped[0] = normed[0]*c0 - normed[2]*s0;             // d=lane     (lower half)
      roped[2] = normed[2]*c0 + normed[0]*s0;             // d=lane+64  (upper partner)
      roped[1] = normed[1]*c1 - normed[3]*s1;             // d=lane+32  (lower half)
      roped[3] = normed[3]*c1 + normed[1]*s1;             // d=lane+96  (upper partner)
    }

    // ---- write out ----
    if (is_q) {
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++)
        out_q[head_local * HEAD_DIM + c * 32 + lane] = roped[c];
    } else { // K head -> quantize into cache slot
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane;
        int slot = head_local * HEAD_DIM + d;             // index into [KV_DIM]
        float s = kv_k_scale ? kv_k_scale[slot] : 1.f;
        kv_k[slot] = fp8(roped[c] / s);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Back-compat single-kernel entry point (correctness baseline; preserves the original signature).
// ---------------------------------------------------------------------------------------------
// The production path is the two-kernel split above (k1_launch chains k1_qkv_gemv then k1_epilogue),
// which is what makes the GEMV hit near-peak HBM bandwidth.  This single-kernel version is kept as a
// simple, self-contained reference with the ORIGINAL public name/signature: it does the same fused
// prologue in one launch (warp-per-head, sequential per-channel dots) — bandwidth-suboptimal, but
// numerically identical.  Launch with dynamic smem = HIDDEN*sizeof(float).
extern "C" __global__ void k1_attn_prologue(
    const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ out_q, fp8* __restrict__ kv_k, fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale) {
  extern __shared__ float xs[];                           // [HIDDEN]
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
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) xs[i] = h[i] * rinv * w_in_norm[i];
  __syncthreads();

  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  for (int row = gwarp; row < Q3_HEAD_ROWS; row += nwarp) {
    int is_q = (row < N_Q_HEADS);
    int is_k = (!is_q && row < N_Q_HEADS + N_KV_HEADS);
    int out_base, head_local;
    if (is_q)      { head_local = row;                          out_base = head_local * HEAD_DIM; }
    else if (is_k) { head_local = row - N_Q_HEADS;              out_base = Q_DIM + head_local*HEAD_DIM; }
    else           { head_local = row - N_Q_HEADS - N_KV_HEADS; out_base = Q_DIM + KV_DIM + head_local*HEAD_DIM; }

    float chan[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) chan[c] = 0.f;
    for (int d = 0; d < HEAD_DIM; d++) {
      int o = out_base + d;
      float r = k1_warp_dot(Wqkv + (size_t)o * HIDDEN, xs, HIDDEN, lane);
      r = __shfl_sync(0xffffffffu, r, 0) * Wqkv_scale[o];     // broadcast lane0 -> all lanes
      if (lane == (d & 31)) chan[d >> 5] = r;
    }
    if (!is_q && !is_k) {
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane;
        int slot = head_local * HEAD_DIM + d;
        float s = kv_v_scale ? kv_v_scale[slot] : 1.f;
        kv_v[slot] = fp8(chan[c] / s);
      }
      continue;
    }
    float ss = 0.f;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) ss += chan[c] * chan[c];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    ss = __shfl_sync(0xffffffffu, ss, 0);
    float hn = rsqrtf(ss / HEAD_DIM + RMS_EPS);
    const float* wn = is_q ? q_norm : k_norm;
    float normed[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) normed[c] = chan[c] * hn * wn[c * 32 + lane];
    float roped[HEAD_DIM / 32];
    {
      float c0 = rope_cos[lane],      s0 = rope_sin[lane];
      float c1 = rope_cos[lane + 32], s1 = rope_sin[lane + 32];
      roped[0] = normed[0]*c0 - normed[2]*s0;
      roped[2] = normed[2]*c0 + normed[0]*s0;
      roped[1] = normed[1]*c1 - normed[3]*s1;
      roped[3] = normed[3]*c1 + normed[1]*s1;
    }
    if (is_q) {
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++)
        out_q[head_local * HEAD_DIM + c * 32 + lane] = roped[c];
    } else {
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane;
        int slot = head_local * HEAD_DIM + d;
        float s = kv_k_scale ? kv_k_scale[slot] : 1.f;
        kv_k[slot] = fp8(roped[c] / s);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Launch helper — chains the fast GEMV (kernel A) then the cheap epilogue (kernel B).
// ---------------------------------------------------------------------------------------------
// Same public name/signature/arg-order as before; decode_step.cu and k12_bench.cu call it unchanged.
//
// Kernel A is launched with a grid that lightly oversubscribes the 132 SMs with resident warps so the
// 9216-row GEMV hides HBM latency (this is where the bandwidth is won).  Kernel B is tiny (72 head
// rows) and runs in a single small CTA.  Both kernels run back-to-back on the SAME stream, so the
// shared `proj` scratch is hazard-free (kernel B reads only after kernel A completes), and chaining
// many K1 calls (one per layer in decode_step.cu) reuses the same scratch safely.
//
// The `proj[QKV_OUT]` scratch is allocated ONCE, lazily, on the first call (which decode_step.cu
// performs as a warm-up OUTSIDE its CUDA-graph capture region), and reused thereafter — so nothing
// allocates during graph capture (cudaMalloc is not stream-capturable).
#ifdef Q3_K1_LAUNCH_HELPER
#include <cuda_runtime.h>
static inline void k1_launch(
    const float* h, const float* w_in_norm, const fp8* Wqkv, const float* Wqkv_scale,
    const float* q_norm, const float* k_norm, const float* rope_cos, const float* rope_sin,
    float* out_q, fp8* kv_k, fp8* kv_v, const float* kv_k_scale, const float* kv_v_scale,
    cudaStream_t stream = 0) {
  // One-time scratch for the raw QKV projection (q | k | v), allocated outside any graph capture.
  static float* proj = nullptr;
  if (proj == nullptr) cudaMalloc(&proj, (size_t)QKV_OUT * sizeof(float));

  // ---- Kernel A: RMSNorm + fused QKV GEMV (HBM-bandwidth-bound) ----
  const int   blockA = 256;                                // 8 warps / CTA
  const int   warpsA = blockA >> 5;
  // Cover all 9216 output rows; oversubscribe the 132 SMs with resident warps (cap ~2 CTAs/SM = 264)
  // to hide HBM latency, but never launch more CTAs than there is work for.
  int needA = (QKV_OUT + warpsA - 1) / warpsA;             // CTAs for 1 warp per row (= 1152)
  int ctasA = needA < 264 ? needA : 264;                   // -> 264 CTAs * 8 warps = 2112 warps
  const size_t smemA = (size_t)HIDDEN * sizeof(float);
  cudaFuncSetAttribute(k1_qkv_gemv, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA);
  k1_qkv_gemv<<<ctasA, blockA, smemA, stream>>>(h, w_in_norm, Wqkv, Wqkv_scale, proj);

  // ---- Kernel B: cheap per-head QK-norm + RoPE + KV-cache write ----
  // 72 head rows; one small CTA of 128 warps' worth is overkill, so use 3 CTAs * 256 threads = 24
  // warps (covers 72 rows in 3 grid-stride steps) — negligible next to the GEMV.
  const int blockB = 256, ctasB = 3;                       // 24 warps -> 72 rows / 24 = 3 steps
  k1_epilogue<<<ctasB, blockB, 0, stream>>>(
      proj, q_norm, k_norm, rope_cos, rope_sin, out_q, kv_k, kv_v, kv_k_scale, kv_v_scale);
}
#endif
