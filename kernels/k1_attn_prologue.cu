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
    const float4* xx4 = reinterpret_cast<const float4*>(xs + (v << 4));
    #pragma unroll
    for (int q = 0; q < 4; ++q) {                         // 4 x 32-bit words = 4 x (2 fp8 pairs)
      unsigned wq = wu[q];
      __nv_fp8x2_e4m3 lo, hi;
      lo.__x = (unsigned short)(wq & 0xffffu);
      hi.__x = (unsigned short)(wq >> 16);
      float2 fl = __half22float2((__half2)lo);
      float2 fh = __half22float2((__half2)hi);
      float4 xq = xx4[q];
      a0 += xq.x*fl.x;  a1 += xq.y*fl.y;
      a0 += xq.z*fh.x;  a1 += xq.w*fh.y;
    }
  }
  float acc = a0 + a1;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                             // valid on lane 0
}

// ---------------------------------------------------------------------------------------------
// cp.async pipeline primitives (sm_80+) — port of the proven k5_experts_v3.cu GEMV core, which
// reaches 58% MBU on this box.  16-byte cache-global async copy + commit/wait groups so the SM keeps
// many weight loads in flight per warp while the dequant+FMA of the previous tile runs.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void k1_cp_async_16(void* smem_dst, const void* gmem_src) {
#if __CUDA_ARCH__ >= 800
  unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_src));
#else
  *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
#endif
}
__device__ __forceinline__ void k1_cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N> __device__ __forceinline__ void k1_cp_async_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Dequant+FMA of one staged uint4 (16 fp8) into 4 fp32 accumulators (ILP; hardware fp8x2->half2).
// x is read as 128-bit float4 loads (4 per uint4) instead of 16 scalar loads -> 4x fewer smem
// transactions in the hot loop (x lives in shared, so this is a pure issue/LSU win).
__device__ __forceinline__ void k1_fma_uint4_fp8(const uint4& p, const float* __restrict__ yy,
                                                 float& a0, float& a1, float& a2, float& a3) {
  const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
  const float4* yy4 = reinterpret_cast<const float4*>(yy);
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    unsigned wq = wu[q];
    __nv_fp8x2_e4m3 lo, hi;
    lo.__x = (unsigned short)(wq & 0xffffu);
    hi.__x = (unsigned short)(wq >> 16);
    float2 fl = __half22float2((__half2)lo);
    float2 fh = __half22float2((__half2)hi);
#ifdef K1_SCALARX
    const float* xq = yy + (q << 2);
    a0 += xq[0]*fl.x;  a1 += xq[1]*fl.y;
    a2 += xq[2]*fh.x;  a3 += xq[3]*fh.y;
#else
    float4 xq = yy4[q];
    a0 += xq.x*fl.x;  a1 += xq.y*fl.y;
    a2 += xq.z*fh.x;  a3 += xq.w*fh.y;
#endif
  }
}
__device__ __forceinline__ float k1_warp_reduce(float acc) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;
}

// Warp computes ROWS dot-products <w_row, ys> over `n` (= HIDDEN) with a STAGES-deep cp.async ring.
// wbuf is this warp's [STAGES][ROWS][32] uint4 scratch in shared. out[r] gets lane-0's reduced dot.
template <int ROWS, int STAGES>
__device__ __forceinline__ void k1_warp_dot_rows_pipe(const fp8* __restrict__ W0, int n, int lane,
                                                      const float* __restrict__ ys,
                                                      uint4* __restrict__ wbuf, float out[ROWS]) {
  constexpr int TILE_V = 32;
  const int nv = n >> 4;
  const int ntile = (nv + TILE_V - 1) / TILE_V;
  const uint4* __restrict__ Wv[ROWS];
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) Wv[r] = reinterpret_cast<const uint4*>(W0 + (size_t)r * n);
  auto slot = [&](int st, int r) -> uint4* { return wbuf + ((size_t)st * ROWS + r) * TILE_V; };

  int fetch = 0;
  #pragma unroll 1
  for (; fetch < STAGES && fetch < ntile; ++fetch) {
    const int v = fetch * TILE_V + lane;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) if (v < nv) k1_cp_async_16(slot(fetch, r) + lane, Wv[r] + v);
    k1_cp_async_commit();
  }
  float acc[ROWS][4];
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) { acc[r][0]=acc[r][1]=acc[r][2]=acc[r][3]=0.f; }

  #pragma unroll 1
  for (int t = 0; t < ntile; ++t) {
    k1_cp_async_wait<STAGES - 1>();
    __syncwarp();
    const int st = t % STAGES;
    const int v  = t * TILE_V + lane;
    if (v < nv) {
      const float* yy = ys + (v << 4);
      #pragma unroll
      for (int r = 0; r < ROWS; ++r)
        k1_fma_uint4_fp8(*(slot(st, r) + lane), yy, acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    }
    const int nf = t + STAGES;
    if (nf < ntile) {
      const int fv = nf * TILE_V + lane;
      __syncwarp();
      #pragma unroll
      for (int r = 0; r < ROWS; ++r) if (fv < nv) k1_cp_async_16(slot(st, r) + lane, Wv[r] + fv);
    }
    k1_cp_async_commit();
  }
  k1_cp_async_wait<0>();
  __syncwarp();
  #pragma unroll
  for (int r = 0; r < ROWS; ++r)
    out[r] = k1_warp_reduce((acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]));
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
// Kernel A' — multi-row GEMV: each warp owns MR independent output rows and keeps all MR partial
// dots in flight at once.  The fp8 loads for the MR rows at a given uint4 phase are mutually
// independent (different HBM rows), so the load pipeline never drains between rows -> much higher
// memory-level parallelism per warp than warp-per-row (which issued ~8 loads then stalled on a
// reduction).  Each lane loads its uint4 of all MR rows, MACs into MR register accumulators; one
// fused reduce+scale+store per row at the very end.  This is the same MLP idea as k1k2_mbu_v2 but
// WITHOUT the cp.async/smem ring (which added overhead and lost): weights go straight reg<-HBM.
// ---------------------------------------------------------------------------------------------
template <int MR>
static __device__ __forceinline__ void k1_qkv_gemv_mr_body(
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale,
    const float* __restrict__ xs, float* __restrict__ proj, int lane, int gwarp, int nwarp) {
  const uint4* __restrict__ Wv = reinterpret_cast<const uint4*>(Wqkv);
  const int nv = HIDDEN >> 4;                              // 256 uint4 chunks per row
  const int row_groups = (QKV_OUT + MR - 1) / MR;
  for (int rg = gwarp; rg < row_groups; rg += nwarp) {
    const int o0 = rg * MR;
    float a0[MR], a1[MR];
    #pragma unroll
    for (int r = 0; r < MR; r++) { a0[r] = 0.f; a1[r] = 0.f; }
    for (int v = lane; v < nv; v += 32) {
      const float* xx = xs + (v << 4);
      uint4 wp[MR];
      #pragma unroll
      for (int r = 0; r < MR; r++)                         // issue MR independent loads first
        if (o0 + r < QKV_OUT) wp[r] = Wv[(size_t)(o0 + r) * nv + v];
      #pragma unroll
      for (int r = 0; r < MR; r++) {                       // then consume (loads overlap in flight)
        if (o0 + r >= QKV_OUT) continue;
        const unsigned* wu = reinterpret_cast<const unsigned*>(&wp[r]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          unsigned wq = wu[q];
          __nv_fp8x2_e4m3 lo, hi;
          lo.__x = (unsigned short)(wq & 0xffffu);
          hi.__x = (unsigned short)(wq >> 16);
          float2 fl = __half22float2((__half2)lo);
          float2 fh = __half22float2((__half2)hi);
          const float* xq = xx + (q << 2);
          a0[r] += xq[0]*fl.x;  a1[r] += xq[1]*fl.y;
          a0[r] += xq[2]*fh.x;  a1[r] += xq[3]*fh.y;
        }
      }
    }
    #pragma unroll
    for (int r = 0; r < MR; r++) {
      float acc = a0[r] + a1[r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
      if (lane == 0 && o0 + r < QKV_OUT) proj[o0 + r] = acc * Wqkv_scale[o0 + r];
    }
  }
}

// RMSNorm + stage x[] (shared with kernel A), then dispatch to the MR body.
template <int MR>
static __device__ __forceinline__ void k1_qkv_gemv_mr_impl(
    const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  extern __shared__ float k1_xs[];
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
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  k1_qkv_gemv_mr_body<MR>(Wqkv, Wqkv_scale, k1_xs, proj, lane, gwarp, nwarp);
}

extern "C" __global__ void k1_qkv_gemv_mr2(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_mr_impl<2>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_mr3(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_mr_impl<3>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_mr4(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_mr_impl<4>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_mr6(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_mr_impl<6>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_mr8(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_mr_impl<8>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}

// ---------------------------------------------------------------------------------------------
// Kernel A'' — cp.async double-buffered, ROWS-per-warp QKV GEMV (the K5v3 58%-MBU recipe ported).
// dynamic smem layout: [ float x[HIDDEN] ][ uint4 wbuf[warps_per_cta][STAGES][ROWS][32] ]
// ---------------------------------------------------------------------------------------------
template <int ROWS, int STAGES>
static __device__ __forceinline__ void k1_qkv_gemv_pipe_impl(
    const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  extern __shared__ float k1_xs[];                        // [HIDDEN] staged normed input
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int warps_per_cta = blockDim.x >> 5;

  // ---- input RMSNorm ----
  float part = 0.f;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) { float v = h[i]; part += v*v; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
  __shared__ float warp_ss[32];
  if (lane == 0) warp_ss[wid] = part;
  __syncthreads();
  __shared__ float rinv_sh;
  if (threadIdx.x == 0) {
    float ss = 0.f;
    for (int i = 0; i < warps_per_cta; i++) ss += warp_ss[i];
    rinv_sh = rsqrtf(ss / HIDDEN + RMS_EPS);
  }
  __syncthreads();
  const float rinv = rinv_sh;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) k1_xs[i] = h[i] * rinv * w_in_norm[i];
  __syncthreads();

  // per-warp cp.async ring, after x[HIDDEN] floats.
  uint4* wbuf = reinterpret_cast<uint4*>(k1_xs + HIDDEN) + (size_t)wid * STAGES * ROWS * 32;

  const int gwarp = blockIdx.x * warps_per_cta + wid;
  const int nwarp = gridDim.x * warps_per_cta;
  const int row_groups = (QKV_OUT + ROWS - 1) / ROWS;
  for (int rg = gwarp; rg < row_groups; rg += nwarp) {
    const int o0 = rg * ROWS;
    float out[ROWS];
    // last group may be short — point W0 at a safe row but mask the store.
    const int safe0 = (o0 + ROWS <= QKV_OUT) ? o0 : (QKV_OUT - ROWS);
    k1_warp_dot_rows_pipe<ROWS, STAGES>(Wqkv + (size_t)safe0 * HIDDEN, HIDDEN, lane, k1_xs, wbuf, out);
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      int o = safe0 + r;
      if (lane == 0 && o >= o0 && o < QKV_OUT) proj[o] = out[r] * Wqkv_scale[o];
    }
  }
}

extern "C" __global__ void k1_qkv_gemv_pipe_r4s2(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_pipe_impl<4, 2>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_pipe_r4s3(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_pipe_impl<4, 3>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_pipe_r2s3(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_pipe_impl<2, 3>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_pipe_r8s2(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_pipe_impl<8, 2>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_pipe_r6s2(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_pipe_impl<6, 2>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_pipe_r12s2(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_pipe_impl<12, 2>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_pipe_r16s2(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_pipe_impl<16, 2>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_pipe_r6s3(const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_pipe_impl<6, 3>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
}

// ---------------------------------------------------------------------------------------------
// Kernel A''' — GLOBAL-X recipe: decouple RMSNorm from the GEMV so the GEMV CTAs carry ZERO of the
// 16 KB x[HIDDEN] in shared memory.  In the fused-smem pipe (above) every GEMV CTA paid 16 KB of
// dynamic smem to stage x AND recomputed the whole input-RMSNorm; with block=384/512 that 16 KB is
// what caps occupancy (only ~2 CTAs/SM fit), so the HBM-latency-hiding warp pool is too shallow.
//
// Here a tiny one-CTA kernel computes x[i] = h[i]*rinv*w_in_norm[i] ONCE into a global scratch xg[].
// xg is 16 KB and read by every GEMV warp -> it lives entirely in L2 (50 MB on H100), so the
// re-reads are ~free.  The GEMV kernel's ONLY dynamic smem is the per-warp cp.async weight ring, so
// occupancy is bounded by registers/ring alone -> many more resident warps -> deeper load pipeline.
// Numerics are identical (same fp32 RMSNorm, same dot order).
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void k1_rmsnorm_x(const float* __restrict__ h,
                                        const float* __restrict__ w_in_norm,
                                        float* __restrict__ xg) {
  // single CTA; block-reduce sum-of-squares then write the normed input to global xg[HIDDEN].
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
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) xg[i] = h[i] * rinv * w_in_norm[i];
}

// GEMV reading the pre-normed x from global (L2-resident).  Dynamic smem = ONLY the per-warp ring.
template <int ROWS, int STAGES>
static __device__ __forceinline__ void k1_qkv_gemv_gx_impl(
    const float* __restrict__ xg, const fp8* __restrict__ Wqkv,
    const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  extern __shared__ uint4 k1_ring[];                      // [warps_per_cta][STAGES][ROWS][32] uint4
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int warps_per_cta = blockDim.x >> 5;
  uint4* wbuf = k1_ring + (size_t)wid * STAGES * ROWS * 32;

  const int gwarp = blockIdx.x * warps_per_cta + wid;
  const int nwarp = gridDim.x * warps_per_cta;
  const int row_groups = (QKV_OUT + ROWS - 1) / ROWS;
  for (int rg = gwarp; rg < row_groups; rg += nwarp) {
    const int o0 = rg * ROWS;
    float out[ROWS];
    const int safe0 = (o0 + ROWS <= QKV_OUT) ? o0 : (QKV_OUT - ROWS);
    k1_warp_dot_rows_pipe<ROWS, STAGES>(Wqkv + (size_t)safe0 * HIDDEN, HIDDEN, lane, xg, wbuf, out);
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      int o = safe0 + r;
      if (lane == 0 && o >= o0 && o < QKV_OUT) proj[o] = out[r] * Wqkv_scale[o];
    }
  }
}

extern "C" __global__ void k1_qkv_gemv_gx_r4s2(const float* __restrict__ xg,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_gx_impl<4, 2>(xg, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_gx_r6s2(const float* __restrict__ xg,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_gx_impl<6, 2>(xg, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_gx_r6s3(const float* __restrict__ xg,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_gx_impl<6, 3>(xg, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_gx_r4s3(const float* __restrict__ xg,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_gx_impl<4, 3>(xg, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_gx_r8s2(const float* __restrict__ xg,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_gx_impl<8, 2>(xg, Wqkv, Wqkv_scale, proj);
}
extern "C" __global__ void k1_qkv_gemv_gx_r8s3(const float* __restrict__ xg,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale, float* __restrict__ proj) {
  k1_qkv_gemv_gx_impl<8, 3>(xg, Wqkv, Wqkv_scale, proj);
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
#include <cstdlib>
static inline void k1_launch(
    const float* h, const float* w_in_norm, const fp8* Wqkv, const float* Wqkv_scale,
    const float* q_norm, const float* k_norm, const float* rope_cos, const float* rope_sin,
    float* out_q, fp8* kv_k, fp8* kv_v, const float* kv_k_scale, const float* kv_v_scale,
    cudaStream_t stream = 0) {
  // One-time scratch for the raw QKV projection (q | k | v), allocated outside any graph capture.
  static float* proj = nullptr;
  if (proj == nullptr) cudaMalloc(&proj, (size_t)QKV_OUT * sizeof(float));
  // One-time scratch for the pre-normed input x[HIDDEN] used by the GLOBAL-X GEMV path.
  static float* xg = nullptr;
  if (xg == nullptr) cudaMalloc(&xg, (size_t)HIDDEN * sizeof(float));

  // ---- GLOBAL-X path (K1_GLOBALX=1): tiny RMSNorm kernel -> xg, then a smem-lean cp.async GEMV ----
  // The GEMV CTAs carry NO x[] in shared, so occupancy is bounded only by the per-warp weight ring
  // -> deeper resident-warp pool -> better HBM-latency hiding.  Opt-in so the proven fused default is
  // untouched; selected config via K1_GXPIPE (62/42/63/43/82/83), K1_BLOCK, K1_CAP.
  static int gx = -1;
  if (gx < 0) { const char* eg = getenv("K1_GLOBALX"); gx = eg ? atoi(eg) : 0; }
  if (gx > 0) {
    static int gblock = 0, gcap = 0, gpipe = 0;
    if (gpipe == 0) {
      const char* eb = getenv("K1_BLOCK");  gblock = eb ? atoi(eb) : 512;
      const char* ec = getenv("K1_CAP");    gcap   = ec ? atoi(ec) : 100000;
      const char* ep = getenv("K1_GXPIPE"); gpipe  = ep ? atoi(ep) : 62;
      if (gblock <= 0) gblock = 512;
      if (gcap   <= 0) gcap   = 100000;
      if (gpipe  <= 0) gpipe  = 62;
    }
    const int   gwarps = gblock >> 5;
    const int   ROWS = gpipe / 10, STAGES = gpipe % 10;
    const size_t smemR = (size_t)gwarps * STAGES * ROWS * 32 * sizeof(uint4);
    int row_groups = (QKV_OUT + ROWS - 1) / ROWS;
    int needA = (row_groups + gwarps - 1) / gwarps;
    int ctasA = needA < gcap ? needA : gcap;
    auto setmaxg = [&](const void* f){ cudaFuncSetAttribute((const void*)f,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemR); };
    setmaxg((const void*)k1_qkv_gemv_gx_r4s2); setmaxg((const void*)k1_qkv_gemv_gx_r6s2);
    setmaxg((const void*)k1_qkv_gemv_gx_r6s3); setmaxg((const void*)k1_qkv_gemv_gx_r4s3);
    setmaxg((const void*)k1_qkv_gemv_gx_r8s2); setmaxg((const void*)k1_qkv_gemv_gx_r8s3);
    k1_rmsnorm_x<<<1, 1024, 0, stream>>>(h, w_in_norm, xg);
    #define K1_GX_LAUNCH(F) F<<<ctasA, gblock, smemR, stream>>>(xg, Wqkv, Wqkv_scale, proj)
    if      (gpipe == 42) K1_GX_LAUNCH(k1_qkv_gemv_gx_r4s2);
    else if (gpipe == 43) K1_GX_LAUNCH(k1_qkv_gemv_gx_r4s3);
    else if (gpipe == 63) K1_GX_LAUNCH(k1_qkv_gemv_gx_r6s3);
    else if (gpipe == 82) K1_GX_LAUNCH(k1_qkv_gemv_gx_r8s2);
    else if (gpipe == 83) K1_GX_LAUNCH(k1_qkv_gemv_gx_r8s3);
    else                  K1_GX_LAUNCH(k1_qkv_gemv_gx_r6s2);
    #undef K1_GX_LAUNCH
    k1_epilogue<<<3, 256, 0, stream>>>(
        proj, q_norm, k_norm, rope_cos, rope_sin, out_q, kv_k, kv_v, kv_k_scale, kv_v_scale);
    return;
  }

  // ---- Kernel A: RMSNorm + fused QKV GEMV (HBM-bandwidth-bound) ----
  // Tunable via env for the on-box sweep; defaults below are the winners.
  //   K1_PIPE  = cp.async pipeline config: 0 = simple warp-per-row(MR); else 42(R4S2) 43(R4S3)
  //              23(R2S3) 82(R8S2).  (Best on the box: 42 with block=512.)
  //   K1_MR    = rows/warp for the non-pipe path (1 = warp-per-row; 2..8 multi-row MLP).
  //   K1_BLOCK = threads/CTA;  K1_CAP = max CTAs.
  // WINNER on the box (H100, ctx4096, clean idle-trough min over 20 trials): PIPE=62 (R6 S2),
  // block=512, with the 128-bit float4 x-read in k1_fma_uint4_fp8 -> 22.30 us/token, 50.5% MBU
  // (vs scalar-x 22.59 us / 49.9% at the same config; vs the original warp-per-head 491 us / 2.3%).
  // The box is multi-tenant (bursty TP=8 across all 8 GPUs) so single runs are noisy — measure as the
  // MIN over many short runs (least-contended run == true kernel speed).  See /root/charles_results/k1_opt.txt.
  static int blockA = 0, capA = 0, mrA = 0, pipeA = -1;
  if (pipeA < 0) {
    const char* eb = getenv("K1_BLOCK"); blockA = eb ? atoi(eb) : 512;
    const char* ec = getenv("K1_CAP");   capA   = ec ? atoi(ec) : 264;
    const char* em = getenv("K1_MR");    mrA    = em ? atoi(em) : 1;
    const char* ep = getenv("K1_PIPE");  pipeA  = ep ? atoi(ep) : 62;
    if (blockA <= 0) blockA = 512;
    if (capA   <= 0) capA   = 264;
    if (mrA    <= 0) mrA    = 1;
    if (pipeA  <  0) pipeA  = 62;
  }
  const int   warpsA = blockA >> 5;
  const size_t smem_x = (size_t)HIDDEN * sizeof(float);

  if (pipeA > 0) {
    int ROWS   = pipeA / 10;
    int STAGES = pipeA % 10;
    size_t smemP = smem_x + (size_t)warpsA * STAGES * ROWS * 32 * sizeof(uint4);
    int row_groups = (QKV_OUT + ROWS - 1) / ROWS;
    int needA = (row_groups + warpsA - 1) / warpsA;
    int ctasA = needA < capA ? needA : capA;
    auto setmax = [&](const void* f){ cudaFuncSetAttribute((const void*)f,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemP); };
    setmax((const void*)k1_qkv_gemv_pipe_r4s2); setmax((const void*)k1_qkv_gemv_pipe_r4s3);
    setmax((const void*)k1_qkv_gemv_pipe_r2s3); setmax((const void*)k1_qkv_gemv_pipe_r8s2);
    setmax((const void*)k1_qkv_gemv_pipe_r6s2); setmax((const void*)k1_qkv_gemv_pipe_r12s2);
    setmax((const void*)k1_qkv_gemv_pipe_r16s2); setmax((const void*)k1_qkv_gemv_pipe_r6s3);
    #define K1_PIPE_LAUNCH(F) F<<<ctasA, blockA, smemP, stream>>>(h,w_in_norm,Wqkv,Wqkv_scale,proj)
    if      (pipeA == 42) K1_PIPE_LAUNCH(k1_qkv_gemv_pipe_r4s2);
    else if (pipeA == 43) K1_PIPE_LAUNCH(k1_qkv_gemv_pipe_r4s3);
    else if (pipeA == 23) K1_PIPE_LAUNCH(k1_qkv_gemv_pipe_r2s3);
    else if (pipeA == 62) K1_PIPE_LAUNCH(k1_qkv_gemv_pipe_r6s2);
    else if (pipeA == 63) K1_PIPE_LAUNCH(k1_qkv_gemv_pipe_r6s3);
    else if (pipeA == 122)K1_PIPE_LAUNCH(k1_qkv_gemv_pipe_r12s2);
    else if (pipeA == 162)K1_PIPE_LAUNCH(k1_qkv_gemv_pipe_r16s2);
    else                  K1_PIPE_LAUNCH(k1_qkv_gemv_pipe_r8s2);
    #undef K1_PIPE_LAUNCH
  } else {
    int row_groups = (QKV_OUT + mrA - 1) / mrA;
    int needA = (row_groups + warpsA - 1) / warpsA;
    int ctasA = needA < capA ? needA : capA;
    cudaFuncSetAttribute(k1_qkv_gemv,     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_x);
    cudaFuncSetAttribute(k1_qkv_gemv_mr2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_x);
    cudaFuncSetAttribute(k1_qkv_gemv_mr4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_x);
    switch (mrA) {
      case 1:  k1_qkv_gemv    <<<ctasA, blockA, smem_x, stream>>>(h, w_in_norm, Wqkv, Wqkv_scale, proj); break;
      case 2:  k1_qkv_gemv_mr2<<<ctasA, blockA, smem_x, stream>>>(h, w_in_norm, Wqkv, Wqkv_scale, proj); break;
      case 3:  k1_qkv_gemv_mr3<<<ctasA, blockA, smem_x, stream>>>(h, w_in_norm, Wqkv, Wqkv_scale, proj); break;
      case 6:  k1_qkv_gemv_mr6<<<ctasA, blockA, smem_x, stream>>>(h, w_in_norm, Wqkv, Wqkv_scale, proj); break;
      case 8:  k1_qkv_gemv_mr8<<<ctasA, blockA, smem_x, stream>>>(h, w_in_norm, Wqkv, Wqkv_scale, proj); break;
      default: k1_qkv_gemv_mr4<<<ctasA, blockA, smem_x, stream>>>(h, w_in_norm, Wqkv, Wqkv_scale, proj); break;
    }
  }

  // ---- Kernel B: cheap per-head QK-norm + RoPE + KV-cache write ----
  // 72 head rows; one small CTA of 128 warps' worth is overkill, so use 3 CTAs * 256 threads = 24
  // warps (covers 72 rows in 3 grid-stride steps) — negligible next to the GEMV.
  const int blockB = 256, ctasB = 3;                       // 24 warps -> 72 rows / 24 = 3 steps
  k1_epilogue<<<ctasB, blockB, 0, stream>>>(
      proj, q_norm, k_norm, rope_cos, rope_sin, out_q, kv_k, kv_v, kv_k_scale, kv_v_scale);
}
#endif
