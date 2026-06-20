// K1 — fused attention prologue (ONE kernel per layer, B=1 decode).
// Fuses, with NO HBM round-trip for the q/k/v intermediates:
//   input-RMSNorm(h, w_norm)
//     -> fused QKV GEMV  (W [9216,4096] fp8 e4m3, K-major, per-out-channel scale)
//     -> per-head QK-norm (RMSNorm over the 128-dim, fp32-accumulated) on q and k
//     -> RoPE(theta=1e6) on q,k
//     -> write k,v into the KV-cache slot for this position (fp8 or bf16).
//
// Layout / parallelism (matches the repo's warp-per-row + split-K idiom, k5_experts_warp.cu):
//   * The whole prologue is ONE CTA-cooperative kernel launch.  We stage the normed input x[HIDDEN]
//     in shared memory once (so the 9216-row GEMV reads it from smem, not HBM).
//   * WARP-PER-HEAD for the epilogue: a warp owns one head's 128 output channels, so the per-head
//     RMSNorm reduction (sum of squares over HEAD_DIM=128 = 4 elems/lane) is a warp-local shuffle.
//   * GEMV contraction (HIDDEN=4096) is split coalesced across the 32 lanes of the warp: consecutive
//     lanes read consecutive 16-byte (uint4 = 16xfp8) chunks of the SAME weight row -> coalesced HBM.
//
// One warp computes one of the 72 "head rows": 64 Q-heads + 4 K-heads + 4 V-heads, each 128 channels.
// Q/K heads get QK-norm + RoPE; V heads are written straight through (no norm/rope).  Grid-stride over
// the 72 head-rows lets a small grid fill the machine while keeping x resident per CTA.
//
// Build:  nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k1_attn_prologue.cu -o /tmp/k1
//         (the file is also #included by the shared *_bench.cu main below)
#include "common.cuh"
using namespace q3;

#ifndef Q3_K1_DEFS
#define Q3_K1_DEFS

// 72 "head rows" = 64 Q + 4 K + 4 V, each owning HEAD_DIM=128 contiguous output channels of Wqkv.
//   row  0..63 : Q head r,     out base = r*HEAD_DIM,                     channels q[r*128 .. ]
//   row 64..67 : K head r-64,  out base = Q_DIM + (r-64)*HEAD_DIM
//   row 68..71 : V head r-68,  out base = Q_DIM + KV_DIM + (r-68)*HEAD_DIM
constexpr int Q3_HEAD_ROWS = N_Q_HEADS + 2 * N_KV_HEADS;   // 72

// Coalesced split-K dot of one fp8 weight row w[0..n) with the staged x[0..n) (smem), across a warp.
// Dequant uses the hardware fp8x2->half2 path (8 vector converts per 128-bit load).  Returns the
// *unscaled* sum on every lane (warp-reduced); the caller multiplies by the per-channel scale.
static __device__ __forceinline__ float k1_warp_dot(const fp8* __restrict__ w,
                                                     const float* __restrict__ xs, int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;                                  // 16 fp8 per uint4
  for (int v = lane; v < nv; v += 32) {                   // consecutive lanes -> consecutive uint4
    uint4 p = wv[v];
    const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
    const float* xx = xs + (v << 4);
    #pragma unroll
    for (int q = 0; q < 4; q++) {
      unsigned wq = wu[q];
      __nv_fp8x2_e4m3 lo, hi;
      lo.__x = (unsigned short)(wq & 0xffffu);
      hi.__x = (unsigned short)(wq >> 16);
      float2 fl = __half22float2((__half2)lo), fh = __half22float2((__half2)hi);
      const float* xq = xx + (q << 2);
      a0 += xq[0]*fl.x; a1 += xq[1]*fl.y; a0 += xq[2]*fh.x; a1 += xq[3]*fh.y;
    }
  }
  float acc = a0 + a1;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  acc = __shfl_sync(0xffffffffu, acc, 0);                 // broadcast lane0 -> all lanes
  return acc;
}

#endif // Q3_K1_DEFS

// h:        [HIDDEN] residual-stream input (fp32)
// w_in_norm:[HIDDEN] input RMSNorm weights
// Wqkv:     fp8 [QKV_OUT, HIDDEN] (K-major / in contiguous), scale [QKV_OUT] per out-channel
// q_norm,k_norm: [HEAD_DIM] per-head QK-norm weights (shared across heads)
// rope_cos,rope_sin: [HEAD_DIM/2] cos/sin for THIS position (theta=1e6, "rotate-half" GPT-NeoX layout)
// out_q:    [Q_DIM] normed+roped query (kept resident in HBM for K2)
// kv_k/kv_v: KV-cache write slot for this token; [KV_DIM] each (caller offsets to pos*KV_DIM)
// kv_scale: per-channel quant scale for the fp8 KV cache write (length KV_DIM); pass nullptr for bf16 path.
extern "C" __global__ void k1_attn_prologue(
    const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ out_q, fp8* __restrict__ kv_k, fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale) {
  // ---- 1) input RMSNorm -> x[HIDDEN] staged in shared memory (block-wide, no HBM round-trip) ----
  extern __shared__ float xs[];                           // HIDDEN floats
  // block reduction of sum-of-squares of h.
  float part = 0.f;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) { float v = h[i]; part += v*v; }
  // warp reduce
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

  // ---- 2/3/4) warp-per-head: GEMV for this head's 128 channels, then QK-norm + RoPE + write ----
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

  for (int row = gwarp; row < Q3_HEAD_ROWS; row += nwarp) {
    // classify the head-row.
    int is_q = (row < N_Q_HEADS);
    int is_k = (!is_q && row < N_Q_HEADS + N_KV_HEADS);
    int out_base, head_local;
    if (is_q)      { head_local = row;                      out_base = head_local * HEAD_DIM; }
    else if (is_k) { head_local = row - N_Q_HEADS;          out_base = Q_DIM + head_local*HEAD_DIM; }
    else           { head_local = row - N_Q_HEADS - N_KV_HEADS; out_base = Q_DIM + KV_DIM + head_local*HEAD_DIM; }

    // GEMV: k1_warp_dot collaborates across the whole 32-lane warp to compute ONE output channel
    // (split-K coalesced over HIDDEN), broadcasting that channel's value to all lanes.  We iterate the
    // head's 128 channels and have each lane KEEP only the channels it owns: lane L owns {L,L+32,L+64,
    // L+96} in chan[0..3].  This keeps the per-head data warp-local for the QK-norm/RoPE epilogue.
    float chan[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) chan[c] = 0.f;
    for (int d = 0; d < HEAD_DIM; d++) {
      int o = out_base + d;
      float r = k1_warp_dot(Wqkv + (size_t)o * HIDDEN, xs, HIDDEN, lane) * Wqkv_scale[o];
      if (lane == (d & 31)) chan[d >> 5] = r;               // owning lane stores its channel
    }
    if (!is_q && !is_k) {
      // ---- V head: no norm / no rope; write straight to cache slot ----
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane;
        int slot = head_local * HEAD_DIM + d;               // index into [KV_DIM]
        float s = kv_v_scale ? kv_v_scale[slot] : 1.f;
        kv_v[slot] = fp8(chan[c] / s);                      // quantize: stored = val/scale
      }
      continue;
    }

    // ---- Q or K head: per-head RMSNorm over HEAD_DIM (fp32) ----
    float ss = 0.f;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) ss += chan[c] * chan[c];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    ss = __shfl_sync(0xffffffffu, ss, 0);                   // warp-local reduce + broadcast
    float hn = rsqrtf(ss / HEAD_DIM + RMS_EPS);
    const float* wn = is_q ? q_norm : k_norm;
    float normed[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) {
      int d = c * 32 + lane;
      normed[c] = chan[c] * hn * wn[d];
    }

    // ---- RoPE (theta=1e6, GPT-NeoX "rotate-half": pairs (i, i+HEAD_DIM/2)) ----
    // out[i]            = x[i]*cos[i] - x[i+H/2]*sin[i]
    // out[i+H/2]        = x[i+H/2]*cos[i] + x[i]*sin[i]
    // We need cross-lane partners. Build the head's 128 normed values in smem-free regs via shuffles:
    // lane L owns d in {L, L+32, L+64, L+96}. Partner of d is d ^ (HEAD_DIM/2)=d^64, which lives on the
    // SAME lane (since 64 = 2*32, flipping bit 6 keeps d&31). So partners are in this lane's own regs:
    //   c=0 (d=L)     partner d=L+64    -> c=2
    //   c=1 (d=L+32)  partner d=L+96    -> c=3
    // Thus pairs are (chan slot 0 <-> 2) and (1 <-> 3); fully register-local, no shuffle needed.
    float roped[HEAD_DIM / 32];
    {
      int half = HEAD_DIM / 2;                              // 64
      // slot c, channel d=c*32+lane. cos/sin indexed by min(d, d-half) in [0,half).
      // pair (0,2): d0=lane (<half since lane<32<64), d2=lane+64 (>=half). rope index = lane.
      // pair (1,3): d1=lane+32 (<half), d3=lane+96 (>=half). rope index = lane+32.
      float c0 = rope_cos[lane],     s0 = rope_sin[lane];
      float c1 = rope_cos[lane + 32], s1 = rope_sin[lane + 32];
      roped[0] = normed[0]*c0 - normed[2]*s0;               // d=lane  (lower half)
      roped[2] = normed[2]*c0 + normed[0]*s0;               // d=lane+64 (upper partner)
      roped[1] = normed[1]*c1 - normed[3]*s1;               // d=lane+32 (lower half)
      roped[3] = normed[3]*c1 + normed[1]*s1;               // d=lane+96 (upper partner)
      (void)half;
    }

    // ---- write out ----
    if (is_q) {
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane;
        out_q[head_local * HEAD_DIM + d] = roped[c];
      }
    } else { // K head -> quantize into cache slot
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane;
        int slot = head_local * HEAD_DIM + d;               // index into [KV_DIM]
        float s = kv_k_scale ? kv_k_scale[slot] : 1.f;
        kv_k[slot] = fp8(roped[c] / s);
      }
    }
  }
}

// Launch helper: stage x[HIDDEN] in dynamic smem; pick a grid that covers 72 head-rows with spare warps
// so a CTA's resident x is reused.  One CTA of 256 threads = 8 warps; 9 CTAs = 72 warps = 1 warp/head-row.
#ifdef Q3_K1_LAUNCH_HELPER
static inline void k1_launch(
    const float* h, const float* w_in_norm, const fp8* Wqkv, const float* Wqkv_scale,
    const float* q_norm, const float* k_norm, const float* rope_cos, const float* rope_sin,
    float* out_q, fp8* kv_k, fp8* kv_v, const float* kv_k_scale, const float* kv_v_scale,
    cudaStream_t stream = 0) {
  const int block = 256;                          // 8 warps/CTA
  const int ctas  = (Q3_HEAD_ROWS + 7) / 8;       // 9 CTAs -> 72 warps -> warp-per-head-row
  const size_t smem = (size_t)HIDDEN * sizeof(float);
  cudaFuncSetAttribute(k1_attn_prologue, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  k1_attn_prologue<<<ctas, block, smem, stream>>>(
      h, w_in_norm, Wqkv, Wqkv_scale, q_norm, k_norm, rope_cos, rope_sin,
      out_q, kv_k, kv_v, kv_k_scale, kv_v_scale);
}
#endif
