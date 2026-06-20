// decode_step.cu — fused, CUDA-graph-captured single-token DECODE STEP for Qwen3-235B-A22B.
// Target: sm_90a / H100.  Standard CUDA only.
//
// THE POINT
// ---------
// At batch size 1 a single decode token issues a long chain of tiny GEMV / flash-decode kernels —
// 7 launches per transformer layer x 94 layers + a final norm/lm_head/argmax = ~660 kernel
// launches per token.  Each launch costs a few microseconds of CPU->GPU dispatch latency, and at
// B=1 the kernels themselves are HBM-bandwidth-bound GEMVs that finish in microseconds, so the
// launch overhead is a *first-order* cost — the model runs at only ~11% of the HBM roofline,
// launch/overhead-bound rather than bandwidth-bound.
//
// The fix is a CUDA GRAPH: we record the entire 94-layer step ONCE into a cudaGraph, instantiate
// it, and then replay the whole thing with a single cudaGraphLaunch.  The driver pre-resolves every
// node's launch parameters, so per-token CPU dispatch overhead collapses to one launch instead of
// ~660.  This file builds that graph from the repo's existing per-layer kernels and benchmarks
// graph-replay vs. the identical step issued as individual launches, to expose the overhead delta.
//
// WHAT THIS BUILDS (one full decode layer, chained on a stream):
//   K1  k1_attn_prologue        RMSNorm + fused QKV GEMV + QK-norm + RoPE + KV-cache write
//   K2a k2_flash_decode_partial split-KV GQA flash-decode, pass 1 (online softmax partials)
//   K2b k2_flash_decode_reduce  split-KV reduce, pass 2 (log-sum-exp combine -> attn_out)
//   K3  k3_attn_epilogue        O-proj GEMV (Wo @ attn_out) + fused residual add
//   K4  k4_router               post-RMSNorm + gate GEMV + softmax + top-8 + renorm (sel_idx/sel_w)
//   K5a k5a_gateup              fused fp8 MoE gate+up SwiGLU  (a = silu(gate(y)) * up(y))
//   K5b k5b_down                fp8 MoE down-proj + routed residual accumulate
//   ... repeated for all N_LAYERS=94 ...
//   final: RMSNorm + lm_head GEMV (Wlm [VOCAB, HIDDEN]) + on-device argmax -> next token id.
//
// LATENCY-PROXY DISCLAIMER (important, read this):
//   This is a *latency/launch-overhead proxy*, NOT a numerically faithful forward pass.  To keep
//   the resident memory modest we allocate ONE layer's worth of dummy fp8 weights and REUSE the
//   same buffers for all 94 layers.  The kernel CHAIN, launch COUNT, grid/block shapes, dynamic
//   smem, and per-token HBM read VOLUME are identical to the real model, so the measured us/token
//   and the graph-vs-eager launch-overhead delta are representative.  The produced logits/token id
//   are meaningless (same weights every layer).  To make it the real model you would point each
//   layer's pointers at that layer's distinct weights — the host enqueue code is unchanged.
//
// Build (must compile):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/decode_step.cu -o /tmp/ds
//
// =================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

// ---- Pull in the existing compiled kernels as a single translation unit. -------------------------
// K5 owns the common host machinery (#include <cuda_runtime.h>, the fp8 dot primitive, the k5_plan
// launcher); guard out its main().  K1/K2 expose CTA-cooperative / split-KV launch helpers.  K3/K4
// are the fleshed-out epilogue + router.  Each kernel's device helper has a unique name
// (warp_dot_fp8 / k1_warp_dot / k2_warp_sum / k3_warp_dot / k4_warp_dot) and each extern __shared__
// array a distinct name, so they compose cleanly into ONE TU (nvcc keys dynamic smem per-kernel).
#define K5_NO_MAIN
#define Q3_K1_LAUNCH_HELPER
#define Q3_K2_LAUNCH_HELPER
#define Q3_K3_LAUNCH_HELPER
#define Q3_K4_LAUNCH_HELPER
#include "k5_experts.cu"        // k5a_gateup, k5b_down, warp_dot_fp8, k5_plan, CK-free host helpers
#include "k1_attn_prologue.cu"  // k1_attn_prologue + k1_launch
#include "k2_flash_decode.cu"   // k2_flash_decode_partial/_reduce + k2_launch + k2_pick_splits
#include "k3_attn_epilogue.cu"  // k3_attn_epilogue + k3_launch
#include "k4_router.cu"         // k4_router + k4_launch
#include "k6_graph_capture.cu"  // K6: build_decode_graph / replay_decode_step / destroy_decode_graph

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                       \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));           \
  exit(1); } } while (0)

// =================================================================================================
// Final head: RMSNorm + lm_head GEMV (Wlm [VOCAB, HIDDEN] fp8) + on-device argmax.
// =================================================================================================
//
// lm_head is the single biggest matrix touched per token: VOCAB(151936) x HIDDEN(4096) fp8 = ~622 MB.
// We compute it as the same warp-per-output-row coalesced fp8 GEMV (one warp dots one vocab row
// against the staged, final-normed hidden), then a 2-stage on-device argmax (per-block max into a
// scratch buffer, then a final reduce to one token id) — all on the stream, no host sync.

#ifndef Q3_DS_DEFS
#define Q3_DS_DEFS
// Coalesced split-K fp8 dot, identical idiom to the other kernels (unique name for this TU).
static __device__ __forceinline__ float ds_warp_dot(const fp8* __restrict__ w,
                                                     const float* __restrict__ xs,
                                                     int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;
  for (int v = lane; v < nv; v += 32) {
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
  return acc;
}
#endif // Q3_DS_DEFS

// Final RMSNorm of the residual stream into a staged normed vector in global memory.
// One CTA; block-reduce sum-of-squares, then scale.  hn_out[HIDDEN] feeds the lm_head GEMV.
extern "C" __global__ void ds_final_norm(const float* __restrict__ h,
                                         const float* __restrict__ w_final_norm,
                                         float* __restrict__ hn_out) {
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
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) hn_out[i] = h[i] * rinv * w_final_norm[i];
}

// lm_head GEMV + per-block partial argmax.  Warp-per-vocab-row, split-K coalesced over HIDDEN.
// Each warp computes one vocab logit; lane 0 reduces a CTA-local (max_logit, argmax) into the
// block_max/block_arg scratch (one entry per CTA).  hn is read from global (VOCAB rows reuse it).
extern "C" __global__ void ds_lmhead_argmax_partial(
    const float* __restrict__ hn,
    const fp8*  __restrict__ Wlm, const float* __restrict__ Wlm_scale,
    float* __restrict__ block_max, int* __restrict__ block_arg) {
  // stage hn[HIDDEN] in shared memory once per CTA.
  extern __shared__ float hs[];                            // [HIDDEN]
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) hs[k] = hn[k];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int wid   = threadIdx.x >> 5;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int nwc   = blockDim.x >> 5;                       // warps per CTA

  float my_max = -3.0e38f; int my_arg = -1;
  for (int row = gwarp; row < VOCAB; row += nwarp) {
    float v = ds_warp_dot(Wlm + (size_t)row * HIDDEN, hs, HIDDEN, lane);
    if (lane == 0) { v *= Wlm_scale[row]; if (v > my_max) { my_max = v; my_arg = row; } }
  }
  // CTA-local reduction over warps (lane 0 of each warp holds the warp's best).
  __shared__ float smax[32];
  __shared__ int   sarg[32];
  if (lane == 0) { smax[wid] = my_max; sarg[wid] = my_arg; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float bm = -3.0e38f; int ba = -1;
    for (int w = 0; w < nwc; ++w) if (smax[w] > bm) { bm = smax[w]; ba = sarg[w]; }
    block_max[blockIdx.x] = bm;
    block_arg[blockIdx.x] = ba;
  }
}

// Final argmax reduce over the per-block partials -> a single token id in tok_out[0].
extern "C" __global__ void ds_argmax_final(const float* __restrict__ block_max,
                                           const int* __restrict__ block_arg,
                                           int nblocks, int* __restrict__ tok_out) {
  if (threadIdx.x != 0) return;                            // tiny final reduce, single thread
  float bm = -3.0e38f; int ba = -1;
  for (int b = 0; b < nblocks; ++b) if (block_max[b] > bm) { bm = block_max[b]; ba = block_arg[b]; }
  tok_out[0] = ba;
}

// =================================================================================================
// Device-side state for one decode step (one layer's reused dummy weights — see proxy disclaimer).
// =================================================================================================
struct DecodeState {
  // residual stream (ping-pong so K3 can read h_in and write h_out without a hazard).
  float *h_a = nullptr, *h_b = nullptr;     // [HIDDEN]
  // K1 weights / params (one layer, reused).
  float *w_in_norm = nullptr;               // [HIDDEN]
  fp8   *Wqkv = nullptr;  float *Wqkv_scale = nullptr;   // [QKV_OUT, HIDDEN], [QKV_OUT]
  float *q_norm = nullptr, *k_norm = nullptr;            // [HEAD_DIM]
  float *rope_cos = nullptr, *rope_sin = nullptr;        // [HEAD_DIM/2]
  float *out_q = nullptr;                   // [Q_DIM]
  // KV cache (one big context buffer; K1 writes the current slot, K2 reads [0,ctx_len)).
  fp8   *kv_k = nullptr, *kv_v = nullptr;   // [ctx_len, KV_DIM]
  float *kv_k_scale = nullptr, *kv_v_scale = nullptr;    // [KV_DIM]
  int    ctx_len = 0, n_splits = 0;
  // K2 partials.
  float *part_m = nullptr, *part_l = nullptr, *part_acc = nullptr;
  float *attn_out = nullptr;                // [Q_DIM]
  // K3 (O-proj).
  fp8   *Wo = nullptr;  float *Wo_scale = nullptr;       // [HIDDEN, Q_DIM], [HIDDEN]
  // K4 (router).
  float *w_post_norm = nullptr;             // [HIDDEN]
  fp8   *Wgate = nullptr; float *Wgate_scale = nullptr;  // [N_EXPERTS, HIDDEN], [N_EXPERTS]
  int   *sel_idx = nullptr;  float *sel_w = nullptr;     // [TOP_K]
  // K5 (experts) — arrays of per-expert pointers (TOP_K active).
  const fp8   **Wgu_d = nullptr;  const float **Wgu_scale_d = nullptr;
  const fp8   **Wd_d  = nullptr;  const float **Wd_scale_d  = nullptr;
  float *a_glb = nullptr;                   // [TOP_K * MOE_INTER]
  // final head.
  float *w_final_norm = nullptr;            // [HIDDEN]
  float *hn = nullptr;                      // [HIDDEN]
  fp8   *Wlm = nullptr;  float *Wlm_scale = nullptr;     // [VOCAB, HIDDEN], [VOCAB]
  float *block_max = nullptr;  int *block_arg = nullptr; // [lm_blocks]
  int   *tok_out = nullptr;                 // [1]
  int    lm_blocks = 0;
  // cached launch plan for K5.
  K5Launch k5;
};

// ---- enqueue ONE decode layer (7 kernels) onto a stream -----------------------------------------
// Ping-pong: reads from h_src, writes the post-attn residual into h_dst, then K4/K5 update h_dst
// in place.  Returns the residual buffer holding this layer's output (h_dst).
static float* enqueue_layer(DecodeState& S, float* h_src, float* h_dst, cudaStream_t s) {
  // K1: RMSNorm + QKV GEMV + QK-norm + RoPE + KV write.  Reads h_src, writes out_q + KV slot.
  k1_launch(h_src, S.w_in_norm, S.Wqkv, S.Wqkv_scale, S.q_norm, S.k_norm,
            S.rope_cos, S.rope_sin, S.out_q, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale, s);

  // K2: split-KV flash-decode (partial + reduce) -> attn_out.
  k2_launch(S.out_q, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale, S.ctx_len,
            S.part_m, S.part_l, S.part_acc, S.attn_out, S.n_splits, s);

  // K3: O-proj GEMV + fused residual add.  h_dst = h_src + Wo @ attn_out.
  k3_launch(S.attn_out, S.Wo, S.Wo_scale, h_src, h_dst, s);

  // K4: router -> sel_idx[8], sel_w[8] (reads the post-attn residual h_dst).
  k4_launch(h_dst, S.w_post_norm, S.Wgate, S.Wgate_scale, S.sel_idx, S.sel_w, s);

  // K5: fused MoE experts; gate+up then down accumulates straight into h_dst.
  //   k5b uses atomicAdd into h_dst (the residual), so h_dst already holds the post-attn residual
  //   from K3 — the experts add the MLP contribution on top.  We pass the SAME normed activation
  //   that K4 produced implicitly via y; here we feed the post-attn residual h_dst as the expert
  //   input (proxy: in the real model K5 reads the post-RMSNorm y; reusing h_dst keeps the read
  //   volume / kernel shape identical, which is all the latency proxy needs).
  //   (cudaFuncSetAttribute for the >48KB dynamic smem is done once in alloc_state, off the hot path
  //    and outside any capture region.)
  k5a_gateup<<<S.k5.ctasA, S.k5.block, S.k5.smemA, s>>>(
      h_dst, S.sel_idx, S.Wgu_d, S.Wgu_scale_d, S.a_glb, TOP_K);
  k5b_down<<<S.k5.ctasB, S.k5.block, S.k5.smemB, s>>>(
      S.sel_idx, S.sel_w, S.Wd_d, S.Wd_scale_d, S.a_glb, h_dst, TOP_K);

  return h_dst;
}

// ---- enqueue the FULL decode step: 94 layers + final norm + lm_head + argmax --------------------
static void enqueue_decode_step(DecodeState& S, cudaStream_t s) {
  // Residual ping-pong: layer L reads `cur`, writes its output into `nxt`; then swap.
  float* cur = S.h_a;
  float* nxt = S.h_b;
  for (int layer = 0; layer < N_LAYERS; ++layer) {
    // Latency proxy: every layer reuses the same dummy weights; only the residual ping-pongs.
    float* out = enqueue_layer(S, cur, nxt, s);   // out == nxt (this layer's residual output)
    cur = out;
    nxt = (cur == S.h_a) ? S.h_b : S.h_a;         // next layer writes into the other buffer
  }
  // final RMSNorm + lm_head GEMV + on-device argmax (reads the last layer's residual `cur`).
  // (lm_head dynamic-smem opt-in is set once in alloc_state, outside any capture region.)
  const size_t lm_smem = (size_t)HIDDEN * sizeof(float);
  ds_final_norm<<<1, 256, 0, s>>>(cur, S.w_final_norm, S.hn);
  ds_lmhead_argmax_partial<<<S.lm_blocks, 256, lm_smem, s>>>(
      S.hn, S.Wlm, S.Wlm_scale, S.block_max, S.block_arg);
  ds_argmax_final<<<1, 32, 0, s>>>(S.block_max, S.block_arg, S.lm_blocks, S.tok_out);
}

// =================================================================================================
// Allocation + dummy weight init (ONE layer reused — latency proxy).
// =================================================================================================
static inline unsigned hashu(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static inline float frnd(unsigned seed, size_t i, float scale) {
  unsigned h = hashu((unsigned)(i * 2654435761u) ^ (seed * 40503u));
  return (((h % 2001) / 1000.0f) - 1.0f) * scale;
}
// Fill a device fp8 buffer with deterministic small values (host build then H2D).
static void fill_fp8(fp8* dptr, size_t n, unsigned seed) {
  std::vector<fp8> host(n);
  for (size_t i = 0; i < n; ++i) host[i] = (fp8)frnd(seed, i, 0.25f);
  CK(cudaMemcpy(dptr, host.data(), n * sizeof(fp8), cudaMemcpyHostToDevice));
}
static void fill_f32(float* dptr, size_t n, unsigned seed, float scale, bool positive) {
  std::vector<float> host(n);
  for (size_t i = 0; i < n; ++i) { float v = frnd(seed, i, scale); host[i] = positive ? (fabsf(v)+1e-3f) : v; }
  CK(cudaMemcpy(dptr, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

static void alloc_state(DecodeState& S, int ctx_len, int n_layers_weights /*=1 proxy*/) {
  S.ctx_len  = ctx_len;
  S.n_splits = k2_pick_splits(ctx_len);

  // residual ping-pong.
  CK(cudaMalloc(&S.h_a, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&S.h_b, HIDDEN * sizeof(float)));
  fill_f32(S.h_a, HIDDEN, 99u, 1.0f, false);
  CK(cudaMemset(S.h_b, 0, HIDDEN * sizeof(float)));

  // K1.
  CK(cudaMalloc(&S.w_in_norm, HIDDEN * sizeof(float)));   fill_f32(S.w_in_norm, HIDDEN, 1u, 0.5f, true);
  CK(cudaMalloc(&S.Wqkv, (size_t)QKV_OUT * HIDDEN * sizeof(fp8)));  fill_fp8(S.Wqkv, (size_t)QKV_OUT*HIDDEN, 2u);
  CK(cudaMalloc(&S.Wqkv_scale, QKV_OUT * sizeof(float))); fill_f32(S.Wqkv_scale, QKV_OUT, 3u, 0.02f, true);
  CK(cudaMalloc(&S.q_norm, HEAD_DIM * sizeof(float)));    fill_f32(S.q_norm, HEAD_DIM, 4u, 0.5f, true);
  CK(cudaMalloc(&S.k_norm, HEAD_DIM * sizeof(float)));    fill_f32(S.k_norm, HEAD_DIM, 5u, 0.5f, true);
  CK(cudaMalloc(&S.rope_cos, (HEAD_DIM/2) * sizeof(float)));
  CK(cudaMalloc(&S.rope_sin, (HEAD_DIM/2) * sizeof(float)));
  {
    std::vector<float> rc(HEAD_DIM/2), rs(HEAD_DIM/2);
    for (int i = 0; i < HEAD_DIM/2; ++i) { float f = powf(ROPE_THETA, -2.f*i/HEAD_DIM)*7.f; rc[i]=cosf(f); rs[i]=sinf(f); }
    CK(cudaMemcpy(S.rope_cos, rc.data(), (HEAD_DIM/2)*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(S.rope_sin, rs.data(), (HEAD_DIM/2)*sizeof(float), cudaMemcpyHostToDevice));
  }
  CK(cudaMalloc(&S.out_q, Q_DIM * sizeof(float)));

  // KV cache (ctx_len positions).
  CK(cudaMalloc(&S.kv_k, (size_t)ctx_len * KV_DIM * sizeof(fp8)));  fill_fp8(S.kv_k, (size_t)ctx_len*KV_DIM, 20u);
  CK(cudaMalloc(&S.kv_v, (size_t)ctx_len * KV_DIM * sizeof(fp8)));  fill_fp8(S.kv_v, (size_t)ctx_len*KV_DIM, 21u);
  CK(cudaMalloc(&S.kv_k_scale, KV_DIM * sizeof(float))); fill_f32(S.kv_k_scale, KV_DIM, 22u, 0.04f, true);
  CK(cudaMalloc(&S.kv_v_scale, KV_DIM * sizeof(float))); fill_f32(S.kv_v_scale, KV_DIM, 23u, 0.04f, true);

  // K2 partials.
  CK(cudaMalloc(&S.part_m,  (size_t)N_Q_HEADS * S.n_splits * sizeof(float)));
  CK(cudaMalloc(&S.part_l,  (size_t)N_Q_HEADS * S.n_splits * sizeof(float)));
  CK(cudaMalloc(&S.part_acc,(size_t)N_Q_HEADS * S.n_splits * HEAD_DIM * sizeof(float)));
  CK(cudaMalloc(&S.attn_out, Q_DIM * sizeof(float)));

  // K3.
  CK(cudaMalloc(&S.Wo, (size_t)HIDDEN * Q_DIM * sizeof(fp8)));  fill_fp8(S.Wo, (size_t)HIDDEN*Q_DIM, 30u);
  CK(cudaMalloc(&S.Wo_scale, HIDDEN * sizeof(float)));          fill_f32(S.Wo_scale, HIDDEN, 31u, 0.02f, true);

  // K4.
  CK(cudaMalloc(&S.w_post_norm, HIDDEN * sizeof(float)));       fill_f32(S.w_post_norm, HIDDEN, 40u, 0.5f, true);
  CK(cudaMalloc(&S.Wgate, (size_t)N_EXPERTS * HIDDEN * sizeof(fp8)));  fill_fp8(S.Wgate, (size_t)N_EXPERTS*HIDDEN, 41u);
  CK(cudaMalloc(&S.Wgate_scale, N_EXPERTS * sizeof(float)));    fill_f32(S.Wgate_scale, N_EXPERTS, 42u, 0.02f, true);
  CK(cudaMalloc(&S.sel_idx, TOP_K * sizeof(int)));
  CK(cudaMalloc(&S.sel_w,   TOP_K * sizeof(float)));
  // seed sel_idx/sel_w so K5 is valid even before K4 runs (K4 overwrites them in-graph anyway).
  { std::vector<int> si(TOP_K); std::vector<float> sw(TOP_K, 1.0f/TOP_K);
    for (int i=0;i<TOP_K;++i) si[i]=i;
    CK(cudaMemcpy(S.sel_idx, si.data(), TOP_K*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(S.sel_w,   sw.data(), TOP_K*sizeof(float), cudaMemcpyHostToDevice)); }

  // K5 experts — allocate TOP_K distinct expert weight sets (proxy: reused across all layers).
  const size_t gu_n = (size_t)2 * MOE_INTER * HIDDEN;     // gate+up per expert
  const size_t d_n  = (size_t)HIDDEN * MOE_INTER;         // down per expert
  std::vector<fp8*>   Wgu_dp(TOP_K), Wd_dp(TOP_K);
  std::vector<float*> Sgu_dp(TOP_K), Sd_dp(TOP_K);
  for (int e = 0; e < TOP_K; ++e) {
    CK(cudaMalloc(&Wgu_dp[e], gu_n * sizeof(fp8)));  fill_fp8(Wgu_dp[e], gu_n, 50u + e);
    CK(cudaMalloc(&Wd_dp[e],  d_n  * sizeof(fp8)));  fill_fp8(Wd_dp[e],  d_n,  70u + e);
    CK(cudaMalloc(&Sgu_dp[e], 2 * MOE_INTER * sizeof(float))); fill_f32(Sgu_dp[e], 2*MOE_INTER, 90u+e, 0.02f, true);
    CK(cudaMalloc(&Sd_dp[e],  HIDDEN * sizeof(float)));        fill_f32(Sd_dp[e],  HIDDEN,       110u+e, 0.02f, true);
  }
  // K5 indexes these pointer arrays by EXPERT ID (sel_idx, 0..N_EXPERTS-1, written by K4) — NOT by
  // slot. The proxy keeps only TOP_K physical weight sets, so build N_EXPERTS-wide pointer arrays
  // that round-robin into them: valid for any routed expert id, still only ~845 MB resident.
  std::vector<fp8*>   Wgu_full(N_EXPERTS), Wd_full(N_EXPERTS);
  std::vector<float*> Sgu_full(N_EXPERTS), Sd_full(N_EXPERTS);
  for (int e = 0; e < N_EXPERTS; ++e) { int p = e % TOP_K;
    Wgu_full[e] = Wgu_dp[p]; Wd_full[e] = Wd_dp[p]; Sgu_full[e] = Sgu_dp[p]; Sd_full[e] = Sd_dp[p]; }
  CK(cudaMalloc(&S.Wgu_d,       N_EXPERTS * sizeof(fp8*)));   CK(cudaMemcpy(S.Wgu_d,       Wgu_full.data(), N_EXPERTS*sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.Wd_d,        N_EXPERTS * sizeof(fp8*)));   CK(cudaMemcpy(S.Wd_d,        Wd_full.data(),  N_EXPERTS*sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.Wgu_scale_d, N_EXPERTS * sizeof(float*))); CK(cudaMemcpy(S.Wgu_scale_d, Sgu_full.data(), N_EXPERTS*sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.Wd_scale_d,  N_EXPERTS * sizeof(float*))); CK(cudaMemcpy(S.Wd_scale_d,  Sd_full.data(),  N_EXPERTS*sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.a_glb, (size_t)TOP_K * MOE_INTER * sizeof(float)));
  S.k5 = k5_plan(TOP_K);

  // final head: lm_head is the biggest single matrix (VOCAB x HIDDEN fp8 ~622 MB).
  CK(cudaMalloc(&S.w_final_norm, HIDDEN * sizeof(float)));  fill_f32(S.w_final_norm, HIDDEN, 130u, 0.5f, true);
  CK(cudaMalloc(&S.hn, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&S.Wlm, (size_t)VOCAB * HIDDEN * sizeof(fp8)));  fill_fp8(S.Wlm, (size_t)VOCAB*HIDDEN, 131u);
  CK(cudaMalloc(&S.Wlm_scale, VOCAB * sizeof(float)));          fill_f32(S.Wlm_scale, VOCAB, 132u, 0.02f, true);
  // lm_head launch: warp-per-vocab-row; pick blocks to lightly oversubscribe the SMs.
  S.lm_blocks = 264;                                       // 264 CTAs x 8 warps = 2112 warps
  CK(cudaMalloc(&S.block_max, S.lm_blocks * sizeof(float)));
  CK(cudaMalloc(&S.block_arg, S.lm_blocks * sizeof(int)));
  CK(cudaMalloc(&S.tok_out, sizeof(int)));

  // ---- opt in to the dynamic-smem sizes ONCE (host calls; done before any graph capture). ----
  //   k5a stages HIDDEN floats (16 KB), k5b stages TOP_K*MOE_INTER floats (48 KB), lm_head stages
  //   HIDDEN floats (16 KB).  The k1/k3/k4 launch helpers set their own attributes internally.
  CK(cudaFuncSetAttribute(k5a_gateup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S.k5.smemA));
  CK(cudaFuncSetAttribute(k5b_down,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S.k5.smemB));
  CK(cudaFuncSetAttribute(ds_lmhead_argmax_partial,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(HIDDEN*sizeof(float))));

  (void)n_layers_weights;  // proxy: only 1 layer's weights are physically resident.
  CK(cudaDeviceSynchronize());
}

// =================================================================================================
// main() — microbench: (a) captured-graph decode step, (b) eager (per-launch) step, (c) GB/s.
// =================================================================================================
int main(int argc, char** argv) {
  const int    ctx_len = (argc > 1) ? atoi(argv[1]) : 4096;
  const int    IT      = (argc > 2) ? atoi(argv[2]) : 200;
  const double PEAK    = (argc > 3) ? atof(argv[3]) : 3350.0;   // GB/s, single H100 HBM3
  const int    WARM    = 20;

  int ndev = 0, dev = 0; cudaDeviceProp prop;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("No CUDA device found.\n"); return 1; }
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("== Qwen3-235B-A22B fused decode step (latency proxy) ==\n");
  printf("device: %s  SMs=%d  HBM peak=%.0f GB/s\n", prop.name, prop.multiProcessorCount, PEAK);
  printf("ctx_len=%d  layers=%d  iters=%d\n", ctx_len, N_LAYERS, IT);

  DecodeState S;
  alloc_state(S, ctx_len, /*n_layers_weights=*/1);

  // ---- per-token kernel launch count (the thing the graph collapses) -------------------------
  //   per layer: K1 (1) + K2 partial+reduce (2) + K3 (1) + K4 (1) + K5a+K5b (2) = 7
  //   final: norm (1) + lm_head (1) + argmax (1) = 3
  const int launches_per_layer = 7;
  const int total_launches = launches_per_layer * N_LAYERS + 3;
  printf("kernel launches / token (eager): %d  (= %d/layer x %d + 3 head)\n",
         total_launches, launches_per_layer, N_LAYERS);

  // ---- per-token ACTIVE HBM read volume (the bytes that MUST move per token) -----------------
  // Per layer (fp8 weights dominate; activations/scales are < 1 MB and ignored):
  //   K1 Wqkv  = QKV_OUT*HIDDEN          K2 KV     = 2*ctx_len*KV_DIM
  //   K3 Wo    = HIDDEN*Q_DIM            K4 Wgate  = N_EXPERTS*HIDDEN
  //   K5 experts = TOP_K*(2*MOE_INTER*HIDDEN + HIDDEN*MOE_INTER)
  const double b_qkv   = (double)QKV_OUT * HIDDEN;
  const double b_kv    = 2.0 * (double)ctx_len * KV_DIM;
  const double b_o     = (double)HIDDEN * Q_DIM;
  const double b_gate  = (double)N_EXPERTS * HIDDEN;
  const double b_exp   = (double)TOP_K * ((double)2*MOE_INTER*HIDDEN + (double)HIDDEN*MOE_INTER);
  const double b_layer = b_qkv + b_kv + b_o + b_gate + b_exp;          // bytes/layer (fp8 = 1 B each)
  const double b_lm    = (double)VOCAB * HIDDEN;                       // lm_head, once per token
  const double b_token = b_layer * N_LAYERS + b_lm;                    // full single-GPU read/token
  printf("\nper-token ACTIVE read (full single-GPU model): %.2f GB\n", b_token / 1e9);
  printf("  per layer %.1f MB  (experts %.1f + Wqkv %.1f + Wo %.1f + KV %.1f + gate %.2f) x %d layers\n",
         b_layer/1e6, b_exp/1e6, b_qkv/1e6, b_o/1e6, b_kv/1e6, b_gate/1e6, N_LAYERS);
  printf("  + lm_head %.1f MB once.  (A 1/8 EP+TP shard would read ~%.2f GB/token.)\n",
         b_lm/1e6, b_token/8.0/1e9);
  printf("NOTE: only ONE layer's dummy weights are resident (~%.0f MB) and reused 94x — this is a\n",
         (b_layer - b_kv)/1e6 + b_lm/1e6);
  printf("      LATENCY/launch-overhead PROXY; the read VOLUME above is the real per-token traffic.\n");

  // NOTE (numerical proxy): K5b atomicAdds the MLP contribution into the residual buffers, which are
  // NOT re-initialized between steps, so across repeated replays/iters the residual grows to inf/nan.
  // That is expected and HARMLESS for this benchmark: every kernel still issues identical grids, reads
  // the identical byte volume, and runs in identical time regardless of the (meaningless) values.

  // =============================================================================================
  // (1) CUDA-GRAPH CAPTURE of the whole 94-layer step + head.
  // =============================================================================================
  // Whole-step capture via K6 (kernels/k6_graph_capture.cu): warm-up outside capture, then record
  // K1..K5 x94 + final-norm + lm_head + argmax into one graph and instantiate it.
  DecodeGraph g;
  if (!build_decode_graph(g, [&](cudaStream_t s) { enqueue_decode_step(S, s); })) {
    printf("graph capture FAILED (see CUDA error above)\n"); return 1;
  }
  cudaStream_t    cap  = g.stream;              // K6-owned capture/replay stream
  cudaGraphExec_t exec = g.exec;
  printf("\ncaptured graph: %zu nodes instantiated.\n", decode_graph_nodes(g));

  cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));

  // ---- (a) graph REPLAY timing ----
  for (int i = 0; i < WARM; ++i) CK(cudaGraphLaunch(exec, cap));
  CK(cudaStreamSynchronize(cap));
  CK(cudaEventRecord(ev0, cap));
  for (int i = 0; i < IT; ++i) CK(cudaGraphLaunch(exec, cap));
  CK(cudaEventRecord(ev1, cap));
  CK(cudaEventSynchronize(ev1));
  float ms_graph; CK(cudaEventElapsedTime(&ms_graph, ev0, ev1)); ms_graph /= IT;

  // ---- (b) EAGER (individual launches) timing — same kernels, no graph ----
  for (int i = 0; i < WARM; ++i) enqueue_decode_step(S, cap);
  CK(cudaStreamSynchronize(cap));
  CK(cudaEventRecord(ev0, cap));
  for (int i = 0; i < IT; ++i) enqueue_decode_step(S, cap);
  CK(cudaEventRecord(ev1, cap));
  CK(cudaEventSynchronize(ev1));
  float ms_eager; CK(cudaEventElapsedTime(&ms_eager, ev0, ev1)); ms_eager /= IT;

  // =============================================================================================
  // (3) report.
  // =============================================================================================
  auto tokps = [](float ms) { return 1.0e3 / ms; };
  auto gbps  = [&](float ms) { return b_token / 1e6 / ms; };   // bytes/ms = GB/s
  printf("\n  %-28s %12s %12s %12s %12s\n", "mode", "us/token", "tok/s", "GB/s", "%HBMpeak");
  printf("  %-28s %12.2f %12.1f %12.1f %11.1f%%\n", "captured graph (replay)",
         ms_graph*1e3, tokps(ms_graph), gbps(ms_graph), 100.0*gbps(ms_graph)/PEAK);
  printf("  %-28s %12.2f %12.1f %12.1f %11.1f%%\n", "eager (individual launches)",
         ms_eager*1e3, tokps(ms_eager), gbps(ms_eager), 100.0*gbps(ms_eager)/PEAK);

  float delta_us = (ms_eager - ms_graph) * 1e3;
  printf("\n  launch-overhead delta (eager - graph): %.2f us/token  (%.1f%% faster with graph)\n",
         delta_us, 100.0 * (ms_eager - ms_graph) / ms_eager);
  printf("  implied per-launch dispatch cost: %.2f us  (delta / %d launches)\n",
         delta_us / total_launches, total_launches);

  // ---- cleanup (best-effort; OS reclaims on exit) ----
  destroy_decode_graph(g);                      // frees exec, graph, and the K6 capture stream (cap)
  CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1));
  printf("\n== done ==\n");
  return 0;
}
