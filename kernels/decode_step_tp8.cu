// decode_step_tp8.cu — TENSOR-PARALLEL (TP=8) single-token DECODE STEP for Qwen3-235B-A22B.
// Target: 8x H100 (sm_90a), one process driving all 8 GPUs, NCCL for the per-layer all-reduces.
// Standard CUDA + NCCL only.  Reuses the repo's validated single-GPU kernels (k1..k5) UNCHANGED,
// launched on per-rank SHARDED sub-ranges; the only new code is the sharded weight layout, the two
// all-reduces/layer, and the multi-GPU host driver.
//
// THE POINT
// ---------
// A single H100 reads the full ~22 GB of active fp8 weights/token and tops out at ~153 tok/s (HBM
// bound).  The path past that cap is to SHARD the model across all 8 GPUs so each reads only ~1/8 of
// the per-token weight volume (~2.75 GB) IN PARALLEL, then stitch the partial results back together
// with two tiny NCCL all-reduces per layer.  This file measures the REAL B=1 latency of that TP=8
// step: 94 layers x (sharded attention + sharded MoE), each layer paying two ~8 KB all-reduces, plus
// a vocab-sharded lm_head + a cross-rank argmax.
//
// TP=8 SHARDING (matches docs/b1-tp8-moe-rearchitecture-h200.md and research/comms_floor.md)
// -------------------------------------------------------------------------------------------------
//   Attention (row-parallel O-proj):
//     * 64 Q heads / 8 ranks = 8 Q heads/rank.  4 KV heads < 8 ranks, so KV is REPLICATED (every
//       rank holds all 4 KV heads).  Each rank's Wqkv shard is [8 Q-head rows + 4 K + 4 V rows] =
//       (8 + 4 + 4) * HEAD_DIM = 2048 rows of HIDDEN  (vs the full 9216) -> ~1/8 of Q + full KV.
//     * K1 (sharded QKV GEMV + QK-norm + RoPE + KV write) on this rank's 2048-row shard.
//     * K2 flash-decode over this rank's 8 Q heads only (each rank owns Q heads [r*8, r*8+8)).
//     * K3 O-proj: this rank holds the Wo COLUMN-slice [HIDDEN, 8*HEAD_DIM=1024] that consumes its 8
//       heads' attn_out -> produces a PARTIAL hidden[HIDDEN].  (Row-parallel: the O-proj contraction
//       dim Q_DIM=8192 is split 8 ways, 1024/rank; the partials SUM to the true O-proj output.)
//     * NCCL all-reduce(SUM) the partial O-proj output [HIDDEN] -> full post-attn residual on EVERY
//       rank (the fused residual add is done locally AFTER the all-reduce so every rank adds h_in once).
//
//   MoE (column/intermediate-parallel, TP within every expert — NO expert-parallel imbalance):
//     * Each rank holds 1536/8 = 192 of every expert's intermediate columns.  gate/up shard
//       Wgu_shard[2*192, HIDDEN]; down shard Wd_shard[HIDDEN, 192].  Every rank reads exactly 8/8 of
//       each active expert's 192-col share -> zero balls-in-bins gamble (the EP8 busiest-rank loss is
//       structurally eliminated).
//     * K4 router runs REPLICATED on every rank (cheap, 0.05 GB; identical input after AR#1 -> every
//       rank computes the same top-8, so no comms is needed to agree on sel_idx/sel_w).
//     * K5a (gate+up SwiGLU) produces this rank's 192-slice a[TOP_K, 192] for the 8 active experts.
//     * K5b (down-proj) over the 192-wide contraction -> PARTIAL hidden[HIDDEN] (the 8 experts'
//       contributions sum into it locally; the cross-RANK column-sum is the all-reduce below).
//     * NCCL all-reduce(SUM) the partial MoE-down output [HIDDEN] -> full MoE contribution; added to
//       the residual locally on every rank.
//
//   Head:  lm_head [VOCAB, HIDDEN] sharded by VOCAB rows (VOCAB/8 + remainder on rank 0).  Each rank
//          argmaxes its vocab slice locally, then a tiny all-reduce-MAX over (logit,globalid) pairs
//          picks the global next token.  (Final RMSNorm is replicated — cheap.)
//
//   Per step: 2 all-reduces/layer x 94 = 188 NCCL all-reduces on ~16 KB ([HIDDEN] fp32) payloads +
//   1 head all-reduce.  Tiny messages -> pure latency, not bandwidth (comms_floor.md: ~7-16 us each).
//
// LATENCY-PROXY DISCLAIMER (same as decode_step.cu): only ONE layer's worth of SHARDED dummy fp8
// weights is resident per GPU and reused for all 94 layers, so the produced logits are meaningless.
// But every rank's per-token HBM READ VOLUME is the real ~1/8 shard (~2.75 GB), the kernel chain /
// grid shapes are the real ones, and the all-reduces are real NCCL collectives on real streams — so
// the measured us/token, tok/s, and all-reduce overhead are representative of the real TP=8 step.
//
// CORRECTNESS GATE (NEW): before the bench, run_correctness_check() validates ONE sharded layer
// against a single-GPU FULL reference assembled from the per-rank shards (the exact decode_step.cu
// math).  It compares the two residual stitch-points the all-reduces produce — post-attention
// (after AR#1) and post-MoE (after AR#2) — and asserts max|ref-shd| < 1e-2.  This proves: the
// Wqkv row-shard + Wo column-shard + expert intermediate-shard layouts are right, the AR(SUM) of
// the partials reconstructs the full projection, and the residual is added EXACTLY ONCE (not 8x).
// A failure aborts with exit code 2 before any (bogus) throughput is reported.  Pass argv[4]==0 to
// skip the check (timing-only runs).
//
// BUILD (on the 8xH100 box; NCCL via pip `nvidia-nccl-cu12`, headers/lib under the pip nvidia dir):
//   NCCL_INC=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'include'))")
//   NCCL_LIB=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'lib'))")
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -I "$NCCL_INC" \
//        kernels/decode_step_tp8.cu -L "$NCCL_LIB" -lnccl -o /tmp/dstp8
//   LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" /tmp/dstp8
//   (If NCCL is system-installed instead, plain `-lnccl` suffices; nccl.h is then on the default path.)
//   ARGS: [ctx_len=4096] [iters=200] [HBM_GBs=3350] [run_check=1]   (run_check=0 -> timing only)
//
// =================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>
#include <nccl.h>
#include "common.cuh"
#include "nvls_engine.cuh"     // NVLS multimem in-switch all-reduce (replaces NCCL for the 188 layer ARs)
using namespace q3;

// ---- NVLS toggle: build with -DUSE_NVLS=0 to fall back to the NCCL all-reduces (A/B comparison). ---
#ifndef USE_NVLS
#define USE_NVLS 1
#endif

// ---- GEMM toggle: build with -DUSE_GEMM=0 to fall back to the hand-rolled M=1 GEMV kernels (A/B). --
//   USE_GEMM=1 (default) routes the dense per-rank projections (K1 QKV, K3 Oproj, K4 router gate,
//   K5 experts gate+up/down, lm_head) through cuBLASLt fp8 tensor-core GEMM (the validated fast path,
//   spec_verify_forward_gemm.cu).  K2 flash-decode stays as-is.  The 2 NVLS all-reduces/layer + the
//   cross-rank correctness gate are unchanged.
#ifndef USE_GEMM
#define USE_GEMM 1
#endif
// Max verify columns the GEMM panels are built/padded for (B=1 decode uses M=1; spec verify M<=16).
#ifndef GEMM_MMAX
#define GEMM_MMAX 16
#endif

// ---- Pull in the existing validated kernels as ONE translation unit (same recipe as decode_step.cu).
#define K5_NO_MAIN
#define Q3_K1_LAUNCH_HELPER
#define Q3_K2_LAUNCH_HELPER
#define Q3_K3_LAUNCH_HELPER
#define Q3_K4_LAUNCH_HELPER
#include "k5_experts.cu"        // k5a_gateup, k5b_down, K5Launch, k5_plan
#include "k1_attn_prologue.cu"  // k1_attn_prologue + k1_launch (sharded by row count)
#include "k2_flash_decode.cu"   // k2_flash_decode_partial/_reduce + k2_launch + k2_pick_splits
#include "k3_attn_epilogue.cu"  // k3_attn_epilogue + k3_launch
#include "k4_router.cu"         // k4_router + k4_launch

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                       \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));           \
  exit(1); } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) {                      \
  printf("NCCL err %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r_));           \
  exit(1); } } while (0)

#if USE_GEMM
#include <cublasLt.h>
#include "gemm_engine.cuh"      // LtPanel (cuBLASLt fp8 TN-GEMM) + gemm_rmsnorm_quant / gemm_quant
#endif

// =================================================================================================
// TP=8 shard geometry (compile-time constants derived from common.cuh).
// =================================================================================================
constexpr int TP            = 8;
constexpr int Q_HEADS_RANK  = N_Q_HEADS / TP;                 // 8 Q heads / rank  (64/8)
static_assert(N_Q_HEADS % TP == 0, "Q heads must split evenly across TP ranks");
constexpr int Q_DIM_RANK    = Q_HEADS_RANK * HEAD_DIM;        // 1024  (this rank's Q-head output)
// Wqkv row shard: this rank's 8 Q-head rows + the 4 K rows + the 4 V rows (KV replicated, 4<8).
constexpr int QKV_OUT_RANK  = Q_DIM_RANK + 2 * KV_DIM;        // 1024 + 512 + 512 = 2048 rows
// MoE intermediate shard: 1536 / 8 = 192 cols of EVERY expert (TP within experts).
constexpr int MOE_INTER_RANK = MOE_INTER / TP;               // 192
static_assert(MOE_INTER % TP == 0, "MoE intermediate must split evenly across TP ranks");
// lm_head vocab shard (VOCAB not divisible by 8 -> rank 0 absorbs the remainder).
static inline int vocab_rows_for(int rank)  { int base = VOCAB / TP; int rem = VOCAB % TP;
                                              return base + (rank == 0 ? rem : 0); }
static inline int vocab_offset_for(int rank){ int base = VOCAB / TP; int rem = VOCAB % TP;
                                              return rank == 0 ? 0 : rem + rank * base; }

// =================================================================================================
// Final head: replicated RMSNorm + VOCAB-sharded lm_head GEMV + local argmax over the rank's slice.
// (RMSNorm is identical to decode_step.cu; lm_head/argmax write a per-rank (max,arg) so the host can
//  do the cross-rank reduce — or we all-reduce-max the pair below.)
// =================================================================================================
#ifndef Q3_DSTP8_DEFS
#define Q3_DSTP8_DEFS
static __device__ __forceinline__ float tp8_warp_dot(const fp8* __restrict__ w,
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
#endif // Q3_DSTP8_DEFS

#if USE_GEMM
// =================================================================================================
// GEMM OUTPUT EPILOGUES.  cuBLASLt writes raw fp8 dots D[Mpad,N] (col-major bf16).  These tiny
// epilogues read column 0 (the B=1 activation), apply act_scale[0] * per-channel Wscale[n], and
// write the M=1 fp32 result the rest of the chain expects — identical to how the GEMV kernels
// applied `r * Wscale[o]`.  (Mpad is the col-major leading dim = GEMM_MMAX rounded to 16 = 16.)
// =================================================================================================
// Generic: out[n] = D[col0,n] * act_scale[0] * Wscale[n].  Used for K1 QKV proj + K3 O-proj.
extern "C" __global__ void gemm_epi_scale(
    const __nv_bfloat16* __restrict__ D, const float* __restrict__ Wscale,
    const float* __restrict__ act_scale, float* __restrict__ out, int N, int Mpad) {
  const float as = act_scale[0];
  for (int n = blockIdx.x*blockDim.x + threadIdx.x; n < N; n += gridDim.x*blockDim.x)
    out[n] = (float)D[(size_t)n * Mpad] * as * Wscale[n];
}
// Router gate logits: g_logits[e] = D[col0,e] * act_scale[0]  (per-expert Wgate_scale applied in
// tp8_k4_select, exactly as the split-K gate GEMV deferred its scale).  N == N_EXPERTS.
extern "C" __global__ void gemm_epi_gate(
    const __nv_bfloat16* __restrict__ D, const float* __restrict__ act_scale,
    float* __restrict__ g_logits, int N, int Mpad) {
  const float as = act_scale[0];
  for (int e = blockIdx.x*blockDim.x + threadIdx.x; e < N; e += gridDim.x*blockDim.x)
    g_logits[e] = (float)D[(size_t)e * Mpad] * as;
}
// K5a gate+up SwiGLU epilogue.  D is [Mpad, TOP_K*2*MOE_INTER_RANK] for the 8 routed experts' fused
// gate+up rows (rows [0,192) gate, [192,384) up, per slot).  a_glb[slot*192+j] =
//   silu(act_scale*Dgate * Sgu[e][j]) * (act_scale*Dup * Sgu[e][192+j]).  Mirrors tp8_k5a_gateup's
// final-scale math byte-for-byte (just the dot came from the GEMM instead of the warp loop).
extern "C" __global__ void gemm_epi_k5a(
    const __nv_bfloat16* __restrict__ D, const int* __restrict__ sel_idx,
    const float* const* __restrict__ Wgu_scale, const float* __restrict__ act_scale,
    float* __restrict__ a_glb, int nslot, int Mpad) {
  const float as = act_scale[0];
  const int total = nslot * MOE_INTER_RANK;
  for (int it = blockIdx.x*blockDim.x + threadIdx.x; it < total; it += gridDim.x*blockDim.x) {
    const int slot = it / MOE_INTER_RANK;
    const int j    = it - slot * MOE_INTER_RANK;
    const int e    = sel_idx[slot];
    const float* S = Wgu_scale[e];
    const int gcol = slot * (2*MOE_INTER_RANK) + j;             // gate row j of this slot
    const int ucol = slot * (2*MOE_INTER_RANK) + MOE_INTER_RANK + j;  // up row j
    float gacc = (float)D[(size_t)gcol * Mpad] * as;
    float uacc = (float)D[(size_t)ucol * Mpad] * as;
    a_glb[(size_t)slot * MOE_INTER_RANK + j] = silu(gacc * S[j]) * (uacc * S[MOE_INTER_RANK + j]);
  }
}
// K5b down epilogue.  D is [Mpad, TOP_K*HIDDEN]: row o of slot s = the down dot for channel o.
//   h_io[o] += sel_w[s] * act_scale[s] * D[s*HIDDEN+o] * Sd[e][o].  atomicAdd (8 slots contribute).
//   act_scale is PER-SLOT (each slot's a_glb quantized with its own amax scale).
extern "C" __global__ void gemm_epi_k5b(
    const __nv_bfloat16* __restrict__ D, const int* __restrict__ sel_idx,
    const float* __restrict__ sel_w, const float* const* __restrict__ Wd_scale,
    const float* __restrict__ act_scale, float* __restrict__ h_io, int nslot, int Mpad) {
  const int total = nslot * HIDDEN;
  for (int it = blockIdx.x*blockDim.x + threadIdx.x; it < total; it += gridDim.x*blockDim.x) {
    const int slot = it / HIDDEN;
    const int o    = it - slot * HIDDEN;
    const int e    = sel_idx[slot];
    const float gw = sel_w[slot];
    float acc = (float)D[(size_t)(slot*HIDDEN + o) * Mpad] * act_scale[slot];
    atomicAdd(&h_io[o], gw * acc * Wd_scale[e][o]);
  }
}
// lm_head epilogue + partial argmax over this rank's vocab slice.  D is [Mpad, v_rows].
//   logit[row] = D[col0,row] * act_scale[0] * Wlm_scale[row].  Same block-partial argmax as
//   tp8_lmhead_argmax_partial (one block -> (max,arg) over its row stride).
extern "C" __global__ void gemm_epi_lmhead_argmax(
    const __nv_bfloat16* __restrict__ D, const float* __restrict__ Wlm_scale,
    const float* __restrict__ act_scale, int n_rows, int row_offset, int Mpad,
    float* __restrict__ block_max, int* __restrict__ block_arg) {
  const float as = act_scale[0];
  float my_max = -3.0e38f; int my_arg = -1;
  for (int row = blockIdx.x*blockDim.x + threadIdx.x; row < n_rows; row += gridDim.x*blockDim.x) {
    float v = (float)D[(size_t)row * Mpad] * as * Wlm_scale[row];
    if (v > my_max) { my_max = v; my_arg = row_offset + row; }
  }
  // block reduce (max with arg)
  __shared__ float smax[32]; __shared__ int sarg[32];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  #pragma unroll
  for (int o=16;o>0;o>>=1){ float om=__shfl_down_sync(0xffffffffu,my_max,o); int oa=__shfl_down_sync(0xffffffffu,my_arg,o);
                            if (om>my_max){ my_max=om; my_arg=oa; } }
  if (lane==0){ smax[wid]=my_max; sarg[wid]=my_arg; }
  __syncthreads();
  if (threadIdx.x==0){ float bm=-3.0e38f; int ba=-1; int nwc=(blockDim.x+31)>>5;
                       for(int w=0;w<nwc;w++) if(smax[w]>bm){bm=smax[w];ba=sarg[w];}
                       block_max[blockIdx.x]=bm; block_arg[blockIdx.x]=ba; }
}
#endif // USE_GEMM

// Replicated final RMSNorm (same as decode_step.cu's ds_final_norm).
extern "C" __global__ void tp8_final_norm(const float* __restrict__ h,
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

// Sharded lm_head GEMV + per-block partial argmax over THIS RANK'S vocab slice.
//   Wlm is the rank's [n_rows, HIDDEN] slice; `row_offset` maps a local row back to its global vocab
//   id so the final argmax reports the true token id.  Identical warp-per-row idiom to decode_step.cu.
extern "C" __global__ void tp8_lmhead_argmax_partial(
    const float* __restrict__ hn,
    const fp8*  __restrict__ Wlm, const float* __restrict__ Wlm_scale,
    int n_rows, int row_offset,
    float* __restrict__ block_max, int* __restrict__ block_arg) {
  extern __shared__ float hs[];                            // [HIDDEN]
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) hs[k] = hn[k];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int wid   = threadIdx.x >> 5;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int nwc   = blockDim.x >> 5;

  float my_max = -3.0e38f; int my_arg = -1;
  for (int row = gwarp; row < n_rows; row += nwarp) {
    float v = tp8_warp_dot(Wlm + (size_t)row * HIDDEN, hs, HIDDEN, lane);
    if (lane == 0) { v *= Wlm_scale[row]; if (v > my_max) { my_max = v; my_arg = row_offset + row; } }
  }
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

// Reduce this rank's per-block partials into a single (max_logit, token_id) at slot [0].
// We pack it as two parallel device scalars so the host (or an all-reduce-max) can pick the global max.
extern "C" __global__ void tp8_argmax_final(const float* __restrict__ block_max,
                                            const int* __restrict__ block_arg,
                                            int nblocks,
                                            float* __restrict__ rank_max, int* __restrict__ rank_arg) {
  if (threadIdx.x != 0) return;
  float bm = -3.0e38f; int ba = -1;
  for (int b = 0; b < nblocks; ++b) if (block_max[b] > bm) { bm = block_max[b]; ba = block_arg[b]; }
  rank_max[0] = bm;
  rank_arg[0] = ba;
}

// =================================================================================================
// Per-RANK device state (one rank == one GPU; the SHARDED reuse of one layer's dummy weights).
// =================================================================================================
struct RankState {
  int rank = 0, dev = 0;
  cudaStream_t stream = nullptr;
  ncclComm_t   comm   = nullptr;
  NvlsCtx*     nvls   = nullptr;   // NVLS multimem AR context (null -> NCCL fallback).  attn_partial/
                                   // moe_partial are repointed to nvls->uc halves when NVLS is active.

  // residual ping-pong (full [HIDDEN] on every rank after each all-reduce).
  float *h_a = nullptr, *h_b = nullptr;

  // ---- K1 (attention prologue), SHARDED to this rank's QKV rows ----
  float *w_in_norm = nullptr;                                // [HIDDEN] (replicated; cheap)
  fp8   *Wqkv = nullptr;  float *Wqkv_scale = nullptr;       // [QKV_OUT_RANK, HIDDEN], [QKV_OUT_RANK]
  float *q_norm = nullptr, *k_norm = nullptr;                // [HEAD_DIM]
  float *rope_cos = nullptr, *rope_sin = nullptr;            // [HEAD_DIM/2]
  float *out_q = nullptr;                                    // [Q_DIM_RANK]  (this rank's 8 Q heads)
  float *qkv_proj = nullptr;                                 // [QKV_OUT_RANK] K1 GEMV scratch (per-rank)
  // KV cache: REPLICATED (all 4 KV heads), since 4 < 8 ranks.
  fp8   *kv_k = nullptr, *kv_v = nullptr;                    // [ctx_len, KV_DIM]
  float *kv_k_scale = nullptr, *kv_v_scale = nullptr;        // [KV_DIM]
  int    ctx_len = 0, n_splits = 0, k2r_warps = 8;   // k2r_warps: parallel split-folds in the K2 reduce
  // K2 partials (sized for this rank's Q_HEADS_RANK heads).
  float *part_m = nullptr, *part_l = nullptr, *part_acc = nullptr;
  float *attn_out = nullptr;                                 // [Q_DIM_RANK]

  // ---- K3 (O-proj), SHARDED: this rank holds the Wo column-slice for its 8 heads ----
  // Wo_shard logical [HIDDEN, Q_DIM_RANK]: dots this rank's attn_out[Q_DIM_RANK] -> PARTIAL hidden.
  fp8   *Wo = nullptr;  float *Wo_scale = nullptr;           // [HIDDEN, Q_DIM_RANK], [HIDDEN]
  float *attn_partial = nullptr;                             // [HIDDEN] partial O-proj (AR INPUT; K3 writes)
  float *attn_reduced = nullptr;                             // [HIDDEN] AR OUTPUT (residual-add reads).
                                                             // NVLS out-of-place: OUT half; NCCL: == attn_partial.

  // ---- K4 (router), REPLICATED ----
  float *w_post_norm = nullptr;                              // [HIDDEN]
  fp8   *Wgate = nullptr; float *Wgate_scale = nullptr;      // [N_EXPERTS, HIDDEN], [N_EXPERTS]
  float *g_logits = nullptr;                                 // [N_EXPERTS] gate-GEMV split-K accumulator
  int   *sel_idx = nullptr;  float *sel_w = nullptr;         // [TOP_K]
  float *y_norm = nullptr;                                   // [HIDDEN] post-norm MoE input (staged)

  // ---- K5 (experts), SHARDED to MOE_INTER_RANK=192 intermediate cols ----
  const fp8   **Wgu_d = nullptr;  const float **Wgu_scale_d = nullptr;  // gate+up shard [2*192, HIDDEN]
  const fp8   **Wd_d  = nullptr;  const float **Wd_scale_d  = nullptr;  // down  shard [HIDDEN, 192]
  float *a_glb = nullptr;                                    // [TOP_K * MOE_INTER_RANK]
  float *moe_partial = nullptr;                              // [HIDDEN] partial MoE-down (AR INPUT; K5b writes)
  float *moe_reduced = nullptr;                              // [HIDDEN] AR OUTPUT (residual-add reads).
                                                             // NVLS out-of-place: OUT half; NCCL: == moe_partial.

  // ---- final head, VOCAB-sharded ----
  float *w_final_norm = nullptr;                             // [HIDDEN]
  float *hn = nullptr;                                       // [HIDDEN]
  fp8   *Wlm = nullptr;  float *Wlm_scale = nullptr;         // [vrows, HIDDEN], [vrows]
  int    v_rows = 0, v_off = 0, lm_blocks = 0;
  float *block_max = nullptr;  int *block_arg = nullptr;
  float *rank_max = nullptr;   int *rank_arg = nullptr;      // [1] each (this rank's best)

  K5Launch k5;                                               // K5 plan for nslot=TOP_K, inter=192

#if USE_GEMM
  // ---- cuBLASLt fp8 TN-GEMM engine (replaces the dense GEMVs; K2 flash-decode unchanged) ----------
  cublasLtHandle_t lt = nullptr;
  // one autotuned panel per dense projection (K,N fixed; M padded to GEMM_MMAX=16).
  LtPanel p_qkv;     // [K=HIDDEN, N=QKV_OUT_RANK]  K1 QKV
  LtPanel p_oproj;   // [K=Q_DIM_RANK, N=HIDDEN]    K3 O-proj
  LtPanel p_gate;    // [K=HIDDEN, N=N_EXPERTS]     K4 router gate
  LtPanel p_k5gu;    // [K=HIDDEN, N=TOP_K*2*MOE_INTER_RANK]  K5 gate+up (8 experts fused)
  LtPanel p_k5d;     // [K=MOE_INTER_RANK, N=TOP_K*HIDDEN]    K5 down   (8 experts fused)
  LtPanel p_lm;      // [K=HIDDEN, N=v_rows]        lm_head
  // activation quant scratch (col-major [K, GEMM_MMAX] fp8, zero-padded) + per-tensor act scale.
  __nv_fp8_e4m3 *xq_hidden=nullptr;   // [HIDDEN*GEMM_MMAX]   K1/K4/K5a/lm_head input
  __nv_fp8_e4m3 *xq_qdim=nullptr;     // [Q_DIM_RANK*GEMM_MMAX] K3 input (attn_out)
  __nv_fp8_e4m3 *xq_a=nullptr;        // [TOP_K*MOE_INTER_RANK*GEMM_MMAX] K5b input (a_glb), per-slot
  float *act_scale=nullptr;           // [1] per-tensor activation dequant scale (reused per GEMM)
  float *act_scale_a=nullptr;         // [TOP_K] per-slot activation scale for the K5b down GEMMs
  // GEMM bf16 output buffers (col-major [GEMM_MMAX, N]).
  __nv_bfloat16 *d_qkv=nullptr, *d_oproj=nullptr, *d_gate=nullptr, *d_k5gu=nullptr, *d_k5d=nullptr, *d_lm=nullptr;
  // K5 is GROUPED: 8 routed experts, each a separate (X or W) operand -> a per-slot GEMM.  We keep
  // the TOP_K physical expert shard HOST pointers (the N_EXPERTS table round-robins e -> e%TOP_K),
  // so given the routed sel_idx[slot] we pick Wgu_phys[sel_idx[slot]%TOP_K] for that slot's GEMM.
  fp8*   Wgu_phys_h[TOP_K] = {};   // gate+up shard [2*192, HIDDEN] per physical expert (host ptr)
  fp8*   Wd_phys_h[TOP_K]  = {};   // down    shard [HIDDEN, 192]   per physical expert (host ptr)
  int    sel_h[TOP_K]      = {};   // host copy of routed expert ids (D2H once, eager path only)
  // PACKED gate+up: the 8 routed experts' [2*192,HIDDEN] shards concatenated into ONE contiguous
  // [TOP_K*2*192, HIDDEN] buffer so K5a is a SINGLE GEMM (1 launch, not 8).  In the proxy/graphed
  // path the pack is built once (fixed slot->shard); the correctness path re-packs the routed shards.
  fp8*   Wgu_pack=nullptr;          // [TOP_K*2*MOE_INTER_RANK, HIDDEN] fp8
  LtPanel p_k5gu_pack;              // [K=HIDDEN, N=TOP_K*2*MOE_INTER_RANK]  single grouped gate+up GEMM
  fp8*   Wd_pack=nullptr;           // [TOP_K*HIDDEN, MOE_INTER_RANK] fp8  (down shards concatenated)
  LtPanel p_k5d_pack;               // [K=MOE_INTER_RANK, N=TOP_K*HIDDEN]  single grouped down GEMM (flatness)
#endif

  // ---- CUDA-graph capture of this rank's full token (kernels + its NCCL collectives) ----
  cudaGraph_t     graph = nullptr;
  cudaGraphExec_t exec  = nullptr;
  // ---- kernels-only segment graphs (NCCL all-reduces stay eager between them) ----
  cudaGraphExec_t exec_segA = nullptr, exec_segB = nullptr, exec_seghead = nullptr;
};

// Forward declarations of the sharded launch helpers (defined after enqueue_tp8_layer, which calls them).
static void tp8_k1_launch(RankState& S, const float* h, cudaStream_t s);
static void tp8_k2_launch(RankState& S, cudaStream_t s);
static void tp8_k3_launch(RankState& S, cudaStream_t s);

// =================================================================================================
// A 192-intermediate K5 plan (k5_plan assumes MOE_INTER; we override the row counts + smem for 192).
// k5a/k5b read MOE_INTER from common.cuh, so for the SHARDED launch we cannot reuse k5a_gateup/k5b_down
// verbatim with a different inner width.  Instead we provide thin 192-aware kernels below that mirror
// the exact warp-per-row coalesced-fp8 idiom (identical math, just MOE_INTER_RANK as the inner dim).
// =================================================================================================
// ---------------------------------------------------------------------------------------------
// SHARDED K5: MULTIPLE-ROWS-PER-WARP (the measured win for the TP=8 shard; see k5_sharded_bench.cu).
// ---------------------------------------------------------------------------------------------
// At TP=8 each rank reads only ~1/8 of the experts (gate+up 12.6 MB, down 6.3 MB).  The original
// R=1 warp-per-row kernels measured only ~15% MBU fused on this shard — STARVED.  Two distinct
// starvations, both fixed by giving each warp R adjacent output rows (R independent in-flight loads
// per lane, no cp.async — cp.async measured SLOWER at M=1, see k5_experts_v3.cu sweep):
//   * Kernel A (gate+up): inner dim is HIDDEN=4096 (unchanged), but the shard has only 192 rows/
//     expert -> 1536 row-items total under-fill the grid.  R=2 rows/warp (gate+up = 4 rows) keeps
//     warps busy longer and lifts A from ~26% -> ~39% MBU.
//   * Kernel B (down): inner dim shrinks 1536 -> 192 = 12 uint4 < 32 lanes (20 idle lanes/load).
//     R=8 rows/warp packs 8 independent loads per lane -> lifts B from ~9% -> ~27% MBU.
// Measured fused: 15% -> ~34% MBU on the 192-shard (2.2x), correctness max_rel 1.3e-5.  Compile-time
// R; warp_dot math is byte-identical to tp8_warp_dot (split-K, hw fp8x2->half2, 2 accumulators/row).
#ifndef TP8_K5_RA
#define TP8_K5_RA 2     // gate+up rows per warp  (dots 2*RA = 4 weight rows/warp) — sweep best (42% MBU)
#define TP8_K5_RB 16    // down    rows per warp  — sweep: R=16 (22.3% MBU) slightly beats R=8 on the shard
#endif

// gate+up shard: a_glb[slot*192 + j] = silu(s_g*<y,gate_j>) * (s_u*<y,up_j>), RA channels/warp.
// Wgu shard layout [2*192, HIDDEN]: rows [0,192) gate, [192,384) up.
template <int R>
__device__ __forceinline__ void tp8_k5a_gateup_t(
    const float* __restrict__ y, const int* __restrict__ sel_idx,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    float* __restrict__ a_glb, int nslot) {
  extern __shared__ float ys[];                              // [HIDDEN]
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) ys[k] = y[k];
  __syncthreads();
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int njrow = (MOE_INTER_RANK + R - 1) / R;
  const int total = nslot * njrow;
  const int nv    = HIDDEN >> 4;
  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / njrow;
    const int j0   = (item - slot * njrow) * R;
    const int e    = sel_idx[slot];
    const fp8*   W = Wgu[e];
    const float* S = Wgu_scale[e];
    const uint4* gv[R]; const uint4* uv[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      gv[r] = reinterpret_cast<const uint4*>(W + (size_t)(j0 + r) * HIDDEN);
      uv[r] = reinterpret_cast<const uint4*>(W + (size_t)(MOE_INTER_RANK + j0 + r) * HIDDEN);
    }
    float g0[R], g1[R], u0[R], u1[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) { g0[r]=g1[r]=u0[r]=u1[r]=0.f; }
    for (int v = lane; v < nv; v += 32) {
      const float* yy = ys + (v << 4);
      uint4 gp[R], up[R];
      #pragma unroll
      for (int r = 0; r < R; ++r) { gp[r] = gv[r][v]; up[r] = uv[r][v]; }  // 2R loads in flight
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        const unsigned* gu = reinterpret_cast<const unsigned*>(&gp[r]);
        const unsigned* uu = reinterpret_cast<const unsigned*>(&up[r]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float* yq = yy + (q << 2);
          unsigned gq = gu[q]; __nv_fp8x2_e4m3 gl,gh; gl.__x=(unsigned short)(gq&0xffffu); gh.__x=(unsigned short)(gq>>16);
          float2 gfl=__half22float2((__half2)gl), gfh=__half22float2((__half2)gh);
          g0[r]+=yq[0]*gfl.x; g1[r]+=yq[1]*gfl.y; g0[r]+=yq[2]*gfh.x; g1[r]+=yq[3]*gfh.y;
          unsigned uq = uu[q]; __nv_fp8x2_e4m3 ul,uh; ul.__x=(unsigned short)(uq&0xffffu); uh.__x=(unsigned short)(uq>>16);
          float2 ufl=__half22float2((__half2)ul), ufh=__half22float2((__half2)uh);
          u0[r]+=yq[0]*ufl.x; u1[r]+=yq[1]*ufl.y; u0[r]+=yq[2]*ufh.x; u1[r]+=yq[3]*ufh.y;
        }
      }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float gacc = g0[r]+g1[r], uacc = u0[r]+u1[r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) { gacc+=__shfl_down_sync(0xffffffffu,gacc,o); uacc+=__shfl_down_sync(0xffffffffu,uacc,o); }
      if (lane == 0) { const int j = j0 + r;
        if (j < MOE_INTER_RANK)
          a_glb[(size_t)slot * MOE_INTER_RANK + j] = silu(gacc * S[j]) * (uacc * S[MOE_INTER_RANK + j]);
      }
    }
  }
}
// down shard: h_io[o] += sel_w * s_d * <a_slot[0,192), down_o[0,192)>, R output channels/warp.
// Wd shard layout [HIDDEN, 192]: row o is the 192-wide (12 uint4) down contraction for channel o.
template <int R>
__device__ __forceinline__ void tp8_k5b_down_t(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot) {
  extern __shared__ float as[];                              // [nslot*192]
  const int na = nslot * MOE_INTER_RANK;
  for (int i = threadIdx.x; i < na; i += blockDim.x) as[i] = a_glb[i];
  __syncthreads();
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int norow = (HIDDEN + R - 1) / R;
  const int total = nslot * norow;
  const int nv    = MOE_INTER_RANK >> 4;                     // 12 at the shard
  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / norow;
    const int o0   = (item - slot * norow) * R;
    const int e    = sel_idx[slot];
    const float gw = sel_w[slot];
    const fp8*   W = Wd[e];
    const float* S = Wd_scale[e];
    const float* asl = as + (size_t)slot * MOE_INTER_RANK;
    const uint4* wv[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) wv[r] = reinterpret_cast<const uint4*>(W + (size_t)(o0 + r) * MOE_INTER_RANK);
    float a0[R], a1[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) { a0[r]=a1[r]=0.f; }
    for (int v = lane; v < nv; v += 32) {
      const float* yy = asl + (v << 4);
      uint4 p[R];
      #pragma unroll
      for (int r = 0; r < R; ++r) p[r] = wv[r][v];           // R loads in flight per lane
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        const unsigned* wu = reinterpret_cast<const unsigned*>(&p[r]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          unsigned wq = wu[q];
          __nv_fp8x2_e4m3 lo,hi; lo.__x=(unsigned short)(wq&0xffffu); hi.__x=(unsigned short)(wq>>16);
          float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi);
          const float* yq = yy + (q << 2);
          a0[r]+=yq[0]*fl.x; a1[r]+=yq[1]*fl.y; a0[r]+=yq[2]*fh.x; a1[r]+=yq[3]*fh.y;
        }
      }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
      float acc = a0[r] + a1[r];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
      if (lane == 0) { const int o = o0 + r; if (o < HIDDEN) atomicAdd(&h_io[o], gw * acc * S[o]); }
    }
  }
}
// Concrete entry points the launch sites + cudaFuncSetAttribute reference (template args baked in).
extern "C" __global__ void tp8_k5a_gateup(
    const float* __restrict__ y, const int* __restrict__ sel_idx,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    float* __restrict__ a_glb, int nslot) {
  tp8_k5a_gateup_t<TP8_K5_RA>(y, sel_idx, Wgu, Wgu_scale, a_glb, nslot);
}
extern "C" __global__ void tp8_k5b_down(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot) {
  tp8_k5b_down_t<TP8_K5_RB>(sel_idx, sel_w, Wd, Wd_scale, a_glb, h_io, nslot);
}

// Local fused residual add: h_dst[i] = h_src[i] + reduced[i]  (run AFTER the all-reduce of `reduced`).
extern "C" __global__ void tp8_residual_add(const float* __restrict__ h_src,
                                            const float* __restrict__ reduced,
                                            float* __restrict__ h_dst) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < HIDDEN; i += gridDim.x * blockDim.x)
    h_dst[i] = h_src[i] + reduced[i];
}

#if USE_GEMM
// forward decls of the existing kernels the GEMM helpers reuse (defined later in this TU).
extern "C" __global__ void tp8_k1_epilogue(
    const float* __restrict__ proj, const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ out_q, fp8* __restrict__ kv_k, fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale);
extern "C" __global__ void tp8_k4_select(
    const float* __restrict__ g_logits, const float* __restrict__ Wgate_scale,
    int* __restrict__ sel_idx, float* __restrict__ sel_w);
// =================================================================================================
// cuBLASLt fp8 GEMM LAUNCH HELPERS — drop-in replacements for the dense GEMVs (K2 attn unchanged).
// Each: (1) RMSNorm+quantize OR quantize the fp32 activation to a col-major [K,Mpad] fp8 X (col 0 =
// the B=1 token; cols 1..15 are zero-padded by the memset in alloc_rank), (2) cuBLASLt GEMM
// D[Mpad,N] = X^T @ W on tensor cores, (3) a tiny epilogue applying act_scale*Wscale (folding the
// activation-quant scale back in).  ZERO repack of W (already K-major).  Stream-capturable EXCEPT
// the K5 grouped path's host sel read (kept on the eager path).
// =================================================================================================
// K1 QKV: RMSNorm(h) -> fp8 X[HIDDEN], GEMM -> D[*, QKV_OUT_RANK], scale -> qkv_proj[QKV_OUT_RANK].
//   Then the EXISTING tp8_k1_epilogue (QK-norm + RoPE + KV write) runs unchanged on qkv_proj.
static void gemm_k1_launch(RankState& S, const float* h, cudaStream_t s) {
  const size_t smem = (size_t)HIDDEN * sizeof(float);
  gemm_rmsnorm_quant<<<1, 1024, smem, s>>>(h, S.w_in_norm, S.xq_hidden, S.act_scale, HIDDEN);
  S.p_qkv.run(S.xq_hidden, S.Wqkv, S.d_qkv, s);
  gemm_epi_scale<<<32, 256, 0, s>>>(S.d_qkv, S.Wqkv_scale, S.act_scale, S.qkv_proj,
                                    QKV_OUT_RANK, S.p_qkv.Mpad);
  tp8_k1_epilogue<<<1, 256, 0, s>>>(S.qkv_proj, S.q_norm, S.k_norm, S.rope_cos, S.rope_sin,
                                    S.out_q, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale);
}
// K3 O-proj: quantize attn_out -> fp8 X[Q_DIM_RANK], GEMM -> D[*,HIDDEN], scale -> attn_partial.
static void gemm_k3_launch(RankState& S, cudaStream_t s) {
  gemm_quant<<<1, 256, 0, s>>>(S.attn_out, S.xq_qdim, S.act_scale, Q_DIM_RANK);
  S.p_oproj.run(S.xq_qdim, S.Wo, S.d_oproj, s);
  gemm_epi_scale<<<32, 256, 0, s>>>(S.d_oproj, S.Wo_scale, S.act_scale, S.attn_partial,
                                    HIDDEN, S.p_oproj.Mpad);
}
// K4 router gate: RMSNorm(h) -> fp8 X[HIDDEN], GEMM -> D[*,N_EXPERTS], scale -> g_logits; then the
//   EXISTING tp8_k4_select (softmax + top-8 + Wgate_scale) runs unchanged.  (Reuses xq_hidden.)
static void gemm_k4_launch(RankState& S, const float* h, cudaStream_t s) {
  const size_t smem = (size_t)HIDDEN * sizeof(float);
  gemm_rmsnorm_quant<<<1, 1024, smem, s>>>(h, S.w_post_norm, S.xq_hidden, S.act_scale, HIDDEN);
  S.p_gate.run(S.xq_hidden, S.Wgate, S.d_gate, s);
  gemm_epi_gate<<<1, 128, 0, s>>>(S.d_gate, S.act_scale, S.g_logits, N_EXPERTS, S.p_gate.Mpad);
  tp8_k4_select<<<1, 32, 0, s>>>(S.g_logits, S.Wgate_scale, S.sel_idx, S.sel_w);
}
// Pack the routed experts' gate+up shards into the contiguous Wgu_pack buffer (8 D2D copies on the
// same device).  Only needed when the routing differs from the prebuilt fixed pack (correctness).
static void gemm_k5_pack_gateup(RankState& S, const int* sel_phys, cudaStream_t s) {
  const size_t shard_n = (size_t)2 * MOE_INTER_RANK * HIDDEN;   // [2*192, HIDDEN] fp8
  for (int slot = 0; slot < TOP_K; ++slot)
    CK(cudaMemcpyAsync(S.Wgu_pack + (size_t)slot * shard_n, S.Wgu_phys_h[sel_phys[slot]],
                       shard_n * sizeof(fp8), cudaMemcpyDeviceToDevice, s));
}
// K5 grouped MoE — HYBRID: gate+up = ONE packed cuBLASLt fp8 GEMM (8 experts' shards concatenated;
// shared X = quant(y) -> D[*, TOP_K*2*192]); SiLU epilogue -> a_glb.  down = the EXISTING fast
// multi-row-per-warp GEMV (k5b: a_glb[192] per slot -> moe_partial), which is already ~22% MBU and
// far cheaper than 8 tiny down GEMMs.  `Wgu_pack` must already hold the routed (or fixed) shards.
static void gemm_k5_launch(RankState& S, const float* y, const int* /*sel_phys*/, cudaStream_t s) {
  // ---- gate+up: quantize y, ONE GEMM over the packed [TOP_K*2*192, HIDDEN] weights ----
  gemm_quant<<<32, 256, 0, s>>>(y, S.xq_hidden, S.act_scale, HIDDEN);
  S.p_k5gu_pack.run(S.xq_hidden, S.Wgu_pack, S.d_k5gu, s);
  // SiLU epilogue: D layout is [Mpad, TOP_K*2*192], slot s at column block s*384 -> a_glb (uses the
  // per-routed-expert gate/up scales via sel_idx).
  gemm_epi_k5a<<<64, 256, 0, s>>>(S.d_k5gu, S.sel_idx, S.Wgu_scale_d, S.act_scale,
                                  S.a_glb, TOP_K, S.p_k5gu_pack.Mpad);
  // ---- down: the existing fast GEMV (graph-safe, ~22% MBU) ----
  tp8_k5b_down<<<S.k5.ctasB, S.k5.block, S.k5.smemB, s>>>(
      S.sel_idx, S.sel_w, S.Wd_d, S.Wd_scale_d, S.a_glb, S.moe_partial, TOP_K);
}
// lm_head: quantize hn -> fp8 X[HIDDEN], GEMM -> D[*,v_rows], scale+partial-argmax over the slice.
static void gemm_lmhead_launch(RankState& S, const float* hn, cudaStream_t s) {
  gemm_quant<<<1, 256, 0, s>>>(hn, S.xq_hidden, S.act_scale, HIDDEN);
  S.p_lm.run(S.xq_hidden, S.Wlm, S.d_lm, s);
  gemm_epi_lmhead_argmax<<<S.lm_blocks, 256, 0, s>>>(S.d_lm, S.Wlm_scale, S.act_scale,
      S.v_rows, S.v_off, S.p_lm.Mpad, S.block_max, S.block_arg);
}
#endif // USE_GEMM

// =================================================================================================
// FAST TP=8 ROUTER (the #1 measured bottleneck fix).
// -------------------------------------------------------------------------------------------------
// The stock k4_router runs the ENTIRE gate GEMV (128 experts x 4096 fp8) on a SINGLE CTA of 8 warps
// -> 1 of 132 SMs active, ~108 us/launch x 94 layers = ~10.2 ms/token = 59% of the kernels-only floor.
// It is GROSSLY occupancy-starved, not bandwidth bound (only 0.52 MB read).  Fix: spread the gate GEMV
// across the whole GPU with split-K (one warp per (expert, k-split)), then do the tiny softmax/top-8
// selection in a second 1-CTA kernel.
//
//   Kernel A (tp8_k4_gate_gemv): grid of CTAs, each stages y=RMSNorm(h) into smem (redundant per-CTA
//     RMSNorm — cheap, avoids a global y round-trip), then each warp computes a K-SLICE of one expert's
//     dot and atomicAdds the partial into a pre-zeroed global g_logits[expert].  With K4_KSPLIT=4 that
//     is 128*4=512 warps -> 128 CTAs of 4 warps -> fills all 132 SMs many times over.
//   Kernel B (tp8_k4_select): 1 CTA, reads g_logits[128], applies per-expert scale, fp32 softmax,
//     top-8 + renormalize -> sel_idx[8]/sel_w[8].  Identical selection math to k4_router (bit-for-bit).
// Per-channel scale is applied in B (after the cross-split atomic sum), so the split-K is exact.
// =================================================================================================
#ifndef TP8_K4_KSPLIT
#define TP8_K4_KSPLIT 8        // K-splits per expert (128 experts * 8 = 1024 warps; KSPLIT 4..32 flat)
#endif

// Kernel A: split-K gate GEMV.  Each warp owns (expert e, ksplit ks); atomicAdds its partial dot into
// g_logits[e] (pre-zeroed by the caller).  y is staged per-CTA via a block-wide RMSNorm of h.
// NOTE: a precompute-y-once variant was tried (microbench 16->13 us standalone) but in the 94x/token
// launch chain the EXTRA 1-CTA rmsnorm launch (~8.7 us launch overhead each) cost MORE than the
// per-CTA redundancy it removed, so the fused-norm form below is kept.
extern "C" __global__ void tp8_k4_gate_gemv(
    const float* __restrict__ h, const float* __restrict__ w_post_norm,
    const fp8* __restrict__ Wgate, float* __restrict__ g_logits) {
  extern __shared__ float ys[];                            // [HIDDEN] staged y
  // ---- block-wide RMSNorm(h) -> y (redundant across CTAs; cheap, kills the global y round-trip) ----
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

  // ---- split-K gate dot: warp (gwarp) -> expert e = gwarp / KSPLIT, ksplit ks = gwarp % KSPLIT ----
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int total = N_EXPERTS * TP8_K4_KSPLIT;
  const int nv    = HIDDEN >> 4;                            // 256 uint4 over HIDDEN
  const int chunk = (nv + TP8_K4_KSPLIT - 1) / TP8_K4_KSPLIT;
  for (int item = gwarp; item < total; item += nwarp) {
    const int e  = item / TP8_K4_KSPLIT;
    const int ks = item - e * TP8_K4_KSPLIT;
    const int v0 = ks * chunk, v1 = min(v0 + chunk, nv);
    const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(Wgate + (size_t)e * HIDDEN);
    float a0 = 0.f, a1 = 0.f;
    for (int v = v0 + lane; v < v1; v += 32) {
      uint4 p = wv[v];
      const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
      const float* yy = ys + (v << 4);
      #pragma unroll
      for (int q = 0; q < 4; ++q) {
        unsigned wq = wu[q];
        __nv_fp8x2_e4m3 lo, hi; lo.__x=(unsigned short)(wq&0xffffu); hi.__x=(unsigned short)(wq>>16);
        float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi);
        const float* yq = yy + (q << 2);
        a0 += yq[0]*fl.x; a1 += yq[1]*fl.y; a0 += yq[2]*fh.x; a1 += yq[3]*fh.y;
      }
    }
    float acc = a0 + a1;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&g_logits[e], acc);            // cross-split sum; scale applied in select
  }
}

// Kernel B: scale + softmax + top-8 + renormalize over g_logits[128] -> sel_idx[8]/sel_w[8].
// A whole warp loads/scales the 128 logits (4/lane) cooperatively; lane 0 does the tiny top-8.  The
// per-expert probability is computed ONCE (vs 8x in the stock k4_router), and a running "taken" mask
// removes the inner O(s) rescan.  Selection is argmax-equivalent to softmax-prob argmax (prob is a
// monotone function of the logit), so we top-8 on the logit directly and softmax only the 8 winners.
// Numerically identical SELECTION to k4_router; the 8 returned weights match to fp rounding.
extern "C" __global__ void tp8_k4_select(
    const float* __restrict__ g_logits, const float* __restrict__ Wgate_scale,
    int* __restrict__ sel_idx, float* __restrict__ sel_w) {
  __shared__ float logits[N_EXPERTS];
  const int lane = threadIdx.x & 31;
  for (int e = lane; e < N_EXPERTS; e += 32) logits[e] = g_logits[e] * Wgate_scale[e];
  __syncwarp();
  if (lane != 0) return;
  // full-128 softmax denom FIRST (matches k4_router exactly), before any masking.
  float mx = -FLT_MAX;
  for (int e = 0; e < N_EXPERTS; ++e) mx = fmaxf(mx, logits[e]);
  float sum = 0.f;
  for (int e = 0; e < N_EXPERTS; ++e) sum += __expf(logits[e] - mx);
  const float inv_sum = 1.f / sum;
  // top-8 by logit (monotone in prob); mask each pick to -inf in place so it isn't repicked.
  float logit_sel[TOP_K];
  for (int s = 0; s < TOP_K; ++s) {
    int bi = -1; float bv = -FLT_MAX;
    for (int e = 0; e < N_EXPERTS; ++e) if (logits[e] > bv) { bv = logits[e]; bi = e; }
    if (bi < 0) bi = s;
    sel_idx[s]   = bi;
    logit_sel[s] = logits[bi];
    logits[bi]   = -FLT_MAX;
  }
  // renormalize the 8 selected probs to sum 1 (norm_topk_prob=true).
  float chosen = 0.f;
  for (int s = 0; s < TOP_K; ++s) { float p = __expf(logit_sel[s] - mx) * inv_sum;
                                    sel_w[s] = p; chosen += p; }
  const float inv_chosen = 1.f / chosen;
  for (int s = 0; s < TOP_K; ++s) sel_w[s] *= inv_chosen;
}

// Capture-safe FAST router launch.  Spreads the gate GEMV over the whole GPU (split-K, 512 warps) then
// selects in a tiny 1-CTA kernel.  g_logits must be pre-zeroed each call (cheap 512 B memset).  No
// cudaFuncSetAttribute on the hot path (opted-in once in alloc_rank) -> stream-capturable.
static inline void tp8_k4_launch(const float* h, const float* w_post_norm,
                                 const fp8* Wgate, const float* Wgate_scale,
                                 int* sel_idx, float* sel_w,
                                 float* g_logits, cudaStream_t s) {
  const int block = 128, warps = block >> 5;                // 4 warps/CTA
  const int need_warps = N_EXPERTS * TP8_K4_KSPLIT;         // 1024 @ KSPLIT=8
  const int ctas = (need_warps + warps - 1) / warps;        // 256 CTAs (capped by sched to 132 SMs)
  const size_t smem = (size_t)HIDDEN * sizeof(float);
  CK(cudaMemsetAsync(g_logits, 0, N_EXPERTS * sizeof(float), s));
  tp8_k4_gate_gemv<<<ctas, block, smem, s>>>(h, w_post_norm, Wgate, g_logits);
  tp8_k4_select<<<1, 32, 0, s>>>(g_logits, Wgate_scale, sel_idx, sel_w);
}

// =================================================================================================
// Enqueue ONE TP=8 decode layer on a rank's stream.  Returns the residual buffer holding this layer's
// output.  TWO NCCL all-reduces (after O-proj, after MoE-down) stitch the row-parallel partials.
// =================================================================================================
//   The all-reduces MUST be issued by every rank in the SAME order on their own stream + comm; NCCL
//   matches them across ranks.  We bracket each with ncclGroupStart/End so the 8 per-rank launches are
//   coalesced into one collective.  Because K3/K5b produce a PARTIAL hidden, we all-reduce(SUM) the
//   partial, then add the residual locally (so the residual is added exactly once, not 8x).
// AR dispatch: NVLS multimem in-switch reduce when wired, else the original NCCL all-reduce.  `buf`
// MUST be the NVLS unicast view (S.attn_partial / S.moe_partial are repointed to it in alloc_rank when
// NVLS is active), and `elt_off` selects which packed half (NVLS_OFF_ATTN / NVLS_OFF_MOE).  The
// SUM is identical to NCCL's (exact fp32 add in the switch), so the correctness gate is preserved.
static inline void ar_sum_hidden(RankState& S, float* buf, int nvls_off, cudaStream_t s) {
#if USE_NVLS
  if (S.nvls && S.nvls->ready) {
    nvls_allreduce_launch(*S.nvls, nvls_off, s);
    return;
  }
#endif
  NK(ncclGroupStart());
  NK(ncclAllReduce(buf, buf, HIDDEN, ncclFloat32, ncclSum, S.comm, s));
  NK(ncclGroupEnd());
}

#if USE_GEMM
// Fixed slot->physical-shard map for the latency-proxy / graphed timing path (slot s -> shard s).
// Valid for timing (dummy weights; the routed values are meaningless in the proxy) and graph-safe
// (no per-layer D2H of sel_idx).  The CORRECTNESS path resolves the REAL routed mapping host-side.
static const int K5_SEL_PHYS_FIXED[TOP_K] = {0,1,2,3,4,5,6,7};
#endif

static float* enqueue_tp8_layer(RankState& S, float* h_src, float* h_dst) {
  cudaStream_t s = S.stream;

#if USE_GEMM
  // ---- K1: cuBLASLt fp8 QKV GEMM (RMSNorm+quant -> GEMM -> scale) + the unchanged QK-norm/RoPE epi.
  gemm_k1_launch(S, h_src, s);
  // ---- K2: flash-decode (unchanged) ----
  tp8_k2_launch(S, s);
  // ---- K3: cuBLASLt fp8 O-proj GEMM -> PARTIAL hidden (h_in=0; residual added post-all-reduce) ----
  CK(cudaMemsetAsync(S.attn_partial, 0, HIDDEN * sizeof(float), s));
  gemm_k3_launch(S, s);
  // ---- AR#1 ---- (out-of-place: result lands in S.attn_reduced; residual-add reads THAT)
  ar_sum_hidden(S, S.attn_partial, NVLS_OFF_ATTN, s);
  tp8_residual_add<<<32, 256, 0, s>>>(h_src, S.attn_reduced, h_dst);
  // ---- K4: cuBLASLt fp8 router gate GEMM -> g_logits -> select ----
  gemm_k4_launch(S, h_dst, s);
  // ---- K5: cuBLASLt fp8 GROUPED gate+up/down GEMM -> PARTIAL MoE-down hidden ----
  CK(cudaMemsetAsync(S.moe_partial, 0, HIDDEN * sizeof(float), s));
  gemm_k5_launch(S, h_dst, K5_SEL_PHYS_FIXED, s);   // proxy/graphed: fixed slot->shard (graph-safe)
  // ---- AR#2 ----
  ar_sum_hidden(S, S.moe_partial, NVLS_OFF_MOE, s);
  tp8_residual_add<<<32, 256, 0, s>>>(h_dst, S.moe_reduced, h_dst);
  return h_dst;
#else
  // ---- K1: sharded RMSNorm + QKV GEMV (this rank's 2048 rows) + QK-norm + RoPE + KV write ----
  tp8_k1_launch(S, h_src, s);
  // ---- K2: flash-decode over this rank's Q_HEADS_RANK=8 heads (KV is the replicated full cache) ----
  tp8_k2_launch(S, s);
  // ---- K3: O-proj on the [HIDDEN, Q_DIM_RANK] column-shard -> PARTIAL hidden (NO residual add yet) ---
  CK(cudaMemsetAsync(S.attn_partial, 0, HIDDEN * sizeof(float), s));   // h_in = 0 -> pure partial
  tp8_k3_launch(S, s);
  // ---- AR#1: all-reduce(SUM) the partial O-proj output across the 8 ranks -> full O-proj ----
  ar_sum_hidden(S, S.attn_partial, NVLS_OFF_ATTN, s);
  tp8_residual_add<<<32, 256, 0, s>>>(h_src, S.attn_reduced, h_dst);
  // ---- K4: router (REPLICATED) -> sel_idx[8], sel_w[8] ----
  tp8_k4_launch(h_dst, S.w_post_norm, S.Wgate, S.Wgate_scale, S.sel_idx, S.sel_w, S.g_logits, s);
  // ---- K5: sharded gate+up (192) then sharded down -> PARTIAL MoE-down hidden ----
  CK(cudaMemsetAsync(S.moe_partial, 0, HIDDEN * sizeof(float), s));    // accumulate partial from 0
  tp8_k5a_gateup<<<S.k5.ctasA, S.k5.block, S.k5.smemA, s>>>(
      h_dst, S.sel_idx, S.Wgu_d, S.Wgu_scale_d, S.a_glb, TOP_K);
  tp8_k5b_down<<<S.k5.ctasB, S.k5.block, S.k5.smemB, s>>>(
      S.sel_idx, S.sel_w, S.Wd_d, S.Wd_scale_d, S.a_glb, S.moe_partial, TOP_K);
  // ---- AR#2: all-reduce(SUM) the partial MoE-down across ranks -> full MoE contribution ----
  ar_sum_hidden(S, S.moe_partial, NVLS_OFF_MOE, s);
  tp8_residual_add<<<32, 256, 0, s>>>(h_dst, S.moe_reduced, h_dst);
  return h_dst;
#endif
}

// ---- enqueue the FULL TP=8 step on a rank: 94 layers + final norm + sharded lm_head + argmax ----
// NCCL CORRECTNESS CONTRACT (single process, ncclCommInitAll, 8 ranks):
//   ncclAllReduce + ncclGroupStart/End ENQUEUE onto this rank's stream.  NCCL matches the i-th
//   collective issued on comm-rank-r with the i-th on every other comm-rank.  Every rank here issues
//   the IDENTICAL ordered sequence of 189 collectives (AR#1, AR#2 per layer, then the head argmax-max),
//   so the matching is unambiguous.
//   DRIVER REQUIREMENT (one comm per host thread — see run_all_ranks / main):
//     ncclGroupEnd() on a single-thread driver issuing per-rank groups SEQUENTIALLY can BLOCK: NVIDIA's
//     Group-Calls doc requires all ranks' i-th collective to be in ONE group when one thread owns
//     multiple comms.  We instead give EACH rank its OWN host thread (each owns exactly one
//     communicator), which is the documented exception — per-rank individual groups are then legal and
//     ncclGroupEnd cannot block waiting on peers, because the 8 threads reach their i-th collective
//     concurrently.  run_all_ranks() launches 8 std::threads, each calling fn(R[r]) then syncing its
//     own stream, and joins them; never enqueue rank 0's whole step before rank 1's on one thread.
static void enqueue_tp8_step(RankState& S) {
  cudaStream_t s = S.stream;
  float* cur = S.h_a;
  float* nxt = S.h_b;
  for (int layer = 0; layer < N_LAYERS; ++layer) {
    float* out = enqueue_tp8_layer(S, cur, nxt);
    cur = out;
    nxt = (cur == S.h_a) ? S.h_b : S.h_a;
  }
  // final RMSNorm (replicated) + VOCAB-sharded lm_head + local argmax -> (rank_max, rank_arg).
  tp8_final_norm<<<1, 256, 0, s>>>(cur, S.w_final_norm, S.hn);
#if USE_GEMM
  gemm_lmhead_launch(S, S.hn, s);                  // cuBLASLt fp8 lm_head GEMM + scale + partial argmax
#else
  const size_t lm_smem = (size_t)HIDDEN * sizeof(float);
  tp8_lmhead_argmax_partial<<<S.lm_blocks, 256, lm_smem, s>>>(
      S.hn, S.Wlm, S.Wlm_scale, S.v_rows, S.v_off, S.block_max, S.block_arg);
#endif
  tp8_argmax_final<<<1, 32, 0, s>>>(S.block_max, S.block_arg, S.lm_blocks, S.rank_max, S.rank_arg);
  // Cross-rank argmax: all-reduce-MAX over the per-rank best logits; the matching token id is then
  // resolved host-side from the gathered (max,arg) pairs (a 2-int all-gather would also work).  We
  // all-reduce the logit so every rank learns the global max; the host picks the arg of the winner.
  NK(ncclGroupStart());
  NK(ncclAllReduce(S.rank_max, S.rank_max, 1, ncclFloat32, ncclMax, S.comm, s));
  NK(ncclGroupEnd());
}

// =================================================================================================
// KERNELS-ONLY GRAPH SEGMENTS (the robust fallback when NCCL-in-graph replay is slow on this build).
// -------------------------------------------------------------------------------------------------
// The two all-reduces/layer chop the per-token stream into kernel-only segments.  We capture the
// kernels of ONE layer into two segment graphs (segA = K1,K2,K3 -> attn_partial; segB = residual_add,
// K4,K5 -> moe_partial) plus a head segment, then at replay time the host launches:
//   [segA_graph, eager AR#1, segB_graph, eager AR#2] x 94  +  [head_graph, eager head-AR-max]
// so every layer's ~9 kernel launches collapse to 2 graph launches, while the 189 NCCL all-reduces
// stay EAGER (NCCL's fast path).  Host ops/token: 2*94 graph launches + 188 ARs + 2 head = ~378
// (vs ~1100 eager), removing the bulk of the launch overhead without the slow captured-NCCL path.
//
// LATENCY-PROXY NOTE: a captured graph pins fixed buffer pointers, so the graphed path drops the
// per-layer residual ping-pong and reuses FIXED h_a (in) / h_b (out) every layer.  In this proxy the
// residual values are meaningless (they grow to nan — explicitly harmless), and the kernel shapes /
// grids / per-token HBM byte volume are byte-identical to the eager path, so the measured us/token is
// faithful to the launch-structure being benchmarked.
// =================================================================================================
// segA: K1 -> K2 -> memset(attn_partial) -> K3.  Reads h_in, writes attn_partial (partial O-proj).
static void enqueue_tp8_segA(RankState& S, float* h_in, cudaStream_t s) {
#if USE_GEMM
  gemm_k1_launch(S, h_in, s);
  tp8_k2_launch(S, s);
  CK(cudaMemsetAsync(S.attn_partial, 0, HIDDEN * sizeof(float), s));
  gemm_k3_launch(S, s);
#else
  tp8_k1_launch(S, h_in, s);
  tp8_k2_launch(S, s);
  CK(cudaMemsetAsync(S.attn_partial, 0, HIDDEN * sizeof(float), s));
  tp8_k3_launch(S, s);
#endif
}
// segB: residual_add(h_in + attn_reduced -> h_out) -> K4 -> memset(moe_partial) -> K5a -> K5b.
//   attn_reduced is the AR#1 result (eager AR ran before this segment replays; out-of-place -> OUT half).
static void enqueue_tp8_segB(RankState& S, float* h_in, float* h_out, cudaStream_t s) {
  tp8_residual_add<<<32, 256, 0, s>>>(h_in, S.attn_reduced, h_out);
#if USE_GEMM
  gemm_k4_launch(S, h_out, s);
  CK(cudaMemsetAsync(S.moe_partial, 0, HIDDEN * sizeof(float), s));
  gemm_k5_launch(S, h_out, K5_SEL_PHYS_FIXED, s);   // graph-safe fixed slot->shard
#else
  tp8_k4_launch(h_out, S.w_post_norm, S.Wgate, S.Wgate_scale, S.sel_idx, S.sel_w, S.g_logits, s);
  CK(cudaMemsetAsync(S.moe_partial, 0, HIDDEN * sizeof(float), s));
  tp8_k5a_gateup<<<S.k5.ctasA, S.k5.block, S.k5.smemA, s>>>(
      h_out, S.sel_idx, S.Wgu_d, S.Wgu_scale_d, S.a_glb, TOP_K);
  tp8_k5b_down<<<S.k5.ctasB, S.k5.block, S.k5.smemB, s>>>(
      S.sel_idx, S.sel_w, S.Wd_d, S.Wd_scale_d, S.a_glb, S.moe_partial, TOP_K);
#endif
}
// head: residual_add(h_in + moe_reduced -> h_in) -> final norm -> lm_head -> argmax_final.
//   moe_reduced is the AR#2 result (eager AR ran before this segment replays; out-of-place -> OUT half).
//   Writes rank_max/rank_arg; the eager head-AR-max then picks the global token.
static void enqueue_tp8_seghead(RankState& S, float* h_in, cudaStream_t s) {
  tp8_residual_add<<<32, 256, 0, s>>>(h_in, S.moe_reduced, h_in);
  tp8_final_norm<<<1, 256, 0, s>>>(h_in, S.w_final_norm, S.hn);
#if USE_GEMM
  gemm_lmhead_launch(S, S.hn, s);
#else
  const size_t lm_smem = (size_t)HIDDEN * sizeof(float);
  tp8_lmhead_argmax_partial<<<S.lm_blocks, 256, lm_smem, s>>>(
      S.hn, S.Wlm, S.Wlm_scale, S.v_rows, S.v_off, S.block_max, S.block_arg);
#endif
  tp8_argmax_final<<<1, 32, 0, s>>>(S.block_max, S.block_arg, S.lm_blocks, S.rank_max, S.rank_arg);
}
// Capture a kernel-only segment (no NCCL, no setattr inside) into an instantiated graph exec.
template <typename Fn>
static cudaGraphExec_t capture_segment(cudaStream_t s, Fn fn) {
  // warm-up the segment once outside capture (first-touch), then capture.
  fn(s); CK(cudaStreamSynchronize(s));
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  fn(s);
  cudaGraph_t g = nullptr; CK(cudaStreamEndCapture(s, &g));
  cudaGraphExec_t e = nullptr; CK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
  cudaGraphDestroy(g);
  return e;
}
// Replay ONE token in the kernels-only-graph + eager-AR mode (host launches segments, eager ARs).
static void replay_tp8_step_kgraph(RankState& S) {
  cudaStream_t s = S.stream;
  for (int layer = 0; layer < N_LAYERS; ++layer) {
    CK(cudaGraphLaunch(S.exec_segA, s));                          // K1,K2,K3 -> attn_partial
    ar_sum_hidden(S, S.attn_partial, NVLS_OFF_ATTN, s);          // AR#1 (NVLS or NCCL)
    CK(cudaGraphLaunch(S.exec_segB, s));                          // resid,K4,K5 -> moe_partial
    ar_sum_hidden(S, S.moe_partial, NVLS_OFF_MOE, s);            // AR#2 (NVLS or NCCL)
  }
  CK(cudaGraphLaunch(S.exec_seghead, s));                         // resid,norm,lm_head,argmax
  NK(ncclGroupStart());
  NK(ncclAllReduce(S.rank_max, S.rank_max, 1, ncclFloat32, ncclMax, S.comm, s));                 // eager head AR
  NK(ncclGroupEnd());
}

// =================================================================================================
// A tiny reusable sense-reversing spin barrier for the N rank-threads.  Used to make all 8 ranks
// ENTER and EXIT cudaStreamBeginCapture/EndCapture CONCURRENTLY — the NCCL collectives recorded
// during capture need every rank live, or ncclGroupEnd would block / capture would be inconsistent.
// =================================================================================================
struct SpinBarrier {
  int n;
  std::atomic<int> count{0};
  std::atomic<int> sense{0};
  explicit SpinBarrier(int n_) : n(n_) {}
  void wait() {
    int s = sense.load(std::memory_order_acquire);
    if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
      count.store(0, std::memory_order_release);
      sense.store(s ^ 1, std::memory_order_release);          // release all waiters
    } else {
      while (sense.load(std::memory_order_acquire) == s) { /* spin */ }
    }
  }
};

// =================================================================================================
// MULTI-THREAD DRIVER — one host thread per rank/communicator (the NCCL deadlock fix).
// =================================================================================================
// Each thread sets ITS device, runs fn(R[r]) (which enqueues that rank's collectives in its own
// per-rank groups), then synchronizes its own stream — so all 8 ranks reach their i-th collective
// concurrently and ncclGroupEnd never blocks on un-enqueued peers.  This is the documented
// one-comm-per-thread NCCL usage; it replaces the previous single-thread "enqueue all 8, then sync"
// loop that could deadlock at ncclGroupEnd (see CONTRACT note above).
template <typename Fn>
static void run_all_ranks(std::vector<RankState>& R, Fn fn, bool sync_each = true) {
  std::vector<std::thread> th;
  th.reserve(R.size());
  for (size_t r = 0; r < R.size(); ++r) {
    th.emplace_back([&R, r, fn, sync_each]() {
      CK(cudaSetDevice(R[r].dev));
      fn(R[r]);
      if (sync_each) CK(cudaStreamSynchronize(R[r].stream));
    });
  }
  for (auto& t : th) t.join();
}

// =================================================================================================
// Sharded launch helpers — launch the existing k1/k2/k3 device kernels on the rank's sub-ranges.
// =================================================================================================
// K1: the stock k1_qkv_gemv loops `o < QKV_OUT` (a common.cuh constant), so it cannot be reused on a
// sub-range without reading the full 9216 rows.  These two kernels mirror k1_qkv_gemv / k1_epilogue
// EXACTLY (same RMSNorm, same coalesced fp8 warp-dot, same per-head QK-norm+RoPE), but bound the GEMV
// loop to this rank's QKV_OUT_RANK=2048 rows and use the sharded [8Q|4K|4V] row->head map — so the
// per-rank read is ~1/8 of Q + the (replicated) KV, exactly the TP=8 attention shard.
extern "C" __global__ void tp8_k1_qkv_gemv(
    const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale,
    float* __restrict__ proj /*[QKV_OUT_RANK]*/) {
  extern __shared__ float xs[];                              // [HIDDEN]
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
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int o = gwarp; o < QKV_OUT_RANK; o += nwarp) {        // <-- SHARDED row bound (2048, ~1/8 of 9216 Q + full KV)
    float r = tp8_warp_dot(Wqkv + (size_t)o * HIDDEN, xs, HIDDEN, lane);
    if (lane == 0) proj[o] = r * Wqkv_scale[o];
  }
}
// K1 epilogue for the shard: rows [0,Q_DIM_RANK) are this rank's 8 Q heads; [Q_DIM_RANK, +KV_DIM) are
// the 4 K heads; [+KV_DIM, +KV_DIM) the 4 V heads.  Mirrors k1_epilogue's per-head QK-norm/RoPE/write.
extern "C" __global__ void tp8_k1_epilogue(
    const float* __restrict__ proj,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ out_q, fp8* __restrict__ kv_k, fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale) {
  const int HEAD_ROWS = Q_HEADS_RANK + 2 * N_KV_HEADS;       // 8 Q + 4 K + 4 V = 16
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int row = gwarp; row < HEAD_ROWS; row += nwarp) {
    const int is_q = (row < Q_HEADS_RANK);
    const int is_k = (!is_q && row < Q_HEADS_RANK + N_KV_HEADS);
    int proj_base, head_local;
    if (is_q)      { head_local = row;                          proj_base = head_local * HEAD_DIM; }
    else if (is_k) { head_local = row - Q_HEADS_RANK;           proj_base = Q_DIM_RANK + head_local*HEAD_DIM; }
    else           { head_local = row - Q_HEADS_RANK - N_KV_HEADS; proj_base = Q_DIM_RANK + KV_DIM + head_local*HEAD_DIM; }

    float chan[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) chan[c] = proj[proj_base + c * 32 + lane];

    if (!is_q && !is_k) {                                     // V head -> quantize into the cache slot
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
    if (is_q) {                                               // -> this rank's local out_q slice
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++)
        out_q[head_local * HEAD_DIM + c * 32 + lane] = roped[c];
    } else {                                                  // K head -> quantize into cache slot
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
static void tp8_k1_launch(RankState& S, const float* h, cudaStream_t s) {
  // S.qkv_proj is a per-rank device scratch (allocated in alloc_rank on this rank's device).
  // NOTE: the cudaFuncSetAttribute opt-in for the >48KB dynamic smem is done ONCE in alloc_rank
  //   (off the hot path, outside any graph-capture region) — calling it here would abort capture.
  const int blockA = 256, warpsA = blockA >> 5;
  int needA = (QKV_OUT_RANK + warpsA - 1) / warpsA;           // 2048/8 = 256 CTAs for 1 warp/row
  int ctasA = needA < 264 ? needA : 264;
  const size_t smemA = (size_t)HIDDEN * sizeof(float);
  tp8_k1_qkv_gemv<<<ctasA, blockA, smemA, s>>>(h, S.w_in_norm, S.Wqkv, S.Wqkv_scale, S.qkv_proj);
  // epilogue: 16 head rows -> 1 small CTA
  tp8_k1_epilogue<<<1, 256, 0, s>>>(S.qkv_proj, S.q_norm, S.k_norm, S.rope_cos, S.rope_sin,
                                    S.out_q, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale);
}

// K2: flash-decode over THIS RANK'S 8 Q heads.  The stock k2 kernels loop `qh < N_Q_HEADS`; we launch
// a thin sharded variant that maps the local head index (0..7) to the GLOBAL kv head via the global
// head id `S.rank*Q_HEADS_RANK + local`, then GQA-broadcasts to one of the 4 (replicated) KV heads.
extern "C" __global__ void tp8_k2_partial(
    const float* __restrict__ q /*[Q_DIM_RANK]*/,
    const fp8*  __restrict__ kv_k, const fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale,
    int ctx_len, int n_splits, int rank,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc) {
  const int lane  = threadIdx.x & 31;
  const int wid   = threadIdx.x >> 5;
  const int lqh   = blockIdx.y * (blockDim.x >> 5) + wid;     // local query head 0..7
  if (lqh >= Q_HEADS_RANK) return;
  const int gqh   = rank * Q_HEADS_RANK + lqh;                // global query head 0..63
  const int split = blockIdx.x;
  const int kvh   = gqh / GQA_GROUP;                          // GQA broadcast -> KV head 0..3 (replicated)
  const int chunk = (ctx_len + n_splits - 1) / n_splits;
  const int t0 = split * chunk, t1 = min(t0 + chunk, ctx_len);
  const float scale = rsqrtf((float)HEAD_DIM);
  const int kv_base = kvh * HEAD_DIM;
  const int c0 = kv_base + lane * K2_VPL;
  float qreg[K2_VPL], ksc[K2_VPL], vsc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) {
    qreg[c] = q[lqh * HEAD_DIM + lane * K2_VPL + c];          // <-- LOCAL q index (this rank's slice)
    ksc[c]  = kv_k_scale ? kv_k_scale[c0 + c] : 1.f;
    vsc[c]  = kv_v_scale ? kv_v_scale[c0 + c] : 1.f;
  }
  float m = -FLT_MAX, l = 0.f, acc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) acc[c] = 0.f;
  const unsigned* __restrict__ k32 = reinterpret_cast<const unsigned*>(kv_k);
  const unsigned* __restrict__ v32 = reinterpret_cast<const unsigned*>(kv_v);
  const int row_words = KV_DIM / 4, base_words = kv_base / 4;
  for (int t = t0; t < t1; t++) {
    const unsigned* krow = k32 + (size_t)t * row_words + base_words;
    float kv[K2_VPL]; k2_load4(krow, lane, ksc, kv);
    float p = 0.f;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) p += qreg[c] * kv[c];
    float sft = k2_warp_sum(p) * scale;
    float m_new = fmaxf(m, sft);
    float corr  = __expf(m - m_new);
    float pexp  = __expf(sft - m_new);
    l = l * corr + pexp;
    const unsigned* vrow = v32 + (size_t)t * row_words + base_words;
    float vv[K2_VPL]; k2_load4(vrow, lane, vsc, vv);
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c] * corr + pexp * vv[c];
    m = m_new;
  }
  const size_t pidx = (size_t)lqh * n_splits + split;        // partials indexed by LOCAL head
  if (lane == 0) { part_m[pidx] = m; part_l[pidx] = l; }
  float* ao = part_acc + pidx * HEAD_DIM + lane * K2_VPL;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) ao[c] = acc[c];
}
// K2 reduce — PARALLEL: ONE CTA per (local) q_head, K2R_WARPS warps tree-combine the n_splits partials.
// The old reduce launched only Q_HEADS_RANK/4 = 2 CTAs on 132 SMs, each warp SERIALLY looping all 64
// splits with a dependent log-sum-exp chain reading 64x128 floats from HBM -> it was SLOWER than the
// well-parallelized partial pass (the K2 bottleneck).  Now each CTA's W warps each fold a STRIDED
// subset of splits into a register (m,l,acc[4/lane]); warp 0 folds the W warp-partials in smem.  Same
// log-sum-exp math, bit-identical output (verified in microbench: max|serial-parallel| < 1e-10), but
// the dependent chain per warp is n_splits/W instead of n_splits, and the W=8 warps give intra-CTA
// latency hiding.  Microbench (ctx 4096, nsp 64, 8 heads): combined K2 33.6us -> 20.5us (1.6x).
#ifndef K2R_WARPS
#define K2R_WARPS 32   // Fix B: wide reduce (each warp folds n_splits/32 partials) — best with 192 splits
#endif
extern "C" __global__ void tp8_k2_reduce(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, int n_splits, float* __restrict__ attn_out /*[Q_DIM_RANK]*/) {
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  const int W    = blockDim.x >> 5;             // warps per CTA = parallel split-folds
  const int lqh  = blockIdx.x;                  // ONE CTA per local q_head
  if (lqh >= Q_HEADS_RANK) return;
  // each warp folds a strided subset {wid, wid+W, ...} of the n_splits partials.
  float m = -FLT_MAX, l = 0.f, acc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) acc[c] = 0.f;
  for (int sp = wid; sp < n_splits; sp += W) {
    const size_t pidx = (size_t)lqh * n_splits + sp;
    float ms = part_m[pidx], ls = part_l[pidx];
    if (ls <= 0.f) continue;
    const float* ai = part_acc + pidx * HEAD_DIM + lane * K2_VPL;
    float m_new = fmaxf(m, ms);
    float corr_o = __expf(m - m_new), corr_s = __expf(ms - m_new);
    l = l * corr_o + ls * corr_s;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c] * corr_o + ai[c] * corr_s;
    m = m_new;
  }
  // fold the W warp-partials in smem (warp 0).  smem: m[W] + l[W] + acc[W*HEAD_DIM].
  extern __shared__ float k2r_sm[];
  float* sm_m = k2r_sm;                 // [W]
  float* sm_l = sm_m + W;               // [W]
  float* sm_a = sm_l + W;               // [W*HEAD_DIM]
  if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
  float* sao = sm_a + (size_t)wid * HEAD_DIM + lane * K2_VPL;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) sao[c] = acc[c];
  __syncthreads();
  if (wid == 0) {
    float rm = -FLT_MAX, rl = 0.f, racc[K2_VPL];
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) racc[c] = 0.f;
    for (int w = 0; w < W; w++) {
      float ms = sm_m[w], ls = sm_l[w];
      if (ls <= 0.f) continue;
      const float* ai = sm_a + (size_t)w * HEAD_DIM + lane * K2_VPL;
      float mn = fmaxf(rm, ms), co = __expf(rm - mn), cs = __expf(ms - mn);
      rl = rl * co + ls * cs;
      #pragma unroll
      for (int c = 0; c < K2_VPL; c++) racc[c] = racc[c] * co + ai[c] * cs;
      rm = mn;
    }
    float inv = (rl > 0.f) ? (1.f / rl) : 0.f;
    float* o = attn_out + lqh * HEAD_DIM + lane * K2_VPL;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) o[c] = racc[c] * inv;
  }
}
// =================================================================================================
// FIX B — FUSED TP=8 flash-decode: partial + reduce in ONE launch, partials on-chip (no HBM round-trip,
// no 2nd kernel).  The TP=8 shard has a STRUCTURAL property the generic K2 ignored: a rank owns 8 Q
// heads [8r,8r+8) which ALL map to the SINGLE KV head (8r)/GQA_GROUP (GQA_GROUP=16 > 8) -> every CTA
// reads ONE KV head's cache.  We load each KV row's K and V ONCE per warp and reuse the dequant across
// the head's online-softmax recurrence; W warps split the [0,ctx) time range W ways (short dependent
// chains -> latency hiding); the W partials combine in shared memory.  grid = Q_HEADS_RANK (8) CTAs *
// W warps; high W keeps the 132 SMs busy while each warp's slice stays long enough to amortize setup.
//
// vs the old 2-kernel split-K path this removes: (a) the 2nd kernel launch (tp8_k2_reduce ~5.7us), and
// (b) the part_m/l/acc HBM write+read (n_splits*8*130 floats round-trip).  Same online-softmax math,
// bit-identical to the serial reference (verified by the engine correctness gate post-attn residual).
// =================================================================================================
#ifndef TP8_K2F_WARPS
#define TP8_K2F_WARPS 16   // warps/CTA = time-splits; 8 heads * 16 = 128 warps over 132 SMs, each warp
                           // streams ctx/16 ~256 timesteps (4096) -> short dependent chain, full latency hide
#endif
extern "C" __global__ void tp8_k2_fused(
    const float* __restrict__ q /*[Q_DIM_RANK]*/,
    const fp8*  __restrict__ kv_k, const fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale,
    int ctx_len, int rank, float* __restrict__ attn_out /*[Q_DIM_RANK]*/) {
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  const int W    = blockDim.x >> 5;             // warps per CTA = number of time-splits
  const int lqh  = blockIdx.x;                  // ONE CTA per LOCAL q_head 0..Q_HEADS_RANK-1
  if (lqh >= Q_HEADS_RANK) return;
  const int gqh  = rank * Q_HEADS_RANK + lqh;   // global q_head 0..63
  const int kvh  = gqh / GQA_GROUP;             // GQA broadcast -> the rank's single KV head
  const float scale = rsqrtf((float)HEAD_DIM);
  const int kv_base = kvh * HEAD_DIM;
  const int c0 = kv_base + lane * K2_VPL;

  float qreg[K2_VPL], ksc[K2_VPL], vsc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) {
    qreg[c] = q[lqh * HEAD_DIM + lane * K2_VPL + c];           // LOCAL q index (this rank's slice)
    ksc[c]  = kv_k_scale ? kv_k_scale[c0 + c] : 1.f;
    vsc[c]  = kv_v_scale ? kv_v_scale[c0 + c] : 1.f;
  }

  // this warp's contiguous KV time slice [t0,t1).
  const int chunk = (ctx_len + W - 1) / W;
  const int t0 = wid * chunk, t1 = min(t0 + chunk, ctx_len);

  float m = -FLT_MAX, l = 0.f, acc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) acc[c] = 0.f;

  const unsigned* __restrict__ k32 = reinterpret_cast<const unsigned*>(kv_k);
  const unsigned* __restrict__ v32 = reinterpret_cast<const unsigned*>(kv_v);
  const int row_words = KV_DIM / 4, base_words = kv_base / 4;

  // 2x time-unroll: two independent coalesced K (and V) loads in flight per iteration (hide HBM latency).
  int t = t0;
  for (; t + 1 < t1; t += 2) {
    float kv0[K2_VPL], kv1[K2_VPL];
    k2_load4(k32 + (size_t)t       * row_words + base_words, lane, ksc, kv0);
    k2_load4(k32 + (size_t)(t + 1) * row_words + base_words, lane, ksc, kv1);
    float p0 = 0.f, p1 = 0.f;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) { p0 += qreg[c]*kv0[c]; p1 += qreg[c]*kv1[c]; }
    float s0 = k2_warp_sum(p0) * scale, s1 = k2_warp_sum(p1) * scale;
    float vv0[K2_VPL], vv1[K2_VPL];
    k2_load4(v32 + (size_t)t       * row_words + base_words, lane, vsc, vv0);
    k2_load4(v32 + (size_t)(t + 1) * row_words + base_words, lane, vsc, vv1);
    float mn = fmaxf(m, s0), corr = __expf(m - mn), pe = __expf(s0 - mn);
    l = l * corr + pe;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c]*corr + pe*vv0[c];
    m = mn;
    mn = fmaxf(m, s1); corr = __expf(m - mn); pe = __expf(s1 - mn);
    l = l * corr + pe;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c]*corr + pe*vv1[c];
    m = mn;
  }
  for (; t < t1; t++) {
    float kv[K2_VPL];
    k2_load4(k32 + (size_t)t * row_words + base_words, lane, ksc, kv);
    float p = 0.f;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) p += qreg[c]*kv[c];
    float s = k2_warp_sum(p) * scale;
    float vv[K2_VPL];
    k2_load4(v32 + (size_t)t * row_words + base_words, lane, vsc, vv);
    float mn = fmaxf(m, s), corr = __expf(m - mn), pe = __expf(s - mn);
    l = l * corr + pe;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c]*corr + pe*vv[c];
    m = mn;
  }

  // combine the W warp-partials in shared memory (warp 0).  smem: m[W] + l[W] + acc[W*HEAD_DIM].
  extern __shared__ float k2f_sm[];
  float* sm_m = k2f_sm;            // [W]
  float* sm_l = sm_m + W;          // [W]
  float* sm_a = sm_l + W;          // [W*HEAD_DIM]
  if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
  float* sao = sm_a + (size_t)wid * HEAD_DIM + lane * K2_VPL;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) sao[c] = acc[c];
  __syncthreads();
  if (wid == 0) {
    float rm = -FLT_MAX, rl = 0.f, racc[K2_VPL];
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) racc[c] = 0.f;
    for (int w = 0; w < W; w++) {
      float ms = sm_m[w], ls = sm_l[w];
      if (ls <= 0.f) continue;
      const float* ai = sm_a + (size_t)w * HEAD_DIM + lane * K2_VPL;
      float mn = fmaxf(rm, ms), co = __expf(rm - mn), cs = __expf(ms - mn);
      rl = rl * co + ls * cs;
      #pragma unroll
      for (int c = 0; c < K2_VPL; c++) racc[c] = racc[c]*co + ai[c]*cs;
      rm = mn;
    }
    float inv = (rl > 0.f) ? (1.f / rl) : 0.f;
    float* o = attn_out + lqh * HEAD_DIM + lane * K2_VPL;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) o[c] = racc[c] * inv;
  }
}

// Pick warps/CTA (= time-splits) for the fused kernel: 8 heads * W warps over the SMs, slice long enough.
static inline int tp8_k2f_warps(int ctx_len) {
  int w = TP8_K2F_WARPS;
  if (ctx_len > 16384) w = 24;
  if (ctx_len <= 512)  w = 4;
  int max_by_chunk = (ctx_len + 31) / 32;       // keep each warp's slice >= ~32 timesteps
  if (w > max_by_chunk) w = max_by_chunk;
  if (w < 1) w = 1;
  return w;
}

// ---- TP=8 K2 toggle: USE_K2_FUSED=1 the single-launch fused kernel (one CTA/head -> only 8 CTAs,
//      occupancy-starved on 132 SMs, MEASURED SLOWER: 68us vs 21us).  Default 0 = the split-K 2-kernel
//      path (128 CTAs in the partial -> fills the SMs; the win is from MORE splits, not fewer launches).
#ifndef USE_K2_FUSED
#define USE_K2_FUSED 0
#endif
static void tp8_k2_launch(RankState& S, cudaStream_t s) {
#if USE_K2_FUSED
  const int warps = tp8_k2f_warps(S.ctx_len);
  const int block = warps * 32;
  const size_t smem = (size_t)(2 * warps + warps * HEAD_DIM) * sizeof(float);   // m[W]+l[W]+acc[W*128]
  tp8_k2_fused<<<Q_HEADS_RANK, block, smem, s>>>(
      S.out_q, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale, S.ctx_len, S.rank, S.attn_out);
#else
  const int warps_per_cta = 4, block = warps_per_cta * 32;
  dim3 gP(S.n_splits, (Q_HEADS_RANK + warps_per_cta - 1) / warps_per_cta);
  tp8_k2_partial<<<gP, block, 0, s>>>(S.out_q, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale,
                                      S.ctx_len, S.n_splits, S.rank, S.part_m, S.part_l, S.part_acc);
  // parallel reduce: ONE CTA per head, S.k2r_warps warps; smem = (2*W + W*HEAD_DIM) floats.  More warps
  // = each folds n_splits/W partials (shorter dependent log-sum-exp chain) — matters when n_splits is high.
  const int rw = S.k2r_warps, rblock = rw * 32;
  const size_t rsmem = (size_t)(2 * rw + rw * HEAD_DIM) * sizeof(float);
  tp8_k2_reduce<<<Q_HEADS_RANK, rblock, rsmem, s>>>(S.part_m, S.part_l, S.part_acc, S.n_splits, S.attn_out);
#endif
}

// K3: O-proj on the [HIDDEN, Q_DIM_RANK] column-shard -> PARTIAL hidden (h_in = ZERO, see caller).
extern "C" __global__ void tp8_k3_oproj(
    const float* __restrict__ attn_out /*[Q_DIM_RANK]*/,
    const fp8*  __restrict__ Wo, const float* __restrict__ Wo_scale,
    float* __restrict__ h_partial /*[HIDDEN], pre-zeroed*/) {
  extern __shared__ float xs[];                              // [Q_DIM_RANK]
  for (int k = threadIdx.x; k < Q_DIM_RANK; k += blockDim.x) xs[k] = attn_out[k];
  __syncthreads();
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int o = gwarp; o < HIDDEN; o += nwarp) {
    float acc = tp8_warp_dot(Wo + (size_t)o * Q_DIM_RANK, xs, Q_DIM_RANK, lane);
    if (lane == 0) h_partial[o] = acc * Wo_scale[o];          // partial; residual added post-all-reduce
  }
}
static void tp8_k3_launch(RankState& S, cudaStream_t s) {
  // cudaFuncSetAttribute opt-in done ONCE in alloc_rank (4KB smem here is < 48KB so it isn't even
  //   required, but keeping it out of the hot path is mandatory for graph capture).
  const int block = 256, warps_per_cta = block >> 5;
  int ctas = (HIDDEN + warps_per_cta - 1) / warps_per_cta;
  if (ctas > 264) ctas = 264;
  const size_t smem = (size_t)Q_DIM_RANK * sizeof(float);     // 1024 floats = 4 KB
  tp8_k3_oproj<<<ctas, block, smem, s>>>(S.attn_out, S.Wo, S.Wo_scale, S.attn_partial);
}

// =================================================================================================
// Allocation + dummy SHARDED weights per rank (one layer reused — latency proxy).
// =================================================================================================
static inline unsigned hashu(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static inline float frnd(unsigned seed, size_t i, float scale) {
  unsigned h = hashu((unsigned)(i * 2654435761u) ^ (seed * 40503u));
  return (((h % 2001) / 1000.0f) - 1.0f) * scale;
}
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

// Fix B: TP=8 K2 split count.  Each rank runs only Q_HEADS_RANK=8 heads (all mapping to ONE KV head),
// so the generic 64 splits leaves a long dependent online-softmax chain (64 timesteps/split @ ctx4096)
// -> latency-bound partial.  Sweep (4096): 64 -> 21.5us, 128 -> 18.6us (BEST), 192 -> 19.6 (partial keeps
// dropping but the reduce's HBM partial-read grows).  128 is the partial/reduce balance for the shard.
static inline int tp8_k2_splits(int ctx_len) {
  // Joint sweep (4096) of n_splits x reduce-warps (k2r_warps=32): 64/8 -> 21.5us, 128/32 -> 15.6us,
  // 192/32 -> 14.6us (BEST), 256/32 -> 14.8us.  192 splits with the 32-warp reduce is the balance:
  // the partial's dependent chain is short (4096/192 ~21 steps) and the wide reduce hides the partials' read.
  int s = 192;
  if (ctx_len <= 1024) s = 96;                   // short ctx: fewer splits keep each chunk amortized
  if (ctx_len > 16384) s = 256;                  // long ctx: more parallelism, chunks stay long
  int max_by_chunk = (ctx_len + 31) / 32;
  if (s > max_by_chunk) s = max_by_chunk;
  if (s < 1) s = 1;
  return s;
}
static void alloc_rank(RankState& S, int ctx_len) {
  CK(cudaSetDevice(S.dev));
  S.ctx_len  = ctx_len;
  S.n_splits = tp8_k2_splits(ctx_len);
  // K2 n_splits override (sweep the partial's split-K parallelism without a rebuild): K2_SPLITS env.
  if (const char* e = getenv("K2_SPLITS")) { int v = atoi(e); if (v > 0) S.n_splits = v; }
  // K2 reduce warps/CTA (more = shorter per-warp fold chain of the n_splits partials).  K2R_WARPS env.
  S.k2r_warps = K2R_WARPS;
  if (const char* e = getenv("K2R_W")) { int v = atoi(e); if (v >= 1 && v <= 32) S.k2r_warps = v; }

  CK(cudaMalloc(&S.h_a, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&S.h_b, HIDDEN * sizeof(float)));
  fill_f32(S.h_a, HIDDEN, 99u, 1.0f, false);
  CK(cudaMemset(S.h_b, 0, HIDDEN * sizeof(float)));

  // ---- K1 SHARD: Wqkv[QKV_OUT_RANK=2048, HIDDEN] (this rank's 8 Q + 4 K + 4 V rows) ----
  CK(cudaMalloc(&S.w_in_norm, HIDDEN * sizeof(float)));   fill_f32(S.w_in_norm, HIDDEN, 1u, 0.5f, true);
  CK(cudaMalloc(&S.Wqkv, (size_t)QKV_OUT_RANK * HIDDEN * sizeof(fp8)));  fill_fp8(S.Wqkv, (size_t)QKV_OUT_RANK*HIDDEN, 2u + S.rank);
  CK(cudaMalloc(&S.Wqkv_scale, QKV_OUT_RANK * sizeof(float))); fill_f32(S.Wqkv_scale, QKV_OUT_RANK, 3u, 0.02f, true);
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
  CK(cudaMalloc(&S.out_q, Q_DIM_RANK * sizeof(float)));
  CK(cudaMalloc(&S.qkv_proj, (size_t)QKV_OUT_RANK * sizeof(float)));   // K1 GEMV scratch (per-rank)

  // ---- KV cache: REPLICATED (all 4 KV heads) ----
  CK(cudaMalloc(&S.kv_k, (size_t)ctx_len * KV_DIM * sizeof(fp8)));  fill_fp8(S.kv_k, (size_t)ctx_len*KV_DIM, 20u);
  CK(cudaMalloc(&S.kv_v, (size_t)ctx_len * KV_DIM * sizeof(fp8)));  fill_fp8(S.kv_v, (size_t)ctx_len*KV_DIM, 21u);
  CK(cudaMalloc(&S.kv_k_scale, KV_DIM * sizeof(float))); fill_f32(S.kv_k_scale, KV_DIM, 22u, 0.04f, true);
  CK(cudaMalloc(&S.kv_v_scale, KV_DIM * sizeof(float))); fill_f32(S.kv_v_scale, KV_DIM, 23u, 0.04f, true);

  // K2 partials sized for this rank's Q_HEADS_RANK heads.
  CK(cudaMalloc(&S.part_m,  (size_t)Q_HEADS_RANK * S.n_splits * sizeof(float)));
  CK(cudaMalloc(&S.part_l,  (size_t)Q_HEADS_RANK * S.n_splits * sizeof(float)));
  CK(cudaMalloc(&S.part_acc,(size_t)Q_HEADS_RANK * S.n_splits * HEAD_DIM * sizeof(float)));
  CK(cudaMalloc(&S.attn_out, Q_DIM_RANK * sizeof(float)));

  // ---- K3 SHARD: Wo[HIDDEN, Q_DIM_RANK=1024] column-slice for this rank's heads ----
  CK(cudaMalloc(&S.Wo, (size_t)HIDDEN * Q_DIM_RANK * sizeof(fp8)));  fill_fp8(S.Wo, (size_t)HIDDEN*Q_DIM_RANK, 30u + S.rank);
  CK(cudaMalloc(&S.Wo_scale, HIDDEN * sizeof(float)));               fill_f32(S.Wo_scale, HIDDEN, 31u, 0.02f, true);
  // attn_partial: when NVLS is wired, point it at the MC-bound unicast buffer's ATTN half (so K3 writes
  // there, the multimem reduce sums in-switch, and the residual-add reads the reduced result).  Else a
  // plain cudaMalloc (NCCL reduces it).  moe_partial below points at the MOE half the same way.
#if USE_NVLS
  if (S.nvls && S.nvls->ready) {
    S.attn_partial = S.nvls->uc + NVLS_OFF_ATTN;        // AR INPUT  (K3 writes the partial here)
    S.attn_reduced = S.nvls->uc + NVLS_OFF_ATTN_OUT;    // AR OUTPUT (out-of-place: residual-add reads here)
  } else {
    CK(cudaMalloc(&S.attn_partial, HIDDEN * sizeof(float)));
    S.attn_reduced = S.attn_partial;                    // NCCL reduces in-place
  }
#else
  CK(cudaMalloc(&S.attn_partial, HIDDEN * sizeof(float)));
  S.attn_reduced = S.attn_partial;
#endif

  // ---- K4 REPLICATED ----
  CK(cudaMalloc(&S.w_post_norm, HIDDEN * sizeof(float)));       fill_f32(S.w_post_norm, HIDDEN, 40u, 0.5f, true);
  CK(cudaMalloc(&S.Wgate, (size_t)N_EXPERTS * HIDDEN * sizeof(fp8)));  fill_fp8(S.Wgate, (size_t)N_EXPERTS*HIDDEN, 41u);
  CK(cudaMalloc(&S.Wgate_scale, N_EXPERTS * sizeof(float)));    fill_f32(S.Wgate_scale, N_EXPERTS, 42u, 0.02f, true);
  CK(cudaMalloc(&S.g_logits, N_EXPERTS * sizeof(float)));       // fast-router split-K accumulator
  CK(cudaMalloc(&S.sel_idx, TOP_K * sizeof(int)));
  CK(cudaMalloc(&S.sel_w,   TOP_K * sizeof(float)));
  { std::vector<int> si(TOP_K); std::vector<float> sw(TOP_K, 1.0f/TOP_K);
    for (int i=0;i<TOP_K;++i) si[i]=i;
    CK(cudaMemcpy(S.sel_idx, si.data(), TOP_K*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(S.sel_w,   sw.data(), TOP_K*sizeof(float), cudaMemcpyHostToDevice)); }

  // ---- K5 SHARD: TOP_K physical expert shards (gate+up [2*192,HIDDEN], down [HIDDEN,192]) ----
  const size_t gu_n = (size_t)2 * MOE_INTER_RANK * HIDDEN;    // 2*192*4096 fp8 per expert shard
  const size_t d_n  = (size_t)HIDDEN * MOE_INTER_RANK;        // 4096*192   fp8 per expert shard
  std::vector<fp8*>   Wgu_dp(TOP_K), Wd_dp(TOP_K);
  std::vector<float*> Sgu_dp(TOP_K), Sd_dp(TOP_K);
  for (int e = 0; e < TOP_K; ++e) {
    CK(cudaMalloc(&Wgu_dp[e], gu_n * sizeof(fp8)));  fill_fp8(Wgu_dp[e], gu_n, 50u + e + S.rank);
    CK(cudaMalloc(&Wd_dp[e],  d_n  * sizeof(fp8)));  fill_fp8(Wd_dp[e],  d_n,  70u + e + S.rank);
    CK(cudaMalloc(&Sgu_dp[e], 2 * MOE_INTER_RANK * sizeof(float))); fill_f32(Sgu_dp[e], 2*MOE_INTER_RANK, 90u+e, 0.02f, true);
    CK(cudaMalloc(&Sd_dp[e],  HIDDEN * sizeof(float)));             fill_f32(Sd_dp[e],  HIDDEN,           110u+e, 0.02f, true);
  }
#if USE_GEMM
  // keep the TOP_K physical shard HOST pointers so the grouped-MoE GEMM can pick the routed expert's
  // weight per slot (W operand of a cuBLASLt matmul must be a host-visible device pointer).
  for (int e = 0; e < TOP_K; ++e) { S.Wgu_phys_h[e] = Wgu_dp[e]; S.Wd_phys_h[e] = Wd_dp[e]; }
#endif
  // K5 indexes by EXPERT ID (0..127, written by K4); round-robin N_EXPERTS-wide pointer arrays into
  // the TOP_K physical shards (valid for any routed id, still ~1/8 the resident mass of a full layer).
  std::vector<fp8*>   Wgu_full(N_EXPERTS), Wd_full(N_EXPERTS);
  std::vector<float*> Sgu_full(N_EXPERTS), Sd_full(N_EXPERTS);
  for (int e = 0; e < N_EXPERTS; ++e) { int p = e % TOP_K;
    Wgu_full[e] = Wgu_dp[p]; Wd_full[e] = Wd_dp[p]; Sgu_full[e] = Sgu_dp[p]; Sd_full[e] = Sd_dp[p]; }
  CK(cudaMalloc(&S.Wgu_d,       N_EXPERTS * sizeof(fp8*)));   CK(cudaMemcpy(S.Wgu_d,       Wgu_full.data(), N_EXPERTS*sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.Wd_d,        N_EXPERTS * sizeof(fp8*)));   CK(cudaMemcpy(S.Wd_d,        Wd_full.data(),  N_EXPERTS*sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.Wgu_scale_d, N_EXPERTS * sizeof(float*))); CK(cudaMemcpy(S.Wgu_scale_d, Sgu_full.data(), N_EXPERTS*sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.Wd_scale_d,  N_EXPERTS * sizeof(float*))); CK(cudaMemcpy(S.Wd_scale_d,  Sd_full.data(),  N_EXPERTS*sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.a_glb, (size_t)TOP_K * MOE_INTER_RANK * sizeof(float)));
#if USE_NVLS
  if (S.nvls && S.nvls->ready) {
    S.moe_partial = S.nvls->uc + NVLS_OFF_MOE;          // AR INPUT  (K5b writes the partial here)
    S.moe_reduced = S.nvls->uc + NVLS_OFF_MOE_OUT;      // AR OUTPUT (out-of-place: residual-add reads here)
  } else {
    CK(cudaMalloc(&S.moe_partial, HIDDEN * sizeof(float)));
    S.moe_reduced = S.moe_partial;                      // NCCL reduces in-place
  }
#else
  CK(cudaMalloc(&S.moe_partial, HIDDEN * sizeof(float)));
  S.moe_reduced = S.moe_partial;
#endif

  // K5 plan: MULTIPLE-ROWS-PER-WARP (RA gate+up, RB down) — the measured win on the TP=8 192-shard
  // (k5_sharded_bench.cu: R=1/block=256 -> 15% MBU; RA=2/RB=8/block=512 -> ~34% MBU, 2.2x).  Grid is
  // row-GROUP aware: A has TOP_K*192/RA groups, B has TOP_K*HIDDEN/RB groups.  Block 512 fills the SMs.
  S.k5.block = 512;
  {
    const int warps_per_cta = S.k5.block >> 5;
    auto ctas_for = [&](int rows, int R) { int groups = (rows + R - 1) / R;
                                           int need = (groups + warps_per_cta - 1) / warps_per_cta;
                                           return std::min(std::max(need, 132), 264); };
    S.k5.ctasA = ctas_for(TOP_K * MOE_INTER_RANK, TP8_K5_RA);
    S.k5.ctasB = ctas_for(TOP_K * HIDDEN,         TP8_K5_RB);
    S.k5.smemA = (size_t)HIDDEN * sizeof(float);
    S.k5.smemB = (size_t)TOP_K * MOE_INTER_RANK * sizeof(float);   // 8*192*4 = 6 KB
  }

  // ---- final head: VOCAB-sharded lm_head ----
  S.v_rows = vocab_rows_for(S.rank);
  S.v_off  = vocab_offset_for(S.rank);
  CK(cudaMalloc(&S.w_final_norm, HIDDEN * sizeof(float)));  fill_f32(S.w_final_norm, HIDDEN, 130u, 0.5f, true);
  CK(cudaMalloc(&S.hn, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&S.Wlm, (size_t)S.v_rows * HIDDEN * sizeof(fp8)));  fill_fp8(S.Wlm, (size_t)S.v_rows*HIDDEN, 131u + S.rank);
  CK(cudaMalloc(&S.Wlm_scale, S.v_rows * sizeof(float)));           fill_f32(S.Wlm_scale, S.v_rows, 132u, 0.02f, true);
  S.lm_blocks = 264;
  CK(cudaMalloc(&S.block_max, S.lm_blocks * sizeof(float)));
  CK(cudaMalloc(&S.block_arg, S.lm_blocks * sizeof(int)));
  CK(cudaMalloc(&S.rank_max, sizeof(float)));
  CK(cudaMalloc(&S.rank_arg, sizeof(int)));

  // dynamic-smem opt-ins for the sharded kernels.  ALL cudaFuncSetAttribute calls live HERE (once per
  // rank, off the hot path), so the per-token launch helpers issue ONLY kernel launches + the two
  // all-reduces — which is what makes the whole token stream-capturable into a CUDA graph.  (The k1/k3
  // helpers above and the capture-safe router below therefore no longer call setattr per launch.)
  CK(cudaFuncSetAttribute(tp8_k5a_gateup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S.k5.smemA));
  CK(cudaFuncSetAttribute(tp8_k5b_down,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S.k5.smemB));
  CK(cudaFuncSetAttribute(tp8_lmhead_argmax_partial,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(HIDDEN*sizeof(float))));
  CK(cudaFuncSetAttribute(tp8_k1_qkv_gemv,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(HIDDEN*sizeof(float))));
  CK(cudaFuncSetAttribute(tp8_k3_oproj,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(Q_DIM_RANK*sizeof(float))));
  CK(cudaFuncSetAttribute(k4_router,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(HIDDEN*sizeof(float))));
  CK(cudaFuncSetAttribute(tp8_k4_gate_gemv,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(HIDDEN*sizeof(float))));

#if USE_GEMM
  // =============================================================================================
  // cuBLASLt fp8 TN-GEMM SETUP (replaces the dense GEMVs).  One autotuned panel per projection;
  // activation quant scratch (col-major [K, GEMM_MMAX] fp8, columns 1..15 zero-padded once); GEMM
  // bf16 output buffers (col-major [GEMM_MMAX, N]).  Panels autotuned at GEMM_MMAX=16 then pinned.
  // =============================================================================================
  const int MP = ((GEMM_MMAX + 15) / 16) * 16;     // M padded to the fp8 16-wide tile (=16)
  CL(cublasLtCreate(&S.lt));
  // quant scratch (zeroed -> columns 1..15 stay 0; the per-call quant overwrites column 0 only).
  CK(cudaMalloc(&S.xq_hidden, (size_t)HIDDEN     * MP * sizeof(__nv_fp8_e4m3)));
  CK(cudaMalloc(&S.xq_qdim,   (size_t)Q_DIM_RANK * MP * sizeof(__nv_fp8_e4m3)));
  CK(cudaMalloc(&S.xq_a,      (size_t)TOP_K * MOE_INTER_RANK * MP * sizeof(__nv_fp8_e4m3)));
  CK(cudaMemset(S.xq_hidden, 0, (size_t)HIDDEN     * MP * sizeof(__nv_fp8_e4m3)));
  CK(cudaMemset(S.xq_qdim,   0, (size_t)Q_DIM_RANK * MP * sizeof(__nv_fp8_e4m3)));
  CK(cudaMemset(S.xq_a,      0, (size_t)TOP_K * MOE_INTER_RANK * MP * sizeof(__nv_fp8_e4m3)));
  CK(cudaMalloc(&S.act_scale,   sizeof(float)));
  CK(cudaMalloc(&S.act_scale_a, TOP_K * sizeof(float)));
  // GEMM bf16 outputs (col-major [MP, N]).
  CK(cudaMalloc(&S.d_qkv,   (size_t)MP * QKV_OUT_RANK            * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&S.d_oproj, (size_t)MP * HIDDEN                  * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&S.d_gate,  (size_t)MP * N_EXPERTS               * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&S.d_k5gu,  (size_t)MP * TOP_K * 2*MOE_INTER_RANK* sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&S.d_k5d,   (size_t)MP * TOP_K * HIDDEN          * sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&S.d_lm,    (size_t)MP * S.v_rows                * sizeof(__nv_bfloat16)));
  // autotune events
  cudaEvent_t ge0, ge1; CK(cudaEventCreate(&ge0)); CK(cudaEventCreate(&ge1));
  // init+autotune each panel (X,W,D real buffers; K5 uses physical shard 0 for the autotune).
  S.p_qkv.init  (S.lt, HIDDEN,            QKV_OUT_RANK,             MP, S.xq_hidden, S.Wqkv,  S.d_qkv,   S.stream, ge0, ge1);
  S.p_oproj.init(S.lt, Q_DIM_RANK,        HIDDEN,                   MP, S.xq_qdim,   S.Wo,    S.d_oproj, S.stream, ge0, ge1);
  S.p_gate.init (S.lt, HIDDEN,            N_EXPERTS,                MP, S.xq_hidden, S.Wgate, S.d_gate,  S.stream, ge0, ge1);
  S.p_k5gu.init (S.lt, HIDDEN,            2*MOE_INTER_RANK,         MP, S.xq_hidden, S.Wgu_phys_h[0], S.d_k5gu, S.stream, ge0, ge1);
  S.p_k5d.init  (S.lt, MOE_INTER_RANK,    HIDDEN,                   MP, S.xq_a,      S.Wd_phys_h[0],  S.d_k5d,  S.stream, ge0, ge1);
  S.p_lm.init   (S.lt, HIDDEN,            S.v_rows,                 MP, S.xq_hidden, S.Wlm,   S.d_lm,    S.stream, ge0, ge1);
  // PACKED gate+up: concatenate the 8 physical shards into one [TOP_K*2*192, HIDDEN] buffer so K5a is
  // a SINGLE grouped GEMM.  Fixed slot s -> physical shard s for the proxy/graphed path (graph-safe).
  const size_t gu_shard_n = (size_t)2 * MOE_INTER_RANK * HIDDEN;
  CK(cudaMalloc(&S.Wgu_pack, (size_t)TOP_K * gu_shard_n * sizeof(fp8)));
  for (int slot = 0; slot < TOP_K; ++slot)
    CK(cudaMemcpy(S.Wgu_pack + (size_t)slot * gu_shard_n, S.Wgu_phys_h[slot],
                  gu_shard_n * sizeof(fp8), cudaMemcpyDeviceToDevice));
  S.p_k5gu_pack.init(S.lt, HIDDEN, TOP_K*2*MOE_INTER_RANK, MP, S.xq_hidden, S.Wgu_pack, S.d_k5gu, S.stream, ge0, ge1);
  // PACKED down (for the flatness floor only): 8 down shards [HIDDEN,192] concatenated -> ONE GEMM
  // [K=192, N=TOP_K*HIDDEN].  (The engine runs down as the fast GEMV; this panel measures the
  // achievable single-GEMM grouped-down floor that matches the validated spec_decode_loop reference.)
  const size_t d_shard_n = (size_t)HIDDEN * MOE_INTER_RANK;
  CK(cudaMalloc(&S.Wd_pack, (size_t)TOP_K * d_shard_n * sizeof(fp8)));
  for (int slot = 0; slot < TOP_K; ++slot)
    CK(cudaMemcpy(S.Wd_pack + (size_t)slot * d_shard_n, S.Wd_phys_h[slot],
                  d_shard_n * sizeof(fp8), cudaMemcpyDeviceToDevice));
  S.p_k5d_pack.init(S.lt, MOE_INTER_RANK, TOP_K*HIDDEN, MP, S.xq_a, S.Wd_pack, S.d_k5d, S.stream, ge0, ge1);
  if (!(S.p_qkv.haveAlgo && S.p_oproj.haveAlgo && S.p_gate.haveAlgo &&
        S.p_k5gu.haveAlgo && S.p_k5d.haveAlgo && S.p_lm.haveAlgo &&
        S.p_k5gu_pack.haveAlgo && S.p_k5d_pack.haveAlgo)) {
    printf("rank %d: cuBLASLt fp8 GEMM autotune FAILED for at least one panel.\n", S.rank);
    exit(3);
  }
  CK(cudaEventDestroy(ge0)); CK(cudaEventDestroy(ge1));
#endif

  CK(cudaDeviceSynchronize());
}

// =================================================================================================
// CROSS-RANK CORRECTNESS CHECK (one layer): sharded TP=8 result vs a single-GPU FULL reference.
// -------------------------------------------------------------------------------------------------
// We validate the two stitch points the all-reduces are responsible for:
//   (1) the post-ATTENTION residual  r1 = h_in + Wo @ attn_out   (after AR#1 + the local residual add)
//   (2) the post-MoE        residual  r2 = r1   + MoE(r1)          (after AR#2 + the local residual add)
// The reference is the EXACT single-GPU math of decode_step.cu (k1/k2/k3 full attention, k4 router,
// k5a/k5b full-width experts), run on rank 0 over the FULL weights ASSEMBLED from the 8 per-rank
// shards — so a mismatch can only come from a wrong shard layout, a wrong all-reduce, or a residual
// added the wrong number of times.  Both sides consume the identical h_in and the identical
// (replicated) KV cache, so the only numeric delta is fp32 accumulation order -> we assert < 1e-2.
//
// Shard-assembly contract being verified:
//   * Wqkv: full row r of the Q-block (r in [0,Q_DIM)) == rank (r/Q_DIM_RANK)'s local Q row (r%..);
//           the 4 K and 4 V rows are REPLICATED (taken from rank 0; we force every rank to rank-0's
//           KV rows + KV scales below so the replication invariant the kernel assumes actually holds).
//   * Wo:   full column block [r*Q_DIM_RANK, (r+1)*Q_DIM_RANK) of Wo[HIDDEN, Q_DIM] == rank r's shard;
//           the per-rank PARTIAL O-projs SUM (AR#1) to the full O-proj.
//   * experts: full intermediate columns [r*MOE_INTER_RANK,(r+1)*..) of every expert == rank r's shard;
//           the per-rank PARTIAL MoE-down outputs SUM (AR#2) to the full MoE contribution.
// =================================================================================================

// Copy a device fp8 region rank0.dev -> rank r.dev (peer or staged through host).
static void d2d_fp8(fp8* dst, int dst_dev, const fp8* src, int src_dev, size_t n) {
  CK(cudaMemcpyPeer(dst, dst_dev, src, src_dev, n * sizeof(fp8)));
}
static void d2d_f32(float* dst, int dst_dev, const float* src, int src_dev, size_t n) {
  CK(cudaMemcpyPeer(dst, dst_dev, src, src_dev, n * sizeof(float)));
}

// Run ONE sharded layer on every rank and copy rank 0's two residual stitch-points to host.
//   r1_out <- post-attention residual (h_src + AR(O-proj));  r2_out <- post-MoE residual.
// Mirrors enqueue_tp8_layer EXACTLY but snapshots h_dst right after the AR#1 residual add.
static void sharded_one_layer_capture(std::vector<RankState>& R,
                                      float* r1_out_host, float* r2_out_host) {
  // ONE thread per rank (NCCL one-comm-per-thread): all ranks reach AR#1/AR#2 concurrently so
  // ncclGroupEnd never blocks on un-enqueued peers.  rank 0 snapshots the two stitch points.
  run_all_ranks(R, [&](RankState& S) {
    cudaStream_t s = S.stream;
    float* h_src = S.h_a;          // fresh input (filled in alloc_rank, never mutated before this)
    float* h_dst = S.h_b;

#if USE_GEMM
    // ---- GEMM path (the thing being validated): cuBLASLt fp8 K1/K3/K4/K5/lmhead vs the reference ----
    gemm_k1_launch(S, h_src, s);
    tp8_k2_launch(S, s);
    CK(cudaMemsetAsync(S.attn_partial, 0, HIDDEN * sizeof(float), s));
    gemm_k3_launch(S, s);
    ar_sum_hidden(S, S.attn_partial, NVLS_OFF_ATTN, s);                  // AR#1 (NVLS or NCCL)
    tp8_residual_add<<<32, 256, 0, s>>>(h_src, S.attn_reduced, h_dst);   // r1 = h_src + AR(O-proj)
    if (S.rank == 0) CK(cudaMemcpyAsync(r1_out_host, h_dst, HIDDEN * sizeof(float),
                                        cudaMemcpyDeviceToHost, s));
    gemm_k4_launch(S, h_dst, s);                                          // GEMM gate -> sel_idx/sel_w
    // resolve the REAL routed expert -> physical shard map host-side (eager path: D2H is OK here).
    CK(cudaMemcpyAsync(S.sel_h, S.sel_idx, TOP_K * sizeof(int), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    int sel_phys[TOP_K]; for (int i = 0; i < TOP_K; ++i) sel_phys[i] = ((S.sel_h[i] % TOP_K) + TOP_K) % TOP_K;
    CK(cudaMemsetAsync(S.moe_partial, 0, HIDDEN * sizeof(float), s));
    gemm_k5_pack_gateup(S, sel_phys, s);                                  // re-pack the ROUTED shards
    gemm_k5_launch(S, h_dst, sel_phys, s);                                // grouped fp8 gate+up GEMM + GEMV down
    ar_sum_hidden(S, S.moe_partial, NVLS_OFF_MOE, s);                    // AR#2 (NVLS or NCCL)
    tp8_residual_add<<<32, 256, 0, s>>>(h_dst, S.moe_reduced, h_dst);    // r2 = r1 + AR(MoE)
    if (S.rank == 0) CK(cudaMemcpyAsync(r2_out_host, h_dst, HIDDEN * sizeof(float),
                                        cudaMemcpyDeviceToHost, s));
#else
    tp8_k1_launch(S, h_src, s);
    tp8_k2_launch(S, s);
    CK(cudaMemsetAsync(S.attn_partial, 0, HIDDEN * sizeof(float), s));
    tp8_k3_launch(S, s);
    ar_sum_hidden(S, S.attn_partial, NVLS_OFF_ATTN, s);                  // AR#1 (NVLS or NCCL)
    tp8_residual_add<<<32, 256, 0, s>>>(h_src, S.attn_reduced, h_dst);   // r1 = h_src + AR(O-proj)
    if (S.rank == 0) CK(cudaMemcpyAsync(r1_out_host, h_dst, HIDDEN * sizeof(float),
                                        cudaMemcpyDeviceToHost, s));
    // NEW fast router (split-K gate GEMV + select) — validated here against the reference's stock
    // single-CTA k4_router; identical math, so r2 must still match to < 1e-2.
    tp8_k4_launch(h_dst, S.w_post_norm, S.Wgate, S.Wgate_scale, S.sel_idx, S.sel_w, S.g_logits, s);
    CK(cudaMemsetAsync(S.moe_partial, 0, HIDDEN * sizeof(float), s));
    tp8_k5a_gateup<<<S.k5.ctasA, S.k5.block, S.k5.smemA, s>>>(
        h_dst, S.sel_idx, S.Wgu_d, S.Wgu_scale_d, S.a_glb, TOP_K);
    tp8_k5b_down<<<S.k5.ctasB, S.k5.block, S.k5.smemB, s>>>(
        S.sel_idx, S.sel_w, S.Wd_d, S.Wd_scale_d, S.a_glb, S.moe_partial, TOP_K);
    ar_sum_hidden(S, S.moe_partial, NVLS_OFF_MOE, s);                    // AR#2 (NVLS or NCCL)
    tp8_residual_add<<<32, 256, 0, s>>>(h_dst, S.moe_reduced, h_dst);    // r2 = r1 + AR(MoE)
    if (S.rank == 0) CK(cudaMemcpyAsync(r2_out_host, h_dst, HIDDEN * sizeof(float),
                                        cudaMemcpyDeviceToHost, s));
#endif
  });  // each thread syncs its own stream before joining
}

// Build the FULL reference weights on rank 0's device by assembling the per-rank shards, then run the
// single-GPU decode_step.cu math for ONE layer and snapshot the same two residual stitch-points.
static void reference_one_layer_capture(std::vector<RankState>& R,
                                        float* r1_out_host, float* r2_out_host) {
  const int dev0 = R[0].dev;
  CK(cudaSetDevice(dev0));

  // ---- assemble full Wqkv[QKV_OUT=9216, HIDDEN]: Q rows from each rank, K/V rows from rank 0 ----
  fp8*   Wqkv_full = nullptr;  float* Wqkv_scale_full = nullptr;
  CK(cudaMalloc(&Wqkv_full,       (size_t)QKV_OUT * HIDDEN * sizeof(fp8)));
  CK(cudaMalloc(&Wqkv_scale_full, (size_t)QKV_OUT * sizeof(float)));
  for (int r = 0; r < TP; ++r) {
    // rank r's local rows [0, Q_DIM_RANK) are global Q rows [r*Q_DIM_RANK, (r+1)*Q_DIM_RANK).
    d2d_fp8(Wqkv_full + (size_t)r * Q_DIM_RANK * HIDDEN, dev0,
            R[r].Wqkv, R[r].dev, (size_t)Q_DIM_RANK * HIDDEN);
    d2d_f32(Wqkv_scale_full + (size_t)r * Q_DIM_RANK, dev0,
            R[r].Wqkv_scale, R[r].dev, (size_t)Q_DIM_RANK);
  }
  // K rows: full [Q_DIM, Q_DIM+KV_DIM) <- rank 0's local [Q_DIM_RANK, Q_DIM_RANK+KV_DIM).
  d2d_fp8(Wqkv_full + (size_t)Q_DIM * HIDDEN, dev0,
          R[0].Wqkv + (size_t)Q_DIM_RANK * HIDDEN, R[0].dev, (size_t)KV_DIM * HIDDEN);
  d2d_f32(Wqkv_scale_full + Q_DIM, dev0, R[0].Wqkv_scale + Q_DIM_RANK, R[0].dev, KV_DIM);
  // V rows: full [Q_DIM+KV_DIM, QKV_OUT) <- rank 0's local [Q_DIM_RANK+KV_DIM, QKV_OUT_RANK).
  d2d_fp8(Wqkv_full + (size_t)(Q_DIM + KV_DIM) * HIDDEN, dev0,
          R[0].Wqkv + (size_t)(Q_DIM_RANK + KV_DIM) * HIDDEN, R[0].dev, (size_t)KV_DIM * HIDDEN);
  d2d_f32(Wqkv_scale_full + Q_DIM + KV_DIM, dev0,
          R[0].Wqkv_scale + Q_DIM_RANK + KV_DIM, R[0].dev, KV_DIM);

  // ---- assemble full Wo[HIDDEN, Q_DIM]: rank r owns column block [r*Q_DIM_RANK, (r+1)*Q_DIM_RANK) ----
  fp8*   Wo_full = nullptr;  float* Wo_scale_full = nullptr;
  CK(cudaMalloc(&Wo_full,       (size_t)HIDDEN * Q_DIM * sizeof(fp8)));
  CK(cudaMalloc(&Wo_scale_full, (size_t)HIDDEN * sizeof(float)));
  // Wo is K-major (row o is Q_DIM contiguous); rank r's shard is row-o's columns [r*1024,(r+1)*1024).
  // Stage each rank's shard to host, scatter into the full row-major buffer, then upload once.
  {
    std::vector<fp8> hshard((size_t)HIDDEN * Q_DIM_RANK);
    std::vector<fp8> hfull ((size_t)HIDDEN * Q_DIM);
    for (int r = 0; r < TP; ++r) {
      CK(cudaSetDevice(R[r].dev));
      CK(cudaMemcpy(hshard.data(), R[r].Wo, (size_t)HIDDEN * Q_DIM_RANK * sizeof(fp8),
                    cudaMemcpyDeviceToHost));
      for (int o = 0; o < HIDDEN; ++o)
        memcpy(&hfull[(size_t)o * Q_DIM + r * Q_DIM_RANK],
               &hshard[(size_t)o * Q_DIM_RANK], Q_DIM_RANK * sizeof(fp8));
    }
    CK(cudaSetDevice(dev0));
    CK(cudaMemcpy(Wo_full, hfull.data(), (size_t)HIDDEN * Q_DIM * sizeof(fp8),
                  cudaMemcpyHostToDevice));
    // Wo_scale is per-output-channel [HIDDEN] and identical (seed 31u) across ranks -> copy rank 0's.
    d2d_f32(Wo_scale_full, dev0, R[0].Wo_scale, R[0].dev, HIDDEN);
  }

  // ---- assemble full experts: gate+up[2*MOE_INTER, HIDDEN] and down[HIDDEN, MOE_INTER] per expert ----
  // Each rank holds MOE_INTER_RANK=192 intermediate cols of every expert.  The full gate+up row layout
  // is [gate rows 0..MOE_INTER | up rows MOE_INTER..2*MOE_INTER], each row HIDDEN-wide.  Rank r's gate
  // rows are full gate rows [r*192,(r+1)*192); likewise up.  The full down row o (MOE_INTER-wide) has
  // rank r's 192 contiguous columns at [r*192,(r+1)*192).
  // We assemble only the TOP_K PHYSICAL expert shards (the proxy keeps TOP_K of them; the N_EXPERTS
  // pointer table round-robins into those TOP_K, identically on every rank, so reconstructing the
  // TOP_K physical experts and round-robining the reference table reproduces the exact routed math).
  const size_t gu_full_n = (size_t)2 * MOE_INTER * HIDDEN;
  const size_t d_full_n  = (size_t)HIDDEN * MOE_INTER;
  std::vector<fp8*>   Wgu_phys(TOP_K), Wd_phys(TOP_K);
  std::vector<float*> Sgu_phys(TOP_K), Sd_phys(TOP_K);
  {
    // Per-rank pointer tables live on the rank's device; we need the PHYSICAL shard pointers.  The
    // pointer table round-robins expert id e -> physical p = e % TOP_K, so physical p == table[p].
    // Pull table[0..TOP_K) from each rank to host to get its physical shard device pointers.
    std::vector<std::vector<fp8*>>   rk_Wgu(TP, std::vector<fp8*>(N_EXPERTS));
    std::vector<std::vector<fp8*>>   rk_Wd (TP, std::vector<fp8*>(N_EXPERTS));
    std::vector<std::vector<float*>> rk_Sgu(TP, std::vector<float*>(N_EXPERTS));
    std::vector<std::vector<float*>> rk_Sd (TP, std::vector<float*>(N_EXPERTS));
    for (int r = 0; r < TP; ++r) {
      CK(cudaSetDevice(R[r].dev));
      CK(cudaMemcpy(rk_Wgu[r].data(), R[r].Wgu_d, N_EXPERTS * sizeof(fp8*),   cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(rk_Wd [r].data(), R[r].Wd_d,  N_EXPERTS * sizeof(fp8*),   cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(rk_Sgu[r].data(), R[r].Wgu_scale_d, N_EXPERTS * sizeof(float*), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(rk_Sd [r].data(), R[r].Wd_scale_d,  N_EXPERTS * sizeof(float*), cudaMemcpyDeviceToHost));
    }
    CK(cudaSetDevice(dev0));
    std::vector<fp8> gu_full(gu_full_n), d_full(d_full_n);
    std::vector<fp8> gu_shard((size_t)2 * MOE_INTER_RANK * HIDDEN), d_shard((size_t)HIDDEN * MOE_INTER_RANK);
    std::vector<float> sgu_full(2 * MOE_INTER), sd_full(HIDDEN), sgu_shard(2 * MOE_INTER_RANK);
    for (int p = 0; p < TOP_K; ++p) {
      CK(cudaMalloc(&Wgu_phys[p], gu_full_n * sizeof(fp8)));
      CK(cudaMalloc(&Wd_phys[p],  d_full_n  * sizeof(fp8)));
      CK(cudaMalloc(&Sgu_phys[p], 2 * MOE_INTER * sizeof(float)));
      CK(cudaMalloc(&Sd_phys[p],  HIDDEN * sizeof(float)));
      for (int r = 0; r < TP; ++r) {
        // gate+up shard for physical expert p on rank r (the pointer at table index p): D2H stage.
        CK(cudaSetDevice(R[r].dev));
        CK(cudaMemcpy(gu_shard.data(), rk_Wgu[r][p],
                      (size_t)2 * MOE_INTER_RANK * HIDDEN * sizeof(fp8), cudaMemcpyDeviceToHost));
        // gate rows: shard rows [0,192) -> full gate rows [r*192,(r+1)*192).
        for (int j = 0; j < MOE_INTER_RANK; ++j)
          memcpy(&gu_full[(size_t)(r * MOE_INTER_RANK + j) * HIDDEN],
                 &gu_shard[(size_t)j * HIDDEN], HIDDEN * sizeof(fp8));
        // up rows: shard rows [192,384) -> full up rows [MOE_INTER + r*192, MOE_INTER + (r+1)*192).
        for (int j = 0; j < MOE_INTER_RANK; ++j)
          memcpy(&gu_full[(size_t)(MOE_INTER + r * MOE_INTER_RANK + j) * HIDDEN],
                 &gu_shard[(size_t)(MOE_INTER_RANK + j) * HIDDEN], HIDDEN * sizeof(fp8));
        // gate+up scale shard [2*192] -> full [gate j at r*192+j | up j at MOE_INTER+r*192+j].
        CK(cudaMemcpy(sgu_shard.data(), rk_Sgu[r][p],
                      (size_t)2 * MOE_INTER_RANK * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < MOE_INTER_RANK; ++j) {
          sgu_full[r * MOE_INTER_RANK + j]              = sgu_shard[j];
          sgu_full[MOE_INTER + r * MOE_INTER_RANK + j]  = sgu_shard[MOE_INTER_RANK + j];
        }
        // down shard [HIDDEN, 192] -> full down row o columns [r*192,(r+1)*192).
        CK(cudaMemcpy(d_shard.data(), rk_Wd[r][p],
                      (size_t)HIDDEN * MOE_INTER_RANK * sizeof(fp8), cudaMemcpyDeviceToHost));
        for (int o = 0; o < HIDDEN; ++o)
          memcpy(&d_full[(size_t)o * MOE_INTER + r * MOE_INTER_RANK],
                 &d_shard[(size_t)o * MOE_INTER_RANK], MOE_INTER_RANK * sizeof(fp8));
      }
      // down scale is per-output-channel [HIDDEN], identical across ranks -> take rank 0's physical p.
      CK(cudaSetDevice(R[0].dev));
      CK(cudaMemcpy(sd_full.data(), rk_Sd[0][p], HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
      CK(cudaSetDevice(dev0));
      CK(cudaMemcpy(Wgu_phys[p], gu_full.data(), gu_full_n * sizeof(fp8), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(Wd_phys[p],  d_full.data(),  d_full_n  * sizeof(fp8), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(Sgu_phys[p], sgu_full.data(), 2 * MOE_INTER * sizeof(float), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(Sd_phys[p],  sd_full.data(),  HIDDEN * sizeof(float),        cudaMemcpyHostToDevice));
    }
  }
  // round-robin N_EXPERTS-wide reference pointer tables into the TOP_K physical experts (same map).
  std::vector<fp8*>   Wgu_tab(N_EXPERTS), Wd_tab(N_EXPERTS);
  std::vector<float*> Sgu_tab(N_EXPERTS), Sd_tab(N_EXPERTS);
  for (int e = 0; e < N_EXPERTS; ++e) { int p = e % TOP_K;
    Wgu_tab[e] = Wgu_phys[p]; Wd_tab[e] = Wd_phys[p]; Sgu_tab[e] = Sgu_phys[p]; Sd_tab[e] = Sd_phys[p]; }
  const fp8 **Wgu_d=nullptr,**Wd_d=nullptr; const float **Sgu_d=nullptr,**Sd_d=nullptr;
  CK(cudaMalloc(&Wgu_d, N_EXPERTS*sizeof(fp8*)));   CK(cudaMemcpy(Wgu_d, Wgu_tab.data(), N_EXPERTS*sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd_d,  N_EXPERTS*sizeof(fp8*)));   CK(cudaMemcpy(Wd_d,  Wd_tab.data(),  N_EXPERTS*sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sgu_d, N_EXPERTS*sizeof(float*))); CK(cudaMemcpy(Sgu_d, Sgu_tab.data(), N_EXPERTS*sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd_d,  N_EXPERTS*sizeof(float*))); CK(cudaMemcpy(Sd_d,  Sd_tab.data(),  N_EXPERTS*sizeof(float*), cudaMemcpyHostToDevice));

  // ---- reference scratch (full attention + full MoE), all on rank 0's device ----
  cudaStream_t s = R[0].stream;
  RankState& S0 = R[0];                         // reuse rank-0's replicated inputs/cache/norm weights
  float *h_src=nullptr,*h_dst=nullptr,*out_q=nullptr,*attn_out=nullptr,*a_glb=nullptr;
  float *part_m=nullptr,*part_l=nullptr,*part_acc=nullptr;
  CK(cudaMalloc(&h_src, HIDDEN*sizeof(float)));  CK(cudaMalloc(&h_dst, HIDDEN*sizeof(float)));
  d2d_f32(h_src, dev0, S0.h_a, S0.dev, HIDDEN);  // SAME input as the sharded path (rank 0's h_a)
  CK(cudaMalloc(&out_q,    Q_DIM*sizeof(float)));
  CK(cudaMalloc(&attn_out, Q_DIM*sizeof(float)));
  CK(cudaMalloc(&part_m,   (size_t)N_Q_HEADS*S0.n_splits*sizeof(float)));
  CK(cudaMalloc(&part_l,   (size_t)N_Q_HEADS*S0.n_splits*sizeof(float)));
  CK(cudaMalloc(&part_acc, (size_t)N_Q_HEADS*S0.n_splits*HEAD_DIM*sizeof(float)));
  CK(cudaMalloc(&a_glb,    (size_t)TOP_K*MOE_INTER*sizeof(float)));
  int*   ref_sel_idx = nullptr;  float* ref_sel_w = nullptr;   // reference's OWN routing buffers
  CK(cudaMalloc(&ref_sel_idx, TOP_K*sizeof(int)));
  CK(cudaMalloc(&ref_sel_w,   TOP_K*sizeof(float)));

  // K1 (full QKV GEMV + epilogue) — k1_launch writes out_q[Q_DIM] + the KV-cache current slot.  The
  // KV cache used is rank 0's (replicated copy); we forced every rank to rank 0's KV rows/scales so
  // the sharded K2 reads the identical cache the reference K1 just wrote.
  k1_launch(h_src, S0.w_in_norm, Wqkv_full, Wqkv_scale_full, S0.q_norm, S0.k_norm,
            S0.rope_cos, S0.rope_sin, out_q, S0.kv_k, S0.kv_v, S0.kv_k_scale, S0.kv_v_scale, s);
  k2_launch(out_q, S0.kv_k, S0.kv_v, S0.kv_k_scale, S0.kv_v_scale, S0.ctx_len,
            part_m, part_l, part_acc, attn_out, S0.n_splits, s);
  k3_launch(attn_out, Wo_full, Wo_scale_full, h_src, h_dst, s);   // h_dst = h_src + Wo@attn_out = r1
  CK(cudaMemcpyAsync(r1_out_host, h_dst, HIDDEN*sizeof(float), cudaMemcpyDeviceToHost, s));
  // Router is REPLICATED + deterministic; the reference runs its OWN copy into dedicated buffers (it
  // must agree with every sharded rank's K4 since they share the SAME post-attn residual r1).
  k4_launch(h_dst, S0.w_post_norm, S0.Wgate, S0.Wgate_scale, ref_sel_idx, ref_sel_w, s);
  // full-width experts (MOE_INTER=1536) accumulate straight into h_dst -> r2 = r1 + MoE(r1).
  K5Launch L = k5_plan(TOP_K);
  CK(cudaFuncSetAttribute(k5a_gateup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemA));
  CK(cudaFuncSetAttribute(k5b_down,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemB));
  k5a_gateup<<<L.ctasA, L.block, L.smemA, s>>>(h_dst, ref_sel_idx, Wgu_d, Sgu_d, a_glb, TOP_K);
  k5b_down  <<<L.ctasB, L.block, L.smemB, s>>>(ref_sel_idx, ref_sel_w, Wd_d, Sd_d, a_glb, h_dst, TOP_K);
  CK(cudaMemcpyAsync(r2_out_host, h_dst, HIDDEN*sizeof(float), cudaMemcpyDeviceToHost, s));
  CK(cudaStreamSynchronize(s));

  // free the reference scratch (best-effort; the physical-expert buffers too).
  cudaFree(Wqkv_full); cudaFree(Wqkv_scale_full); cudaFree(Wo_full); cudaFree(Wo_scale_full);
  cudaFree(h_src); cudaFree(h_dst); cudaFree(out_q); cudaFree(attn_out);
  cudaFree(part_m); cudaFree(part_l); cudaFree(part_acc); cudaFree(a_glb);
  cudaFree(ref_sel_idx); cudaFree(ref_sel_w);
  cudaFree(Wgu_d); cudaFree(Wd_d); cudaFree(Sgu_d); cudaFree(Sd_d);
  for (int p = 0; p < TOP_K; ++p) { cudaFree(Wgu_phys[p]); cudaFree(Wd_phys[p]);
                                    cudaFree(Sgu_phys[p]); cudaFree(Sd_phys[p]); }
}

// Force every rank to share rank 0's REPLICATED tensors so the replication invariant the kernels
// assume (identical KV-projection rows, identical KV cache + scales, identical router) actually holds
// for the check.  In the latency-proxy bench these are filled with per-rank seeds (harmless for
// timing); for a meaningful cross-rank numeric comparison they must truly match.
static void unify_replicated_state(std::vector<RankState>& R) {
  const int dev0 = R[0].dev;
  for (int r = 1; r < TP; ++r) {
    RankState& S = R[r];
    // KV-projection rows of Wqkv (local rows [Q_DIM_RANK, QKV_OUT_RANK)) + their scales <- rank 0.
    d2d_fp8(S.Wqkv + (size_t)Q_DIM_RANK * HIDDEN, S.dev,
            R[0].Wqkv + (size_t)Q_DIM_RANK * HIDDEN, dev0, (size_t)2 * KV_DIM * HIDDEN);
    d2d_f32(S.Wqkv_scale + Q_DIM_RANK, S.dev, R[0].Wqkv_scale + Q_DIM_RANK, dev0, 2 * KV_DIM);
    // KV cache + scales (replicated) <- rank 0.
    d2d_fp8(S.kv_k, S.dev, R[0].kv_k, dev0, (size_t)S.ctx_len * KV_DIM);
    d2d_fp8(S.kv_v, S.dev, R[0].kv_v, dev0, (size_t)S.ctx_len * KV_DIM);
    d2d_f32(S.kv_k_scale, S.dev, R[0].kv_k_scale, dev0, KV_DIM);
    d2d_f32(S.kv_v_scale, S.dev, R[0].kv_v_scale, dev0, KV_DIM);
    // router (replicated): w_post_norm, Wgate, Wgate_scale, and the input-norm + RoPE tables <- rank 0.
    d2d_f32(S.w_post_norm, S.dev, R[0].w_post_norm, dev0, HIDDEN);
    d2d_fp8(S.Wgate, S.dev, R[0].Wgate, dev0, (size_t)N_EXPERTS * HIDDEN);
    d2d_f32(S.Wgate_scale, S.dev, R[0].Wgate_scale, dev0, N_EXPERTS);
    d2d_f32(S.w_in_norm, S.dev, R[0].w_in_norm, dev0, HIDDEN);
    d2d_f32(S.q_norm, S.dev, R[0].q_norm, dev0, HEAD_DIM);
    d2d_f32(S.k_norm, S.dev, R[0].k_norm, dev0, HEAD_DIM);
    d2d_f32(S.rope_cos, S.dev, R[0].rope_cos, dev0, HEAD_DIM / 2);
    d2d_f32(S.rope_sin, S.dev, R[0].rope_sin, dev0, HEAD_DIM / 2);
    // same fresh input h_a as rank 0.
    d2d_f32(S.h_a, S.dev, R[0].h_a, dev0, HIDDEN);
  }
  CK(cudaSetDevice(dev0));
  for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(R[r].dev)); CK(cudaDeviceSynchronize()); }
}

// Returns 0 on pass, 1 on fail.  Prints max abs diff at both stitch points.
static int run_correctness_check(std::vector<RankState>& R) {
  printf("\n== cross-rank CORRECTNESS check (1 layer, sharded TP=8 vs single-GPU full reference) ==\n");
  unify_replicated_state(R);

  std::vector<float> ref_r1(HIDDEN), ref_r2(HIDDEN), shd_r1(HIDDEN), shd_r2(HIDDEN);
  // Reference first (it WRITES rank 0's KV-cache current slot via k1_launch); the sharded path then
  // reads that SAME slot.  Both paths recompute K1's KV write identically (same KV weights + input),
  // so order is immaterial, but we run reference first for clarity.
  reference_one_layer_capture(R, ref_r1.data(), ref_r2.data());
  sharded_one_layer_capture (R, shd_r1.data(), shd_r2.data());

  auto maxabsdiff = [](const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)fabsf(a[i]-b[i]));
    return m;
  };
  auto maxabs = [](const std::vector<float>& a) {
    double m = 0.0; for (float v : a) m = std::max(m, (double)fabsf(v)); return m;
  };
  const double d1 = maxabsdiff(ref_r1, shd_r1);
  const double d2 = maxabsdiff(ref_r2, shd_r2);
  const double s1 = maxabs(ref_r1), s2 = maxabs(ref_r2);
#if USE_GEMM
  // The GEMM path ALSO quantizes the ACTIVATION to fp8 e4m3 (the reference GEMV keeps fp32
  // activations), so the delta carries the inherent e4m3 activation-rounding (~3 mantissa bits)
  // ON TOP of the all-reduce stitch this gate checks.  We use the precision-appropriate fp8 tol
  // (the same 8e-2 spec_verify_forward_gemm.cu gated fp8 on); the GEMM/wgmma is bit-exact, the gap
  // is fp8 quant the SHIP model already carries.  The structural invariants (shard layout, AR-SUM,
  // residual-once) are still validated to that tolerance — a wrong layout fails by O(1), not O(1e-2).
  const double TOL = 8e-2;
  const char* tol_note = " [fp8 e4m3 activation+weight quant tol; GEMM is bit-exact wgmma]";
#else
  const double TOL = 1e-2;
  const char* tol_note = "";
#endif
  printf("  post-attention residual : max|ref-shd| = %.3e   (ref max|.| = %.3e)\n", d1, s1);
  printf("  post-MoE        residual : max|ref-shd| = %.3e   (ref max|.| = %.3e)\n", d2, s2);
  const bool pass = (d1 < TOL) && (d2 < TOL);
  printf("  TOL=%.0e%s  ->  %s\n", TOL, tol_note, pass ? "PASS" : "FAIL");
  if (!pass) {
    printf("  CORRECTNESS FAILED: the sharded TP=8 layer does not match the full reference.\n");
    return 1;
  }
  return 0;
}

// =================================================================================================
// PER-KERNEL PROFILER — definitive per-kernel-class ranking on ONE rank (rank 0 is representative).
// -------------------------------------------------------------------------------------------------
// Each kernel class is launched REPS times back-to-back between two cudaEvents on rank 0's stream,
// divided by REPS to get us/launch, then multiplied by its per-token launch count (N_LAYERS for the
// per-layer kernels, 1 for the head kernels) to get the per-token us contribution.  Inputs are the
// rank's already-allocated dummy buffers (values are garbage in the proxy, but the kernel SHAPE / grid
// / HBM byte-traffic are the real ones, so the timing is faithful).  No NCCL, no cross-rank sync — this
// isolates pure kernel-execution cost, which is exactly the ~20 ms kernels-only floor we are dissecting.
// =================================================================================================
static void profile_per_kernel(RankState& S, int PEAK_unused) {
  CK(cudaSetDevice(S.dev));
  cudaStream_t s = S.stream;
  const int REPS = 2000;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  // launch grids/smem identical to the hot-path helpers.
  const int blockA = 256, warpsA = blockA >> 5;
  int needA = (QKV_OUT_RANK + warpsA - 1) / warpsA; int ctasA = needA < 264 ? needA : 264;
  const size_t k1_smem = (size_t)HIDDEN * sizeof(float);
  const int k3_block = 256, k3_wpc = k3_block >> 5;
  int k3_ctas = (HIDDEN + k3_wpc - 1) / k3_wpc; if (k3_ctas > 264) k3_ctas = 264;
  const size_t k3_smem = (size_t)Q_DIM_RANK * sizeof(float);
  const int k2_wpc = 4, k2_block = k2_wpc * 32;
  dim3 k2gP(S.n_splits, (Q_HEADS_RANK + k2_wpc - 1) / k2_wpc);
  dim3 k2gR((Q_HEADS_RANK + k2_wpc - 1) / k2_wpc);
  const size_t lm_smem = (size_t)HIDDEN * sizeof(float);
  const size_t k4_smem = (size_t)HIDDEN * sizeof(float);

  struct Row { const char* name; double us_per; double per_tok; int count; bool in_sum; };
  std::vector<Row> rows;
  auto timeit_x = [&](const char* name, int per_tok_count, bool in_sum, auto launch) -> double {
    // warm
    for (int i = 0; i < 50; ++i) launch();
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < REPS; ++i) launch();
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1));
    double us_per = (double)ms * 1e3 / REPS;
    rows.push_back({name, us_per, us_per * per_tok_count, per_tok_count, in_sum});
    return us_per * per_tok_count;
  };
  auto timeit = [&](const char* name, int per_tok_count, auto launch) -> double {
    return timeit_x(name, per_tok_count, true, launch);
  };

  printf("\n== PER-KERNEL PROFILE (rank 0, %d reps/kernel, per-token = us/launch x launches/token) ==\n", REPS);
#if USE_GEMM
  printf("   (USE_GEMM=1: the dense GEMV rows below are EXCLUDED from the SUM — kept as the baseline\n");
  printf("    reference; the cuBLASLt fp8 GEMM rows are what the engine runs and ARE in the SUM.)\n");
#endif
  // ---- attention ----  (dense GEMV rows: in-sum only when USE_GEMM=0; else baseline reference)
  const bool gemv_in_sum = !USE_GEMM;
  timeit_x("K1 qkv_gemv", N_LAYERS, gemv_in_sum, [&]{ tp8_k1_qkv_gemv<<<ctasA, blockA, k1_smem, s>>>(
      S.h_a, S.w_in_norm, S.Wqkv, S.Wqkv_scale, S.qkv_proj); });
  timeit("K1 epilogue", N_LAYERS, [&]{ tp8_k1_epilogue<<<1, 256, 0, s>>>(
      S.qkv_proj, S.q_norm, S.k_norm, S.rope_cos, S.rope_sin, S.out_q, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale); });
  // K2 OLD 2-kernel path (partial+reduce): in-sum only when the fused path is OFF; else baseline ref.
  const bool k2old_in_sum = !USE_K2_FUSED;
  timeit_x("K2 partial OLD (flash-decode)", N_LAYERS, k2old_in_sum, [&]{ tp8_k2_partial<<<k2gP, k2_block, 0, s>>>(
      S.out_q, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale, S.ctx_len, S.n_splits, S.rank, S.part_m, S.part_l, S.part_acc); });
  const int k2r_w = S.k2r_warps, k2r_block = k2r_w * 32;
  const size_t k2r_smem = (size_t)(2 * k2r_w + k2r_w * HEAD_DIM) * sizeof(float);
  timeit_x("K2 reduce OLD", N_LAYERS, k2old_in_sum, [&]{ tp8_k2_reduce<<<Q_HEADS_RANK, k2r_block, k2r_smem, s>>>(
      S.part_m, S.part_l, S.part_acc, S.n_splits, S.attn_out); });
  // K2 FUSED (Fix B): single launch, partials on-chip — the engine's actual K2 when USE_K2_FUSED=1.
  {
    const int k2f_w = tp8_k2f_warps(S.ctx_len), k2f_block = k2f_w * 32;
    const size_t k2f_smem = (size_t)(2 * k2f_w + k2f_w * HEAD_DIM) * sizeof(float);
    timeit_x("K2 FUSED (Fix B, in-sum)", N_LAYERS, (bool)USE_K2_FUSED, [&]{ tp8_k2_fused<<<Q_HEADS_RANK, k2f_block, k2f_smem, s>>>(
        S.out_q, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale, S.ctx_len, S.rank, S.attn_out); });
  }
  timeit_x("K3 oproj", N_LAYERS, gemv_in_sum, [&]{ tp8_k3_oproj<<<k3_ctas, k3_block, k3_smem, s>>>(
      S.attn_out, S.Wo, S.Wo_scale, S.attn_partial); });
  // ---- MoE ----
  // OLD single-CTA router (the measured bottleneck) — timed for the before->after record (NOT in sum).
  timeit_x("K4 router OLD (1-CTA)", N_LAYERS, false, [&]{ k4_router<<<1, 256, k4_smem, s>>>(
      S.h_a, S.w_post_norm, S.Wgate, S.Wgate_scale, S.sel_idx, S.sel_w); });
  // NEW fast router: fused-RMSNorm split-K gate GEMV across the whole GPU + tiny select kernel.
  {
    const int k4_block = 128, k4_warps = k4_block >> 5;
    const int k4_ctas = (N_EXPERTS * TP8_K4_KSPLIT + k4_warps - 1) / k4_warps;
    timeit_x("K4 gate_gemv NEW", N_LAYERS, gemv_in_sum, [&]{
      CK(cudaMemsetAsync(S.g_logits, 0, N_EXPERTS * sizeof(float), s));
      tp8_k4_gate_gemv<<<k4_ctas, k4_block, k4_smem, s>>>(S.h_a, S.w_post_norm, S.Wgate, S.g_logits); });
    timeit("K4 select NEW", N_LAYERS, [&]{ tp8_k4_select<<<1, 32, 0, s>>>(
      S.g_logits, S.Wgate_scale, S.sel_idx, S.sel_w); });
  }
  timeit_x("K5a gateup", N_LAYERS, gemv_in_sum, [&]{ tp8_k5a_gateup<<<S.k5.ctasA, S.k5.block, S.k5.smemA, s>>>(
      S.h_a, S.sel_idx, S.Wgu_d, S.Wgu_scale_d, S.a_glb, TOP_K); });
  timeit_x("K5b down", N_LAYERS, gemv_in_sum, [&]{ tp8_k5b_down<<<S.k5.ctasB, S.k5.block, S.k5.smemB, s>>>(
      S.sel_idx, S.sel_w, S.Wd_d, S.Wd_scale_d, S.a_glb, S.moe_partial, TOP_K); });
  // ---- residual adds (2 per layer) + head ----
  timeit("residual_add (x2/layer)", 2 * N_LAYERS, [&]{ tp8_residual_add<<<32, 256, 0, s>>>(
      S.h_a, S.attn_partial, S.h_b); });
  timeit("final_norm", 1, [&]{ tp8_final_norm<<<1, 256, 0, s>>>(S.h_a, S.w_final_norm, S.hn); });
  timeit_x("lmhead_argmax_partial", 1, gemv_in_sum, [&]{ tp8_lmhead_argmax_partial<<<S.lm_blocks, 256, lm_smem, s>>>(
      S.hn, S.Wlm, S.Wlm_scale, S.v_rows, S.v_off, S.block_max, S.block_arg); });
  timeit("argmax_final", 1, [&]{ tp8_argmax_final<<<1, 32, 0, s>>>(
      S.block_max, S.block_arg, S.lm_blocks, S.rank_max, S.rank_arg); });

#if USE_GEMM
  // =============================================================================================
  // cuBLASLt fp8 GEMM forward rows — the ENGINE PATH (these ARE in the SUM).  Each row times the
  // full launch sequence the engine runs (quant + GEMM + scale epilogue), so the SUM is the true
  // per-rank GEMM forward us/token.  K5 grouped = 8 gate+up GEMMs + 8 down GEMMs + their quants.
  // =============================================================================================
  const int MP = S.p_qkv.Mpad;
  const size_t qsmem = (size_t)HIDDEN * sizeof(float);
  timeit("GEMM K1 qkv (q+gemm+epi)", N_LAYERS, [&]{
    gemm_rmsnorm_quant<<<1,1024,qsmem,s>>>(S.h_a, S.w_in_norm, S.xq_hidden, S.act_scale, HIDDEN);
    S.p_qkv.run(S.xq_hidden, S.Wqkv, S.d_qkv, s);
    gemm_epi_scale<<<32,256,0,s>>>(S.d_qkv, S.Wqkv_scale, S.act_scale, S.qkv_proj, QKV_OUT_RANK, MP); });
  timeit("GEMM K3 oproj (q+gemm+epi)", N_LAYERS, [&]{
    gemm_quant<<<1,256,0,s>>>(S.attn_out, S.xq_qdim, S.act_scale, Q_DIM_RANK);
    S.p_oproj.run(S.xq_qdim, S.Wo, S.d_oproj, s);
    gemm_epi_scale<<<32,256,0,s>>>(S.d_oproj, S.Wo_scale, S.act_scale, S.attn_partial, HIDDEN, MP); });
  timeit("GEMM K4 gate (q+gemm+epi)", N_LAYERS, [&]{
    gemm_rmsnorm_quant<<<1,1024,qsmem,s>>>(S.h_a, S.w_post_norm, S.xq_hidden, S.act_scale, HIDDEN);
    S.p_gate.run(S.xq_hidden, S.Wgate, S.d_gate, s);
    gemm_epi_gate<<<1,128,0,s>>>(S.d_gate, S.act_scale, S.g_logits, N_EXPERTS, MP); });
  timeit("GEMM K5 gateup(1 GEMM)+down GEMV", N_LAYERS, [&]{ gemm_k5_launch(S, S.h_a, K5_SEL_PHYS_FIXED, s); });
  timeit("GEMM lm_head (q+gemm+epi)", 1, [&]{ gemm_lmhead_launch(S, S.hn, s); });
#endif

  double total = 0.0; for (auto& r : rows) if (r.in_sum) total += r.per_tok;
  // sort descending by per-token contribution.
  std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){ return a.per_tok > b.per_tok; });
  printf("  %-28s %10s %8s %12s %8s\n", "kernel", "us/launch", "x/token", "us/token", "%of-sum");
  for (auto& r : rows)
    printf("  %-28s %10.3f %8d %12.2f %7.1f%%%s\n", r.name, r.us_per, r.count, r.per_tok,
           r.in_sum ? 100.0*r.per_tok/total : 0.0, r.in_sum ? "" : "  (excl from sum)");
  printf("  %-28s %10s %8s %12.2f %7.1f%%\n", "SUM (kernels-only/token)", "", "", total, 100.0);

#if USE_GEMM
  // =============================================================================================
  // GEMM FORWARD T(M) FLATNESS TABLE — sum the GEMM-able panels (x N_LAYERS, + lm_head) timed at
  // M = 1,4,8,16.  fp8 always runs the 16-wide tile so T(M)~=T(1) (the spec-verify property): a
  // ~1.0 ratio proves verify(M) costs ~one decode-forward (M=8 verify == M=1 decode).
  // =============================================================================================
  {
    const int Ms[] = {1,4,8,16}; const int NM = 4;
    const int FW = 10, FIT = 50;
    double fwd_us[NM]; for (int i=0;i<NM;i++) fwd_us[i]=0.0;
    for (int mi=0; mi<NM; ++mi) {
      double per_layer = 0.0;
      per_layer += S.p_qkv.time_at_M (Ms[mi], S.xq_hidden, S.Wqkv,           S.d_qkv,   s, e0, e1, FW, FIT);
      per_layer += S.p_oproj.time_at_M(Ms[mi], S.xq_qdim,   S.Wo,            S.d_oproj, s, e0, e1, FW, FIT);
      per_layer += S.p_gate.time_at_M (Ms[mi], S.xq_hidden, S.Wgate,         S.d_gate,  s, e0, e1, FW, FIT);
      // K5: ONE packed gate+up GEMM + ONE packed down GEMM (the achievable grouped-MoE floor, matching
      // the validated spec_decode_loop reference; both GEMMs for the FLATNESS property).
      per_layer += S.p_k5gu_pack.time_at_M(Ms[mi], S.xq_hidden, S.Wgu_pack, S.d_k5gu, s, e0, e1, FW, FIT);
      per_layer += S.p_k5d_pack.time_at_M (Ms[mi], S.xq_a,      S.Wd_pack,  S.d_k5d,  s, e0, e1, FW, FIT);
      double lm = S.p_lm.time_at_M(Ms[mi], S.xq_hidden, S.Wlm, S.d_lm, s, e0, e1, FW, FIT);
      fwd_us[mi] = per_layer * N_LAYERS + lm;
    }
    printf("\n== GEMM forward T(M) FLATNESS (cuBLASLt fp8, GEMM panels x %d layers + lm_head) ==\n", N_LAYERS);
    printf("  %-6s %14s %12s %14s\n", "M", "us/forward", "tok/s", "ratio vs M=1");
    for (int mi=0; mi<NM; ++mi)
      printf("  M=%-4d %14.1f %12.1f %14.3f\n", Ms[mi], fwd_us[mi], 1e6/fwd_us[mi], fwd_us[mi]/fwd_us[0]);
    printf("  FLAT => verify(M) ~= decode(1): T(16)/T(1) = %.3f.  GEMM single-forward (M=1) = %.0f tok/s;\n",
           fwd_us[NM-1]/fwd_us[0], 1e6/fwd_us[0]);
    printf("  M=8 verify-forward = %.0f tok/s (8 candidates at ~the M=1 wall).  (GEMM-only; +K2 attn in SUM.)\n",
           1e6/fwd_us[2]);
  }
#endif

  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
}

// =================================================================================================
// main() — one process, 8 GPUs, NCCL.  Measures the REAL TP=8 B=1 decode latency + AR overhead.
// Define DSTP8_NO_MAIN before #include-ing this file to reuse RankState/alloc_rank/tp8_k1_launch/
// tp8_k2_launch/tp8_k3_launch as a library (same convention as K5_NO_MAIN for k5_experts.cu).
// =================================================================================================
#ifndef DSTP8_NO_MAIN
int main(int argc, char** argv) {
  const int    ctx_len = (argc > 1) ? atoi(argv[1]) : 4096;
  const int    IT      = (argc > 2) ? atoi(argv[2]) : 200;
  const double PEAK    = (argc > 3) ? atof(argv[3]) : 3350.0;   // GB/s per H100 HBM3
  const int    WARM    = 20;

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < TP) {
    printf("Need >= %d CUDA devices for TP=%d; found %d.\n", TP, TP, ndev); return 1;
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("== Qwen3-235B-A22B TP=8 sharded decode step (latency proxy) ==\n");
  printf("device0: %s  SMs=%d  HBM peak=%.0f GB/s  TP=%d  ctx_len=%d  layers=%d  iters=%d\n",
         prop.name, prop.multiProcessorCount, PEAK, TP, ctx_len, N_LAYERS, IT);

  // ---- enable peer access so NCCL uses NVLink P2P (best-effort; NVSwitch makes all pairs peers) ----
  for (int i = 0; i < TP; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < TP; ++j) if (i != j) {
      int can = 0; cudaDeviceCanAccessPeer(&can, i, j);
      if (can) cudaDeviceEnablePeerAccess(j, 0);   // ignore "already enabled"
    }
  }

  // ---- NCCL: one communicator clique across the 8 local GPUs (single-process, simplest path) ----
  std::vector<RankState> R(TP);
  std::vector<ncclComm_t> comms(TP);
  std::vector<int> devs(TP);
  for (int r = 0; r < TP; ++r) devs[r] = r;
  NK(ncclCommInitAll(comms.data(), TP, devs.data()));   // 8 ranks on 1 process

  // ---- NVLS: wire the in-switch multimem all-reduce over the 8 GPUs (before per-rank alloc, so the
  //      AR buffers attn_partial/moe_partial can be repointed onto the multicast-bound memory).  If
  //      multicast isn't supported (no NVSwitch / driver), nvls_engine_setup returns false and the
  //      engine transparently falls back to the NCCL all-reduces (ar_sum_hidden dispatch). -----------
  std::vector<NvlsCtx> nvls;
  bool nvls_on = false;
#if USE_NVLS
  nvls_on = nvls_engine_setup(nvls, TP);
  if (!nvls_on) printf("NVLS unavailable -> falling back to NCCL all-reduces.\n");
#else
  printf("NVLS disabled at build (USE_NVLS=0) -> NCCL all-reduces.\n");
#endif

  // ---- per-rank state + stream ----
  for (int r = 0; r < TP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    if (nvls_on) R[r].nvls = &nvls[r];
    CK(cudaSetDevice(r));
    CK(cudaStreamCreate(&R[r].stream));
    alloc_rank(R[r], ctx_len);
  }

  // ---- per-token ACTIVE HBM read volume PER GPU (the ~1/8 shard each rank reads) ----------------
  // Per layer per rank (fp8 weights dominate):
  //   Wqkv shard = QKV_OUT_RANK*HIDDEN     KV (replicated!) = 2*ctx_len*KV_DIM
  //   Wo shard   = HIDDEN*Q_DIM_RANK       Wgate (replicated) = N_EXPERTS*HIDDEN
  //   experts shard = TOP_K*(2*MOE_INTER_RANK*HIDDEN + HIDDEN*MOE_INTER_RANK)
  const double b_qkv   = (double)QKV_OUT_RANK * HIDDEN;
  const double b_kv    = 2.0 * (double)ctx_len * KV_DIM;                 // REPLICATED (4 KV heads < 8)
  const double b_o     = (double)HIDDEN * Q_DIM_RANK;
  const double b_gate  = (double)N_EXPERTS * HIDDEN;                     // router replicated (cheap)
  const double b_exp   = (double)TOP_K * ((double)2*MOE_INTER_RANK*HIDDEN + (double)HIDDEN*MOE_INTER_RANK);
  const double b_layer = b_qkv + b_kv + b_o + b_gate + b_exp;
  const double b_lm    = (double)vocab_rows_for(0) * HIDDEN;             // rank-0 (largest) vocab shard
  const double b_token = b_layer * N_LAYERS + b_lm;                      // per-GPU read/token (incl. KV)
  const double b_weight_only = (b_layer - b_kv) * N_LAYERS + b_lm;       // exclude replicated KV
  printf("\nper-token PER-GPU active read (TP=8 shard): %.2f GB  (weight-only %.2f GB + replicated KV @ctx%d)\n",
         b_token / 1e9, b_weight_only / 1e9, ctx_len);
  printf("  per layer/GPU %.2f MB (experts %.2f + Wqkv %.2f + Wo %.2f + KV(replicated) %.2f + gate(repl) %.2f) x %d\n",
         b_layer/1e6, b_exp/1e6, b_qkv/1e6, b_o/1e6, b_kv/1e6, b_gate/1e6, N_LAYERS);
  printf("  + lm_head shard %.1f MB.  experts/layer=%.2f MB = ONE full-expert-equiv (TP8 kills EP busiest-rank gamble).\n",
         b_lm/1e6, b_exp/1e6);
  printf("  full single-GPU read would be ~%.2f GB -> %.1fx more per GPU; weight-only matches the ~2.7 GB spec.\n",
         b_token*8.0/1e9, 8.0);

  // ---- collective accounting ----
  const int ar_per_layer = 2;                                  // O-proj + MoE-down
  const int ar_per_step  = ar_per_layer * N_LAYERS + 1;        // + 1 head argmax-max = 189
  printf("NCCL all-reduces / token: %d  (%d/layer x %d + 1 head)\n", ar_per_step, ar_per_layer, N_LAYERS);
  printf("  payload: [HIDDEN]=%d floats = %.1f KB/all-reduce  (tiny -> latency-bound)\n",
         HIDDEN, HIDDEN*sizeof(float)/1024.0);

  // ---- CORRECTNESS GATE (before the proxy bench, which accumulates garbage into the residuals) ----
  //   Validates the two all-reduce stitch points of ONE sharded layer against a single-GPU full
  //   reference assembled from the per-rank shards.  Fail -> abort before reporting bogus throughput.
  const bool skip_check = (argc > 4) && (atoi(argv[4]) == 0);
  if (!skip_check) {
    if (run_correctness_check(R) != 0) {
      printf("ABORT: correctness check failed; not reporting throughput.\n");
      for (int r = 0; r < TP; ++r) ncclCommDestroy(comms[r]);
      return 2;
    }
  } else {
    printf("\n(correctness check skipped: argv[4]==0)\n");
  }

  // ---- PER-KERNEL PROFILE (rank 0): definitive per-kernel-class us/token ranking -----------------
  profile_per_kernel(R[0], (int)PEAK);

  // ---- warm up once OUTSIDE timing (lazy module load, cudaFuncSetAttribute, NCCL channel setup) ----
  //   Driven via run_all_ranks (one thread/rank) so the per-rank NCCL groups can't deadlock.
  run_all_ranks(R, [](RankState& S){ enqueue_tp8_step(S); });

  // ---- (a) FULL TP=8 step timing.  Time on rank 0's stream with events recorded INSIDE rank 0's
  //          thread; every rank syncs its own stream each iter so the measured time is the
  //          slowest-rank step (the real B=1 cost).  WARM iters then IT timed iters, all in-thread.
  cudaEvent_t ev0, ev1;
  CK(cudaSetDevice(0)); CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));

  float ms_full = 0.f;
  {
    std::vector<std::thread> th; th.reserve(TP);
    for (int r = 0; r < TP; ++r) {
      th.emplace_back([&, r]() {
        CK(cudaSetDevice(R[r].dev));
        for (int i = 0; i < WARM; ++i) { enqueue_tp8_step(R[r]); CK(cudaStreamSynchronize(R[r].stream)); }
        if (r == 0) CK(cudaEventRecord(ev0, R[0].stream));
        for (int i = 0; i < IT; ++i) { enqueue_tp8_step(R[r]); CK(cudaStreamSynchronize(R[r].stream)); }
        if (r == 0) CK(cudaEventRecord(ev1, R[0].stream));
      });
    }
    for (auto& t : th) t.join();
  }
  CK(cudaSetDevice(0)); CK(cudaEventSynchronize(ev1));
  CK(cudaEventElapsedTime(&ms_full, ev0, ev1)); ms_full /= IT;

  // ---- (b) all-reduce-only timing: same 189 collectives/step, NO kernels, to isolate AR overhead.
  //      Uses the SAME dispatch as the engine (NVLS when wired, else NCCL) so this measures the comms
  //      cost actually being shipped — the before->after the report cites. ---------------------------
  auto enqueue_ar_only = [](RankState& S, int n) {
    for (int l = 0; l < n; ++l)
      ar_sum_hidden(S, S.attn_partial, (l & 1) ? NVLS_OFF_MOE : NVLS_OFF_ATTN, S.stream);
  };
  float ms_ar = 0.f;
  {
    std::vector<std::thread> th; th.reserve(TP);
    for (int r = 0; r < TP; ++r) {
      th.emplace_back([&, r]() {
        CK(cudaSetDevice(R[r].dev));
        for (int i = 0; i < WARM; ++i) { enqueue_ar_only(R[r], ar_per_step); CK(cudaStreamSynchronize(R[r].stream)); }
        if (r == 0) CK(cudaEventRecord(ev0, R[0].stream));
        for (int i = 0; i < IT; ++i) { enqueue_ar_only(R[r], ar_per_step); CK(cudaStreamSynchronize(R[r].stream)); }
        if (r == 0) CK(cudaEventRecord(ev1, R[0].stream));
      });
    }
    for (auto& t : th) t.join();
  }
  CK(cudaSetDevice(0)); CK(cudaEventSynchronize(ev1));
  CK(cudaEventElapsedTime(&ms_ar, ev0, ev1)); ms_ar /= IT;

  // =============================================================================================
  // (c) CUDA-GRAPH path: capture the WHOLE token (94 layers' kernels + 188 NCCL all-reduces + head
  //     + the head argmax-max AR) into ONE per-rank graph, then replay all 8 graphs IT times with a
  //     single cudaGraphLaunch/rank.  This collapses the ~1100 host launches/token into 1 replay/rank.
  //     NCCL>=2.9 records collectives enqueued during capture as graph nodes; all 8 ranks must capture
  //     CONCURRENTLY (the collectives need every rank live) -> the SpinBarrier brackets begin/end.
  // =============================================================================================
  SpinBarrier cap_bar(TP);
  std::atomic<int> capture_ok{1};
  {
    std::vector<std::thread> th; th.reserve(TP);
    for (int r = 0; r < TP; ++r) {
      th.emplace_back([&, r]() {
        CK(cudaSetDevice(R[r].dev));
        // warm-up one eager step OUTSIDE capture (lazy module load / any first-touch), then sync.
        enqueue_tp8_step(R[r]);
        CK(cudaStreamSynchronize(R[r].stream));
        // All 8 ranks enter capture together.  ThreadLocal mode -> each thread captures only the work
        // IT issues on its own stream (the 8 captures don't see each other's launches), while the NCCL
        // collectives are matched across the concurrently-capturing ranks.
        cap_bar.wait();
        cudaError_t e = cudaStreamBeginCapture(R[r].stream, cudaStreamCaptureModeThreadLocal);
        if (e != cudaSuccess) { capture_ok.store(0); printf("rank %d BeginCapture: %s\n", r, cudaGetErrorString(e)); }
        enqueue_tp8_step(R[r]);                       // record ONE full token (kernels + this rank's ARs)
        e = cudaStreamEndCapture(R[r].stream, &R[r].graph);
        if (e != cudaSuccess) { capture_ok.store(0); printf("rank %d EndCapture: %s\n", r, cudaGetErrorString(e)); }
        cap_bar.wait();                               // all ranks finished capture before instantiate
        if (R[r].graph) {
          e = cudaGraphInstantiate(&R[r].exec, R[r].graph, nullptr, nullptr, 0);
          if (e != cudaSuccess) { capture_ok.store(0); printf("rank %d Instantiate: %s\n", r, cudaGetErrorString(e)); }
        }
      });
    }
    for (auto& t : th) t.join();
  }

  float ms_graph = 0.f;
  size_t graph_nodes = 0;
  if (capture_ok.load() && R[0].graph) {
    cudaGraphGetNodes(R[0].graph, nullptr, &graph_nodes);
    printf("\ncaptured per-rank graph: %zu nodes (rank 0).  Replaying %d iters x 8 ranks.\n",
           graph_nodes, IT);
    std::vector<std::thread> th; th.reserve(TP);
    for (int r = 0; r < TP; ++r) {
      th.emplace_back([&, r]() {
        CK(cudaSetDevice(R[r].dev));
        for (int i = 0; i < WARM; ++i) { CK(cudaGraphLaunch(R[r].exec, R[r].stream)); CK(cudaStreamSynchronize(R[r].stream)); }
        if (r == 0) CK(cudaEventRecord(ev0, R[0].stream));
        for (int i = 0; i < IT; ++i) { CK(cudaGraphLaunch(R[r].exec, R[r].stream)); CK(cudaStreamSynchronize(R[r].stream)); }
        if (r == 0) CK(cudaEventRecord(ev1, R[0].stream));
      });
    }
    for (auto& t : th) t.join();
    CK(cudaSetDevice(0)); CK(cudaEventSynchronize(ev1));
    CK(cudaEventElapsedTime(&ms_graph, ev0, ev1)); ms_graph /= IT;
  } else {
    printf("\nCUDA-GRAPH capture FAILED on at least one rank (see errors above) — graphed result skipped.\n");
  }

  // =============================================================================================
  // (d) KERNELS-ONLY graphs + EAGER all-reduces (the robust fallback).  Each layer's ~9 kernel
  //     launches collapse into 2 graph launches (segA, segB); the 188 NCCL all-reduces stay on
  //     NCCL's fast EAGER path.  Host ops/token: 2*94 + 188 + 2 = ~378 (vs ~1100 eager).  No
  //     cross-rank barrier is needed during capture here — the segments contain NO collectives —
  //     so each rank captures its 3 kernel segments independently, then all 8 ranks replay together
  //     with the eager ARs interleaved (which IS where the ranks rendezvous, exactly as in eager).
  // =============================================================================================
  float ms_kgraph = 0.f, ms_konly = 0.f;
  cudaEvent_t g_ev_k0, g_ev_k1;
  CK(cudaSetDevice(0)); CK(cudaEventCreate(&g_ev_k0)); CK(cudaEventCreate(&g_ev_k1));
  {
    std::vector<std::thread> th; th.reserve(TP);
    for (int r = 0; r < TP; ++r) {
      th.emplace_back([&, r]() {
        CK(cudaSetDevice(R[r].dev));
        RankState& S = R[r];
        // Capture the 3 kernel-only segments (fixed buffers: h_a in, h_b out — proxy drops ping-pong).
        S.exec_segA    = capture_segment(S.stream, [&](cudaStream_t s){ enqueue_tp8_segA(S, S.h_a, s); });
        S.exec_segB    = capture_segment(S.stream, [&](cudaStream_t s){ enqueue_tp8_segB(S, S.h_a, S.h_b, s); });
        S.exec_seghead = capture_segment(S.stream, [&](cudaStream_t s){ enqueue_tp8_seghead(S, S.h_b, s); });
        CK(cudaStreamSynchronize(S.stream));
        // time: WARM then IT replays, per-iter sync (same methodology as the eager/full-graph paths).
        for (int i = 0; i < WARM; ++i) { replay_tp8_step_kgraph(S); CK(cudaStreamSynchronize(S.stream)); }
        if (r == 0) CK(cudaEventRecord(ev0, R[0].stream));
        for (int i = 0; i < IT; ++i)  { replay_tp8_step_kgraph(S); CK(cudaStreamSynchronize(S.stream)); }
        if (r == 0) CK(cudaEventRecord(ev1, R[0].stream));
        // ---- DIAGNOSTIC: kernels-only graphs replayed back-to-back with NO eager ARs and only ONE
        //      sync at the end (isolates pure collapsed-kernel execution time / launch floor). ----
        CK(cudaStreamSynchronize(S.stream));
        if (r == 0) CK(cudaEventRecord(g_ev_k0, R[0].stream));
        for (int i = 0; i < IT; ++i) {
          for (int l = 0; l < N_LAYERS; ++l) { CK(cudaGraphLaunch(S.exec_segA, S.stream));
                                               CK(cudaGraphLaunch(S.exec_segB, S.stream)); }
          CK(cudaGraphLaunch(S.exec_seghead, S.stream));
        }
        if (r == 0) CK(cudaEventRecord(g_ev_k1, R[0].stream));
        CK(cudaStreamSynchronize(S.stream));
      });
    }
    for (auto& t : th) t.join();
    CK(cudaSetDevice(0)); CK(cudaEventSynchronize(ev1));
    CK(cudaEventElapsedTime(&ms_kgraph, ev0, ev1)); ms_kgraph /= IT;
    CK(cudaEventSynchronize(g_ev_k1));
    CK(cudaEventElapsedTime(&ms_konly, g_ev_k0, g_ev_k1)); ms_konly /= IT;
    printf("\nkernels-only segment graphs captured (segA+segB+head/rank); 188 ARs eager.  Replayed %d iters x 8 ranks.\n", IT);
    printf("DIAGNOSTIC kernels-only graph replay (NO ARs, 1 sync/IT): %.2f us/token  (pure collapsed-kernel exec floor)\n",
           ms_konly*1e3);
    CK(cudaEventDestroy(g_ev_k0)); CK(cudaEventDestroy(g_ev_k1));
  }

  // =============================================================================================
  // report.
  // =============================================================================================
  auto tokps = [](float ms) { return 1.0e3 / ms; };
  auto gbps  = [&](float ms) { return b_token / 1e6 / ms; };   // per-GPU bytes/ms = GB/s
  printf("\n  %-34s %12s %12s %12s %12s\n", "metric", "us/token", "tok/s", "GB/s/GPU", "%HBMpeak");
  printf("  %-34s %12.2f %12.1f %12.1f %11.1f%%\n", "TP=8 EAGER step (baseline)",
         ms_full*1e3, tokps(ms_full), gbps(ms_full), 100.0*gbps(ms_full)/PEAK);
  if (ms_graph > 0.f)
    printf("  %-34s %12.2f %12.1f %12.1f %11.1f%%\n", "TP=8 full NCCL-in-graph (replay)",
           ms_graph*1e3, tokps(ms_graph), gbps(ms_graph), 100.0*gbps(ms_graph)/PEAK);
  printf("  %-34s %12.2f %12.1f %12.1f %11.1f%%\n", "TP=8 kernels-graph + eager AR",
         ms_kgraph*1e3, tokps(ms_kgraph), gbps(ms_kgraph), 100.0*gbps(ms_kgraph)/PEAK);

  // Pick the best graphed path as the headline result.
  float ms_best = ms_kgraph;
  const char* best_name = "kernels-graph + eager AR";
  if (ms_graph > 0.f && ms_graph < ms_best) { ms_best = ms_graph; best_name = "full NCCL-in-graph"; }
  printf("\n  >>> BEST graphed path: %s  ->  %.2f us/token  =  %.1f tok/s\n",
         best_name, ms_best*1e3, tokps(ms_best));
  printf("  >>> SPEEDUP vs EAGER baseline: %.2fx  (%.1f%% faster; %.2f us/tok saved; %.1f -> %.1f tok/s)\n",
         ms_full / ms_best, 100.0*(ms_full - ms_best)/ms_full, (ms_full - ms_best)*1e3,
         tokps(ms_full), tokps(ms_best));

  printf("\n  %-34s %12.2f %12s %12s %12s\n", "  all-reduces only (189)",
         ms_ar*1e3, "-", "-", "-");
  printf("  %-34s %12.2f\n", "  -> per-all-reduce", ms_ar*1e3 / ar_per_step);
  printf("  %-34s %12.2f  (%.1f%% of EAGER step)\n", "  -> AR overhead / token",
         ms_ar*1e3, 100.0 * ms_ar / ms_full);
  printf("  %-34s %12.2f\n", "  compute-only (eager - AR)", (ms_full - ms_ar)*1e3);
  printf("  %-34s %12.2f  (%.1f%% of BEST graphed step -> the new dominant cost)\n",
         "  AR-share of BEST graphed step", ms_ar*1e3, 100.0 * ms_ar / ms_best);

  // ideal weight-only tok/s if each GPU streamed its WEIGHT shard at ~45% of HBM peak (no comms, no KV):
  const double ideal_ms = (b_weight_only / 1e9) / (PEAK * 0.45 / 1e3);   // GB / (GB/s) -> ms
  printf("\n  single-GPU cap was ~153 tok/s; TP=8 weight-only ideal (per-GPU %.2f GB @ ~45%% peak)"
         " ~ %.0f tok/s (~%.2f ms); the all-reduces + replicated-KV read add the overhead measured above.\n",
         b_weight_only/1e9, 1.0e3 / ideal_ms, ideal_ms);
  printf("== done ==\n");

  // ---- cleanup (best-effort) ----
  for (int r = 0; r < TP; ++r) {
    if (R[r].exec)  cudaGraphExecDestroy(R[r].exec);
    if (R[r].graph) cudaGraphDestroy(R[r].graph);
    if (R[r].exec_segA)    cudaGraphExecDestroy(R[r].exec_segA);
    if (R[r].exec_segB)    cudaGraphExecDestroy(R[r].exec_segB);
    if (R[r].exec_seghead) cudaGraphExecDestroy(R[r].exec_seghead);
  }
  for (int r = 0; r < TP; ++r) { ncclCommDestroy(comms[r]); }
  CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1));
  return 0;
}
#endif // DSTP8_NO_MAIN
