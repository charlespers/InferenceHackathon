// megakernel_decode.cu — PERSISTENT, grid-resident MEGAKERNEL decode step for Qwen3-235B-A22B,
// B=1, TP=8 / EP=8 across 8x H100 (single node, NVLink P2P), sm_90a.
//
// =================================================================================================
// WHY A MEGAKERNEL (the structural comms lever)
// -------------------------------------------------------------------------------------------------
// At B=1 the sharded decode step is overhead-bound on TWO axes that both scale with the number of
// per-step launches/syncs (188 collectives/token = 2 all-reduces/layer x 94, plus ~7 kernel
// launches/layer):
//
//   1) PER-KERNEL LAUNCH overhead.  Each of the ~660 GEMV/flash-decode launches/token costs a few us
//      of CPU->GPU dispatch; a CUDA graph (decode_step.cu) collapses that to ONE launch but still
//      issues every collective as its own host-driven NCCL/NVSHMEM call.
//
//   2) PER-COLLECTIVE LAUNCH + BARRIER overhead.  Measured on this box: NCCL all-reduce 35 us;
//      NVSHMEM put+barrier ~17 us — of which a 16 KB NVLink transfer is ~0.1 us, so ~17 us is almost
//      ENTIRELY host-launch + barrier handshake.  188 x 17 us = ~3.2 ms -> ~310 tok/s hard cap.
//
// A graph removes (1) but NOT (2): the collectives are still separate device-reachable calls whose
// barrier handshakes are serialized end-to-end with kernel boundaries between them.  The structural
// fix is to put EVERYTHING — every GEMV, every flash-decode, every all-reduce — inside ONE persistent
// grid-resident kernel that is launched ONCE and loops over the layers internally.  Then:
//   * there is no per-kernel relaunch between layers (the grid stays resident);
//   * the in-kernel all-reduce is a few device-side NVLink puts + ONE barrier that the resident grid
//     already owns — no host round-trip, no re-acquire of NVSHMEM sync state per collective.
// The barrier still costs something, but it is the in-kernel device barrier cost (single-digit us),
// not the host-launch-dominated ~17 us.  This file MEASURES that: it times the single persistent
// launch over N_LAYERS_TEST layer-steps and reports us/layer-step + the projection to 94 layers.
//
// =================================================================================================
// WHAT THIS KERNEL DOES (per layer, all as __device__ inline functions, NO host relaunch between)
// -------------------------------------------------------------------------------------------------
//   K1  mk_rmsnorm_qkv_gemv : input-RMSNorm + fused QKV GEMV (warp-per-row coalesced fp8, k1 idiom)
//   K2  mk_flash_decode     : split-KV single-query GQA online-softmax (warp-per-head, k2 idiom)
//   K3  mk_oproj_residual   : O-proj GEMV + fused residual add (warp-per-row, k3 idiom)
//   AR1 mk_allreduce_recd   : IN-KERNEL recursive-doubling NVSHMEM all-reduce of the residual [HIDDEN]
//   K4  mk_router           : post-RMSNorm + gate GEMV + softmax + top-8 + renorm (k4 idiom)
//   K5  mk_expert_gateup / mk_expert_down : fused fp8 MoE SwiGLU + routed down-accumulate (k5 idiom)
//   AR2 mk_allreduce_recd   : SECOND in-kernel all-reduce of the post-MoE residual [HIDDEN]
//
// Each PE owns its TP/EP slice (its head range, its expert weights); the two in-kernel all-reduces
// per layer are exactly the comms the megakernel exists to make cheap.  After N_LAYERS_TEST layers a
// final RMSNorm + lm_head GEMV slice + (partial) argmax close the step (head done once, not per layer).
//
// SYNCHRONIZATION
//   * INTRA-PE (across this PE's CTAs):  cooperative-groups grid_group::sync() — the whole grid is
//     co-resident (we size the grid <= the cooperative-launch occupancy limit), so a grid barrier is
//     legal and cheap.  This orders the producer stage (e.g. K1 GEMV) before the consumer (K2).
//   * INTER-PE (the TP all-reduce):  the recursive-doubling put+barrier idiom validated in
//     nvshmem_comms.cu (nvshmemx_float_put_block + nvshmemx_barrier_all_block), but issued FROM INSIDE
//     the persistent kernel.  We use ONLY device-side put/get + barrier (NOT the library block
//     collectives, which fail collective_launch occupancy on this build — see MEMORY).
//   The kernel is launched with nvshmemx_collective_launch so NVSHMEM's inter-PE device-barrier state
//   is set up (the <<<>>> syntax would hang on the first device barrier).
//
// LATENCY-PROXY DISCLAIMER (same contract as decode_step.cu)
//   ONE layer's worth of dummy fp8 weights is resident and REUSED for all N_LAYERS_TEST layers.  The
//   work shape, smem, in-kernel barrier/collective COUNT, and per-layer HBM read VOLUME are identical
//   to the real model, so us/layer-step and the projected tok/s are representative.  The numbers
//   produced are meaningless (same weights every layer); a correctness check on the in-kernel
//   all-reduce against a CPU reference proves the COMMS primitive is right (the load-bearing claim).
//
// IP: public NVSHMEM/CUDA + the in-repo k1/k2/k3/k4/k5 warp-per-row idioms only.  Edits nothing else.
//
// ================================ BUILD (8xH100 box, cu12 NVSHMEM) ==============================
//   /usr/local/cuda/bin/nvcc -arch=sm_90a -O3 -rdc=true \
//      -I kernels/ -I /root/nv12/nvidia/nvshmem/include \
//      kernels/megakernel_decode.cu \
//      -L /root/nv12/nvidia/nvshmem/lib -lnvshmem_host -lnvshmem_device -lnvidia-ml -lcuda \
//      -o /tmp/mega
//   (--use_fast_math is intentionally OMITTED: it can perturb the all-reduce correctness check; add
//    it back for a pure-speed run.)
//
// ================================ RUN (8 PEs, 1 node) ===========================================
//   LD_LIBRARY_PATH=/root/nv12/nvidia/nvshmem/lib:$LD_LIBRARY_PATH \
//   NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_DISABLE_IB_NATIVE=1 NVSHMEM_BOOTSTRAP=MPI \
//   mpirun -np 8 --allow-run-as-root /tmp/mega
// ================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cfloat>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "common.cuh"
using namespace q3;
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                         \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));             \
  exit(1); } } while (0)

// ------------------------------------------------------------------------------------------------
// How many transformer layers the persistent kernel loops over in ONE launch.  Start at 4 (a correct,
// timed, in-kernel-comms proof); the orchestrator scales this toward 94 by recompiling with a larger
// value.  us/layer-step is reported so 94-layer tok/s is a trivial extrapolation regardless.
// ------------------------------------------------------------------------------------------------
#ifndef N_LAYERS_TEST
#define N_LAYERS_TEST 4
#endif

// TP/EP slicing.  TP=8: each PE owns 1/8 of the Q heads for attention and 1/8 of the QKV / O-proj
// output rows; the residual is all-reduced across PEs so every PE holds the full [HIDDEN] stream.
// EP=8: each PE owns a disjoint set of experts; for this latency proxy each PE runs TOP_K/?-style
// expert slice work sized so the per-PE expert read volume mirrors a 1/8 EP shard.
constexpr int NPES_EXPECT = 8;
constexpr int QH_PER_PE   = N_Q_HEADS / NPES_EXPECT;     // 8 query heads / PE (64/8)
constexpr int QKV_ROWS_PE = QKV_OUT  / NPES_EXPECT;      // 1152 fused-QKV output rows / PE
constexpr int OROWS_PE    = HIDDEN   / NPES_EXPECT;      // 512 O-proj output rows / PE
// Experts: the model routes TOP_K=8 experts/token over EP=8 PEs.  As a balanced latency proxy each PE
// owns ONE active-expert slot's worth of gate/up/down work (so the 8 PEs cover the 8 routed experts).
constexpr int EXP_PER_PE  = 1;                           // active-expert slots handled by this PE

// ================================================================================================
// Device dot primitive — the repo's coalesced split-K fp8 GEMV inner loop (k5/k1/k3/k4 share it).
// Consecutive lanes read consecutive uint4 (16 fp8) of the SAME weight row -> coalesced 128-bit HBM;
// hardware fp8x2->half2 dequant; 2 accumulators for ILP.  n a multiple of 16.  Valid on lane 0.
// ================================================================================================
static __device__ __forceinline__ float mk_warp_dot(const fp8* __restrict__ w,
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

static __device__ __forceinline__ float mk_silu(float x) { return x / (1.f + __expf(-x)); }

// ================================================================================================
// Resident per-layer state (pointers into HBM + symmetric heap).  ONE layer's dummy weights, reused.
// All buffers are device pointers passed to the persistent kernel as a single struct by value.
// ================================================================================================
struct MegaState {
  // Residual stream on the SYMMETRIC heap (so the in-kernel all-reduce can put/get it across PEs).
  float* h_sym  = nullptr;     // [HIDDEN] residual, symmetric — REPLICATED on every PE (NOT reduced)
  float* delta_sym = nullptr;  // [HIDDEN] this PE's PARTIAL projection (O-proj / MoE), symmetric;
                               //          all-reduced (sum of 8 disjoint partials) then added to h_sym
  float* ar_recv= nullptr;     // [HIDDEN] all-reduce scratch, symmetric

  // staged-activation scratch in plain HBM (per-PE local; produced+consumed within a layer)
  float* y_norm = nullptr;     // [HIDDEN] staged normed activation (K1 / K4 input)
  float* proj   = nullptr;     // [QKV_ROWS_PE] this PE's slice of the QKV projection (q-heads only here)
  float* out_q  = nullptr;     // [QH_PER_PE * HEAD_DIM] this PE's query heads, normed+roped
  float* attn_o = nullptr;     // [QH_PER_PE * HEAD_DIM] this PE's attention output
  float* a_glb  = nullptr;     // [EXP_PER_PE * MOE_INTER] expert gate*up activation

  // K1 weights (this PE's QKV slice).
  float* w_in_norm = nullptr;  // [HIDDEN]
  fp8*   Wqkv = nullptr;  float* Wqkv_scale = nullptr;   // [QKV_ROWS_PE, HIDDEN]
  float* q_norm = nullptr;     // [HEAD_DIM]
  float* rope_cos = nullptr, *rope_sin = nullptr;        // [HEAD_DIM/2]

  // KV cache (this PE's KV heads — TP shards KV with the Q heads; KV_DIM/8 channels here).
  fp8*   kv_k = nullptr, *kv_v = nullptr;                // [ctx_len, KV_DIM] (we read a 1/8 head slice)
  float* kv_k_scale = nullptr, *kv_v_scale = nullptr;    // [KV_DIM]
  int    ctx_len = 0, n_splits = 0;
  float* part_m = nullptr, *part_l = nullptr, *part_acc = nullptr;

  // K3 O-proj (this PE's HIDDEN output slice).
  fp8*   Wo = nullptr;  float* Wo_scale = nullptr;       // [OROWS_PE, Q_DIM]

  // K4 router (replicated; the gate is tiny — 128 x HIDDEN — and every PE picks the same top-8).
  float* w_post_norm = nullptr;                          // [HIDDEN]
  fp8*   Wgate = nullptr;  float* Wgate_scale = nullptr; // [N_EXPERTS, HIDDEN]
  int*   sel_idx = nullptr;  float* sel_w = nullptr;     // [TOP_K]

  // K5 experts (this PE owns EXP_PER_PE expert slots' weights).
  fp8*   Wgu = nullptr;  float* Wgu_scale = nullptr;     // [EXP_PER_PE*2*MOE_INTER, HIDDEN]
  fp8*   Wd  = nullptr;  float* Wd_scale  = nullptr;     // [EXP_PER_PE*HIDDEN, MOE_INTER]

  // final head (lm_head slice on this PE).
  float* w_final_norm = nullptr;                         // [HIDDEN]
  fp8*   Wlm = nullptr;  float* Wlm_scale = nullptr;     // [VOCAB/8, HIDDEN] slice
  int    vocab_pe = 0;
  float* logit_max = nullptr;  int* logit_arg = nullptr; // [grid blocks] per-CTA partial argmax

  int mype = 0, npes = 0;
};

// ================================================================================================
// IN-KERNEL recursive-doubling all-reduce(SUM) over the resident grid + 8 PEs.
// -------------------------------------------------------------------------------------------------
// `acc` (symmetric [n]) starts = this PE's partial, ends = the global sum on every PE.  `recv`
// (symmetric scratch [n]) receives the partner's partial each round.  P=8 -> 3 put+barrier rounds.
// This is the nvshmem_comms.cu idiom, but called INSIDE the persistent kernel — block 0 drives the
// inter-PE put/barrier (one block performs the device barrier), and a grid sync fences the rest of
// this PE's CTAs around it so the residual is consistent before/after.
//
// NOTE: nvshmemx_*_block APIs are block-collective: the whole CALLING block must reach them.  We let
// ONLY block 0 perform the cross-PE exchange (its threads cooperate on the put + barrier), then a
// grid_group::sync() broadcasts the result to all CTAs implicitly (they all read `acc` from the
// symmetric heap afterwards).  All-PE barrier_all_block on one block/PE matches nvshmem_comms.cu.
// ================================================================================================
static __device__ void mk_allreduce_recd(float* __restrict__ acc, float* __restrict__ recv,
                                          int n, int mype, int npes, cg::grid_group& grid) {
  // Ensure every CTA on this PE has finished writing its share of `acc` before the exchange.
  grid.sync();
  if (blockIdx.x == 0) {
    const int tid = threadIdx.x, nthr = blockDim.x;
    for (int mask = 1; mask < npes; mask <<= 1) {
      const int peer = mype ^ mask;
      nvshmemx_float_put_block(recv, acc, n, peer);   // one-sided NVLink put of the whole partial
      nvshmemx_barrier_all_block();                   // every PE delivered this round's partial
      for (int i = tid; i < n; i += nthr) acc[i] += recv[i];
      __syncthreads();
      nvshmemx_barrier_all_block();                   // no PE overwrites recv before peer consumes it
    }
  }
  // Make the reduced `acc` visible to ALL of this PE's CTAs for the next stage.
  grid.sync();
}

// Grid-strided zero of a symmetric [n] buffer across ALL of this PE's CTAs (caller fences with grid.sync).
static __device__ void mk_zero(float* __restrict__ b, int n) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (int i = g; i < n; i += stride) b[i] = 0.f;
}

// Grid-strided residual add: h_sym[i] += delta[i] (the reduced partial), across all of this PE's CTAs.
static __device__ void mk_add_into(float* __restrict__ h, const float* __restrict__ delta, int n) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (int i = g; i < n; i += stride) h[i] += delta[i];
}

// ================================================================================================
// Per-PE block-wide RMSNorm of h_sym[HIDDEN] -> y_norm[HIDDEN] (staged for the GEMVs).
// Every CTA recomputes it locally (HIDDEN=4096 is cheap; avoids a cross-CTA reduce + extra barrier).
// ================================================================================================
static __device__ void mk_rmsnorm(const float* __restrict__ h, const float* __restrict__ w,
                                   float* __restrict__ y_out) {
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
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) y_out[i] = h[i] * rinv * w[i];
}

// ================================================================================================
// K1 — this PE's QKV GEMV slice (Q heads only, the TP shard).  Reads staged y_norm from HBM (already
// produced by mk_rmsnorm into y_norm), warp-per-row over this PE's QKV_ROWS_PE rows, grid-stride.
// We only need this PE's Q-head rows for the local attention; writes raw proj into proj[].
// ================================================================================================
static __device__ void mk_qkv_gemv(const MegaState& S) {
  const float* ys = S.y_norm;                       // staged normed input (HBM; small, hot in L2)
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  // This PE owns QKV_ROWS_PE consecutive QKV output rows (its TP slice); warp-per-row grid-stride.
  for (int o = gwarp; o < QKV_ROWS_PE; o += nwarp) {
    float r = mk_warp_dot(S.Wqkv + (size_t)o * HIDDEN, ys, HIDDEN, lane);
    if (lane == 0) S.proj[o] = r * S.Wqkv_scale[o];
  }
}

// K1 epilogue — per-head QK-norm + RoPE for this PE's QH_PER_PE query heads.  proj[] holds this PE's
// rows; the first QH_PER_PE*HEAD_DIM of them are the Q heads (proxy layout).  Warp-per-head.
static __device__ void mk_q_norm_rope(const MegaState& S) {
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int h = gwarp; h < QH_PER_PE; h += nwarp) {
    const int base = h * HEAD_DIM;
    float chan[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) chan[c] = S.proj[base + c * 32 + lane];
    // per-head RMSNorm over HEAD_DIM (warp-local).
    float ss = 0.f;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) ss += chan[c] * chan[c];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float hn = rsqrtf(ss / HEAD_DIM + RMS_EPS);
    float normed[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) normed[c] = chan[c] * hn * S.q_norm[c * 32 + lane];
    // RoPE (rotate-half; partner of d is d^64 -> same lane, slots 0<->2 and 1<->3).
    float c0 = S.rope_cos[lane],      s0 = S.rope_sin[lane];
    float c1 = S.rope_cos[lane + 32], s1 = S.rope_sin[lane + 32];
    float roped[HEAD_DIM / 32];
    roped[0] = normed[0]*c0 - normed[2]*s0;
    roped[2] = normed[2]*c0 + normed[0]*s0;
    roped[1] = normed[1]*c1 - normed[3]*s1;
    roped[3] = normed[3]*c1 + normed[1]*s1;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++)
      S.out_q[base + c * 32 + lane] = roped[c];
  }
}

// ================================================================================================
// K2 — split-KV flash-decode for this PE's QH_PER_PE query heads (warp-per-(head,split), k2 idiom).
// Pass-1 partials live in part_*; we then run pass-2 reduce.  Both passes are split by gwarp over
// this PE's local heads so the persistent grid stays busy.  KV head = local-qh / GQA within the slice.
// ================================================================================================
static __device__ void mk_flash_partial(const MegaState& S) {
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int n_splits = S.n_splits;
  const int total = QH_PER_PE * n_splits;                 // (head, split) work items on this PE
  const int chunk = (S.ctx_len + n_splits - 1) / n_splits;
  const float scale = rsqrtf((float)HEAD_DIM);
  const fp8* __restrict__ kf8 = S.kv_k;
  const fp8* __restrict__ vf8 = S.kv_v;
  constexpr int CPL = HEAD_DIM / 32;                       // channels per lane (=4)

  for (int item = gwarp; item < total; item += nwarp) {
    const int qh    = item / n_splits;                   // this PE's local query head 0..QH_PER_PE-1
    const int split = item - qh * n_splits;
    // GQA: within this PE's slice the KV head index follows the same q/GQA mapping (proxy uses head 0
    // of the local KV slice so the read shape/volume matches a TP-sharded KV cache).
    const int kvh   = (qh / GQA_GROUP);                  // 0 for a 8-head slice (GQA_GROUP=16)
    const int kv_base = kvh * HEAD_DIM;
    const int t0 = split * chunk;
    const int t1 = min(t0 + chunk, S.ctx_len);

    // LAYOUT: producer mk_q_norm_rope writes Q stride-32 (channel c*32+lane in slot c). We read Q the
    // SAME way and gather K/V at the matching channels c*32+lane so the Q.K dot pairs identical channels
    // (the previous lane*4+c read permuted Q vs the producer — fixed here).
    float qreg[CPL], ksc[CPL], vsc[CPL];
    #pragma unroll
    for (int c = 0; c < CPL; c++) {
      const int ch = c * 32 + lane;                      // this lane's channel for slot c (stride-32)
      qreg[c] = S.out_q[qh * HEAD_DIM + ch];
      ksc[c]  = S.kv_k_scale ? S.kv_k_scale[kv_base + ch] : 1.f;
      vsc[c]  = S.kv_v_scale ? S.kv_v_scale[kv_base + ch] : 1.f;
    }
    float m = -FLT_MAX, l = 0.f, acc[CPL];
    #pragma unroll
    for (int c = 0; c < CPL; c++) acc[c] = 0.f;
    for (int t = t0; t < t1; t++) {
      const fp8* krow = kf8 + (size_t)t * KV_DIM + kv_base;
      const fp8* vrow = vf8 + (size_t)t * KV_DIM + kv_base;
      float kc[CPL], vc[CPL];
      #pragma unroll
      for (int c = 0; c < CPL; c++) { kc[c] = (float)krow[c * 32 + lane]; vc[c] = (float)vrow[c * 32 + lane]; }
      float p = 0.f;
      #pragma unroll
      for (int c = 0; c < CPL; c++) p += qreg[c] * kc[c] * ksc[c];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
      float s = p * scale;
      float m_new = fmaxf(m, s), corr = __expf(m - m_new), pexp = __expf(s - m_new);
      l = l * corr + pexp;
      #pragma unroll
      for (int c = 0; c < CPL; c++) acc[c] = acc[c] * corr + pexp * vc[c] * vsc[c];
      m = m_new;
    }
    const size_t pidx = (size_t)qh * n_splits + split;
    if (lane == 0) { S.part_m[pidx] = m; S.part_l[pidx] = l; }
    float* ao = S.part_acc + pidx * HEAD_DIM;
    #pragma unroll
    for (int c = 0; c < CPL; c++) ao[c * 32 + lane] = acc[c];   // store stride-32 to match the reduce
  }
}

static __device__ void mk_flash_reduce(const MegaState& S) {
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int n_splits = S.n_splits;
  constexpr int CPL = HEAD_DIM / 32;                       // channels per lane (=4)
  for (int qh = gwarp; qh < QH_PER_PE; qh += nwarp) {
    float m = -FLT_MAX, l = 0.f, acc[CPL];
    #pragma unroll
    for (int c = 0; c < CPL; c++) acc[c] = 0.f;
    for (int sp = 0; sp < n_splits; sp++) {
      const size_t pidx = (size_t)qh * n_splits + sp;
      float ms = S.part_m[pidx], ls = S.part_l[pidx];
      if (ls <= 0.f) continue;
      // part_acc is stored stride-32 (channel c*32+lane in slot c) by mk_flash_partial; read it the same.
      const float* ai = S.part_acc + pidx * HEAD_DIM;
      float m_new = fmaxf(m, ms), co = __expf(m - m_new), cs = __expf(ms - m_new);
      l = l * co + ls * cs;
      #pragma unroll
      for (int c = 0; c < CPL; c++) acc[c] = acc[c] * co + ai[c * 32 + lane] * cs;
      m = m_new;
    }
    float inv = (l > 0.f) ? (1.f / l) : 0.f;
    float* o = S.attn_o + qh * HEAD_DIM;
    #pragma unroll
    for (int c = 0; c < CPL; c++) o[c * 32 + lane] = acc[c] * inv;
  }
}

// ================================================================================================
// K3 — O-proj GEMV for this PE's HIDDEN output slice + partial residual.  Each PE produces its
// OROWS_PE rows of (Wo @ attn_local); after the all-reduce sums the 8 partials, every PE holds the
// full O-proj output.  We accumulate into the symmetric residual h_sym (this PE's slice rows) and the
// in-kernel all-reduce(SUM) completes the TP reduction across the 8 head shards.
//
// PROXY: attn_o holds only this PE's QH_PER_PE heads (Q_DIM/8 = 1024 elems), but a full O-proj row is
// Q_DIM=8192 wide.  To keep the GEMV read VOLUME (the bandwidth-bound cost) equal to the real per-PE
// O-proj shard we dot the full Q_DIM weight row against a Q_DIM-wide staged vector built by replicating
// this PE's 1024 attn elems 8x (the values are a proxy; the bytes read = real shard).  We stage that in
// shared memory once per CTA.
// ================================================================================================
static __device__ void mk_oproj_partial(const MegaState& S, float* __restrict__ xs_q /*smem [Q_DIM]*/) {
  // stage a Q_DIM-wide activation from this PE's QH_PER_PE*HEAD_DIM = Q_DIM/8 attn elems (replicate 8x).
  const int local_n = QH_PER_PE * HEAD_DIM;              // 1024
  for (int k = threadIdx.x; k < Q_DIM; k += blockDim.x) xs_q[k] = S.attn_o[k % local_n];
  __syncthreads();
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  // TP O-proj: each PE owns a DISJOINT block of HIDDEN output rows (mype*OROWS_PE .. +OROWS_PE).  It
  // writes ONLY those rows of delta_sym; the all-reduce(SUM) then GATHERS the 8 PEs' disjoint blocks
  // into the full O-proj output on every PE.  delta_sym is zeroed before this stage, so the rows this
  // PE does NOT own stay 0 and the sum is a gather (NOT a multiply of a replicated residual — the v1
  // bug, where partials were atomicAdded onto the already-replicated h_sym and then summed x8).
  const int row_base = S.mype * OROWS_PE;
  for (int o = gwarp; o < OROWS_PE; o += nwarp) {
    float acc = mk_warp_dot(S.Wo + (size_t)o * Q_DIM, xs_q, Q_DIM, lane);
    if (lane == 0) S.delta_sym[row_base + o] = acc * S.Wo_scale[o];
  }
}

// ================================================================================================
// K4 — router: post-RMSNorm(h) staged in y_norm already; gate GEMV over 128 experts (warp-per-expert)
// + softmax + top-8 + renorm.  Replicated on every PE (the gate is tiny and every PE must agree on the
// routing to do its EP slice).  Single CTA does the selection; we run it on block 0 only and use the
// grid sync to publish sel_idx/sel_w to all CTAs (they read it from HBM next stage).
// ================================================================================================
static __device__ void mk_router(const MegaState& S, float* __restrict__ logits /*smem [N_EXPERTS]*/) {
  if (blockIdx.x != 0) return;                            // selection is replicated; one CTA suffices
  const int lane  = threadIdx.x & 31;
  const int gwarp = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;
  const float* ys = S.y_norm;                             // post-RMSNorm activation (staged this stage)
  for (int e = gwarp; e < N_EXPERTS; e += nwarp) {
    float acc = mk_warp_dot(S.Wgate + (size_t)e * HIDDEN, ys, HIDDEN, lane);
    if (lane == 0) logits[e] = acc * S.Wgate_scale[e];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float mx = -FLT_MAX;
    for (int e = 0; e < N_EXPERTS; ++e) mx = fmaxf(mx, logits[e]);
    float sum = 0.f;
    for (int e = 0; e < N_EXPERTS; ++e) sum += __expf(logits[e] - mx);
    const float inv_sum = 1.f / sum;
    float chosen = 0.f;
    for (int s = 0; s < TOP_K; ++s) {
      int bi = -1; float bv = -1.f;
      for (int e = 0; e < N_EXPERTS; ++e) {
        bool taken = false;
        for (int j = 0; j < s; ++j) if (S.sel_idx[j] == e) { taken = true; break; }
        if (taken) continue;
        float p = __expf(logits[e] - mx) * inv_sum;
        if (p > bv) { bv = p; bi = e; }
      }
      S.sel_idx[s] = (bi >= 0 ? bi : s);
      S.sel_w[s]   = (bv >= 0.f ? bv : 0.f);
      chosen      += S.sel_w[s];
    }
    const float inv_chosen = 1.f / chosen;
    for (int s = 0; s < TOP_K; ++s) S.sel_w[s] *= inv_chosen;
  }
}

// ================================================================================================
// K5 — this PE's expert slice.  gate+up SwiGLU into a_glb, then down-proj accumulated into the
// residual h_sym (the EP combine is the SECOND in-kernel all-reduce that follows this stage).
// One PE handles EXP_PER_PE active-expert slots; warp-per-output-row, grid-stride.  y_norm is the
// post-RMSNorm activation (same one the router scored), staged in shared memory per CTA.
// ================================================================================================
static __device__ void mk_expert_gateup(const MegaState& S, float* __restrict__ ys /*smem [HIDDEN]*/) {
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) ys[k] = S.y_norm[k];
  __syncthreads();
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int total = EXP_PER_PE * MOE_INTER;
  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / MOE_INTER;
    const int j    = item - slot * MOE_INTER;
    const fp8*   W = S.Wgu  + (size_t)slot * (2 * MOE_INTER) * HIDDEN;
    const float* Sc= S.Wgu_scale + (size_t)slot * (2 * MOE_INTER);
    float g = mk_warp_dot(W + (size_t)j * HIDDEN,               ys, HIDDEN, lane);
    float u = mk_warp_dot(W + (size_t)(MOE_INTER + j) * HIDDEN, ys, HIDDEN, lane);
    if (lane == 0)
      S.a_glb[(size_t)slot * MOE_INTER + j] = mk_silu(g * Sc[j]) * (u * Sc[MOE_INTER + j]);
  }
}

static __device__ void mk_expert_down(const MegaState& S, float* __restrict__ as /*smem [EXP_PER_PE*MOE_INTER]*/) {
  const int na = EXP_PER_PE * MOE_INTER;
  for (int i = threadIdx.x; i < na; i += blockDim.x) as[i] = S.a_glb[i];
  __syncthreads();
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int total = EXP_PER_PE * HIDDEN;
  // EP down-proj: this PE owns EXP_PER_PE expert slots, each contributing to ALL HIDDEN channels, so
  // this is a genuine SUM of partials across PEs (unlike O-proj's disjoint rows).  Write the partial
  // into delta_sym (which the O-proj all-reduce already drained back to 0); the second all-reduce(SUM)
  // combines the 8 PEs' expert contributions.  atomicAdd handles EXP_PER_PE>1 slots on this PE.
  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / HIDDEN;
    const int o    = item - slot * HIDDEN;
    const float gw = S.sel_w[slot];                       // routed weight for this PE's slot
    const fp8*   W = S.Wd + (size_t)slot * HIDDEN * MOE_INTER;
    const float* Sc= S.Wd_scale + (size_t)slot * HIDDEN;
    float d = mk_warp_dot(W + (size_t)o * MOE_INTER, as + (size_t)slot * MOE_INTER, MOE_INTER, lane);
    if (lane == 0) atomicAdd(&S.delta_sym[o], gw * d * Sc[o]);  // EP partial; all-reduce sums across PEs
  }
}

// ================================================================================================
// THE PERSISTENT MEGAKERNEL — launched ONCE, loops over N_LAYERS_TEST layers, then the head.
// All stages are __device__ calls; the ONLY syncs are intra-PE grid_group::sync() and the in-kernel
// NVSHMEM all-reduce.  NO host relaunch, NO per-collective host launch between layers.
// ================================================================================================
extern "C" __global__ void megakernel_decode(MegaState S, int n_layers) {
  cg::grid_group grid = cg::this_grid();

  // Dynamic shared memory shared by the stages that need a staged activation (sized to the max:
  // Q_DIM floats for the O-proj stage).  Stages that need less just use the prefix.
  extern __shared__ float smem[];
  float* xs_q = smem;                                     // up to [Q_DIM]

  for (int layer = 0; layer < n_layers; ++layer) {
    // ---- K1: input RMSNorm -> y_norm, then QKV GEMV (this PE's slice) ----
    mk_rmsnorm(S.h_sym, S.w_in_norm, S.y_norm);
    grid.sync();                                          // y_norm fully written before the GEMV reads
    mk_qkv_gemv(S);
    grid.sync();                                          // proj fully written before the epilogue
    mk_q_norm_rope(S);
    grid.sync();                                          // out_q ready for flash-decode

    // ---- K2: split-KV flash-decode (this PE's heads) ----
    mk_flash_partial(S);
    grid.sync();                                          // partials ready
    mk_flash_reduce(S);
    grid.sync();                                          // attn_o ready

    // ---- K3: O-proj PARTIAL into delta_sym (this PE's disjoint HIDDEN rows) ----
    mk_zero(S.delta_sym, HIDDEN);                         // rows this PE doesn't own stay 0 -> AR gathers
    grid.sync();
    mk_oproj_partial(S, xs_q);
    // ---- AR1: IN-KERNEL all-reduce(SUM) of the PARTIAL (the TP comm) -> full O-proj output everywhere ----
    mk_allreduce_recd(S.delta_sym, S.ar_recv, HIDDEN, S.mype, S.npes, grid);
    mk_add_into(S.h_sym, S.delta_sym, HIDDEN);            // residual += reduced O-proj partial (added ONCE)
    grid.sync();

    // ---- K4: router (post-RMSNorm staged into y_norm, then gate+top8) ----
    mk_rmsnorm(S.h_sym, S.w_post_norm, S.y_norm);
    grid.sync();
    mk_router(S, xs_q /*reused as [N_EXPERTS] logits scratch in smem prefix*/);
    grid.sync();                                          // sel_idx/sel_w published

    // ---- K5: expert slice (this PE's EP shard) -> PARTIAL into delta_sym ----
    mk_zero(S.delta_sym, HIDDEN);
    grid.sync();
    mk_expert_gateup(S, xs_q /*[HIDDEN] staged y*/);
    grid.sync();                                          // a_glb ready
    mk_expert_down(S, xs_q /*[EXP_PER_PE*MOE_INTER] staged a*/);
    // ---- AR2: SECOND in-kernel all-reduce(SUM) (the EP combine: sum of per-PE expert partials) ----
    mk_allreduce_recd(S.delta_sym, S.ar_recv, HIDDEN, S.mype, S.npes, grid);
    mk_add_into(S.h_sym, S.delta_sym, HIDDEN);            // residual += reduced MoE partial (added ONCE)
    grid.sync();
  }

  // ---- final head (done ONCE, not per layer): RMSNorm + lm_head GEMV slice + per-CTA partial argmax.
  mk_rmsnorm(S.h_sym, S.w_final_norm, S.y_norm);
  grid.sync();
  {
    const int lane  = threadIdx.x & 31;
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarp = (gridDim.x * blockDim.x) >> 5;
    const float* ys = S.y_norm;
    float my_max = -3.0e38f; int my_arg = -1;
    for (int row = gwarp; row < S.vocab_pe; row += nwarp) {
      float v = mk_warp_dot(S.Wlm + (size_t)row * HIDDEN, ys, HIDDEN, lane);
      if (lane == 0) { v *= S.Wlm_scale[row]; if (v > my_max) { my_max = v; my_arg = row; } }
    }
    __shared__ float smax[32]; __shared__ int sarg[32];
    const int wid = threadIdx.x >> 5, nwc = blockDim.x >> 5;
    if (lane == 0) { smax[wid] = my_max; sarg[wid] = my_arg; }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bm = -3.0e38f; int ba = -1;
      for (int w = 0; w < nwc; ++w) if (smax[w] > bm) { bm = smax[w]; ba = sarg[w]; }
      S.logit_max[blockIdx.x] = bm; S.logit_arg[blockIdx.x] = ba;
    }
  }
  // (final cross-CTA + cross-PE argmax reduce is a tiny host/extra step; omitted from the hot loop.)
}

// ================================================================================================
// Host helpers — deterministic fill (same splitmix idiom as decode_step.cu / k5_experts.cu).
// ================================================================================================
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

// Forward decl: standalone in-kernel all-reduce proof (defined at end of file).
__global__ void mk_ar_proof(float* acc, float* recv, int n, int mype, int npes);

// Smoke test: a multi-block grid_group::sync() launched via nvshmemx_collective_launch.  The WHOLE
// megakernel design rests on collective_launch performing a cooperative launch (so grid.sync() is
// legal); grid.sync() is UB on a non-cooperative launch and would HANG.  This tiny kernel proves the
// composition works (or fails fast) BEFORE we trust any timing.  flag[0] is set 1 only after the sync.
__global__ void mk_coop_smoke(int* flag) {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) *flag = 1;
}

// ================================================================================================
// main — runs on every PE.  Bootstraps NVSHMEM, allocates the resident state, launches the persistent
// megakernel ONCE per timed iter (the launch IS the whole decode step), and reports us/layer-step +
// the projection to 94 layers + lm_head.  Also validates the in-kernel all-reduce vs a CPU reference.
// ================================================================================================
int main(int argc, char** argv) {
  const int ctx_len = (argc > 1) ? atoi(argv[1]) : 4096;
  const int iters   = (argc > 2) ? atoi(argv[2]) : 100;
  const int warmup  = 10;
  const double PEAK = 3350.0;   // GB/s, single H100 HBM3 (for context only)

  nvshmem_init();
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  int n_dev = 0; CK(cudaGetDeviceCount(&n_dev));
  const int dev = (n_dev > 0) ? (mype % n_dev) : 0;
  CK(cudaSetDevice(dev));

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  if (mype == 0) {
    printf("== Qwen3-235B-A22B PERSISTENT MEGAKERNEL decode (latency proxy, TP=%d/EP=%d) ==\n", npes, npes);
    printf("device: %s  SMs=%d  ctx_len=%d  N_LAYERS_TEST=%d  iters=%d\n",
           prop.name, prop.multiProcessorCount, ctx_len, N_LAYERS_TEST, iters);
    if (npes != NPES_EXPECT)
      printf("  NOTE: expected %d PEs (8x H100); running with %d (slice constants assume 8).\n",
             NPES_EXPECT, npes);
  }

  cudaStream_t s; CK(cudaStreamCreate(&s));

  // ---- allocate the resident state (ONE layer's dummy weights; reused — latency proxy) ----------
  MegaState S{}; S.mype = mype; S.npes = npes; S.ctx_len = ctx_len;
  S.n_splits = 64; { int mbc = (ctx_len + 31) / 32; if (S.n_splits > mbc) S.n_splits = mbc; if (S.n_splits < 1) S.n_splits = 1; }

  // symmetric heap: residual + partial-delta + all-reduce scratch (must be symmetric for the in-kernel put).
  S.h_sym     = (float*)nvshmem_malloc(sizeof(float) * HIDDEN);
  S.delta_sym = (float*)nvshmem_malloc(sizeof(float) * HIDDEN);
  S.ar_recv   = (float*)nvshmem_malloc(sizeof(float) * HIDDEN);
  if (!S.h_sym || !S.delta_sym || !S.ar_recv) { printf("PE %d: nvshmem_malloc failed\n", mype); nvshmem_global_exit(2); }

  // plain HBM scratch / weights.
  CK(cudaMalloc(&S.y_norm, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&S.proj,   QKV_ROWS_PE * sizeof(float)));
  CK(cudaMalloc(&S.out_q,  QH_PER_PE * HEAD_DIM * sizeof(float)));
  CK(cudaMalloc(&S.attn_o, QH_PER_PE * HEAD_DIM * sizeof(float)));
  CK(cudaMalloc(&S.a_glb,  EXP_PER_PE * MOE_INTER * sizeof(float)));

  CK(cudaMalloc(&S.w_in_norm, HIDDEN * sizeof(float)));        fill_f32(S.w_in_norm, HIDDEN, 1u, 0.5f, true);
  CK(cudaMalloc(&S.Wqkv, (size_t)QKV_ROWS_PE * HIDDEN * sizeof(fp8)));  fill_fp8(S.Wqkv, (size_t)QKV_ROWS_PE*HIDDEN, 2u + mype);
  CK(cudaMalloc(&S.Wqkv_scale, QKV_ROWS_PE * sizeof(float)));  fill_f32(S.Wqkv_scale, QKV_ROWS_PE, 3u, 0.02f, true);
  CK(cudaMalloc(&S.q_norm, HEAD_DIM * sizeof(float)));         fill_f32(S.q_norm, HEAD_DIM, 4u, 0.5f, true);
  CK(cudaMalloc(&S.rope_cos, (HEAD_DIM/2) * sizeof(float)));
  CK(cudaMalloc(&S.rope_sin, (HEAD_DIM/2) * sizeof(float)));
  { std::vector<float> rc(HEAD_DIM/2), rs(HEAD_DIM/2);
    for (int i = 0; i < HEAD_DIM/2; ++i) { float f = powf(ROPE_THETA, -2.f*i/HEAD_DIM)*7.f; rc[i]=cosf(f); rs[i]=sinf(f); }
    CK(cudaMemcpy(S.rope_cos, rc.data(), (HEAD_DIM/2)*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(S.rope_sin, rs.data(), (HEAD_DIM/2)*sizeof(float), cudaMemcpyHostToDevice)); }

  CK(cudaMalloc(&S.kv_k, (size_t)ctx_len * KV_DIM * sizeof(fp8)));  fill_fp8(S.kv_k, (size_t)ctx_len*KV_DIM, 20u);
  CK(cudaMalloc(&S.kv_v, (size_t)ctx_len * KV_DIM * sizeof(fp8)));  fill_fp8(S.kv_v, (size_t)ctx_len*KV_DIM, 21u);
  CK(cudaMalloc(&S.kv_k_scale, KV_DIM * sizeof(float))); fill_f32(S.kv_k_scale, KV_DIM, 22u, 0.04f, true);
  CK(cudaMalloc(&S.kv_v_scale, KV_DIM * sizeof(float))); fill_f32(S.kv_v_scale, KV_DIM, 23u, 0.04f, true);
  CK(cudaMalloc(&S.part_m,  (size_t)QH_PER_PE * S.n_splits * sizeof(float)));
  CK(cudaMalloc(&S.part_l,  (size_t)QH_PER_PE * S.n_splits * sizeof(float)));
  CK(cudaMalloc(&S.part_acc,(size_t)QH_PER_PE * S.n_splits * HEAD_DIM * sizeof(float)));

  CK(cudaMalloc(&S.Wo, (size_t)OROWS_PE * Q_DIM * sizeof(fp8)));  fill_fp8(S.Wo, (size_t)OROWS_PE*Q_DIM, 30u + mype);
  CK(cudaMalloc(&S.Wo_scale, OROWS_PE * sizeof(float)));          fill_f32(S.Wo_scale, OROWS_PE, 31u, 0.02f, true);

  CK(cudaMalloc(&S.w_post_norm, HIDDEN * sizeof(float)));         fill_f32(S.w_post_norm, HIDDEN, 40u, 0.5f, true);
  CK(cudaMalloc(&S.Wgate, (size_t)N_EXPERTS * HIDDEN * sizeof(fp8)));  fill_fp8(S.Wgate, (size_t)N_EXPERTS*HIDDEN, 41u);
  CK(cudaMalloc(&S.Wgate_scale, N_EXPERTS * sizeof(float)));      fill_f32(S.Wgate_scale, N_EXPERTS, 42u, 0.02f, true);
  CK(cudaMalloc(&S.sel_idx, TOP_K * sizeof(int)));
  CK(cudaMalloc(&S.sel_w,   TOP_K * sizeof(float)));
  { std::vector<int> si(TOP_K); std::vector<float> sw(TOP_K, 1.0f/TOP_K);
    for (int i=0;i<TOP_K;++i) si[i]=i;
    CK(cudaMemcpy(S.sel_idx, si.data(), TOP_K*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(S.sel_w,   sw.data(), TOP_K*sizeof(float), cudaMemcpyHostToDevice)); }

  const size_t gu_n = (size_t)EXP_PER_PE * 2 * MOE_INTER * HIDDEN;
  const size_t d_n  = (size_t)EXP_PER_PE * HIDDEN * MOE_INTER;
  CK(cudaMalloc(&S.Wgu, gu_n * sizeof(fp8)));  fill_fp8(S.Wgu, gu_n, 50u + mype);
  CK(cudaMalloc(&S.Wd,  d_n  * sizeof(fp8)));  fill_fp8(S.Wd,  d_n,  70u + mype);
  CK(cudaMalloc(&S.Wgu_scale, EXP_PER_PE * 2 * MOE_INTER * sizeof(float))); fill_f32(S.Wgu_scale, EXP_PER_PE*2*MOE_INTER, 90u, 0.02f, true);
  CK(cudaMalloc(&S.Wd_scale,  EXP_PER_PE * HIDDEN * sizeof(float)));        fill_f32(S.Wd_scale,  EXP_PER_PE*HIDDEN,       110u, 0.02f, true);

  // lm_head slice (VOCAB/8 rows per PE).
  S.vocab_pe = (VOCAB + npes - 1) / npes;
  CK(cudaMalloc(&S.w_final_norm, HIDDEN * sizeof(float))); fill_f32(S.w_final_norm, HIDDEN, 130u, 0.5f, true);
  CK(cudaMalloc(&S.Wlm, (size_t)S.vocab_pe * HIDDEN * sizeof(fp8)));  fill_fp8(S.Wlm, (size_t)S.vocab_pe*HIDDEN, 131u);
  CK(cudaMalloc(&S.Wlm_scale, S.vocab_pe * sizeof(float)));           fill_f32(S.Wlm_scale, S.vocab_pe, 132u, 0.02f, true);

  // ---- pick a grid that is CO-RESIDENT (required for cg::grid_group::sync + collective_launch) ----
  // The megakernel must have ALL its CTAs simultaneously resident for the grid barrier to be legal.
  // Query the max blocks/SM for this kernel at our block size + dynamic smem, then cap the grid to
  // (blocks/SM * #SMs).  Dynamic smem = Q_DIM floats (the O-proj stage's staged activation).
  const int block = 256;
  const size_t smem = (size_t)Q_DIM * sizeof(float);     // 32 KB
  CK(cudaFuncSetAttribute(megakernel_decode, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int blocks_per_sm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, megakernel_decode, block, smem));
  if (blocks_per_sm < 1) blocks_per_sm = 1;
  int grid_blocks = blocks_per_sm * prop.multiProcessorCount;
  if (grid_blocks < 1) grid_blocks = prop.multiProcessorCount;

  // per-CTA argmax partials (one entry per resident block).
  CK(cudaMalloc(&S.logit_max, grid_blocks * sizeof(float)));
  CK(cudaMalloc(&S.logit_arg, grid_blocks * sizeof(int)));

  if (mype == 0)
    printf("co-resident grid: %d blocks (%d/SM x %d SMs), block=%d, dyn smem=%zu KB\n",
           grid_blocks, blocks_per_sm, prop.multiProcessorCount, block, smem / 1024);

  // ============== COOPERATIVE-LAUNCH SMOKE TEST (validate the load-bearing assumption) ==========
  // The megakernel uses cg::grid_group::sync() across `grid_blocks` blocks, which is legal ONLY if
  // nvshmemx_collective_launch performs a cooperative launch on this build.  If it does not, grid.sync()
  // is UB and the persistent kernel would HANG.  Launch a tiny multi-block grid.sync proof FIRST and
  // confirm it both launches (rc==0) and completes (flag==1) before trusting the megakernel.
  {
    int* coop_flag = nullptr; CK(cudaMalloc(&coop_flag, sizeof(int)));
    CK(cudaMemset(coop_flag, 0, sizeof(int)));
    const int smoke_blocks = (grid_blocks >= 2) ? grid_blocks : 1;   // need >=2 blocks to exercise it
    void* sargs[] = { (void*)&coop_flag };
    dim3 sg(smoke_blocks), sb(32);
    int rc = nvshmemx_collective_launch((const void*)mk_coop_smoke, sg, sb, sargs, 0, s);
    if (rc != 0) {
      printf("PE %d: collective_launch(coop_smoke) rc=%d — collective_launch did NOT accept a %d-block\n"
             "        cooperative grid.  The megakernel's grid.sync() is UNSAFE on this build; fall back\n"
             "        to cudaLaunchCooperativeKernel + a single-block NVSHMEM AR.  Aborting.\n",
             mype, rc, smoke_blocks);
      nvshmem_global_exit(4);
    }
    CK(cudaStreamSynchronize(s));   // if collective_launch were non-cooperative, grid.sync() hangs here
    int hf = 0; CK(cudaMemcpy(&hf, coop_flag, sizeof(int), cudaMemcpyDeviceToHost));
    if (hf != 1) { printf("PE %d: coop smoke flag=%d (grid.sync did not complete)\n", mype, hf); nvshmem_global_exit(4); }
    nvshmem_barrier_all();
    if (mype == 0) printf("  [check] cooperative grid.sync() under collective_launch OK (%d blocks)\n", smoke_blocks);
    CK(cudaFree(coop_flag));
  }

  // ================================ CORRECTNESS (in-kernel all-reduce) =========================
  // Seed the residual deterministically per-PE, run ONE layer-step's worth via a 0-layer launch that
  // still performs... actually run a dedicated single all-reduce by launching the megakernel with
  // n_layers=0 won't exercise AR.  Instead: seed h_sym, launch a tiny 1-round proof kernel? — simpler:
  // launch the full megakernel with n_layers=1 from a KNOWN residual and check that after the two
  // all-reduces every PE's residual equals the SUM over PEs of the per-PE O-proj+MoE partials. That is
  // hard to predict in closed form (depends on fp8 weights), so we instead validate the all-reduce
  // PRIMITIVE directly with a standalone seeded reduce, identical to nvshmem_comms.cu's check.
  {
    std::vector<float> seed(HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) seed[i] = 0.001f * (float)(i % 257) + 0.5f * (float)mype + 1.0f;
    CK(cudaMemcpy(S.h_sym, seed.data(), HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
    nvshmem_barrier_all();
    // Launch a 1-block proof that runs the SAME recursive-doubling put+barrier path the megakernel's
    // mk_allreduce_recd uses (block 0 drives it).  A single block needs no grid sync, so this isolates
    // and validates the in-kernel COMMS primitive against a closed-form CPU reference.
    int n_h = HIDDEN;
    void* args2[] = { (void*)&S.h_sym, (void*)&S.ar_recv, (void*)&n_h, (void*)&S.mype, (void*)&S.npes };
    dim3 g1(1), b1(256);
    int rc = nvshmemx_collective_launch((const void*)mk_ar_proof, g1, b1, args2, 0, s);
    if (rc != 0) { printf("PE %d: collective_launch(ar_proof) rc=%d\n", mype, rc); nvshmem_global_exit(3); }
    CK(cudaStreamSynchronize(s)); nvshmem_barrier_all();
    std::vector<float> got(HIDDEN);
    CK(cudaMemcpy(got.data(), S.h_sym, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
    double maxerr = 0.0; int bad = -1;
    for (int i = 0; i < HIDDEN; ++i) {
      double ref = 0.0; for (int p = 0; p < npes; ++p) ref += 0.001*(double)(i%257) + 0.5*(double)p + 1.0;
      double e = fabs((double)got[i] - ref);
      if (e > maxerr) { maxerr = e; if (e > 1e-2) bad = i; }
    }
    if (bad >= 0) { printf("PE %d: in-kernel all-reduce MISMATCH at i=%d got=%g maxerr=%g\n",
                           mype, bad, got[bad], maxerr); nvshmem_global_exit(2); }
    if (mype == 0) printf("  [check] in-kernel recdouble all-reduce OK (maxerr=%.2e)\n", maxerr);
  }

  // re-seed residual to something finite before timing (values don't affect launch-bound timing).
  fill_f32(S.h_sym, HIDDEN, 99u, 1.0f, false);
  nvshmem_barrier_all();

  // ================================ LAUNCH + TIME ==============================================
  // The persistent megakernel IS the whole decode step.  We launch it ONCE per timed iter with
  // n_layers = N_LAYERS_TEST and measure wall time; us/layer-step = time / N_LAYERS_TEST.
  int n_layers = N_LAYERS_TEST;
  void* kargs[] = { (void*)&S, (void*)&n_layers };
  dim3 grid(grid_blocks), blk(block);
  auto launch = [&]() {
    int rc = nvshmemx_collective_launch((const void*)megakernel_decode, grid, blk, kargs, smem, s);
    if (rc != 0) { printf("PE %d: collective_launch(megakernel) rc=%d\n", mype, rc); nvshmem_global_exit(3); }
  };

  for (int it = 0; it < warmup; ++it) launch();
  CK(cudaStreamSynchronize(s)); nvshmem_barrier_all();

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0, s));
  for (int it = 0; it < iters; ++it) launch();
  CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
  float ms_total; CK(cudaEventElapsedTime(&ms_total, e0, e1));
  const double us_step  = (double)ms_total * 1e3 / iters;          // us per persistent launch (N_LAYERS_TEST layers + head)
  const double us_layer = us_step / N_LAYERS_TEST;                 // us per layer-step (the headline)
  nvshmem_barrier_all();

  if (mype == 0) {
    // Project to the full 94-layer model: 94 layer-steps + the single lm_head/head (folded into the
    // per-step measurement's tail; we extrapolate the layer cost and add one head pass ~= 1 layer-step).
    const double us_94   = us_layer * N_LAYERS + us_layer;         // 94 layers + ~1 head pass
    const double tok_per_s = 1.0e6 / us_94;
    const int    AR_PER_TOK = 2 * N_LAYERS;                        // 188 in-kernel all-reduces / token
    printf("\n  %-40s %12.2f us\n", "persistent launch (N_LAYERS_TEST + head)", us_step);
    printf("  %-40s %12.3f us\n", "-> per LAYER-STEP (K1..AR2)", us_layer);
    printf("  %-40s %12.3f ms\n", "projected 94-layer step", us_94 / 1e3);
    printf("  %-40s %12.1f tok/s  (pre-spec)\n", "projected decode throughput", tok_per_s);
    printf("\n  in-kernel all-reduces/token: %d (2/layer x %d) — issued with ZERO host relaunches.\n",
           AR_PER_TOK, N_LAYERS);
    printf("  baselines this session: NCCL AR 35us -> %.2f ms/tok just in comms (~%.0f tok/s cap);\n",
           35.0 * AR_PER_TOK / 1e3, 1.0e6 / (35.0 * AR_PER_TOK));
    printf("                          NVSHMEM put+barrier 17us -> %.2f ms/tok (~%.0f tok/s cap).\n",
           17.0 * AR_PER_TOK / 1e3, 1.0e6 / (17.0 * AR_PER_TOK));
    printf("  the megakernel folds BOTH the per-kernel AND per-collective launch overhead into ONE\n");
    printf("  resident launch; remaining cost is the in-kernel barrier + the bandwidth-bound GEMVs.\n");
    printf("  (latency proxy: 1 layer's dummy weights reused %dx; read VOLUME/shape = real per-PE shard.)\n",
           N_LAYERS_TEST);
    (void)PEAK;
    fflush(stdout);
  }

  // ---- teardown (best-effort) ----
  nvshmem_barrier_all();
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  nvshmem_free(S.h_sym); nvshmem_free(S.delta_sym); nvshmem_free(S.ar_recv);
  CK(cudaStreamDestroy(s));
  nvshmem_finalize();
  return 0;
}

// ================================================================================================
// Standalone correctness proof for the in-kernel all-reduce primitive (same recursive-doubling path
// the megakernel uses).  ONE block; uses the device-side put/barrier idiom directly (no grid sync
// needed for a single block).  Launched via nvshmemx_collective_launch from the correctness section.
// ================================================================================================
__global__ void mk_ar_proof(float* acc, float* recv, int n, int mype, int npes) {
  const int tid = threadIdx.x, nthr = blockDim.x;
  for (int mask = 1; mask < npes; mask <<= 1) {
    const int peer = mype ^ mask;
    nvshmemx_float_put_block(recv, acc, n, peer);
    nvshmemx_barrier_all_block();
    for (int i = tid; i < n; i += nthr) acc[i] += recv[i];
    __syncthreads();
    nvshmemx_barrier_all_block();
  }
}
