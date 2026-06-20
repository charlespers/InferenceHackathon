// ep_moe_sharded.cu — EXPERT-PARALLEL (EP=8) sharded MoE for Qwen3-235B-A22B, B=1 DECODE.
// Target: 8x H100 (sm_90a), one process driving all 8 GPUs, NCCL for the per-layer combine.
// Standard CUDA + NCCL only.  This is the EP=8 ALTERNATIVE to the TP=8 intermediate-shard
// (decode_step_tp8.cu) for the MoE term, written so the two can be benched head-to-head at B=1.
//
// THE TWO WAYS TO SHARD A QWEN3 MoE LAYER ACROSS 8 GPUs (contrast)
// ---------------------------------------------------------------------------------------------
//   TP=8 (intermediate / column parallel, see decode_step_tp8.cu):
//     * EVERY rank holds 1536/8 = 192 of EVERY expert's intermediate columns.  For the 8 active
//       experts each rank runs 192-wide SKINNY GEMVs (gate/up [2*192, HIDDEN], down [HIDDEN, 192]).
//     * Perfectly balanced: every rank reads exactly 8/8 of each active expert's 1/8 share -> the
//       per-rank MoE byte volume is deterministic, ZERO balls-in-bins gamble.
//     * Cost: the down-proj contraction is only 192 wide (vs 1536), so the warp-per-row dot is
//       short — less work to amortize launch/latency, and the inner reduction is 12x shorter.
//
//   EP=8 (expert parallel, THIS FILE):
//     * The 128 experts are DISTRIBUTED across the 8 ranks, 16 experts/rank: rank r owns the
//       contiguous block experts [r*16, r*16+16).  Each rank stores the FULL gate/up/down for its
//       16 experts at the full MOE_INTER=1536.
//     * For the top-8 active experts, each rank computes the FULL expert (gate+up SwiGLU at 1536,
//       then down at 1536->HIDDEN) for ONLY the active experts it owns, reusing the repo's proven
//       warp-per-output-row coalesced-fp8 GEMV (k5_experts.cu's warp_dot_fp8, the in-repo fast
//       GEMV) at the full width — IDENTICAL to the single-GPU k5 inner loop.
//     * A rank that owns 0 active experts does no expert math; a rank that owns k active experts
//       does k FULL experts.  Then ONE NCCL all-reduce(SUM) over [HIDDEN] combines the per-rank
//       partial MoE outputs into the residual (every rank ends with the full MoE contribution).
//     * Cost: LOAD IMBALANCE.  8 active experts dropped into 8 owner-bins is balls-in-bins:
//         - uniform routing: E[max bin] ~ 2.6 active experts on the busiest rank (vs 1 if perfect),
//           so the step is paced by ~2-3 full experts even though the average rank does 1.
//         - REAL Qwen3 routing is correlated/peaked: a hot rank can hold 5-8 of the 8 active experts
//           (worst case all 8 land on one rank -> that rank does the ENTIRE MoE, 8x the average and
//           no faster than single-GPU for the MoE term).  EP needs expert-placement/balancing or a
//           token-dispatch all-to-all to be competitive at B=1; TP8 sidesteps this entirely.
//
// WHY MEASURE BOTH:  at B=1 the MoE is HBM-bandwidth-bound.  TP8 reads a fixed 1/8 of the 8 active
// experts per rank (~1.78 MB/layer/rank); EP8 reads (active-experts-owned-by-this-rank) FULL experts
// — average 1/8 but with a heavy tail.  This file reports the per-rank byte volume, the busiest-rank
// multiplier under uniform vs adversarial routing, and the real us/token of the EP=8 combine so the
// orchestrator can compare the MoE term against the TP8 number.
//
// FUSION / BANDWIDTH STRATEGY (reused verbatim from k5_experts.cu, the validated fast path):
//   * warp-per-output-row with split-K across the warp's 32 lanes: consecutive lanes read
//     consecutive 16-byte (uint4 = 16 fp8) chunks of the SAME weight row -> fully coalesced HBM.
//   * 128-bit vectorized fp8 loads + hardware fp8x2->half2 dequant + 2 FP accumulators for ILP.
//   * gate+up fused so the staged activation y[HIDDEN] is read once; down folds the routing weight
//     and the per-out-channel scale into the epilogue and accumulates into the partial via atomicAdd.
//
// LATENCY-PROXY DISCLAIMER (same convention as decode_step_tp8.cu): only one layer's worth of dummy
// fp8 expert weights is resident per GPU and reused; the produced hidden is meaningless, but the
// per-rank HBM read volume, the kernel/grid shapes, and the NCCL combine are the real ones — so the
// measured us/token and the all-reduce overhead are representative.  Correctness is checked against a
// CPU fp32 reference that mirrors the kernel exactly after the fp8 round-trip (< 1e-2).
//
// BUILD (on the 8xH100 box; NCCL via pip `nvidia-nccl-cu12`):
//   NCCL_INC=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'include'))")
//   NCCL_LIB=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'lib'))")
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -I "$NCCL_INC" \
//        kernels/ep_moe_sharded.cu -L "$NCCL_LIB" -lnccl -o /tmp/epmoe
//   LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" /tmp/epmoe
//   (If NCCL is system-installed, plain `-lnccl` suffices and nccl.h is on the default include path.)
//
// =================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>
#include <cuda_runtime.h>
#include <nccl.h>
#include "common.cuh"
using namespace q3;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                       \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));           \
  exit(1); } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) {                      \
  printf("NCCL err %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r_));           \
  exit(1); } } while (0)

// =================================================================================================
// EP=8 geometry.
// =================================================================================================
constexpr int EP            = 8;
constexpr int EXPERTS_RANK  = N_EXPERTS / EP;                 // 16 experts owned per rank (128/8)
static_assert(N_EXPERTS % EP == 0, "experts must distribute evenly across EP ranks");
// rank r owns experts [r*EXPERTS_RANK, (r+1)*EXPERTS_RANK).
static __host__ __device__ __forceinline__ int owner_of(int expert) { return expert / EXPERTS_RANK; }
static __host__ __device__ __forceinline__ int local_of(int expert) { return expert % EXPERTS_RANK; }

// =================================================================================================
// Device dot primitive — the in-repo fast GEMV (k5_experts.cu warp_dot_fp8), reproduced locally so
// this file is one self-contained translation unit and never edits k5/common.cuh.
//   Warp dots a contiguous K-major fp8 weight row against a staged f32 activation, split-K across the
//   32 lanes (consecutive lanes -> consecutive uint4 = 16 fp8 -> coalesced 128-bit HBM), hardware
//   fp8x2->half2 dequant, 2 accumulators for ILP.  n must be a multiple of 16 (HIDDEN=4096,
//   MOE_INTER=1536 both are).  Result valid on lane 0.
// =================================================================================================
static __device__ __forceinline__ float ep_warp_dot_fp8(const fp8* __restrict__ w,
                                                         const float* __restrict__ ys,
                                                         int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;                                      // 16 fp8 per uint4
  for (int v = lane; v < nv; v += 32) {                       // lanes 0..31 -> consecutive uint4
    uint4 p = wv[v];
    const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
    const float* yy = ys + (v << 4);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {                             // 4 x 32-bit words = 4 x (2 fp8 pairs)
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
  return acc;                                                 // valid on lane 0
}

// =================================================================================================
// EP kernels — FULL-WIDTH experts (MOE_INTER=1536), but only over the active experts THIS RANK OWNS.
// ---------------------------------------------------------------------------------------------
// The host pre-builds, on each rank, a compact list of the active slots that map to locally-owned
// experts: `loc_slot[i]` (which of the 8 top-k slots) and `loc_expert[i]` (its LOCAL expert index
// 0..15) for i in [0, n_local).  n_local is the number of active experts that landed on this rank
// (the balls-in-bins load for this rank this token).  The kernels iterate only those i, so a rank
// with n_local=0 launches a grid that immediately exits.
// =================================================================================================

// Kernel A — fused gate+up at FULL width:  a[i][j] = silu(s_g*<y,gate_j>) * (s_u*<y,up_j>), j in [0,1536).
//   Wgu[le] is the LOCAL expert le's stacked [2*MOE_INTER, HIDDEN] gate|up matrix (rows [0,1536) gate,
//   [1536,3072) up).  One warp per (i, j); grid-stride over n_local*MOE_INTER.  y staged once per CTA.
//   a_glb is indexed by the DENSE local-active index i (0..n_local-1), NOT the global slot — compact.
extern "C" __global__ void ep_k5a_gateup(
    const float* __restrict__ y,
    const int* __restrict__ loc_expert,                       // [n_local] LOCAL expert id 0..15
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    float* __restrict__ a_glb, int n_local) {
  extern __shared__ float ys[];                               // [HIDDEN]
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) ys[k] = y[k];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int total = n_local * MOE_INTER;
  for (int item = gwarp; item < total; item += nwarp) {
    const int i  = item / MOE_INTER;
    const int j  = item - i * MOE_INTER;
    const int le = loc_expert[i];
    const fp8*   W = Wgu[le];
    const float* S = Wgu_scale[le];
    const float g = ep_warp_dot_fp8(W + (size_t)j * HIDDEN,                ys, HIDDEN, lane);
    const float u = ep_warp_dot_fp8(W + (size_t)(MOE_INTER + j) * HIDDEN,  ys, HIDDEN, lane);
    if (lane == 0)
      a_glb[(size_t)i * MOE_INTER + j] = silu(g * S[j]) * (u * S[MOE_INTER + j]);
  }
}

// Kernel B — full-width down + routed accumulate:  h_part[o] += sel_w * s_d * <a[i], down_o>, o in [0,HIDDEN).
//   Wd[le] is LOCAL expert le's [HIDDEN, MOE_INTER] down matrix.  One warp per (i, o); grid-stride.
//   The full a buffer (n_local*MOE_INTER floats) is staged once per CTA; the routing weight sel_w[i]
//   and the per-out-channel down scale are folded into the epilogue, accumulated into the PARTIAL
//   MoE output via atomicAdd (the rank's active experts race on the same HIDDEN rows, then the cross-
//   rank sum is the NCCL all-reduce).  Launch with dynamic smem = n_local*MOE_INTER*sizeof(float).
extern "C" __global__ void ep_k5b_down(
    const int* __restrict__ loc_expert,                       // [n_local] LOCAL expert id 0..15
    const float* __restrict__ loc_w,                          // [n_local] routing weight for this slot
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_part, int n_local) {
  extern __shared__ float as[];                               // [n_local*MOE_INTER]
  const int na = n_local * MOE_INTER;
  for (int i = threadIdx.x; i < na; i += blockDim.x) as[i] = a_glb[i];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int total = n_local * HIDDEN;
  for (int item = gwarp; item < total; item += nwarp) {
    const int i  = item / HIDDEN;
    const int o  = item - i * HIDDEN;
    const int le = loc_expert[i];
    const float gw = loc_w[i];
    const fp8*   W = Wd[le];
    const float* S = Wd_scale[le];
    const float d = ep_warp_dot_fp8(W + (size_t)o * MOE_INTER, as + (size_t)i * MOE_INTER,
                                    MOE_INTER, lane);
    if (lane == 0) atomicAdd(&h_part[o], gw * d * S[o]);
  }
}

// =================================================================================================
// Per-rank device state (one rank == one GPU; owns EXPERTS_RANK=16 full experts of one dummy layer).
// =================================================================================================
struct RankState {
  int rank = 0, dev = 0;
  cudaStream_t stream = nullptr;
  ncclComm_t   comm   = nullptr;

  // 16 owned full experts: gate+up [2*MOE_INTER, HIDDEN], down [HIDDEN, MOE_INTER], per-row scales.
  const fp8   **Wgu_d = nullptr;  const float **Wgu_scale_d = nullptr;   // [EXPERTS_RANK] ptr arrays
  const fp8   **Wd_d  = nullptr;  const float **Wd_scale_d  = nullptr;
  std::vector<fp8*>   Wgu_dp, Wd_dp;                          // owning storage (for cleanup)
  std::vector<float*> Sgu_dp, Sd_dp;

  float *y = nullptr;                                         // [HIDDEN] post-norm MoE input (staged)
  float *a_glb = nullptr;                                     // [EXPERTS_RANK*MOE_INTER] (caps n_local)
  float *moe_partial = nullptr;                               // [HIDDEN] this rank's partial MoE output

  // compact local-active lists (rebuilt per token from the global top-8 routing).
  int   *loc_expert = nullptr;                                // [EXPERTS_RANK] local expert ids
  float *loc_w      = nullptr;                                // [EXPERTS_RANK] routing weights
  int    n_local    = 0;                                      // active experts owned this token

  int    block = 256;
  size_t smemA = 0;                                           // HIDDEN floats (staged y)
};

// =================================================================================================
// CPU fp32 reference — full MoE over all 8 active experts (NOT sharded), mirroring the kernels after
// the fp8 round-trip.  Used to validate the cross-rank EP sum == the single-GPU answer.
//   For each active expert e with weight gw:
//     a_j = silu( s_g[j] * sum_k y_k deq(Wgu[gate_j,k]) ) * ( s_u[j] * sum_k y_k deq(Wgu[up_j,k]) )
//     h_o += gw * s_d[o] * sum_j a_j deq(Wd[o,j])
// Weights are passed as the global [N_EXPERTS] fp8 arrays (the same bytes uploaded to the GPUs).
// =================================================================================================
static void ep_reference(const float* y, const int* sel_idx, const float* sel_w,
                         const std::vector<std::vector<fp8>>&   Wgu,
                         const std::vector<std::vector<float>>& Sgu,
                         const std::vector<std::vector<fp8>>&   Wd,
                         const std::vector<std::vector<float>>& Sd,
                         float* h_out) {
  std::vector<float> a(MOE_INTER);
  for (int o = 0; o < HIDDEN; ++o) h_out[o] = 0.f;
  for (int s = 0; s < TOP_K; ++s) {
    const int   e  = sel_idx[s];
    const float gw = sel_w[s];
    const fp8*   Wg = Wgu[e].data();
    const float* Sg = Sgu[e].data();
    for (int j = 0; j < MOE_INTER; ++j) {
      const fp8* grow = Wg + (size_t)j * HIDDEN;
      const fp8* urow = Wg + (size_t)(MOE_INTER + j) * HIDDEN;
      double g = 0.0, u = 0.0;
      for (int k = 0; k < HIDDEN; ++k) {
        g += (double)y[k] * (double)(float)grow[k];
        u += (double)y[k] * (double)(float)urow[k];
      }
      float gs = (float)g * Sg[j];
      float us = (float)u * Sg[MOE_INTER + j];
      a[j] = (gs / (1.0f + expf(-gs))) * us;
    }
    const fp8*   Wdn = Wd[e].data();
    const float* Sdn = Sd[e].data();
    for (int o = 0; o < HIDDEN; ++o) {
      const fp8* drow = Wdn + (size_t)o * MOE_INTER;
      double acc = 0.0;
      for (int j = 0; j < MOE_INTER; ++j) acc += (double)a[j] * (double)(float)drow[j];
      h_out[o] += gw * (float)acc * Sdn[o];
    }
  }
}

// =================================================================================================
// Build each rank's compact local-active list from the global top-8 routing (host-side; the routing
// is replicated, so every rank derives its own slice independently with no comms).  Returns the
// busiest-rank n_local for load-imbalance reporting.
// =================================================================================================
static int build_local_lists(std::vector<RankState>& R, const std::vector<int>& sel_idx,
                             const std::vector<float>& sel_w) {
  int busiest = 0;
  for (int r = 0; r < EP; ++r) {
    std::vector<int>   le; std::vector<float> lw;
    for (int s = 0; s < TOP_K; ++s) {
      if (owner_of(sel_idx[s]) == r) { le.push_back(local_of(sel_idx[s])); lw.push_back(sel_w[s]); }
    }
    R[r].n_local = (int)le.size();
    busiest = std::max(busiest, R[r].n_local);
    CK(cudaSetDevice(R[r].dev));
    if (R[r].n_local > 0) {
      CK(cudaMemcpy(R[r].loc_expert, le.data(), le.size()*sizeof(int),   cudaMemcpyHostToDevice));
      CK(cudaMemcpy(R[r].loc_w,      lw.data(), lw.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  return busiest;
}

// =================================================================================================
// Enqueue ONE EP=8 MoE layer on a rank's stream: full-width experts for the owned active experts,
// then ONE NCCL all-reduce(SUM) over [HIDDEN] to combine partials.  Mirrors decode_step_tp8.cu's
// collective-ordering contract: every rank issues the SAME single collective per layer so NCCL
// matches them; the driver enqueues ALL ranks before any sync.
// =================================================================================================
static int ep_grid(int rows, int warps_per_cta) {
  if (rows <= 0) return 1;                                    // empty grid still launches 1 CTA -> exits
  int need = (rows + warps_per_cta - 1) / warps_per_cta;
  return std::min(std::max(need, 1), 264);                   // cap ~2 CTAs/SM at 1024 threads
}

static void enqueue_ep_moe_layer(RankState& S) {
  cudaStream_t s = S.stream;
  const int warps = S.block >> 5;

  // pure partial: accumulate from 0 (the residual add lives in the full decode step, not here).
  CK(cudaMemsetAsync(S.moe_partial, 0, HIDDEN * sizeof(float), s));

  if (S.n_local > 0) {
    const int    ctasA = ep_grid(S.n_local * MOE_INTER, warps);
    const int    ctasB = ep_grid(S.n_local * HIDDEN,   warps);
    const size_t smemB = (size_t)S.n_local * MOE_INTER * sizeof(float);
    // smemB grows with n_local; opt in to the max we could need (busiest rank: EXPERTS_RANK experts,
    // but at B=1 only TOP_K can ever be active -> cap at min(EXPERTS_RANK, TOP_K)).  Set per launch.
    CK(cudaFuncSetAttribute(ep_k5b_down, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB));
    ep_k5a_gateup<<<ctasA, S.block, S.smemA, s>>>(
        S.y, S.loc_expert, S.Wgu_d, S.Wgu_scale_d, S.a_glb, S.n_local);
    ep_k5b_down<<<ctasB, S.block, smemB, s>>>(
        S.loc_expert, S.loc_w, S.Wd_d, S.Wd_scale_d, S.a_glb, S.moe_partial, S.n_local);
  }

  // ---- COMBINE: ONE all-reduce(SUM) of the partial MoE output across the 8 ranks -> full MoE term.
  //   Every rank participates EVERY layer (even n_local==0 ranks contribute their zeroed partial), so
  //   the collective always matches across the clique.  An all-to-all token dispatch would be the
  //   alternative for B>1; at B=1 with a single token the all-reduce of the [HIDDEN] partial is the
  //   simplest correct combine and the cheapest collective (one tiny ~16 KB message).
  NK(ncclGroupStart());
  NK(ncclAllReduce(S.moe_partial, S.moe_partial, HIDDEN, ncclFloat32, ncclSum, S.comm, s));
  NK(ncclGroupEnd());
}

// =================================================================================================
// Allocation + dummy weights per rank (16 owned full experts of ONE layer, reused — latency proxy).
// The SAME deterministic bytes are mirrored on the host so the CPU reference matches the fp8 round-trip.
// =================================================================================================
static inline unsigned hashu(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static inline float frnd(unsigned seed, size_t i, float scale, bool positive) {
  unsigned h = hashu((unsigned)(i * 2654435761u) ^ (seed * 40503u));
  float v = (((h % 2001) / 1000.0f) - 1.0f) * scale;
  return positive ? (fabsf(v) + 1e-3f) : v;
}
// Deterministic per-GLOBAL-expert seeds, so host reference (indexed by global e) and the device shard
// (indexed by local le on owner rank) build byte-identical weights.
static inline unsigned seed_gu(int e) { return 50u + (unsigned)e; }
static inline unsigned seed_d (int e) { return 700u + (unsigned)e; }
static inline unsigned seed_sg(int e) { return 1300u + (unsigned)e; }
static inline unsigned seed_sd(int e) { return 1900u + (unsigned)e; }

static void alloc_rank(RankState& S) {
  CK(cudaSetDevice(S.dev));
  const size_t gu_n = (size_t)2 * MOE_INTER * HIDDEN;         // 3072*4096 fp8 per full expert
  const size_t d_n  = (size_t)HIDDEN * MOE_INTER;            // 4096*1536 fp8 per full expert

  S.Wgu_dp.resize(EXPERTS_RANK); S.Wd_dp.resize(EXPERTS_RANK);
  S.Sgu_dp.resize(EXPERTS_RANK); S.Sd_dp.resize(EXPERTS_RANK);
  std::vector<fp8*>   guh(EXPERTS_RANK), wdh(EXPERTS_RANK);
  std::vector<float*> sgh(EXPERTS_RANK), sdh(EXPERTS_RANK);
  for (int le = 0; le < EXPERTS_RANK; ++le) {
    const int e = S.rank * EXPERTS_RANK + le;                 // GLOBAL expert id owned here
    CK(cudaMalloc(&S.Wgu_dp[le], gu_n * sizeof(fp8)));
    CK(cudaMalloc(&S.Wd_dp[le],  d_n  * sizeof(fp8)));
    CK(cudaMalloc(&S.Sgu_dp[le], 2 * MOE_INTER * sizeof(float)));
    CK(cudaMalloc(&S.Sd_dp[le],  HIDDEN * sizeof(float)));
    { std::vector<fp8> h(gu_n); for (size_t i=0;i<gu_n;++i) h[i]=(fp8)frnd(seed_gu(e), i, 0.25f, false);
      CK(cudaMemcpy(S.Wgu_dp[le], h.data(), gu_n*sizeof(fp8), cudaMemcpyHostToDevice)); }
    { std::vector<fp8> h(d_n);  for (size_t i=0;i<d_n; ++i) h[i]=(fp8)frnd(seed_d(e),  i, 0.25f, false);
      CK(cudaMemcpy(S.Wd_dp[le],  h.data(), d_n*sizeof(fp8),  cudaMemcpyHostToDevice)); }
    { std::vector<float> h(2*MOE_INTER); for (int i=0;i<2*MOE_INTER;++i) h[i]=frnd(seed_sg(e), i, 0.02f, true);
      CK(cudaMemcpy(S.Sgu_dp[le], h.data(), 2*MOE_INTER*sizeof(float), cudaMemcpyHostToDevice)); }
    { std::vector<float> h(HIDDEN); for (int i=0;i<HIDDEN;++i) h[i]=frnd(seed_sd(e), i, 0.02f, true);
      CK(cudaMemcpy(S.Sd_dp[le],  h.data(), HIDDEN*sizeof(float), cudaMemcpyHostToDevice)); }
    guh[le]=S.Wgu_dp[le]; wdh[le]=S.Wd_dp[le]; sgh[le]=S.Sgu_dp[le]; sdh[le]=S.Sd_dp[le];
  }
  CK(cudaMalloc(&S.Wgu_d,       EXPERTS_RANK*sizeof(fp8*)));   CK(cudaMemcpy(S.Wgu_d,       guh.data(), EXPERTS_RANK*sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.Wd_d,        EXPERTS_RANK*sizeof(fp8*)));   CK(cudaMemcpy(S.Wd_d,        wdh.data(), EXPERTS_RANK*sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.Wgu_scale_d, EXPERTS_RANK*sizeof(float*))); CK(cudaMemcpy(S.Wgu_scale_d, sgh.data(), EXPERTS_RANK*sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&S.Wd_scale_d,  EXPERTS_RANK*sizeof(float*))); CK(cudaMemcpy(S.Wd_scale_d,  sdh.data(), EXPERTS_RANK*sizeof(float*), cudaMemcpyHostToDevice));

  CK(cudaMalloc(&S.y, HIDDEN * sizeof(float)));
  { std::vector<float> h(HIDDEN); for (int k=0;k<HIDDEN;++k) h[k]=frnd(99u, k, 1.0f, false);
    CK(cudaMemcpy(S.y, h.data(), HIDDEN*sizeof(float), cudaMemcpyHostToDevice)); }
  // a_glb is sized for the max possible active experts on one rank (cap = min(EXPERTS_RANK, TOP_K)).
  const int cap_local = std::min(EXPERTS_RANK, TOP_K);
  CK(cudaMalloc(&S.a_glb, (size_t)cap_local * MOE_INTER * sizeof(float)));
  CK(cudaMalloc(&S.moe_partial, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&S.loc_expert, EXPERTS_RANK * sizeof(int)));
  CK(cudaMalloc(&S.loc_w,      EXPERTS_RANK * sizeof(float)));

  S.smemA = (size_t)HIDDEN * sizeof(float);
  CK(cudaFuncSetAttribute(ep_k5a_gateup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S.smemA));
  CK(cudaDeviceSynchronize());
}

static void free_rank(RankState& S) {
  CK(cudaSetDevice(S.dev));
  for (int le = 0; le < EXPERTS_RANK; ++le) {
    cudaFree(S.Wgu_dp[le]); cudaFree(S.Wd_dp[le]); cudaFree(S.Sgu_dp[le]); cudaFree(S.Sd_dp[le]);
  }
  cudaFree(S.Wgu_d); cudaFree(S.Wd_d); cudaFree(S.Wgu_scale_d); cudaFree(S.Wd_scale_d);
  cudaFree(S.y); cudaFree(S.a_glb); cudaFree(S.moe_partial); cudaFree(S.loc_expert); cudaFree(S.loc_w);
}

// =================================================================================================
// main() — one process, 8 GPUs, NCCL.  Validates the EP=8 combine vs a single-GPU CPU reference,
// then measures B=1 us/token for (a) a UNIFORM-routing top-8 and (b) an ADVERSARIAL all-on-one-rank
// top-8, to bracket the EP load-imbalance cost against the balanced TP8 number.
// =================================================================================================
int main(int argc, char** argv) {
  const int    IT   = (argc > 1) ? atoi(argv[1]) : 300;
  const double PEAK = (argc > 2) ? atof(argv[2]) : 3350.0;     // GB/s per H100 HBM3
  const int    WARM = 30;

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < EP) {
    printf("Need >= %d CUDA devices for EP=%d; found %d.\n", EP, EP, ndev); return 1;
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("== Qwen3-235B-A22B EP=8 sharded MoE (B=1 decode, latency proxy) ==\n");
  printf("device0: %s  SMs=%d  HBM peak=%.0f GB/s  EP=%d  experts/rank=%d  inter=%d  iters=%d\n",
         prop.name, prop.multiProcessorCount, PEAK, EP, EXPERTS_RANK, MOE_INTER, IT);

  // ---- enable peer access so NCCL uses NVLink P2P (NVSwitch makes all pairs peers) ----
  for (int i = 0; i < EP; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < EP; ++j) if (i != j) {
      int can = 0; cudaDeviceCanAccessPeer(&can, i, j);
      if (can) cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  // ---- NCCL: one communicator clique across the 8 local GPUs (single-process) ----
  std::vector<RankState> R(EP);
  std::vector<ncclComm_t> comms(EP);
  std::vector<int> devs(EP);
  for (int r = 0; r < EP; ++r) devs[r] = r;
  NK(ncclCommInitAll(comms.data(), EP, devs.data()));

  for (int r = 0; r < EP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    CK(cudaSetDevice(r));
    CK(cudaStreamCreate(&R[r].stream));
    alloc_rank(R[r]);
  }

  // =============================================================================================
  // Correctness: pick a top-8 routing, run the EP combine, compare against the CPU fp32 reference.
  // We rebuild the GLOBAL weight set on the host (same deterministic bytes as the device shards) so
  // the reference reads the exact fp8 round-trip the kernels read.
  // =============================================================================================
  std::vector<std::vector<fp8>>   Wgu_h(N_EXPERTS), Wd_h(N_EXPERTS);
  std::vector<std::vector<float>> Sgu_h(N_EXPERTS), Sd_h(N_EXPERTS);
  const size_t gu_n = (size_t)2 * MOE_INTER * HIDDEN, d_n = (size_t)HIDDEN * MOE_INTER;
  for (int e = 0; e < N_EXPERTS; ++e) {
    Wgu_h[e].resize(gu_n); Wd_h[e].resize(d_n); Sgu_h[e].resize(2*MOE_INTER); Sd_h[e].resize(HIDDEN);
    for (size_t i=0;i<gu_n;++i) Wgu_h[e][i]=(fp8)frnd(seed_gu(e), i, 0.25f, false);
    for (size_t i=0;i<d_n; ++i) Wd_h[e][i] =(fp8)frnd(seed_d(e),  i, 0.25f, false);
    for (int i=0;i<2*MOE_INTER;++i) Sgu_h[e][i]=frnd(seed_sg(e), i, 0.02f, true);
    for (int i=0;i<HIDDEN;++i)      Sd_h[e][i] =frnd(seed_sd(e), i, 0.02f, true);
  }
  std::vector<float> y_h(HIDDEN);
  for (int k=0;k<HIDDEN;++k) y_h[k]=frnd(99u, k, 1.0f, false);

  // A spread-out top-8 (one expert on each of the 8 ranks) — exercises every rank's combine path.
  std::vector<int>   sel_spread(TOP_K);
  std::vector<float> sel_w(TOP_K);
  for (int s=0;s<TOP_K;++s) { sel_spread[s] = s * EXPERTS_RANK + (s % EXPERTS_RANK);  // rank s, local s
                              sel_w[s] = 0.05f + 0.02f * s; }
  { float sum=0; for (float w: sel_w) sum+=w; for (float& w: sel_w) w/=sum; }        // renormalize to 1

  std::vector<float> ref(HIDDEN), got(HIDDEN);
  ep_reference(y_h.data(), sel_spread.data(), sel_w.data(), Wgu_h, Sgu_h, Wd_h, Sd_h, ref.data());

  int busiest_spread = build_local_lists(R, sel_spread, sel_w);
  for (int r = 0; r < EP; ++r) { CK(cudaSetDevice(r)); enqueue_ep_moe_layer(R[r]); }
  for (int r = 0; r < EP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(R[r].stream)); }
  CK(cudaSetDevice(0));
  CK(cudaMemcpy(got.data(), R[0].moe_partial, HIDDEN*sizeof(float), cudaMemcpyDeviceToHost));  // post-AR = full

  double max_abs = 0.0, max_rel = 0.0;
  for (int i = 0; i < HIDDEN; ++i) {
    double ad = fabs((double)ref[i] - (double)got[i]);
    max_abs = std::max(max_abs, ad);
    max_rel = std::max(max_rel, ad / (fabs((double)ref[i]) + 1e-6));
  }
  printf("\ncorrectness (EP all-reduce == single-GPU 8-expert reference): max_abs=%.3e max_rel=%.3e -> %s (<1e-2)\n",
         max_abs, max_rel, (max_abs < 1e-2 ? "PASS" : "FAIL"));
  printf("  spread routing busiest-rank n_local=%d (1 active expert/rank, the balanced EP best case)\n",
         busiest_spread);

  // =============================================================================================
  // Per-rank byte accounting.  EP reads FULL experts; the busiest rank's read paces the step.
  // =============================================================================================
  const double b_full_expert = (double)gu_n + (double)d_n;    // bytes (1 byte/fp8) read per FULL expert
  printf("\nper-FULL-expert fp8 read: %.2f MB (gate+up %.2f + down %.2f)\n",
         b_full_expert/1e6, (double)gu_n/1e6, (double)d_n/1e6);
  printf("  EP per-rank read = n_local x %.2f MB.  TP8 reads a fixed 1/8 of all 8 = %.2f MB/rank (balanced).\n",
         b_full_expert/1e6, 8.0 * b_full_expert / 8.0 / 1e6);
  printf("  LOAD IMBALANCE (8 active experts into 8 owner bins):\n");
  printf("    perfect (TP8):            1 expert-equiv/rank -> %.2f MB/rank\n", b_full_expert/1e6);
  printf("    uniform routing E[max]:   ~2.6 experts on busiest rank -> ~%.2f MB/rank (~2.6x)\n", 2.6*b_full_expert/1e6);
  printf("    real Qwen3 (peaked):      5-8 experts on a hot rank -> %.2f-%.2f MB/rank (5-8x; worst=single-GPU MoE)\n",
         5.0*b_full_expert/1e6, 8.0*b_full_expert/1e6);

  // ---- timing helper: rebuild lists for a given routing, enqueue all ranks, sync all, time it. ----
  auto bench_routing = [&](const std::vector<int>& sel, const std::vector<float>& sw) -> std::pair<float,int> {
    std::vector<float> swn = sw; { float s=0; for (float w:swn) s+=w; for (float& w:swn) w/=s; }
    int busiest = build_local_lists(R, sel, swn);
    for (int i = 0; i < WARM; ++i) {
      for (int r = 0; r < EP; ++r) { CK(cudaSetDevice(r)); enqueue_ep_moe_layer(R[r]); }
      for (int r = 0; r < EP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(R[r].stream)); }
    }
    cudaEvent_t e0, e1; CK(cudaSetDevice(0)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, R[0].stream));
    for (int i = 0; i < IT; ++i) {
      for (int r = 0; r < EP; ++r) { CK(cudaSetDevice(r)); enqueue_ep_moe_layer(R[r]); }
      for (int r = 0; r < EP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(R[r].stream)); }
    }
    CK(cudaEventRecord(e1, R[0].stream)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= IT;
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return {ms, busiest};
  };

  // (a) BALANCED EP best case: 1 active expert per rank (spread routing above).
  std::pair<float,int> rb = bench_routing(sel_spread, sel_w);
  float ms_bal = rb.first; int busy_bal = rb.second;

  // (b) ADVERSARIAL worst case: all 8 active experts owned by rank 0 (the balls-in-bins tail).
  std::vector<int>   sel_hot(TOP_K);
  std::vector<float> selw_hot(TOP_K, 1.0f/TOP_K);
  for (int s=0;s<TOP_K;++s) sel_hot[s] = s;                   // experts 0..7 -> all owned by rank 0
  std::pair<float,int> rh = bench_routing(sel_hot, selw_hot);
  float ms_hot = rh.first; int busy_hot = rh.second;

  // (c) A realistic-ish skew: 3 experts on the hot rank, the other 5 spread (E[max]~2.6 ballpark).
  std::vector<int>   sel_skew = {0, 1, 2, 16, 32, 48, 80, 112};   // ranks: 0,0,0,1,2,3,5,7 -> busiest=3
  std::vector<float> selw_skew(TOP_K, 1.0f/TOP_K);
  std::pair<float,int> rs = bench_routing(sel_skew, selw_skew);
  float ms_skew = rs.first; int busy_skew = rs.second;

  // =============================================================================================
  // report.  us/token here is the MoE term only (one layer); x N_LAYERS for the per-token MoE cost.
  // =============================================================================================
  auto busiest_bytes = [&](int n){ return (double)n * b_full_expert; };
  auto gbps = [&](float ms, int n){ return busiest_bytes(n) / 1e6 / ms; };   // busiest-rank GB/s
  printf("\n  %-34s %10s %10s %12s %12s\n", "EP=8 MoE (1 layer, B=1)", "busiest", "us/layer", "GB/s(busy)", "%HBMpeak");
  printf("  %-34s %10d %10.2f %12.1f %11.1f%%\n", "balanced (1 expert/rank)",
         busy_bal, ms_bal*1e3, gbps(ms_bal, busy_bal), 100.0*gbps(ms_bal,busy_bal)/PEAK);
  printf("  %-34s %10d %10.2f %12.1f %11.1f%%\n", "skewed (3 on hot rank)",
         busy_skew, ms_skew*1e3, gbps(ms_skew, busy_skew), 100.0*gbps(ms_skew,busy_skew)/PEAK);
  printf("  %-34s %10d %10.2f %12.1f %11.1f%%\n", "adversarial (all 8 on rank0)",
         busy_hot, ms_hot*1e3, gbps(ms_hot, busy_hot), 100.0*gbps(ms_hot,busy_hot)/PEAK);
  printf("\n  per-token MoE (x %d layers): balanced %.2f ms | skewed %.2f ms | adversarial %.2f ms\n",
         N_LAYERS, ms_bal*N_LAYERS, ms_skew*N_LAYERS, ms_hot*N_LAYERS);
  printf("  CONTRAST vs TP8: TP8's 192-wide skinny GEMVs are balls-in-bins-FREE (fixed 1/8/rank);"
         " EP8 wins only when routing is balanced and loses the tail (adversarial ~= single-GPU MoE).\n");
  printf("  Each layer pays ONE %0.1f-KB NCCL all-reduce (sum) to combine partials (tiny -> latency-bound).\n",
         HIDDEN*sizeof(float)/1024.0);
  printf("== done ==\n");

  for (int r = 0; r < EP; ++r) free_rank(R[r]);
  for (int r = 0; r < EP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamDestroy(R[r].stream)); }
  for (int r = 0; r < EP; ++r) ncclCommDestroy(comms[r]);
  return 0;
}
