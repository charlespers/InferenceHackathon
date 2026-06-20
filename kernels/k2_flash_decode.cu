// K2 — single-query flash-decode attention, B=1, GQA 16:1, head_dim 128, sm_90a.
// Split-KV across CTAs (a lone query underfills 132 SMs) + 2-pass online-softmax reduce.
// In-register KV dequant (fp8 e4m3, per-channel scale).  GQA broadcast: KV head = q_head / 16.
// scale = 1/sqrt(128).
//
// Parallelism (matches the repo's warp idiom):
//   * WARP-PER-HEAD.  One warp owns (one q_head, one KV split).  HEAD_DIM=128 = 4 elems/lane, so the
//     q-dot and the acc[] accumulator are register-resident (4 floats/lane) and reductions are warp
//     shuffles -- no shared memory, no __syncthreads in the hot loop.
//   * SPLIT-KV: grid.x = n_splits CTAs of KV-time-chunks, grid.y groups q_heads.  With 64 q_heads and
//     S splits we launch ~64*S warps; choose S so 64*S >> 132 SMs (see k2_pick_splits).
//   * Pass 1 (partial): each warp streams its [t0,t1) KV chunk with online softmax, emits (m,l,acc).
//   * Pass 2 (reduce):  one warp per q_head merges the S partials via the log-sum-exp trick.
//
// Partial buffers, laid out [q_head][split]:
//   part_m  [N_Q_HEADS * n_splits]
//   part_l  [N_Q_HEADS * n_splits]
//   part_acc[N_Q_HEADS * n_splits * HEAD_DIM]
//
// Build:  nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k2_flash_decode.cu -o /tmp/k2
#include "common.cuh"
#include <cfloat>
using namespace q3;

#ifndef Q3_K2_DEFS
#define Q3_K2_DEFS
// HEAD_DIM=128 split across a 32-lane warp -> 4 elements per lane.
constexpr int K2_VPL = HEAD_DIM / 32;     // 4 values per lane

// Reduce a per-lane partial sum across the warp, broadcast to all lanes.
static __device__ __forceinline__ float k2_warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;                                  // identical on all lanes after xor-fan
}
#endif // Q3_K2_DEFS

// ---- Pass 1: per (q_head, split) online softmax over the KV chunk -> partial (m, l, acc[128]) ----
// q:        [Q_DIM]   normed+roped query (from K1)
// kv_k/kv_v:[ctx_len, KV_DIM]   fp8 cache (KV_DIM = N_KV_HEADS*HEAD_DIM, head g at [.. , g*HEAD_DIM ..])
// kv_k_scale/kv_v_scale: per-channel dequant scale, length KV_DIM (indexed by kv-head channel).
extern "C" __global__ void k2_flash_decode_partial(
    const float* __restrict__ q,
    const fp8*  __restrict__ kv_k, const fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale,
    int ctx_len, int n_splits,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc) {
  const int lane  = threadIdx.x & 31;
  const int wid   = threadIdx.x >> 5;
  const int qh    = blockIdx.y * (blockDim.x >> 5) + wid;   // query head 0..63
  if (qh >= N_Q_HEADS) return;
  const int split = blockIdx.x;                             // KV-chunk 0..n_splits-1
  if (split >= n_splits) return;
  const int kvh   = qh / GQA_GROUP;                         // GQA broadcast -> KV head 0..3
  const int chunk = (ctx_len + n_splits - 1) / n_splits;
  const int t0 = split * chunk, t1 = min(t0 + chunk, ctx_len);
  const float scale = rsqrtf((float)HEAD_DIM);

  // Each lane loads its 4 query elements for this head: d in {lane, lane+32, lane+64, lane+96}.
  float qreg[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) qreg[c] = q[qh * HEAD_DIM + c*32 + lane];

  // KV head base offset (channels) inside the KV_DIM row.
  const int kv_base = kvh * HEAD_DIM;

  // online softmax state (register, per warp): running max m, denom l, weighted acc[4 per lane].
  float m = -FLT_MAX, l = 0.f, acc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) acc[c] = 0.f;

  for (int t = t0; t < t1; t++) {
    const fp8* krow = kv_k + (size_t)t * KV_DIM + kv_base;
    // dot(q, k_t) -- each lane its 4 channels, then warp reduce.
    float p = 0.f;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) {
      int d = c*32 + lane;
      float kv = float(krow[d]) * (kv_k_scale ? kv_k_scale[kv_base + d] : 1.f);
      p += qreg[c] * kv;
    }
    float s = k2_warp_sum(p) * scale;                       // logit, identical on all lanes

    // online softmax update.
    float m_new = fmaxf(m, s);
    float corr  = __expf(m - m_new);                        // rescale old state
    float pexp  = __expf(s - m_new);
    l = l * corr + pexp;
    const fp8* vrow = kv_v + (size_t)t * KV_DIM + kv_base;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) {
      int d = c*32 + lane;
      float vv = float(vrow[d]) * (kv_v_scale ? kv_v_scale[kv_base + d] : 1.f);
      acc[c] = acc[c] * corr + pexp * vv;
    }
    m = m_new;
  }

  // write partials, laid out [qh][split] (and acc as [qh][split][HEAD_DIM]).
  const size_t pidx = (size_t)qh * n_splits + split;
  if (lane == 0) { part_m[pidx] = m; part_l[pidx] = l; }
  float* ao = part_acc + pidx * HEAD_DIM;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) ao[c*32 + lane] = acc[c];
}

// ---- Pass 2: merge n_splits partials per head -> attn_out[Q_DIM] (log-sum-exp combine) ----
// One warp per q_head.  Reads (m,l,acc) for each split, combines online, normalizes by total l.
extern "C" __global__ void k2_flash_decode_reduce(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, int n_splits, float* __restrict__ attn_out) {
  const int lane = threadIdx.x & 31;
  const int qh   = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (qh >= N_Q_HEADS) return;

  float m = -FLT_MAX, l = 0.f, acc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) acc[c] = 0.f;

  for (int sp = 0; sp < n_splits; sp++) {
    const size_t pidx = (size_t)qh * n_splits + sp;
    float ms = part_m[pidx], ls = part_l[pidx];
    if (ls <= 0.f) continue;                                // empty split (t0>=ctx_len) contributes 0
    const float* ai = part_acc + pidx * HEAD_DIM;
    float m_new = fmaxf(m, ms);
    float corr_o = __expf(m - m_new);                       // rescale running
    float corr_s = __expf(ms - m_new);                      // rescale incoming split
    l = l * corr_o + ls * corr_s;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++)
      acc[c] = acc[c] * corr_o + ai[c*32 + lane] * corr_s;
    m = m_new;
  }
  float inv = (l > 0.f) ? (1.f / l) : 0.f;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++)
    attn_out[qh * HEAD_DIM + c*32 + lane] = acc[c] * inv;
}

// ---- launch helpers ----------------------------------------------------------------------------
// Pick a split count so the partial pass fills the 132 SMs.  We have N_Q_HEADS=64 head-warps; with S
// splits there are 64*S warps.  Want enough CTAs: pack W warps/CTA -> 64*S/W CTAs.  Target >= ~2x SMs.
// Also cap so each split has a reasonable chunk (>=64 timesteps) to amortize the per-warp overhead.
static inline int k2_pick_splits(int ctx_len) {
  // ctx 4k -> ~8 splits (512/chunk) ; ctx 32k -> ~16 splits (2k/chunk). Heuristic, clamped.
  int s = 8;
  if (ctx_len > 8192)  s = 16;
  if (ctx_len > 32768) s = 32;
  int max_by_chunk = (ctx_len + 63) / 64;                   // >=64 timesteps/split
  if (s > max_by_chunk) s = max_by_chunk;
  if (s < 1) s = 1;
  return s;
}

#ifdef Q3_K2_LAUNCH_HELPER
// Allocates nothing; caller owns part_* (size with k2_partials_bytes).  Returns the split count used.
static inline size_t k2_partials_elems_m(int n_splits) { return (size_t)N_Q_HEADS * n_splits; }
static inline size_t k2_partials_elems_acc(int n_splits){ return (size_t)N_Q_HEADS * n_splits * HEAD_DIM; }

static inline int k2_launch(
    const float* q, const fp8* kv_k, const fp8* kv_v,
    const float* kv_k_scale, const float* kv_v_scale, int ctx_len,
    float* part_m, float* part_l, float* part_acc, float* attn_out,
    int n_splits = -1, cudaStream_t stream = 0) {
  if (n_splits <= 0) n_splits = k2_pick_splits(ctx_len);
  const int warps_per_cta = 4;                              // 128 threads/CTA -> 4 head-warps
  const int block = warps_per_cta * 32;
  dim3 gP(n_splits, (N_Q_HEADS + warps_per_cta - 1) / warps_per_cta);
  k2_flash_decode_partial<<<gP, block, 0, stream>>>(
      q, kv_k, kv_v, kv_k_scale, kv_v_scale, ctx_len, n_splits, part_m, part_l, part_acc);
  dim3 gR((N_Q_HEADS + warps_per_cta - 1) / warps_per_cta);
  k2_flash_decode_reduce<<<gR, block, 0, stream>>>(part_m, part_l, part_acc, n_splits, attn_out);
  return n_splits;
}
#endif
