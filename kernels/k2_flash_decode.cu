// K2 — single-query flash-decode attention, B=1, GQA 16:1, head_dim 128, sm_90a (H100).
// Split-KV across CTAs (a lone query underfills 132 SMs) + 2-pass online-softmax reduce.
// In-register KV dequant (fp8 e4m3, per-channel scale).  GQA broadcast: KV head = q_head / 16.
// scale = 1/sqrt(128).
//
// =================================================================================================
// WHY THIS WAS SLOW (and what changed)
// -------------------------------------------------------------------------------------------------
// The old kernel read only ~4.2 MB of KV at ctx 4096 yet took ~252 us/token (~17 GB/s, 0.5% of HBM
// peak).  For a read that small the kernel should be latency-bound (tens of us), so it was ~10-60x
// too slow on pure overhead, not bandwidth.  Two first-order problems:
//
//   1) UNDER-PARALLELIZED partial pass.  The launch helper defaulted to 8 splits at ctx 4096, so a
//      single warp streamed a 512-timestep KV chunk *serially* with a dependent online-softmax
//      recurrence (each step waits on the previous m/l/acc).  Latency, not bandwidth, dominated.
//      The autotune sweep already showed "more splits = faster" (64 best) — that is exactly this:
//      more splits = shorter dependent chains = more CTAs in flight to hide HBM latency.
//
//   2) SCALAR fp8 LOADS.  The dot loaded one fp8 at a time via `float(krow[d])`, so the dequant ran
//      at one cast/instruction per byte instead of streaming 128-bit transactions.  We now load each
//      lane's 4 contiguous fp8 channels as one 32-bit word and dequant with the hardware
//      fp8x2->half2 path (the k5_experts.cu idiom), one coalesced 128-byte K (and V) transaction per
//      timestep across the warp.
//
// FIXES
//   * Default split count is now occupancy-driven: target ~enough warps to fill the 132 SMs several
//     times over while keeping each split's chunk long enough to amortize the per-warp epilogue.  At
//     ctx 4096 this picks 64 (matching the sweep): 64 q_heads * 64 splits = 4096 warps in flight.
//   * Vectorized, coalesced fp8 KV loads with fp8x2->half2 dequant; per-channel scales hoisted into
//     registers once (they are constant across the time loop).
//   * Lane L now owns the 4 *contiguous* channels [4L, 4L+4) (a single 32-bit fp8 word) instead of a
//     32-strided pattern, so each timestep's K/V row is a single coalesced transaction.
//
// Parallelism (unchanged shape, faster body):
//   * WARP-PER-HEAD.  One warp owns (one q_head, one KV split).  HEAD_DIM=128 = 4 elems/lane, so the
//     q-dot and acc[] accumulator are register-resident (4 floats/lane); reductions are warp shuffles
//     -- no shared memory, no __syncthreads in the hot loop.
//   * SPLIT-KV: grid.x = n_splits CTAs of KV-time-chunks, grid.y groups q_heads.
//   * Pass 1 (partial): each warp streams its [t0,t1) KV chunk with online softmax -> (m,l,acc).
//   * Pass 2 (reduce):  one warp per q_head merges the S partials via the log-sum-exp trick.
//
// Partial buffers, laid out [q_head][split] (acc as [q_head][split][HEAD_DIM], lane-contiguous):
//   part_m  [N_Q_HEADS * n_splits]
//   part_l  [N_Q_HEADS * n_splits]
//   part_acc[N_Q_HEADS * n_splits * HEAD_DIM]
//
// Public entry points (names + arg order preserved for k12_bench.cu / decode_step.cu):
//   k2_flash_decode_partial / k2_flash_decode_reduce / k2_pick_splits / k2_partials_elems_* / k2_launch
//
// Build:  nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k2_flash_decode.cu -o /tmp/k2
#include "common.cuh"
#include <cfloat>
using namespace q3;

#ifndef Q3_K2_DEFS
#define Q3_K2_DEFS
// HEAD_DIM=128 split across a 32-lane warp -> 4 contiguous elements per lane.
constexpr int K2_VPL = HEAD_DIM / 32;     // 4 values per lane (one 32-bit fp8 word)

// Reduce a per-lane partial sum across the warp, broadcast to all lanes (xor-fan -> identical result).
static __device__ __forceinline__ float k2_warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Load this lane's 4 contiguous fp8 channels (one 32-bit word) and dequant to 4 floats.
//   base32 points at the fp8 row reinterpreted as 32-bit words; lane L reads word L (= channels
//   [4L,4L+4)). Two hardware fp8x2->half2 converts per word. `s` holds the 4 matching per-channel
//   scales for this lane (hoisted out of the time loop). Result written into out[0..3].
static __device__ __forceinline__ void k2_load4(const unsigned* __restrict__ base32, int lane,
                                                 const float* __restrict__ s, float* __restrict__ out) {
  unsigned w = base32[lane];
  __nv_fp8x2_e4m3 lo, hi;
  lo.__x = (unsigned short)(w & 0xffffu);
  hi.__x = (unsigned short)(w >> 16);
  float2 fl = __half22float2((__half2)lo);   // channels 4L+0, 4L+1
  float2 fh = __half22float2((__half2)hi);   // channels 4L+2, 4L+3
  out[0] = fl.x * s[0];
  out[1] = fl.y * s[1];
  out[2] = fh.x * s[2];
  out[3] = fh.y * s[3];
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
  const int kvh   = qh / GQA_GROUP;                         // GQA broadcast -> KV head 0..3
  const int chunk = (ctx_len + n_splits - 1) / n_splits;
  const int t0 = split * chunk;
  const int t1 = min(t0 + chunk, ctx_len);
  const float scale = rsqrtf((float)HEAD_DIM);

  // KV head base offset (channels) inside the KV_DIM row; this lane's 4 contiguous channels.
  const int kv_base = kvh * HEAD_DIM;
  const int c0 = kv_base + lane * K2_VPL;                   // first of this lane's 4 channels

  // This lane's 4 query elements (contiguous) and the 4 K/V per-channel scales (constant in t).
  float qreg[K2_VPL], ksc[K2_VPL], vsc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) {
    qreg[c] = q[qh * HEAD_DIM + lane * K2_VPL + c];
    ksc[c]  = kv_k_scale ? kv_k_scale[c0 + c] : 1.f;
    vsc[c]  = kv_v_scale ? kv_v_scale[c0 + c] : 1.f;
  }

  // online softmax state (register, per warp): running max m, denom l, weighted acc[4 per lane].
  float m = -FLT_MAX, l = 0.f, acc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) acc[c] = 0.f;

  // 32-bit views of the K/V caches; word index for this lane's channels at timestep t is
  // (t*KV_DIM + kv_base)/4 + lane = t*(KV_DIM/4) + kv_base/4 + lane.
  const unsigned* __restrict__ k32 = reinterpret_cast<const unsigned*>(kv_k);
  const unsigned* __restrict__ v32 = reinterpret_cast<const unsigned*>(kv_v);
  const int row_words  = KV_DIM / 4;                        // 32-bit words per KV row
  const int base_words = kv_base / 4;                       // word offset of this kv head

  for (int t = t0; t < t1; t++) {
    const unsigned* krow = k32 + (size_t)t * row_words + base_words;
    float kv[K2_VPL];
    k2_load4(krow, lane, ksc, kv);
    // dot(q, k_t): this lane's 4 channels, then warp reduce.
    float p = 0.f;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) p += qreg[c] * kv[c];
    float s = k2_warp_sum(p) * scale;                       // logit, identical on all lanes

    // online softmax update.
    float m_new = fmaxf(m, s);
    float corr  = __expf(m - m_new);                        // rescale old state
    float pexp  = __expf(s - m_new);
    l = l * corr + pexp;

    const unsigned* vrow = v32 + (size_t)t * row_words + base_words;
    float vv[K2_VPL];
    k2_load4(vrow, lane, vsc, vv);
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c] * corr + pexp * vv[c];
    m = m_new;
  }

  // write partials, laid out [qh][split] (acc as [qh][split][HEAD_DIM], lane-contiguous).
  const size_t pidx = (size_t)qh * n_splits + split;
  if (lane == 0) { part_m[pidx] = m; part_l[pidx] = l; }
  float* ao = part_acc + pidx * HEAD_DIM + lane * K2_VPL;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) ao[c] = acc[c];
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
    const float* ai = part_acc + pidx * HEAD_DIM + lane * K2_VPL;
    float m_new = fmaxf(m, ms);
    float corr_o = __expf(m - m_new);                       // rescale running
    float corr_s = __expf(ms - m_new);                      // rescale incoming split
    l = l * corr_o + ls * corr_s;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++)
      acc[c] = acc[c] * corr_o + ai[c] * corr_s;
    m = m_new;
  }
  float inv = (l > 0.f) ? (1.f / l) : 0.f;
  float* o = attn_out + qh * HEAD_DIM + lane * K2_VPL;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) o[c] = acc[c] * inv;
}

// =================================================================================================
// FUSED single-kernel flash-decode (partial + reduce in ONE launch, partials in shared memory).
//
// WHY: the bench showed K2 at ctx 4096 is OVERHEAD/LAUNCH bound, not bandwidth or chain-latency
// bound (4.2 MB read takes ~42 us vs a ~1.3 us bandwidth floor).  The two-kernel split-KV path pays
// for (a) TWO kernel launches and (b) an HBM round-trip of the (m,l,acc) partials between them, and
// the n_splits sweep showed *more* splits make it slower (reduce cost + launch grow with splits) --
// the opposite of a parallelism-starved kernel.  So the win is to do everything in ONE launch and
// keep partials on-chip.
//
// DESIGN: one CTA owns one q_head.  Its W warps split the [0,ctx) KV range W ways; each warp streams
// its slice with the online-softmax recurrence (2x time-unrolled so two coalesced fp8 loads stay in
// flight) into a register (m,l,acc[4/lane]).  The W partials are combined in shared memory by warp 0
// (W is tiny, e.g. 8), normalized, and attn_out[qh] written directly.  No second launch, no partials
// in HBM.  grid = N_Q_HEADS CTAs (64) * W warps -> e.g. 512 warps in flight across 132 SMs, and each
// CTA's W warps give intra-SM latency hiding.
// =================================================================================================
extern "C" __global__ void k2_flash_decode_fused(
    const float* __restrict__ q,
    const fp8*  __restrict__ kv_k, const fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale,
    int ctx_len, float* __restrict__ attn_out) {
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  const int W    = blockDim.x >> 5;          // warps per CTA = number of time-splits
  const int qh   = blockIdx.x;               // one CTA per q_head
  const int kvh  = qh / GQA_GROUP;
  const float scale = rsqrtf((float)HEAD_DIM);

  const int kv_base = kvh * HEAD_DIM;
  const int c0 = kv_base + lane * K2_VPL;

  float qreg[K2_VPL], ksc[K2_VPL], vsc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) {
    qreg[c] = q[qh * HEAD_DIM + lane * K2_VPL + c];
    ksc[c]  = kv_k_scale ? kv_k_scale[c0 + c] : 1.f;
    vsc[c]  = kv_v_scale ? kv_v_scale[c0 + c] : 1.f;
  }

  // this warp's contiguous KV time slice [t0,t1).
  const int chunk = (ctx_len + W - 1) / W;
  const int t0 = wid * chunk;
  const int t1 = min(t0 + chunk, ctx_len);

  float m = -FLT_MAX, l = 0.f, acc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) acc[c] = 0.f;

  const unsigned* __restrict__ k32 = reinterpret_cast<const unsigned*>(kv_k);
  const unsigned* __restrict__ v32 = reinterpret_cast<const unsigned*>(kv_v);
  const int row_words  = KV_DIM / 4;
  const int base_words = kv_base / 4;

  // 2x time-step unroll: two independent coalesced K (and V) loads in flight per iteration.
  int t = t0;
  for (; t + 1 < t1; t += 2) {
    float kv0[K2_VPL], kv1[K2_VPL];
    k2_load4(k32 + (size_t)t       * row_words + base_words, lane, ksc, kv0);
    k2_load4(k32 + (size_t)(t + 1) * row_words + base_words, lane, ksc, kv1);
    float p0 = 0.f, p1 = 0.f;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) { p0 += qreg[c]*kv0[c]; p1 += qreg[c]*kv1[c]; }
    float s0 = k2_warp_sum(p0) * scale;
    float s1 = k2_warp_sum(p1) * scale;
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

  // ---- combine the W warp-partials in shared memory ----
  // smem: m[W], l[W], acc[W][HEAD_DIM] (lane-contiguous). W is small (<=32).
  extern __shared__ float sm[];
  float* sm_m = sm;                 // [W]
  float* sm_l = sm_m + W;           // [W]
  float* sm_acc = sm_l + W;         // [W*HEAD_DIM]
  if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
  float* sao = sm_acc + (size_t)wid * HEAD_DIM + lane * K2_VPL;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) sao[c] = acc[c];
  __syncthreads();

  // warp 0 merges all W partials (log-sum-exp) and writes the normalized output.
  if (wid == 0) {
    float rm = -FLT_MAX, rl = 0.f, racc[K2_VPL];
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) racc[c] = 0.f;
    for (int w = 0; w < W; w++) {
      float ms = sm_m[w], ls = sm_l[w];
      if (ls <= 0.f) continue;
      const float* ai = sm_acc + (size_t)w * HEAD_DIM + lane * K2_VPL;
      float mn = fmaxf(rm, ms), co = __expf(rm - mn), cs = __expf(ms - mn);
      rl = rl * co + ls * cs;
      #pragma unroll
      for (int c = 0; c < K2_VPL; c++) racc[c] = racc[c]*co + ai[c]*cs;
      rm = mn;
    }
    float inv = (rl > 0.f) ? (1.f / rl) : 0.f;
    float* o = attn_out + qh * HEAD_DIM + lane * K2_VPL;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) o[c] = racc[c] * inv;
  }
}

// ---- launch helpers ----------------------------------------------------------------------------
// Pick a split count so the partial pass keeps the 132 SMs busy several times over while leaving
// each split a chunk long enough to amortize the per-warp setup/epilogue.
//
// The old default (8 @ 4k) left a single warp streaming a 512-timestep dependent online-softmax
// recurrence -> latency-bound.  The autotune sweep found 64 far faster: more splits = shorter
// dependent chains + more warps in flight to hide HBM latency.  We target ~64 splits when the
// context is large enough to give each split >= ~64 timesteps, scaling up for very long contexts and
// capping so no split is starved.
static inline int k2_pick_splits(int ctx_len) {
  // 64 q_heads * S splits = warps in flight; want this comfortably above ~2x SM warp slots.
  int s = 64;                              // matches the K2 sweep's best at ctx 4096
  if (ctx_len > 16384) s = 96;             // longer ctx: a bit more parallelism, chunks stay long
  if (ctx_len > 65536) s = 128;
  // keep each split's chunk reasonably long (>= 32 timesteps) to amortize per-warp overhead.
  int max_by_chunk = (ctx_len + 31) / 32;
  if (s > max_by_chunk) s = max_by_chunk;
  if (s < 1) s = 1;
  return s;
}

#ifdef Q3_K2_LAUNCH_HELPER
// Allocates nothing; caller owns part_* (size with k2_partials_elems_*).  Returns the split count used.
static inline size_t k2_partials_elems_m(int n_splits)  { return (size_t)N_Q_HEADS * n_splits; }
static inline size_t k2_partials_elems_acc(int n_splits){ return (size_t)N_Q_HEADS * n_splits * HEAD_DIM; }

static inline int k2_launch(
    const float* q, const fp8* kv_k, const fp8* kv_v,
    const float* kv_k_scale, const float* kv_v_scale, int ctx_len,
    float* part_m, float* part_l, float* part_acc, float* attn_out,
    int n_splits = -1, cudaStream_t stream = 0) {
  // IMPORTANT: when n_splits<=0 use the SAME k2_pick_splits the caller used to size part_*.
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

// Pick the number of warps/time-splits per CTA for the FUSED kernel (one CTA per q_head).  Want
// W*64 warps comfortably over the SMs while keeping each warp's slice long enough to amortize setup.
static inline int k2_pick_warps(int ctx_len) {
  int w = 8;                                  // 64 heads * 8 = 512 warps in flight
  if (ctx_len > 16384) w = 16;
  if (ctx_len <= 512)  w = 4;
  int max_by_chunk = (ctx_len + 31) / 32;     // keep slice >= ~32 timesteps
  if (w > max_by_chunk) w = max_by_chunk;
  if (w < 1) w = 1;
  return w;
}

// Single-launch fused flash-decode: no partials in HBM, no second kernel.  Returns warps/CTA used.
static inline int k2_launch_fused(
    const float* q, const fp8* kv_k, const fp8* kv_v,
    const float* kv_k_scale, const float* kv_v_scale, int ctx_len,
    float* attn_out, int warps = -1, cudaStream_t stream = 0) {
  if (warps <= 0) warps = k2_pick_warps(ctx_len);
  const int block = warps * 32;
  // smem: m[W] + l[W] + acc[W*HEAD_DIM], all floats.
  const size_t smem = (size_t)(2 * warps + warps * HEAD_DIM) * sizeof(float);
  k2_flash_decode_fused<<<N_Q_HEADS, block, smem, stream>>>(
      q, kv_k, kv_v, kv_k_scale, kv_v_scale, ctx_len, attn_out);
  return warps;
}
#endif
