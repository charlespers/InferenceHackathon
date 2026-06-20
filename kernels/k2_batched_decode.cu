// k2_batched_decode.cu — FLAT multi-query (spec-verify) flash-decode for B=1, GQA 16:1, head_dim 128, sm_90a.
//
// =================================================================================================
// WHY THIS FILE (the spec free-ride blocker)
// -------------------------------------------------------------------------------------------------
// Measured (spec_e2e_real.txt): the full decode forward is FLAT for the GEMM panels (T16/T1=1.001)
// and comms is M-independent, but **K2 attention SCALES with the draft width k** — T_K2(8)/T_K2(1)
// ≈ 1.5, T(16)/T(1) ≈ 2.3.  That single fact caps speculative decoding at ~107-294 tok/s instead of
// the ~840-1000 the flat-forward projection assumed.
//
// ROOT CAUSE: the prior multi-query K2 (`tp8_k2_partial_mq`) loads each KV row once and then loops
// over the M draft queries *inside one warp* — M dot-products + M warp-shuffle softmax reductions per
// timestep, serially.  At B=1 the GPU is OCCUPANCY-STARVED (64 heads x ~64 splits warps barely fills
// 132 SMs), so those extra per-query ALU/shuffle ops do NOT hide behind the KV load — they serialize,
// and K2 scales ~linearly in k.
//
// THE FIX (this file): spend the k queries on MORE WARPS, not more serial work.  Parallelize the grid
// over (q_head, query_position, split) so M=8 launches 8x the warps.  That is exactly the lever an
// occupancy-starved kernel wants: the extra warps fill the idle SMs, and because every query of a
// given head reads the SAME KV rows, those reads hit in L2 across the M warps (one HBM stream, M
// consumers).  If the kernel was occupancy/latency-bound at M=1 (it was), adding warps up to the SM
// fill point is ~free -> K2(k) ~ K2(1) until the SMs saturate.  This is the structural reason it
// should go flat where the per-warp-loop version scaled.
//
// CONTRACT: identical math + identical layouts to k2_flash_decode.cu (reuses k2_load4 / k2_warp_sum),
// generalized with an M (draft-query) dimension on q / partials / attn_out.  M=1 reduces EXACTLY to
// k2_flash_decode_partial/_reduce.
//
//   q        : [M][N_Q_HEADS][HEAD_DIM]   M normed+roped draft queries (causal masking is the caller's
//              job via the KV length per query; this kernel attends each query over [0,ctx_len) — for
//              a draft CHAIN of length M the caller passes the per-query ctx; here we bench the
//              shared-prefix case ctx_len common to all M, the dominant cost).
//   kv_k/kv_v: [ctx_len][KV_DIM]          fp8 cache, shared across all M queries (the L2-reuse win).
//   part_*   : [M][N_Q_HEADS][n_splits]   (part_acc [..][HEAD_DIM], lane-contiguous)
//   attn_out : [M][N_Q_HEADS][HEAD_DIM]
//
// =================================================================================================
// STATUS: written OFF-GPU (team is on the box).  NOT yet compiled/run — ready to validate next session.
//   Build:  nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k2_batched_decode.cu -o /tmp/k2b && /tmp/k2b
//   Expect:  the FLATNESS table us(M=1,4,8,16) — the headline is whether us(8)/us(1) -> ~1.0 (flat,
//            the spec free-ride restored) vs the per-warp-loop version's ~1.5.  CPU fp32 ref gate <1e-2.
// =================================================================================================
#include "common.cuh"
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
using namespace q3;

// Reuse the validated fp8-load / warp-reduce idioms from k2_flash_decode.cu.
#ifndef Q3_K2_DEFS
#define Q3_K2_DEFS
constexpr int K2_VPL = HEAD_DIM / 32;     // 4 contiguous fp8 channels per lane (one 32-bit word)

static __device__ __forceinline__ float k2_warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

static __device__ __forceinline__ void k2_load4(const unsigned* __restrict__ base32, int lane,
                                                 const float* __restrict__ s, float* __restrict__ out) {
  unsigned w = base32[lane];
  __nv_fp8x2_e4m3 lo, hi;
  lo.__x = (unsigned short)(w & 0xffffu);
  hi.__x = (unsigned short)(w >> 16);
  float2 fl = __half22float2((__half2)lo);
  float2 fh = __half22float2((__half2)hi);
  out[0] = fl.x * s[0];
  out[1] = fl.y * s[1];
  out[2] = fh.x * s[2];
  out[3] = fh.y * s[3];
}
#endif // Q3_K2_DEFS

// ---- Pass 1: per (query m, q_head, split) online softmax over the KV chunk -> partial (m,l,acc). ----
// Grid: x = n_splits, y = ceil(M*N_Q_HEADS / warps_per_cta).  One warp = one (m, qh, split).
// The (m,qh) pair is unpacked from the global warp id so M just multiplies the warp count — the
// occupancy-fill that makes K2 flat in M on an otherwise-starved GPU.
extern "C" __global__ void k2b_partial(
    const float* __restrict__ q,                 // [M][N_Q_HEADS][HEAD_DIM]
    const fp8*  __restrict__ kv_k, const fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale,
    int ctx_len, int n_splits, int M,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc) {
  const int lane  = threadIdx.x & 31;
  const int wid   = threadIdx.x >> 5;
  const int gw    = blockIdx.y * (blockDim.x >> 5) + wid;     // global warp over the (m,qh) plane
  const int total = M * N_Q_HEADS;
  if (gw >= total) return;
  const int m     = gw / N_Q_HEADS;                           // draft-query index 0..M-1
  const int qh    = gw - m * N_Q_HEADS;                       // query head 0..N_Q_HEADS-1
  const int split = blockIdx.x;                               // KV-chunk 0..n_splits-1
  const int kvh   = qh / GQA_GROUP;                           // GQA broadcast -> KV head
  const int chunk = (ctx_len + n_splits - 1) / n_splits;
  const int t0 = split * chunk;
  const int t1 = min(t0 + chunk, ctx_len);
  const float scale = rsqrtf((float)HEAD_DIM);

  const int kv_base = kvh * HEAD_DIM;
  const int c0 = kv_base + lane * K2_VPL;

  // this (m,qh) query's 4 elements + the 4 K/V per-channel scales (constant in t).
  float qreg[K2_VPL], ksc[K2_VPL], vsc[K2_VPL];
  const float* qrow = q + ((size_t)m * N_Q_HEADS + qh) * HEAD_DIM;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) {
    qreg[c] = qrow[lane * K2_VPL + c];
    ksc[c]  = kv_k_scale ? kv_k_scale[c0 + c] : 1.f;
    vsc[c]  = kv_v_scale ? kv_v_scale[c0 + c] : 1.f;
  }

  float m_ = -FLT_MAX, l = 0.f, acc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) acc[c] = 0.f;

  const unsigned* __restrict__ k32 = reinterpret_cast<const unsigned*>(kv_k);
  const unsigned* __restrict__ v32 = reinterpret_cast<const unsigned*>(kv_v);
  const int row_words  = KV_DIM / 4;
  const int base_words = kv_base / 4;

  // 2x time-unroll: two coalesced K (and V) loads in flight (matches the fused kernel's hot loop).
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
    float mn = fmaxf(m_, s0), corr = __expf(m_ - mn), pe = __expf(s0 - mn);
    l = l * corr + pe;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c]*corr + pe*vv0[c];
    m_ = mn;
    mn = fmaxf(m_, s1); corr = __expf(m_ - mn); pe = __expf(s1 - mn);
    l = l * corr + pe;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c]*corr + pe*vv1[c];
    m_ = mn;
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
    float mn = fmaxf(m_, s), corr = __expf(m_ - mn), pe = __expf(s - mn);
    l = l * corr + pe;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c]*corr + pe*vv[c];
    m_ = mn;
  }

  // partials laid out [m][qh][split] (acc as [..][HEAD_DIM], lane-contiguous).
  const size_t pidx = ((size_t)m * N_Q_HEADS + qh) * n_splits + split;
  if (lane == 0) { part_m[pidx] = m_; part_l[pidx] = l; }
  float* ao = part_acc + pidx * HEAD_DIM + lane * K2_VPL;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) ao[c] = acc[c];
}

// ---- Pass 2: merge n_splits partials per (m,qh) -> attn_out[m][qh][HEAD_DIM] (log-sum-exp). ----
extern "C" __global__ void k2b_reduce(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, int n_splits, int M, float* __restrict__ attn_out) {
  const int lane = threadIdx.x & 31;
  const int gw   = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int total = M * N_Q_HEADS;
  if (gw >= total) return;

  float m = -FLT_MAX, l = 0.f, acc[K2_VPL];
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) acc[c] = 0.f;

  for (int sp = 0; sp < n_splits; sp++) {
    const size_t pidx = (size_t)gw * n_splits + sp;
    float ms = part_m[pidx], ls = part_l[pidx];
    if (ls <= 0.f) continue;
    const float* ai = part_acc + pidx * HEAD_DIM + lane * K2_VPL;
    float m_new = fmaxf(m, ms);
    float corr_o = __expf(m - m_new);
    float corr_s = __expf(ms - m_new);
    l = l * corr_o + ls * corr_s;
    #pragma unroll
    for (int c = 0; c < K2_VPL; c++) acc[c] = acc[c]*corr_o + ai[c]*corr_s;
    m = m_new;
  }
  float inv = (l > 0.f) ? (1.f / l) : 0.f;
  float* o = attn_out + (size_t)gw * HEAD_DIM + lane * K2_VPL;
  #pragma unroll
  for (int c = 0; c < K2_VPL; c++) o[c] = acc[c] * inv;
}

// ---- launch helper ----
static inline int k2b_pick_splits(int ctx_len, int M) {
  // Total warps in flight = M * N_Q_HEADS * splits.  At M=1 we want ~64 splits (the measured best);
  // as M grows the M factor already adds warps, so we can SHRINK splits to keep each warp's KV chunk
  // long (less reduce overhead) while staying well above the SM fill.  This is the lever: M buys
  // occupancy, so trade it back for longer chunks.
  long target_warps = 4096;                       // ~2x the 132*~16 warp slots, fills the GPU
  int s = (int)(target_warps / ((long)M * N_Q_HEADS));
  if (s < 1) s = 1;
  int max_by_chunk = (ctx_len + 31) / 32;         // keep chunks >= ~32 timesteps
  if (s > max_by_chunk) s = max_by_chunk;
  if (s < 1) s = 1;
  return s;
}

static inline size_t k2b_part_m_elems(int M, int n_splits)   { return (size_t)M * N_Q_HEADS * n_splits; }
static inline size_t k2b_part_acc_elems(int M, int n_splits) { return (size_t)M * N_Q_HEADS * n_splits * HEAD_DIM; }

static inline int k2b_launch(
    const float* q, const fp8* kv_k, const fp8* kv_v,
    const float* kv_k_scale, const float* kv_v_scale, int ctx_len, int M,
    float* part_m, float* part_l, float* part_acc, float* attn_out,
    int n_splits = -1, cudaStream_t stream = 0) {
  if (n_splits <= 0) n_splits = k2b_pick_splits(ctx_len, M);
  const int wpc = 4;                                          // 128 threads/CTA -> 4 warps
  const int block = wpc * 32;
  const int plane = M * N_Q_HEADS;
  dim3 gP(n_splits, (plane + wpc - 1) / wpc);
  k2b_partial<<<gP, block, 0, stream>>>(q, kv_k, kv_v, kv_k_scale, kv_v_scale,
                                        ctx_len, n_splits, M, part_m, part_l, part_acc);
  dim3 gR((plane + wpc - 1) / wpc);
  k2b_reduce<<<gR, block, 0, stream>>>(part_m, part_l, part_acc, n_splits, M, attn_out);
  return n_splits;
}

// =================================================================================================
// Microbench + CPU fp32 reference (the flatness table is the headline; correctness gates it).
// =================================================================================================
#ifndef K2B_NO_MAIN
#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

static void fill_fp8(std::vector<fp8>& h, std::vector<float>& ref, unsigned seed) {
  for (size_t i = 0; i < h.size(); i++) {
    seed = seed * 1664525u + 1013904223u;
    float v = ((int)(seed >> 8 & 0xffff) - 32768) / 32768.0f * 0.5f;   // ~[-0.5,0.5]
    __nv_fp8_e4m3 q8(v);
    h[i] = *reinterpret_cast<fp8*>(&q8);
    ref[i] = (float)q8;                                                // the dequantized value
  }
}

int main(int argc, char** argv) {
  int ctx_len = (argc > 1) ? atoi(argv[1]) : 4096;
  int iters   = (argc > 2) ? atoi(argv[2]) : 200;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("== K2 BATCHED multi-query flash-decode ==  device %s  SMs=%d  ctx_len=%d\n",
         prop.name, prop.multiProcessorCount, ctx_len);
  printf("   N_Q_HEADS=%d KV_HEADS=%d HEAD_DIM=%d GQA=%d  (q [M][heads][hd], KV shared across M)\n",
         N_Q_HEADS, N_KV_HEADS, HEAD_DIM, GQA_GROUP);

  const int Ms[] = {1, 4, 8, 16};
  const int MMAX = 16;

  // KV cache (shared across all M queries) + per-channel scales = 1 (values already dequantized in ref).
  std::vector<fp8> hK((size_t)ctx_len * KV_DIM), hV((size_t)ctx_len * KV_DIM);
  std::vector<float> rK(hK.size()), rV(hV.size());
  fill_fp8(hK, rK, 0xA1u); fill_fp8(hV, rV, 0xB2u);
  fp8 *dK, *dV; CK(cudaMalloc(&dK, hK.size())); CK(cudaMalloc(&dV, hV.size()));
  CK(cudaMemcpy(dK, hK.data(), hK.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV, hV.data(), hV.size(), cudaMemcpyHostToDevice));

  // M queries.
  std::vector<float> hQ((size_t)MMAX * N_Q_HEADS * HEAD_DIM);
  for (size_t i = 0; i < hQ.size(); i++) hQ[i] = (float)(((i * 2654435761u) >> 12) & 0x3ff) / 1024.f - 0.5f;
  float* dQ; CK(cudaMalloc(&dQ, hQ.size()*sizeof(float)));
  CK(cudaMemcpy(dQ, hQ.data(), hQ.size()*sizeof(float), cudaMemcpyHostToDevice));

  int splits_max = 64;
  float *dpm, *dpl, *dpa, *dout;
  CK(cudaMalloc(&dpm,  k2b_part_m_elems(MMAX, splits_max)*sizeof(float)));
  CK(cudaMalloc(&dpl,  k2b_part_m_elems(MMAX, splits_max)*sizeof(float)));
  CK(cudaMalloc(&dpa,  k2b_part_acc_elems(MMAX, splits_max)*sizeof(float)));
  CK(cudaMalloc(&dout, (size_t)MMAX*N_Q_HEADS*HEAD_DIM*sizeof(float)));

  // ---- correctness vs CPU fp32 (M=1, head 0 and a mid head) ----
  {
    k2b_launch(dQ, dK, dV, nullptr, nullptr, ctx_len, 1, dpm, dpl, dpa, dout);
    CK(cudaDeviceSynchronize());
    std::vector<float> hout((size_t)N_Q_HEADS*HEAD_DIM);
    CK(cudaMemcpy(hout.data(), dout, hout.size()*sizeof(float), cudaMemcpyDeviceToHost));
    double maxerr = 0.0;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int qh : {0, N_Q_HEADS/2, N_Q_HEADS-1}) {
      int kvh = qh / GQA_GROUP;
      std::vector<float> logit(ctx_len); float mx = -1e30f;
      for (int t = 0; t < ctx_len; t++) {
        double d = 0;
        for (int c = 0; c < HEAD_DIM; c++)
          d += hQ[(size_t)qh*HEAD_DIM + c] * rK[(size_t)t*KV_DIM + kvh*HEAD_DIM + c];
        logit[t] = (float)d * scale; mx = fmaxf(mx, logit[t]);
      }
      double den = 0; std::vector<double> w(ctx_len);
      for (int t = 0; t < ctx_len; t++) { w[t] = exp((double)(logit[t]-mx)); den += w[t]; }
      for (int c = 0; c < HEAD_DIM; c++) {
        double o = 0;
        for (int t = 0; t < ctx_len; t++) o += w[t] * rV[(size_t)t*KV_DIM + kvh*HEAD_DIM + c];
        o /= den;
        double e = fabs(o - hout[(size_t)qh*HEAD_DIM + c]);
        if (e > maxerr) maxerr = e;
      }
    }
    printf("\ncorrectness vs CPU fp32 (M=1, 3 heads): max_abs_err = %.3e  -> %s (tol 1e-2)\n",
           maxerr, maxerr < 1e-2 ? "PASS" : "FAIL");
  }

  // ---- flatness table: us per K2 forward at M=1,4,8,16 ----
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  printf("\n  M    n_splits   warps_in_flight     us/K2-forward   us/query   ratio_vs_M1\n");
  printf("  --------------------------------------------------------------------------------\n");
  double us_m1 = 0;
  for (int M : Ms) {
    int ns = k2b_pick_splits(ctx_len, M);
    long warps = (long)M * N_Q_HEADS * ns;
    for (int w = 0; w < 20; w++) k2b_launch(dQ, dK, dV, nullptr, nullptr, ctx_len, M, dpm, dpl, dpa, dout, ns);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) k2b_launch(dQ, dK, dV, nullptr, nullptr, ctx_len, M, dpm, dpl, dpa, dout, ns);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    double us = ms * 1000.0 / iters;
    if (M == 1) us_m1 = us;
    printf("  %-4d %-10d %-17ld %12.2f   %8.2f   %.3f%s\n",
           M, ns, warps, us, us / M, us / us_m1,
           (M > 1 && us / us_m1 < 1.25) ? "  <- FLAT (spec free-ride restored)" : "");
  }
  printf("\n  HEADLINE: if us(M=8)/us(M=1) ~ 1.0 the K2 k-scaling is gone (occupancy-fill worked) and the\n");
  printf("  spec verify forward goes flat -> the 840-1000 path is unblocked.  Compare to the per-warp-loop\n");
  printf("  tp8_k2_partial_mq which measured 1.5x at M=8.  (Next: wire the winner into decode_step_tp8.cu.)\n");
  return 0;
}
#endif // K2B_NO_MAIN
