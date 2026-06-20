// k1k2_mbu_v2.cu — push K1 (fused QKV GEMV) and K2 (flash-decode) toward their HBM-bandwidth
// ceilings for Qwen3-235B-A22B, B=1 decode (sm_90a / H100).
//
// =================================================================================================
// CONTEXT — both kernels are HBM-bandwidth / latency bound at B=1, not compute bound.
//
//   K1 (Wqkv [QKV_OUT=9216, HIDDEN=4096] fp8 e4m3, ~38 MB): a pure GEMV (M=1). The whole budget is
//   reading those 38 MB of fp8 weights from HBM exactly once. The in-repo k1_qkv_gemv (warp-per-row,
//   warp_dot_fp8 idiom) reaches ~27% MBU = ~904 GB/s. The ceiling is set by how many independent HBM
//   transactions are in flight per SM: warp-per-row issues only ~8 dependent uint4 loads per lane and
//   then a reduction, so the load pipeline drains between rows. To beat it we need (a) more
//   memory-level parallelism (MLP) per warp — several independent rows in flight so loads overlap
//   across rows, and (b) cp.async double-buffered staging of the weight tiles so the compute on tile
//   t overlaps the HBM fetch of tile t+1. Target: > 45% MBU.
//
//   K2 (split-KV flash-decode): the KV read at ctx 4096 is tiny (~4 MB), so this is overhead /
//   latency bound, not bandwidth bound — the goal is to approach the latency floor. We reduce launch
//   and reduce overhead by (i) FUSING the partial + reduce passes into a SINGLE kernel using a
//   grid-wide cooperative grid sync (no second launch, no HBM round-trip of the partials), and
//   (ii) streaming each split's KV chunk with vectorized coalesced fp8 loads and 2x time-step
//   unrolling so each warp keeps two independent loads in flight (shorter dependent chains).
//
// Every technique here is the SAME family already used by the validated k5_experts.cu / k1 / k2 in
// this repo: warp-per-output-row, coalesced uint4 (16xfp8) loads, hardware fp8x2->half2 dequant,
// warp-shuffle reduce, per-out-channel scale folded once. The new ingredients are cp.async pipelining
// and multi-row ILP for K1, and the single-launch cooperative fusion for K2.
//
// Standard CUDA + common.cuh only. New file; edits nothing else. Builds + self-tests + reports %MBU.
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k1k2_mbu_v2.cu -o /tmp/k1k2 && /tmp/k1k2
// =================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "common.cuh"
using namespace q3;
namespace cg = cooperative_groups;

// =================================================================================================
// Shared device helpers (uniquely named so this file composes into a single-TU build next to the
// existing k1/k2/k5 copies without symbol clashes).
// =================================================================================================

// cp.async: stage 16 bytes (one uint4 = 16 fp8) from global -> shared, non-blocking. On sm_90 this
// issues an async copy whose completion we wait on with a commit/wait group, so the SM can run other
// instructions (including issuing the next async copies) while the HBM transaction is in flight.
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src) {
#if __CUDA_ARCH__ >= 800
  unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
#else
  *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// Dequant one staged uint4 (16 fp8) against 16 staged activation floats, 2 accumulators for ILP.
// `wp` is the uint4 of fp8 weights (already in registers/smem), `xx` points at the matching 16 x[].
__device__ __forceinline__ void mac_uint4_fp8(const uint4& wp, const float* __restrict__ xx,
                                              float& a0, float& a1) {
  const unsigned* wu = reinterpret_cast<const unsigned*>(&wp);
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    unsigned wq = wu[q];
    __nv_fp8x2_e4m3 lo, hi;
    lo.__x = (unsigned short)(wq & 0xffffu);
    hi.__x = (unsigned short)(wq >> 16);
    float2 fl = __half22float2((__half2)lo);
    float2 fh = __half22float2((__half2)hi);
    const float* xq = xx + (q << 2);
    a0 += xq[0] * fl.x;  a1 += xq[1] * fl.y;
    a0 += xq[2] * fh.x;  a1 += xq[3] * fh.y;
  }
}

__device__ __forceinline__ float warp_reduce_sum(float acc) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;
}

// =================================================================================================
// K1 — RMSNorm + fused QKV GEMV, MBU-pushed.
//
// Layout: ONE WARP processes ROWS_PER_WARP (=4) output rows at once. For each "phase" v over the
// row's uint4 chunks (HIDDEN/16 = 256 of them, lanes split them 32-wide => 8 phases), the warp:
//   1) issues cp.async loads of the next phase's ROWS_PER_WARP weight uint4s into a double-buffered
//      smem ring (prefetch),
//   2) computes the MAC of the CURRENT phase's already-staged ROWS_PER_WARP uint4s.
// Because the ROWS_PER_WARP rows are independent, their loads overlap (MLP), and the cp.async ring
// hides the HBM latency of phase t+1 behind the math of phase t. Each warp keeps a separate
// accumulator per row; the per-row warp-reduce + scale + store happens once at the end.
//
// Activations x[HIDDEN] are staged in shared memory once per CTA (RMSNorm-normed input), read from
// smem in the hot loop. cp.async weight ring lives in the same dynamic smem after x[].
// =================================================================================================
#ifndef Q3_K1K2_DEFS
#define Q3_K1K2_DEFS
constexpr int K1_ROWS_PER_WARP = 4;     // independent rows per warp -> MLP across rows
constexpr int K1_NVEC          = HIDDEN / 16;   // 256 uint4 chunks per row
constexpr int K1_PHASES        = K1_NVEC / 32;  // 8 phases (lanes cover 32 chunks/phase)
constexpr int K1_STAGES        = 2;             // double-buffered cp.async ring
#endif

// dynamic smem layout: [ float x[HIDDEN] ][ uint4 ring[K1_STAGES][warps_per_cta][ROWS_PER_WARP][32] ]
extern "C" __global__ void k1k2_qkv_gemv(
    const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale,
    float* __restrict__ proj) {
  extern __shared__ float smem[];
  float* xs = smem;                                  // [HIDDEN]
  const int warps_per_cta = blockDim.x >> 5;
  const int wid  = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // weight ring base (uint4 units), after x[HIDDEN] floats.
  uint4* ring = reinterpret_cast<uint4*>(xs + HIDDEN);
  // ring index for (stage, this warp, row r, lane): contiguous per (stage,warp) block of
  // ROWS_PER_WARP*32 uint4s so consecutive lanes are consecutive smem banks.
  const int ring_warp_stride = K1_ROWS_PER_WARP * 32;
  const int ring_stage_stride = warps_per_cta * ring_warp_stride;

  // ---- input RMSNorm (block reduce of sum-of-squares) ----
  float part = 0.f;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) { float v = h[i]; part += v * v; }
  part = warp_reduce_sum(part);
  __shared__ float warp_ss[32];
  if (lane == 0) warp_ss[wid] = part;
  __syncthreads();
  __shared__ float rinv_sh;
  if (threadIdx.x == 0) {
    float ss = 0.f;
    for (int i = 0; i < warps_per_cta; ++i) ss += warp_ss[i];
    rinv_sh = rsqrtf(ss / HIDDEN + RMS_EPS);
  }
  __syncthreads();
  const float rinv = rinv_sh;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) xs[i] = h[i] * rinv * w_in_norm[i];
  __syncthreads();

  const uint4* __restrict__ Wv = reinterpret_cast<const uint4*>(Wqkv);  // [QKV_OUT * K1_NVEC]
  const int gwarp = blockIdx.x * warps_per_cta + wid;
  const int nwarp = gridDim.x * warps_per_cta;

  // ring slot for (stage, row, lane) of THIS warp.
  auto slot = [&](int stage, int r) -> uint4* {
    return ring + (size_t)stage * ring_stage_stride + (size_t)wid * ring_warp_stride
                + (size_t)r * 32 + lane;
  };

  // grid-stride over groups of ROWS_PER_WARP rows.
  const int row_groups = (QKV_OUT + K1_ROWS_PER_WARP - 1) / K1_ROWS_PER_WARP;
  for (int rg = gwarp; rg < row_groups; rg += nwarp) {
    const int o0 = rg * K1_ROWS_PER_WARP;
    const int nrow = min(K1_ROWS_PER_WARP, QKV_OUT - o0);    // last group may be short

    float a0[K1_ROWS_PER_WARP], a1[K1_ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < K1_ROWS_PER_WARP; ++r) { a0[r] = 0.f; a1[r] = 0.f; }

    // base uint4 index of each row's lane-th chunk in phase 0: row o * K1_NVEC + lane.
    // phase p adds p*32 to the chunk index.
    // ---- prime stage 0 ----
    #pragma unroll
    for (int r = 0; r < K1_ROWS_PER_WARP; ++r) {
      if (r < nrow)
        cp_async16(slot(0, r), &Wv[(size_t)(o0 + r) * K1_NVEC + lane]);
    }
    cp_async_commit();

    #pragma unroll 1
    for (int p = 0; p < K1_PHASES; ++p) {
      const int cur = p & 1;
      const int nxt = (p + 1) & 1;
      // issue prefetch of phase p+1 (if any) into the other buffer.
      if (p + 1 < K1_PHASES) {
        const int chunk = (p + 1) * 32 + lane;
        #pragma unroll
        for (int r = 0; r < K1_ROWS_PER_WARP; ++r) {
          if (r < nrow)
            cp_async16(slot(nxt, r), &Wv[(size_t)(o0 + r) * K1_NVEC + chunk]);
        }
        cp_async_commit();
        cp_async_wait<1>();         // wait until only the just-issued group remains in flight
      } else {
        cp_async_wait<0>();         // last phase: drain everything
      }
      __syncwarp();
      // MAC the current phase: this lane's 16 activation floats start at (p*32 + lane)*16.
      const float* xx = xs + ((size_t)((p * 32 + lane)) << 4);
      #pragma unroll
      for (int r = 0; r < K1_ROWS_PER_WARP; ++r) {
        if (r < nrow) {
          uint4 wp = *slot(cur, r);
          mac_uint4_fp8(wp, xx, a0[r], a1[r]);
        }
      }
    }

    // reduce + scale + store each row.
    #pragma unroll
    for (int r = 0; r < K1_ROWS_PER_WARP; ++r) {
      if (r < nrow) {
        float acc = warp_reduce_sum(a0[r] + a1[r]);
        if (lane == 0) { int o = o0 + r; proj[o] = acc * Wqkv_scale[o]; }
      }
    }
  }
}

// ---- K1 epilogue (per-head QK-norm + RoPE + KV write): identical math to k1_epilogue, renamed. ----
// This part touches only ~8704 elems so it is essentially free; kept here so the file is a complete,
// self-validating K1 path. WARP-PER-HEAD; partner of channel d is d^64 which lives on the same lane.
constexpr int K1K2_HEAD_ROWS = N_Q_HEADS + 2 * N_KV_HEADS;   // 72
extern "C" __global__ void k1k2_epilogue(
    const float* __restrict__ proj,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ out_q, fp8* __restrict__ kv_k, fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale) {
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int row = gwarp; row < K1K2_HEAD_ROWS; row += nwarp) {
    const int is_q = (row < N_Q_HEADS);
    const int is_k = (!is_q && row < N_Q_HEADS + N_KV_HEADS);
    int proj_base, head_local;
    if (is_q)      { head_local = row;                          proj_base = head_local * HEAD_DIM; }
    else if (is_k) { head_local = row - N_Q_HEADS;              proj_base = Q_DIM + head_local*HEAD_DIM; }
    else           { head_local = row - N_Q_HEADS - N_KV_HEADS; proj_base = Q_DIM + KV_DIM + head_local*HEAD_DIM; }

    float chan[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) chan[c] = proj[proj_base + c * 32 + lane];

    if (!is_q && !is_k) {
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane, slot = head_local * HEAD_DIM + d;
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
        int d = c * 32 + lane, slot = head_local * HEAD_DIM + d;
        float s = kv_k_scale ? kv_k_scale[slot] : 1.f;
        kv_k[slot] = fp8(roped[c] / s);
      }
    }
  }
}

// =================================================================================================
// K2 — single-launch cooperative flash-decode (partial + reduce fused).
//
// The bench-measured K2 is overhead/latency bound (tiny ~4 MB KV read). The dominant avoidable cost is
// the SECOND kernel LAUNCH + relaunch latency of the separate reduce pass. We kill it by computing the
// partials, doing ONE grid.sync(), then having one warp per q_head merge the partials straight out of
// grid-resident scratch — no second launch, no relaunch latency. (The partials (m,l,acc) still live in
// device global memory and are written then read back, exactly as in the two-kernel path; they are
// tiny so they stay L2-resident — we do NOT eliminate that round-trip, only the extra launch.) The KV
// stream uses the k2 coalesced fp8 idiom with 2x time unrolling so each warp keeps two independent K
// (and V) loads in flight (shorter dependent online-softmax chain).
//
// Grid is launched cooperatively: grid.x = n_splits, grid.y groups q_heads (4 warps/CTA). The merge
// step is done by the first n_splits-independent set of warps (split 0 CTAs), one warp per q_head.
// =================================================================================================
constexpr int K2_VPL2 = HEAD_DIM / 32;   // 4 contiguous values per lane

__device__ __forceinline__ void k2v2_load4(const unsigned* __restrict__ base32, int lane,
                                            const float* __restrict__ s, float* __restrict__ out) {
  unsigned w = base32[lane];
  __nv_fp8x2_e4m3 lo, hi;
  lo.__x = (unsigned short)(w & 0xffffu);
  hi.__x = (unsigned short)(w >> 16);
  float2 fl = __half22float2((__half2)lo);
  float2 fh = __half22float2((__half2)hi);
  out[0] = fl.x * s[0];  out[1] = fl.y * s[1];
  out[2] = fh.x * s[2];  out[3] = fh.y * s[3];
}
__device__ __forceinline__ float k2v2_warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

extern "C" __global__ void k2k1_flash_decode_coop(
    const float* __restrict__ q,
    const fp8*  __restrict__ kv_k, const fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale,
    int ctx_len, int n_splits,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float* __restrict__ attn_out) {
  cg::grid_group grid = cg::this_grid();
  const int lane  = threadIdx.x & 31;
  const int wid   = threadIdx.x >> 5;
  const int warps_per_cta = blockDim.x >> 5;
  const int qh    = blockIdx.y * warps_per_cta + wid;
  const int split = blockIdx.x;

  // ---- Pass 1: partial online softmax over this (q_head, split) KV chunk ----
  if (qh < N_Q_HEADS) {
    const int kvh   = qh / GQA_GROUP;
    const int chunk = (ctx_len + n_splits - 1) / n_splits;
    const int t0 = split * chunk;
    const int t1 = min(t0 + chunk, ctx_len);
    const float scale = rsqrtf((float)HEAD_DIM);
    const int kv_base = kvh * HEAD_DIM;
    const int c0 = kv_base + lane * K2_VPL2;

    float qreg[K2_VPL2], ksc[K2_VPL2], vsc[K2_VPL2];
    #pragma unroll
    for (int c = 0; c < K2_VPL2; c++) {
      qreg[c] = q[qh * HEAD_DIM + lane * K2_VPL2 + c];
      ksc[c]  = kv_k_scale ? kv_k_scale[c0 + c] : 1.f;
      vsc[c]  = kv_v_scale ? kv_v_scale[c0 + c] : 1.f;
    }
    float m = -FLT_MAX, l = 0.f, acc[K2_VPL2];
    #pragma unroll
    for (int c = 0; c < K2_VPL2; c++) acc[c] = 0.f;

    const unsigned* __restrict__ k32 = reinterpret_cast<const unsigned*>(kv_k);
    const unsigned* __restrict__ v32 = reinterpret_cast<const unsigned*>(kv_v);
    const int row_words  = KV_DIM / 4;
    const int base_words = kv_base / 4;

    // 2x time-step unroll: prefetch K of two consecutive timesteps so two independent coalesced
    // loads are in flight, then process both. Shortens the dependent online-softmax chain's stall.
    int t = t0;
    for (; t + 1 < t1; t += 2) {
      const unsigned* k0 = k32 + (size_t)t * row_words + base_words;
      const unsigned* k1 = k32 + (size_t)(t + 1) * row_words + base_words;
      float kv0[K2_VPL2], kv1[K2_VPL2];
      k2v2_load4(k0, lane, ksc, kv0);
      k2v2_load4(k1, lane, ksc, kv1);          // two independent K loads in flight
      float p0 = 0.f, p1 = 0.f;
      #pragma unroll
      for (int c = 0; c < K2_VPL2; c++) { p0 += qreg[c]*kv0[c]; p1 += qreg[c]*kv1[c]; }
      float s0 = k2v2_warp_sum(p0) * scale;
      float s1 = k2v2_warp_sum(p1) * scale;
      const unsigned* w0 = v32 + (size_t)t * row_words + base_words;
      const unsigned* w1 = v32 + (size_t)(t + 1) * row_words + base_words;
      float vv0[K2_VPL2], vv1[K2_VPL2];
      k2v2_load4(w0, lane, vsc, vv0);
      k2v2_load4(w1, lane, vsc, vv1);
      // step t
      float mn = fmaxf(m, s0); float corr = __expf(m - mn); float pe = __expf(s0 - mn);
      l = l * corr + pe;
      #pragma unroll
      for (int c = 0; c < K2_VPL2; c++) acc[c] = acc[c]*corr + pe*vv0[c];
      m = mn;
      // step t+1
      mn = fmaxf(m, s1); corr = __expf(m - mn); pe = __expf(s1 - mn);
      l = l * corr + pe;
      #pragma unroll
      for (int c = 0; c < K2_VPL2; c++) acc[c] = acc[c]*corr + pe*vv1[c];
      m = mn;
    }
    for (; t < t1; t++) {                       // tail
      const unsigned* krow = k32 + (size_t)t * row_words + base_words;
      float kv[K2_VPL2];
      k2v2_load4(krow, lane, ksc, kv);
      float p = 0.f;
      #pragma unroll
      for (int c = 0; c < K2_VPL2; c++) p += qreg[c]*kv[c];
      float s = k2v2_warp_sum(p) * scale;
      const unsigned* vrow = v32 + (size_t)t * row_words + base_words;
      float vv[K2_VPL2];
      k2v2_load4(vrow, lane, vsc, vv);
      float mn = fmaxf(m, s); float corr = __expf(m - mn); float pe = __expf(s - mn);
      l = l * corr + pe;
      #pragma unroll
      for (int c = 0; c < K2_VPL2; c++) acc[c] = acc[c]*corr + pe*vv[c];
      m = mn;
    }
    const size_t pidx = (size_t)qh * n_splits + split;
    if (lane == 0) { part_m[pidx] = m; part_l[pidx] = l; }
    float* ao = part_acc + pidx * HEAD_DIM + lane * K2_VPL2;
    #pragma unroll
    for (int c = 0; c < K2_VPL2; c++) ao[c] = acc[c];
  }

  // ---- single grid-wide barrier: all partials are now written ----
  grid.sync();

  // ---- Pass 2: merge. Only split==0 CTAs do the reduce, one warp per q_head. ----
  if (split == 0 && qh < N_Q_HEADS) {
    float m = -FLT_MAX, l = 0.f, acc[K2_VPL2];
    #pragma unroll
    for (int c = 0; c < K2_VPL2; c++) acc[c] = 0.f;
    for (int sp = 0; sp < n_splits; sp++) {
      const size_t pidx = (size_t)qh * n_splits + sp;
      float ms = part_m[pidx], ls = part_l[pidx];
      if (ls <= 0.f) continue;
      const float* ai = part_acc + pidx * HEAD_DIM + lane * K2_VPL2;
      float mn = fmaxf(m, ms);
      float co = __expf(m - mn), cs = __expf(ms - mn);
      l = l * co + ls * cs;
      #pragma unroll
      for (int c = 0; c < K2_VPL2; c++) acc[c] = acc[c]*co + ai[c]*cs;
      m = mn;
    }
    float inv = (l > 0.f) ? (1.f / l) : 0.f;
    float* o = attn_out + qh * HEAD_DIM + lane * K2_VPL2;
    #pragma unroll
    for (int c = 0; c < K2_VPL2; c++) o[c] = acc[c] * inv;
  }
}

// =================================================================================================
// Launch helpers.
// =================================================================================================
struct K1Plan { int ctas, block; size_t smem; };
static inline K1Plan k1k2_plan(int block = 256) {
  K1Plan P; P.block = block;
  const int warps = block >> 5;
  const int row_groups = (QKV_OUT + K1_ROWS_PER_WARP - 1) / K1_ROWS_PER_WARP;  // 2304
  int need = (row_groups + warps - 1) / warps;
  int cap  = 264;                                  // ~2 CTAs/SM @256; oversubscribe lightly
  P.ctas = std::min(std::max(need, 132), cap);
  // dynamic smem: x[HIDDEN] floats + double-buffered weight ring.
  size_t ring = (size_t)K1_STAGES * warps * K1_ROWS_PER_WARP * 32 * sizeof(uint4);
  P.smem = (size_t)HIDDEN * sizeof(float) + ring;
  return P;
}

static inline void k1k2_launch(
    const float* h, const float* w_in_norm, const fp8* Wqkv, const float* Wqkv_scale,
    const float* q_norm, const float* k_norm, const float* rope_cos, const float* rope_sin,
    float* out_q, fp8* kv_k, fp8* kv_v, const float* kv_k_scale, const float* kv_v_scale,
    float* proj, cudaStream_t stream = 0) {
  K1Plan P = k1k2_plan();
  cudaFuncSetAttribute(k1k2_qkv_gemv, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P.smem);
  k1k2_qkv_gemv<<<P.ctas, P.block, P.smem, stream>>>(h, w_in_norm, Wqkv, Wqkv_scale, proj);
  k1k2_epilogue<<<3, 256, 0, stream>>>(proj, q_norm, k_norm, rope_cos, rope_sin,
                                       out_q, kv_k, kv_v, kv_k_scale, kv_v_scale);
}

// K2 cooperative launch. Returns split count used. part_* must be sized for n_splits.
static inline size_t k2v2_elems_m(int n_splits)  { return (size_t)N_Q_HEADS * n_splits; }
static inline size_t k2v2_elems_acc(int n_splits){ return (size_t)N_Q_HEADS * n_splits * HEAD_DIM; }
static inline int k2v2_pick_splits(int ctx_len) {
  int s = 64;
  if (ctx_len > 16384) s = 96;
  if (ctx_len > 65536) s = 128;
  int max_by_chunk = (ctx_len + 31) / 32;
  if (s > max_by_chunk) s = max_by_chunk;
  if (s < 1) s = 1;
  return s;
}

// Returns: >0 = split count if cooperative launch succeeded, 0 if the grid doesn't fit (caller can
// fall back to the two-kernel path).
static inline int k2v2_launch_coop(
    const float* q, const fp8* kv_k, const fp8* kv_v,
    const float* kv_k_scale, const float* kv_v_scale, int ctx_len,
    float* part_m, float* part_l, float* part_acc, float* attn_out,
    int n_splits = -1, cudaStream_t stream = 0) {
  if (n_splits <= 0) n_splits = k2v2_pick_splits(ctx_len);
  const int warps_per_cta = 4, block = warps_per_cta * 32;
  dim3 grid(n_splits, (N_Q_HEADS + warps_per_cta - 1) / warps_per_cta);

  // Cooperative launch requires every CTA resident simultaneously. Verify it fits.
  int dev = 0; cudaGetDevice(&dev);
  int max_blocks_per_sm = 0, num_sm = 0;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, k2k1_flash_decode_coop,
                                                block, 0);
  int total_ctas = grid.x * grid.y;
  if (max_blocks_per_sm * num_sm < total_ctas || max_blocks_per_sm == 0) return 0;

  void* args[] = {(void*)&q, (void*)&kv_k, (void*)&kv_v, (void*)&kv_k_scale, (void*)&kv_v_scale,
                  (void*)&ctx_len, (void*)&n_splits, (void*)&part_m, (void*)&part_l,
                  (void*)&part_acc, (void*)&attn_out};
  cudaError_t e = cudaLaunchCooperativeKernel((void*)k2k1_flash_decode_coop, grid, block,
                                              args, 0, stream);
  if (e != cudaSuccess) return 0;
  return n_splits;
}

// =================================================================================================
// CPU fp32 references (mirror the kernels exactly after fp8 round-trip; tolerance < 1e-2).
// =================================================================================================
static void k1_cpu_ref(const std::vector<float>& h, const std::vector<float>& w_in_norm,
                       const std::vector<fp8>& Wqkv, const std::vector<float>& Wscale,
                       const std::vector<float>& q_norm, const std::vector<float>& k_norm,
                       const std::vector<float>& rope_cos, const std::vector<float>& rope_sin,
                       const std::vector<float>& kv_k_scale, const std::vector<float>& kv_v_scale,
                       std::vector<float>& out_q, std::vector<float>& kc_f, std::vector<float>& vc_f) {
  double ss = 0; for (int i = 0; i < HIDDEN; i++) ss += (double)h[i]*h[i];
  float rinv = 1.f / std::sqrt((float)(ss / HIDDEN) + RMS_EPS);
  std::vector<float> x(HIDDEN);
  for (int i = 0; i < HIDDEN; i++) x[i] = h[i] * rinv * w_in_norm[i];
  std::vector<float> proj(QKV_OUT);
  for (int o = 0; o < QKV_OUT; o++) {
    double a = 0; const fp8* wr = &Wqkv[(size_t)o * HIDDEN];
    for (int k = 0; k < HIDDEN; k++) a += (double)((float)wr[k]) * x[k];
    proj[o] = (float)a * Wscale[o];
  }
  auto hnr = [&](float* v, const std::vector<float>& wn) {
    double s2 = 0; for (int d = 0; d < HEAD_DIM; d++) s2 += (double)v[d]*v[d];
    float hn = 1.f / std::sqrt((float)(s2 / HEAD_DIM) + RMS_EPS);
    float nm[HEAD_DIM]; for (int d = 0; d < HEAD_DIM; d++) nm[d] = v[d] * hn * wn[d];
    int half = HEAD_DIM / 2;
    for (int i = 0; i < half; i++) {
      float c = rope_cos[i], sn = rope_sin[i];
      v[i] = nm[i]*c - nm[i+half]*sn; v[i+half] = nm[i+half]*c + nm[i]*sn;
    }
  };
  for (int hd = 0; hd < N_Q_HEADS; hd++) {
    float buf[HEAD_DIM]; for (int d = 0; d < HEAD_DIM; d++) buf[d] = proj[hd*HEAD_DIM + d];
    hnr(buf, q_norm); for (int d = 0; d < HEAD_DIM; d++) out_q[hd*HEAD_DIM + d] = buf[d];
  }
  for (int hd = 0; hd < N_KV_HEADS; hd++) {
    float buf[HEAD_DIM]; for (int d = 0; d < HEAD_DIM; d++) buf[d] = proj[Q_DIM + hd*HEAD_DIM + d];
    hnr(buf, k_norm);
    for (int d = 0; d < HEAD_DIM; d++) { int slot = hd*HEAD_DIM+d; float s = kv_k_scale[slot];
      kc_f[slot] = (float)(fp8)(buf[d]/s) * s; }
  }
  for (int hd = 0; hd < N_KV_HEADS; hd++)
    for (int d = 0; d < HEAD_DIM; d++) { int slot = hd*HEAD_DIM+d; float s = kv_v_scale[slot];
      vc_f[slot] = (float)(fp8)(proj[Q_DIM + KV_DIM + hd*HEAD_DIM + d]/s) * s; }
}

static void k2_cpu_ref(const std::vector<float>& q, const std::vector<float>& kc,
                       const std::vector<float>& vc, int ctx_len, std::vector<float>& out) {
  const float scale = 1.f / std::sqrt((float)HEAD_DIM);
  for (int qh = 0; qh < N_Q_HEADS; qh++) {
    int kvh = qh / GQA_GROUP, kb = kvh * HEAD_DIM;
    std::vector<float> logit(ctx_len); float mx = -1e30f;
    for (int t = 0; t < ctx_len; t++) {
      double d = 0; for (int i = 0; i < HEAD_DIM; i++) d += (double)q[qh*HEAD_DIM+i]*kc[(size_t)t*KV_DIM+kb+i];
      logit[t] = (float)d * scale; mx = std::max(mx, logit[t]);
    }
    double denom = 0; std::vector<float> p(ctx_len);
    for (int t = 0; t < ctx_len; t++) { p[t] = std::exp(logit[t]-mx); denom += p[t]; }
    for (int i = 0; i < HEAD_DIM; i++) {
      double a = 0; for (int t = 0; t < ctx_len; t++) a += (double)p[t]*vc[(size_t)t*KV_DIM+kb+i];
      out[qh*HEAD_DIM+i] = (float)(a/denom);
    }
  }
}

// =================================================================================================
// Microbench main (guarded so this file can be #included as a kernel library elsewhere).
// =================================================================================================
#ifndef K1K2_NO_MAIN
#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)

static float rnd(unsigned& s, float lo, float hi) {
  s = s * 1664525u + 1013904223u;
  float u = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
  return lo + u * (hi - lo);
}

int main(int argc, char** argv) {
  const int ctx_len  = (argc > 1) ? atoi(argv[1]) : 4096;
  const int n_splits = (argc > 2) ? atoi(argv[2]) : -1;
  const double PEAK  = (argc > 3) ? atof(argv[3]) : 3350.0;
  unsigned seed = 0x1234abcdu;

  int ndev = 0, dev = 0; cudaDeviceProp prop;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("No CUDA device.\n"); return 1; }
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("== k1k2_mbu_v2  device=%s SMs=%d  ctx_len=%d  PEAK=%.0f GB/s ==\n",
         prop.name, prop.multiProcessorCount, ctx_len, PEAK);

  // ---------------- K1 host inputs ----------------
  std::vector<float> h(HIDDEN), w_in_norm(HIDDEN), Wscale(QKV_OUT);
  std::vector<float> q_norm(HEAD_DIM), k_norm(HEAD_DIM), rope_cos(HEAD_DIM/2), rope_sin(HEAD_DIM/2);
  std::vector<float> kv_k_scale(KV_DIM), kv_v_scale(KV_DIM);
  std::vector<fp8>   Wqkv((size_t)QKV_OUT * HIDDEN);
  for (auto& v : h)         v = rnd(seed, -1.f, 1.f);
  for (auto& v : w_in_norm) v = rnd(seed, 0.5f, 1.5f);
  for (auto& v : q_norm)    v = rnd(seed, 0.5f, 1.5f);
  for (auto& v : k_norm)    v = rnd(seed, 0.5f, 1.5f);
  for (auto& v : Wscale)    v = rnd(seed, 0.01f, 0.03f);
  for (auto& v : kv_k_scale)v = rnd(seed, 0.02f, 0.05f);
  for (auto& v : kv_v_scale)v = rnd(seed, 0.02f, 0.05f);
  for (int i = 0; i < HEAD_DIM/2; i++) {
    float freq = std::pow(ROPE_THETA, -2.f*i/HEAD_DIM), ang = freq * 7.f;
    rope_cos[i] = std::cos(ang); rope_sin[i] = std::sin(ang);
  }
  for (auto& v : Wqkv) v = (fp8)rnd(seed, -1.f, 1.f);

  std::vector<float> out_q_ref(Q_DIM), kc_ref(KV_DIM), vc_ref(KV_DIM);
  k1_cpu_ref(h, w_in_norm, Wqkv, Wscale, q_norm, k_norm, rope_cos, rope_sin,
             kv_k_scale, kv_v_scale, out_q_ref, kc_ref, vc_ref);

  float *d_h,*d_win,*d_Ws,*d_qn,*d_kn,*d_rc,*d_rs,*d_oq,*d_kks,*d_kvs,*d_proj; fp8 *d_W,*d_kk,*d_kv;
  CK(cudaMalloc(&d_h,HIDDEN*4)); CK(cudaMalloc(&d_win,HIDDEN*4)); CK(cudaMalloc(&d_Ws,QKV_OUT*4));
  CK(cudaMalloc(&d_qn,HEAD_DIM*4)); CK(cudaMalloc(&d_kn,HEAD_DIM*4));
  CK(cudaMalloc(&d_rc,HEAD_DIM/2*4)); CK(cudaMalloc(&d_rs,HEAD_DIM/2*4));
  CK(cudaMalloc(&d_kks,KV_DIM*4)); CK(cudaMalloc(&d_kvs,KV_DIM*4));
  CK(cudaMalloc(&d_W,(size_t)QKV_OUT*HIDDEN*sizeof(fp8)));
  CK(cudaMalloc(&d_oq,Q_DIM*4)); CK(cudaMalloc(&d_kk,KV_DIM*sizeof(fp8))); CK(cudaMalloc(&d_kv,KV_DIM*sizeof(fp8)));
  CK(cudaMalloc(&d_proj,QKV_OUT*4));
  CK(cudaMemcpy(d_h,h.data(),HIDDEN*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_win,w_in_norm.data(),HIDDEN*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_Ws,Wscale.data(),QKV_OUT*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_qn,q_norm.data(),HEAD_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_kn,k_norm.data(),HEAD_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_rc,rope_cos.data(),HEAD_DIM/2*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_rs,rope_sin.data(),HEAD_DIM/2*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_kks,kv_k_scale.data(),KV_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_kvs,kv_v_scale.data(),KV_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_W,Wqkv.data(),(size_t)QKV_OUT*HIDDEN*sizeof(fp8),cudaMemcpyHostToDevice));

  k1k2_launch(d_h,d_win,d_W,d_Ws,d_qn,d_kn,d_rc,d_rs,d_oq,d_kk,d_kv,d_kks,d_kvs,d_proj);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<float> oq(Q_DIM); std::vector<fp8> kk(KV_DIM), kv(KV_DIM);
  CK(cudaMemcpy(oq.data(),d_oq,Q_DIM*4,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(kk.data(),d_kk,KV_DIM*sizeof(fp8),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(kv.data(),d_kv,KV_DIM*sizeof(fp8),cudaMemcpyDeviceToHost));
  double e_oq=0,e_k=0,e_v=0;
  for (int i=0;i<Q_DIM;i++) e_oq=std::max(e_oq,(double)std::fabs(oq[i]-out_q_ref[i]));
  for (int i=0;i<KV_DIM;i++){ float gk=(float)kk[i]*kv_k_scale[i], gv=(float)kv[i]*kv_v_scale[i];
    e_k=std::max(e_k,(double)std::fabs(gk-kc_ref[i])); e_v=std::max(e_v,(double)std::fabs(gv-vc_ref[i])); }
  printf("K1  max-abs-err:  out_q=%.3e  k_cache=%.3e  v_cache=%.3e  -> %s (<1e-2)\n",
         e_oq,e_k,e_v, (e_oq<1e-2 && e_k<1e-2 && e_v<1e-2 ? "PASS":"FAIL"));

  // ---------------- K2 setup ----------------
  std::vector<fp8> KC((size_t)ctx_len*KV_DIM), VC((size_t)ctx_len*KV_DIM);
  std::vector<float> KCf((size_t)ctx_len*KV_DIM), VCf((size_t)ctx_len*KV_DIM);
  for (int t=0;t<ctx_len;t++) for (int c=0;c<KV_DIM;c++){
    float vk=rnd(seed,-1.f,1.f)*kv_k_scale[c], vv=rnd(seed,-1.f,1.f)*kv_v_scale[c];
    size_t idx=(size_t)t*KV_DIM+c;
    KC[idx]=(fp8)(vk/kv_k_scale[c]); VC[idx]=(fp8)(vv/kv_v_scale[c]);
    KCf[idx]=(float)KC[idx]*kv_k_scale[c]; VCf[idx]=(float)VC[idx]*kv_v_scale[c];
  }
  std::vector<float> q2(Q_DIM); for (auto& v:q2) v=rnd(seed,-1.f,1.f);
  std::vector<float> out_ref(Q_DIM);
  k2_cpu_ref(q2, KCf, VCf, ctx_len, out_ref);

  float *d_q2,*d_attn,*d_pm,*d_pl,*d_pacc; fp8 *d_KC,*d_VC;
  int S = (n_splits>0)? n_splits : k2v2_pick_splits(ctx_len);
  CK(cudaMalloc(&d_q2,Q_DIM*4)); CK(cudaMalloc(&d_attn,Q_DIM*4));
  CK(cudaMalloc(&d_KC,(size_t)ctx_len*KV_DIM*sizeof(fp8))); CK(cudaMalloc(&d_VC,(size_t)ctx_len*KV_DIM*sizeof(fp8)));
  CK(cudaMalloc(&d_pm, k2v2_elems_m(S)*4)); CK(cudaMalloc(&d_pl, k2v2_elems_m(S)*4));
  CK(cudaMalloc(&d_pacc, k2v2_elems_acc(S)*4));
  CK(cudaMemcpy(d_q2,q2.data(),Q_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_KC,KC.data(),(size_t)ctx_len*KV_DIM*sizeof(fp8),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_VC,VC.data(),(size_t)ctx_len*KV_DIM*sizeof(fp8),cudaMemcpyHostToDevice));

  int Sused = k2v2_launch_coop(d_q2,d_KC,d_VC,d_kks,d_kvs,ctx_len,d_pm,d_pl,d_pacc,d_attn,n_splits);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  bool coop_ok = (Sused > 0);
  if (coop_ok) {
    // Only read/compare d_attn when the cooperative kernel actually ran and wrote it.
    std::vector<float> attn(Q_DIM); CK(cudaMemcpy(attn.data(),d_attn,Q_DIM*4,cudaMemcpyDeviceToHost));
    double e_at=0; for (int i=0;i<Q_DIM;i++) e_at=std::max(e_at,(double)std::fabs(attn[i]-out_ref[i]));
    printf("K2  max-abs-err:  attn_out=%.3e  (coop splits=%d) -> %s (<1e-2)\n",
           e_at, Sused, (e_at<1e-2?"PASS":"FAIL"));
  } else
    printf("K2  cooperative launch did not fit on this GPU (grid too large for resident-all);\n"
           "    caller should fall back to the two-kernel path. (no coop timing below)\n");

  // ---------------- microbench ----------------
  cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e)); const int WARM=30,IT=300;

  // K1
  for(int i=0;i<WARM;i++) k1k2_launch(d_h,d_win,d_W,d_Ws,d_qn,d_kn,d_rc,d_rs,d_oq,d_kk,d_kv,d_kks,d_kvs,d_proj);
  CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
  for(int i=0;i<IT;i++)   k1k2_launch(d_h,d_win,d_W,d_Ws,d_qn,d_kn,d_rc,d_rs,d_oq,d_kk,d_kv,d_kks,d_kvs,d_proj);
  CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e)); float ms1; CK(cudaEventElapsedTime(&ms1,s,e)); ms1/=IT;
  double k1_bytes=(double)QKV_OUT*HIDDEN*sizeof(fp8);
  double k1_gbps = k1_bytes/1e6/ms1;
  K1Plan KP = k1k2_plan();
  printf("\nK1 (QKV GEMV)  Wqkv read %.1f MB   block=%d CTAs=%d smem=%.1fKB\n",
         k1_bytes/1e6, KP.block, KP.ctas, KP.smem/1024.0);
  // METRIC: weight-movement MBU = Wqkv bytes / time / HBM peak. Apples-to-apples with the 904 GB/s
  // baseline (which also counted only the weight read); it excludes the small activation reads
  // (h + w_in_norm, ~2x16KB/CTA) and the proj write, so true HBM utilization is slightly higher.
  printf("  %.2f us/token   %.0f GB/s   %.1f%% weight-movement MBU   -> %s (target >45%%)\n",
         ms1*1e3, k1_gbps, 100.0*k1_gbps/PEAK, (100.0*k1_gbps/PEAK>=45.0?"HIT":"below"));

  // K2 (cooperative single-launch), only if it fit
  if (coop_ok) {
    auto runK2 = [&]() {
      k2v2_launch_coop(d_q2,d_KC,d_VC,d_kks,d_kvs,ctx_len,d_pm,d_pl,d_pacc,d_attn,n_splits);
    };
    for(int i=0;i<WARM;i++) runK2();
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
    for(int i=0;i<IT;i++)   runK2();
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e)); float ms2; CK(cudaEventElapsedTime(&ms2,s,e)); ms2/=IT;
    double k2_bytes=2.0*(double)ctx_len*KV_DIM*sizeof(fp8);
    printf("\nK2 (coop flash-decode)  KV read %.1f MB  splits=%d\n", k2_bytes/1e6, Sused);
    printf("  %.2f us/token   %.0f GB/s   %.1f%% MBU  (latency-bound at this ctx)\n",
           ms2*1e3, k2_bytes/1e6/ms2, 100.0*(k2_bytes/1e6/ms2)/PEAK);
  }

  printf("\n== done ==\n");
  for (void* p : {(void*)d_h,(void*)d_win,(void*)d_Ws,(void*)d_qn,(void*)d_kn,(void*)d_rc,(void*)d_rs,
                  (void*)d_oq,(void*)d_kks,(void*)d_kvs,(void*)d_proj,(void*)d_W,(void*)d_kk,(void*)d_kv,
                  (void*)d_q2,(void*)d_attn,(void*)d_pm,(void*)d_pl,(void*)d_pacc,(void*)d_KC,(void*)d_VC})
    cudaFree(p);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return 0;
}
#endif // K1K2_NO_MAIN
