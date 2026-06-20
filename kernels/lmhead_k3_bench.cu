// lmhead_k3_bench.cu — the two GEMVs that drag the END of a B=1 decode step once the MoE
// experts are sharded: the lm_head projection and the attention O-proj. Both are fp8 GEMVs at
// M=1, so they are pure HBM-bandwidth problems (H100 HBM3 = 3.35 TB/s) — the goal is to read the
// fp8 weights once, fully coalesced, at the same ~45% peak the in-repo fp8 expert GEMV
// (k5_experts.cu, warp-per-output-row + coalesced uint4 fp8 + fp8x2->half2 dequant) already hits.
//
//   lm_head:  logits[VOCAB] = Wlm @ h ,  Wlm fp8 [VOCAB=151936, HIDDEN=4096]  (~622 MB / token,
//             runs ONCE per token at the very end). Greedy decode only needs argmax(logits), so the
//             top-1 argmax is FUSED into the GEMV epilogue: each warp produces one logit and folds
//             it into a block-local (val,idx) reduction, then one atomic per block updates the
//             global best — logits are never re-read from HBM for the argmax (no 622 KB round-trip,
//             trivial next to the weights but it removes a whole second dispatch).
//   O-proj:   h_out[HIDDEN] = h_in + Wo @ attn_out ,  Wo fp8 [HIDDEN=4096, Q_DIM=8192]  (~33.5 MB),
//             warp-per-output-row with the residual add fused into the epilogue (mirrors
//             k3_attn_epilogue.cu).
//
// THE IDIOM (reused verbatim from k5_experts.cu warp_dot_fp8 — the in-repo fast fp8 GEMV):
//   * WARP-PER-OUTPUT-ROW with split-K across the warp's 32 lanes: consecutive lanes read
//     consecutive 16-byte (uint4 = 16 fp8) chunks of the SAME weight row -> fully coalesced 128-bit
//     HBM loads (thread-per-row instead reads rows HIDDEN apart -> memory-divergent, the naive path).
//   * hardware fp8x2->half2 dequant (8 vector converts per 128-bit load) + 2 FP accumulators (ILP).
//   * the activation (h or attn_out) is staged once into shared memory per CTA.
//   * per-output-channel fp32 scale folded once onto the reduced dot (scale is per row).
//   * grid-stride over thousands of warps to fill the 132 SMs and hide HBM latency.
//   HIDDEN=4096 and Q_DIM=8192 are both multiples of 16, so the uint4 vectorization is exact.
//
// Reports GB/s, %HBMpeak, us/token for each GEMV, validates both against a CPU fp32 reference
// (max_rel < 1e-2), and reports the lm_head speedup of the coalesced warp-per-row path vs a naive
// thread-per-row GEMV.
//
// Build (compiles cleanly on sm_90a; standard CUDA only; uses common.cuh read-only):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/lmhead_k3_bench.cu -o /tmp/lmk3 && /tmp/lmk3
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

// ---------------------------------------------------------------------------------------------
// warp_dot_fp8 — coalesced split-K dot of one K-major fp8 weight row w[0..n) against a staged
// activation xs[0..n) (shared memory), collaborating across a 32-lane warp. n must be a multiple
// of 16. Result valid on lane 0. (Identical to k5_experts.cu warp_dot_fp8, the in-repo fast GEMV.)
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ float warp_dot_fp8(const fp8* __restrict__ w,
                                                      const float* __restrict__ xs,
                                                      int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;                                    // 16 fp8 per uint4
  for (int v = lane; v < nv; v += 32) {                     // consecutive lanes -> consecutive uint4
    uint4 p = wv[v];
    const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
    const float* xx = xs + (v << 4);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {                           // 4 x 32-bit words = 4 x (2 fp8 pairs)
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
  return acc;                                               // valid on lane 0
}

// =============================================================================================
// lm_head GEMV + fused top-1 argmax:  logits[t] = scale[t] * <h, Wlm[t]> ;  out = argmax_t logits[t]
// =============================================================================================
// One warp per vocab row t, grid-stride over VOCAB. h[HIDDEN] is staged once into shared memory.
// Lane 0 of each warp holds the row's logit; it is reduced to a per-CTA (best_val,best_idx) in shared
// memory, then a single atomic per CTA contends for the global best — so the VOCAB logits are never
// written to / re-read from HBM for the argmax. Optionally also writes the full logits vector (for the
// CPU reference / speculative-decode draft scoring); pass logits_out=nullptr to skip that write.
//
// Global argmax via a 64-bit CAS that packs the logit (ordered as a sortable uint) with the index, so
// ties resolve deterministically to the lowest index (matches the CPU reference's first-max rule).
__device__ __forceinline__ unsigned long long pack_argmax(float v, int idx) {
  // Order floats as unsigned so larger float => larger key. Then a SMALLER index wins ties, so we
  // store (~idx) in the low 32 bits and take the MAX of the packed 64-bit value.
  unsigned u = __float_as_uint(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);           // monotonic float->uint ordering
  return ((unsigned long long)u << 32) | (unsigned)(~idx & 0xffffffffu);
}
__device__ __forceinline__ int unpack_idx(unsigned long long packed) {
  return (int)(~(unsigned)(packed & 0xffffffffu));
}

static __device__ unsigned long long g_argmax;             // packed (key,~idx); reset by host each run

extern "C" __global__ void lmhead_gemv_argmax(
    const float* __restrict__ h,                            // [HIDDEN] final-norm hidden state
    const fp8*  __restrict__ Wlm, const float* __restrict__ Wlm_scale,  // [VOCAB,HIDDEN], scale[VOCAB]
    float* __restrict__ logits_out,                         // [VOCAB] or nullptr
    int* __restrict__ argmax_out) {                         // [1] token id of the max logit
  extern __shared__ float s[];                              // [HIDDEN] staged h  (+ reduction tail)
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) s[k] = h[k];
  // per-warp reduction scratch lives just past the staged activation
  float* red_val = s + HIDDEN;                              // [warps_per_cta]
  int*   red_idx = (int*)(red_val + (blockDim.x >> 5));     // [warps_per_cta]
  __syncthreads();

  const int lane   = threadIdx.x & 31;
  const int warp   = threadIdx.x >> 5;
  const int nwcta  = blockDim.x >> 5;
  const int gwarp  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp  = (gridDim.x * blockDim.x) >> 5;

  float best_val = -INFINITY; int best_idx = -1;
  for (int t = gwarp; t < VOCAB; t += nwarp) {
    float logit = warp_dot_fp8(Wlm + (size_t)t * HIDDEN, s, HIDDEN, lane) * Wlm_scale[t];
    if (lane == 0) {
      if (logits_out) logits_out[t] = logit;
      if (logit > best_val) { best_val = logit; best_idx = t; }
    }
  }
  // lane 0 of each warp holds (best_val,best_idx) -> reduce across warps of the CTA -> one atomic.
  if (lane == 0) { red_val[warp] = best_val; red_idx[warp] = best_idx; }
  __syncthreads();
  if (warp == 0) {
    float cv = (lane < nwcta) ? red_val[lane] : -INFINITY;
    int   ci = (lane < nwcta) ? red_idx[lane] : -1;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_down_sync(0xffffffffu, cv, o);
      int   oi = __shfl_down_sync(0xffffffffu, ci, o);
      if (ov > cv || (ov == cv && oi >= 0 && (ci < 0 || oi < ci))) { cv = ov; ci = oi; }
    }
    if (lane == 0 && ci >= 0)
      atomicMax(&g_argmax, pack_argmax(cv, ci));
  }
  // last block to finish publishes the unpacked global argmax (cheap; any block can do it safely
  // since g_argmax is monotonic under atomicMax, but writing once keeps argmax_out clean).
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // not guaranteed to be last; final read happens host-side too. Provide a best-effort write.
    argmax_out[0] = unpack_idx(g_argmax);
  }
}

// Tiny finalizer: unpack the global packed argmax into argmax_out[0]. Runs after the GEMV so the
// value is final regardless of block completion order (the in-kernel write above is best-effort).
extern "C" __global__ void lmhead_finalize_argmax(int* __restrict__ argmax_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) argmax_out[0] = unpack_idx(g_argmax);
}

// ---- NAIVE lm_head (thread-per-row, scalar fp8 casts) — the slow baseline for the speedup number.
// One THREAD per vocab row: 32 threads of a warp own 32 rows HIDDEN apart, so their fp8 loads are
// memory-divergent (no coalescing) and dequant is 1 scalar cast at a time. Same math, writes logits.
extern "C" __global__ void lmhead_gemv_naive(
    const float* __restrict__ h,
    const fp8* __restrict__ Wlm, const float* __restrict__ Wlm_scale,
    float* __restrict__ logits_out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int row = t; row < VOCAB; row += stride) {
    const fp8* w = Wlm + (size_t)row * HIDDEN;
    float acc = 0.f;
    for (int k = 0; k < HIDDEN; ++k) acc += h[k] * (float)w[k];
    logits_out[row] = acc * Wlm_scale[row];
  }
}

// =============================================================================================
// K3 O-proj GEMV + fused residual add:  h_out[o] = h_in[o] + Wo_scale[o] * <attn_out, Wo[o]>
// =============================================================================================
// One warp per HIDDEN output row, grid-stride; attn_out[Q_DIM] staged once into shared memory.
// (Same body as k3_attn_epilogue.cu — duplicated locally so this file never #includes/edits it.)
extern "C" __global__ void oproj_gemv(
    const float* __restrict__ attn_out,                     // [Q_DIM]
    const fp8* __restrict__ Wo, const float* __restrict__ Wo_scale,  // [HIDDEN,Q_DIM], scale[HIDDEN]
    const float* __restrict__ h_in,                         // [HIDDEN] residual in
    float* __restrict__ h_out) {                            // [HIDDEN] residual out (may alias h_in)
  extern __shared__ float xs[];                             // [Q_DIM]
  for (int k = threadIdx.x; k < Q_DIM; k += blockDim.x) xs[k] = attn_out[k];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int o = gwarp; o < HIDDEN; o += nwarp) {
    float acc = warp_dot_fp8(Wo + (size_t)o * Q_DIM, xs, Q_DIM, lane);
    if (lane == 0) h_out[o] = h_in[o] + acc * Wo_scale[o];   // fused residual add
  }
}

// =============================================================================================
// CPU fp32 references (read the exact fp8 bytes uploaded to the GPU, so the round-trip matches).
// =============================================================================================
void lmhead_reference(const float* h, const fp8* Wlm, const float* Wlm_scale,
                      float* logits, int* argmax) {
  double best = -1e300; int bi = -1;
  for (int t = 0; t < VOCAB; ++t) {
    const fp8* w = Wlm + (size_t)t * HIDDEN;
    double acc = 0.0;
    for (int k = 0; k < HIDDEN; ++k) acc += (double)h[k] * (double)(float)w[k];
    float lg = (float)acc * Wlm_scale[t];
    logits[t] = lg;
    if ((double)lg > best) { best = lg; bi = t; }            // first-max wins ties
  }
  *argmax = bi;
}

void oproj_reference(const float* attn_out, const fp8* Wo, const float* Wo_scale,
                     const float* h_in, float* h_out) {
  for (int o = 0; o < HIDDEN; ++o) {
    const fp8* w = Wo + (size_t)o * Q_DIM;
    double acc = 0.0;
    for (int k = 0; k < Q_DIM; ++k) acc += (double)attn_out[k] * (double)(float)w[k];
    h_out[o] = h_in[o] + (float)acc * Wo_scale[o];
  }
}

// =============================================================================================
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                     \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));          \
  exit(1); } } while (0)

// deterministic seeded host-side input generation (matches the k5 hash so inputs are reproducible).
static inline unsigned hash_u(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static inline float rnd(unsigned seed, size_t i, float scale, bool positive) {
  unsigned hh = hash_u((unsigned)(i * 2654435761u) ^ (seed * 40503u));
  float v = (((hh % 2001) / 1000.0f) - 1.0f) * scale;
  return positive ? (fabsf(v) + 1e-3f) : v;
}

int main(int argc, char** argv) {
  const int    BLK  = (argc > 1) ? atoi(argv[1]) : 256;       // 8 warps/CTA
  const double PEAK = (argc > 2) ? atof(argv[2]) : 3350.0;    // GB/s; H100 HBM3 = 3.35 TB/s

  int ndev = 0, dev = 0; cudaDeviceProp prop;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("No CUDA device.\n"); return 1; }
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s  SMs=%d  assumed HBM peak=%.0f GB/s\n", prop.name, prop.multiProcessorCount, PEAK);

  // ---- shapes ----
  const size_t lm_n = (size_t)VOCAB * HIDDEN;                 // 151936*4096 fp8 ~ 622 MB
  const size_t wo_n = (size_t)HIDDEN * Q_DIM;                 // 4096*8192   fp8 ~ 33.5 MB

  // ---- build inputs on the host (so the CPU reference reads the exact uploaded fp8 bytes) ----
  printf("building inputs (lm_head %zu fp8 = %.0f MB, O-proj %zu fp8 = %.0f MB) ...\n",
         lm_n, lm_n / 1e6, wo_n, wo_n / 1e6);
  std::vector<fp8>   Wlm_h(lm_n), Wo_h(wo_n);
  std::vector<float> Slm_h(VOCAB), So_h(HIDDEN), h_h(HIDDEN), attn_h(Q_DIM), hin_h(HIDDEN);
  for (size_t i = 0; i < lm_n; ++i) Wlm_h[i] = (fp8)rnd(1u, i, 0.25f, false);
  for (size_t i = 0; i < wo_n; ++i) Wo_h[i]  = (fp8)rnd(2u, i, 0.25f, false);
  for (int i = 0; i < VOCAB;  ++i) Slm_h[i] = rnd(7u,  i, 0.02f, true);
  for (int i = 0; i < HIDDEN; ++i) So_h[i]  = rnd(13u, i, 0.02f, true);
  for (int k = 0; k < HIDDEN; ++k) { h_h[k] = rnd(99u, k, 1.0f, false); hin_h[k] = rnd(123u, k, 1.0f, false); }
  for (int k = 0; k < Q_DIM;  ++k) attn_h[k] = rnd(55u, k, 1.0f, false);

  // ---- upload ----
  fp8 *Wlm_d, *Wo_d; float *Slm_d, *So_d, *h_d, *attn_d, *hin_d, *hout_d, *logits_d;
  int *argmax_d;
  CK(cudaMalloc(&Wlm_d, lm_n * sizeof(fp8)));   CK(cudaMemcpy(Wlm_d, Wlm_h.data(), lm_n * sizeof(fp8), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wo_d,  wo_n * sizeof(fp8)));   CK(cudaMemcpy(Wo_d,  Wo_h.data(),  wo_n * sizeof(fp8), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Slm_d, VOCAB * sizeof(float)));  CK(cudaMemcpy(Slm_d, Slm_h.data(), VOCAB * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&So_d,  HIDDEN * sizeof(float))); CK(cudaMemcpy(So_d,  So_h.data(),  HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&h_d,   HIDDEN * sizeof(float))); CK(cudaMemcpy(h_d,   h_h.data(),   HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&attn_d, Q_DIM * sizeof(float))); CK(cudaMemcpy(attn_d, attn_h.data(), Q_DIM * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&hin_d, HIDDEN * sizeof(float))); CK(cudaMemcpy(hin_d, hin_h.data(), HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&hout_d, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&logits_d, (size_t)VOCAB * sizeof(float)));
  CK(cudaMalloc(&argmax_d, sizeof(int)));
  CK(cudaDeviceSynchronize());

  // ---- launch geometry: warp-per-row, oversubscribe the 132 SMs lightly (grid-stride covers rest) ---
  const int warps_per_cta = BLK >> 5;
  auto ctas_for = [&](size_t rows) {
    int need = (int)((rows + warps_per_cta - 1) / warps_per_cta);
    return std::min(std::max(need, prop.multiProcessorCount), 4 * prop.multiProcessorCount);
  };
  const int ctas_lm = ctas_for(VOCAB);
  const int ctas_wo = ctas_for(HIDDEN);
  // smem: lm_head stages h[HIDDEN] + (warps_per_cta) floats + ints for the argmax reduction tail.
  const size_t smem_lm = (size_t)HIDDEN * sizeof(float) + (size_t)warps_per_cta * (sizeof(float) + sizeof(int));
  const size_t smem_wo = (size_t)Q_DIM  * sizeof(float);
  CK(cudaFuncSetAttribute(lmhead_gemv_argmax, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_lm));
  CK(cudaFuncSetAttribute(oproj_gemv,         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_wo));

  const unsigned long long ARG_INIT = 0ull;                  // packed key 0 = -inf, ~idx all-ones
  auto reset_argmax = [&]() { CK(cudaMemcpyToSymbol(g_argmax, &ARG_INIT, sizeof(ARG_INIT))); };

  // =========================== correctness ===========================
  // lm_head: GPU warp-per-row (with fused argmax) vs CPU fp32 reference.
  std::vector<float> lm_ref(VOCAB), lm_got(VOCAB);
  int argmax_ref = -1, argmax_got = -1;
  lmhead_reference(h_h.data(), Wlm_h.data(), Slm_h.data(), lm_ref.data(), &argmax_ref);

  reset_argmax();
  lmhead_gemv_argmax<<<ctas_lm, BLK, smem_lm>>>(h_d, Wlm_d, Slm_d, logits_d, argmax_d);
  lmhead_finalize_argmax<<<1, 32>>>(argmax_d);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(lm_got.data(), logits_d, (size_t)VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&argmax_got, argmax_d, sizeof(int), cudaMemcpyDeviceToHost));

  double lm_abs = 0.0, lm_rel = 0.0;
  for (int i = 0; i < VOCAB; ++i) {
    double ad = fabs((double)lm_ref[i] - (double)lm_got[i]);
    lm_abs = std::max(lm_abs, ad);
    lm_rel = std::max(lm_rel, ad / (fabs((double)lm_ref[i]) + 1e-6));
  }
  // argmax agreement: exact index match, OR (robust to fp ties) within tolerance of the max logit.
  bool argmax_ok = (argmax_got == argmax_ref) ||
                   (argmax_got >= 0 && argmax_got < VOCAB &&
                    fabs((double)lm_ref[argmax_got] - (double)lm_ref[argmax_ref]) < 1e-3);
  printf("\nlm_head correctness vs CPU fp32:  max_abs=%.3e  max_rel=%.3e  -> %s (<1e-2)\n",
         lm_abs, lm_rel, (lm_rel < 1e-2 ? "PASS" : "FAIL"));
  printf("lm_head argmax: gpu=%d  cpu=%d  -> %s\n",
         argmax_got, argmax_ref, argmax_ok ? "PASS" : "FAIL");

  // O-proj: GPU warp-per-row (fused residual) vs CPU fp32 reference.
  std::vector<float> o_ref(HIDDEN), o_got(HIDDEN);
  oproj_reference(attn_h.data(), Wo_h.data(), So_h.data(), hin_h.data(), o_ref.data());
  oproj_gemv<<<ctas_wo, BLK, smem_wo>>>(attn_d, Wo_d, So_d, hin_d, hout_d);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(o_got.data(), hout_d, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
  double o_abs = 0.0, o_rel = 0.0;
  for (int i = 0; i < HIDDEN; ++i) {
    double ad = fabs((double)o_ref[i] - (double)o_got[i]);
    o_abs = std::max(o_abs, ad);
    o_rel = std::max(o_rel, ad / (fabs((double)o_ref[i]) + 1e-6));
  }
  printf("O-proj  correctness vs CPU fp32:  max_abs=%.3e  max_rel=%.3e  -> %s (<1e-2)\n",
         o_abs, o_rel, (o_rel < 1e-2 ? "PASS" : "FAIL"));

  // =========================== microbench ===========================
  cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
  const int WARM = 20, IT = 100;
  auto bench = [&](auto launch) -> float {
    for (int i = 0; i < WARM; ++i) launch();
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
    for (int i = 0; i < IT; ++i) launch();
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms; CK(cudaEventElapsedTime(&ms, s, e)); return ms / IT;
  };

  // lm_head fast: fused argmax, NO logits write (greedy decode just needs the token id).
  auto run_lm_fast   = [&]() { lmhead_gemv_argmax<<<ctas_lm, BLK, smem_lm>>>(h_d, Wlm_d, Slm_d, nullptr, argmax_d); };
  // lm_head naive: thread-per-row, scalar casts, writes logits. Cap CTAs so it stays a fair GEMV.
  const int naive_ctas = std::min((VOCAB + BLK - 1) / BLK, 4 * prop.multiProcessorCount);
  auto run_lm_naive  = [&]() { lmhead_gemv_naive<<<naive_ctas, BLK>>>(h_d, Wlm_d, Slm_d, logits_d); };
  auto run_oproj     = [&]() { oproj_gemv<<<ctas_wo, BLK, smem_wo>>>(attn_d, Wo_d, So_d, hin_d, hout_d); };

  float ms_lm_fast  = bench(run_lm_fast);
  float ms_lm_naive = bench(run_lm_naive);
  float ms_oproj    = bench(run_oproj);
  CK(cudaGetLastError());

  // Bytes that MUST stream from HBM = the fp8 weight matrices (activations/scales are negligible:
  // h is 16 KB, attn_out 32 KB, logits 608 KB — all <0.2% of the 622 MB lm_head weight).
  const double lm_bytes = (double)lm_n;     // 1 byte / fp8 weight
  const double wo_bytes = (double)wo_n;
  auto gbps = [](double bytes, float ms) { return bytes / 1e6 / ms; };  // bytes/ms = GB/s

  printf("\nper-token weight reads:  lm_head %.1f MB   O-proj %.1f MB\n", lm_bytes/1e6, wo_bytes/1e6);
  printf("launch: block=%d  CTAs(lm)=%d  CTAs(O)=%d\n", BLK, ctas_lm, ctas_wo);
  printf("  %-24s %10s %10s %10s\n", "kernel", "us/tok", "GB/s", "%HBMpeak");
  printf("  %-24s %10.2f %10.1f %9.1f%%\n", "lm_head warp+argmax", ms_lm_fast  * 1e3, gbps(lm_bytes, ms_lm_fast),  100.0*gbps(lm_bytes, ms_lm_fast)/PEAK);
  printf("  %-24s %10.2f %10.1f %9.1f%%\n", "lm_head naive(thread)", ms_lm_naive * 1e3, gbps(lm_bytes, ms_lm_naive), 100.0*gbps(lm_bytes, ms_lm_naive)/PEAK);
  printf("  %-24s %10.2f %10.1f %9.1f%%\n", "O-proj  warp+residual", ms_oproj    * 1e3, gbps(wo_bytes, ms_oproj),    100.0*gbps(wo_bytes, ms_oproj)/PEAK);
  printf("\nlm_head speedup (warp-per-row coalesced vs naive thread-per-row): %.2fx\n", ms_lm_naive / ms_lm_fast);
  printf("end-of-step (lm_head + O-proj) fast path: %.3f ms/token\n", (ms_lm_fast + ms_oproj));

  // ---- cleanup ----
  cudaFree(Wlm_d); cudaFree(Wo_d); cudaFree(Slm_d); cudaFree(So_d);
  cudaFree(h_d); cudaFree(attn_d); cudaFree(hin_d); cudaFree(hout_d); cudaFree(logits_d); cudaFree(argmax_d);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return 0;
}
