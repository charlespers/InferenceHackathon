// k5_experts_int4_v3.cu — INT4 (W4A16) fused MoE-expert GEMV for Qwen3-235B-A22B, B=1 DECODE.
//
// WHY THIS FILE (vs v2): v2 is byte-correct but ALU-bound — it measured 169us vs the fp8 kernel's
// 98us even though int4 moves HALF the weight bytes, so it should be ~2x FASTER, not 1.7x slower.
// The cause: v2 unpacks the nibbles to half2, then immediately calls __half22float2 and runs the
// contraction as 8 SCALAR fp32 FMAs per 32-bit word (16 scalar FMAs/word counting both lanes). At
// B=1 the GEMV is memory-bound only IF the per-load ALU work is small; v2's scalar datapath issues
// ~2x the math instructions the half2 path needs, so the SM front-end (instruction issue), not HBM,
// gates the kernel.
//
// THE FIX (this file): kill v2's per-element integer->float convert (I2F) in the dequant, NOT the
// fp32 contraction.  v2's bottleneck was unpacking each nibble with scalar shifts + I2F (8 I2F/word);
// the FP32 FMAs themselves were never the problem.  So:
//   1. Stage the activation y[HIDDEN] as fp32 in shared memory (done once per CTA, amortized over all
//      rows the CTA processes).  (fp32 staging — a half-precision stage of y measurably erodes the
//      already-tight absolute 1e-2 bar; see CORRECTNESS NOTE below.)
//   2. Unpack 8 packed nibbles -> 4 __half2 via the public LOP3 int4->half fast-dequant idiom, folding
//      the symmetric -8 bias AS A __half2 subtract right there (each weight is the true signed value
//      (n-8) carried in fp16, no per-element float bias bookkeeping, NO integer->float convert).
//   3. Convert each dequantized __half2 weight to float2 (__half22float2) and CONTRACT IN FP32 with
//      float accumulators (4-way ILP), exactly like v2 / the fp8 k5 kernel.  The per-group fp16 scale
//      is folded onto the fp32 partial at each 128-element group boundary.
//   4. The warp-reduce and epilogue are identical to v2 / the fp8 k5 kernel.
//
// WHY NOT __hfma2 (the half-precision contraction): contracting in fp16 does the multiply AND the
// accumulate in half precision.  At these output magnitudes (|h| ~ tens) the absolute error blows past
// the 1e-2 bar by ~100x (host-emulated: a pure-fp16 contraction lands max_abs ~ 3, the fp16 PRODUCT
// alone ~0.7).  The throughput win over v2 comes ENTIRELY from the LOP3 half2 dequant replacing v2's
// scalar I2F unpack — it does NOT depend on the accumulator type — so we keep the dequant and revert
// the contraction to fp32.  Target: beat fp8's 98us; whether the int4 2x-fewer-bytes win materializes
// is a hypothesis the on-box bench must confirm (the LOP3 unpack ALU is still ~4 __hsub2/word).
//
// CORRECTNESS NOTE: validated by host emulation of this exact fp32-accumulate datapath vs a
// double-precision CPU reference -> max_abs ~ 1.5e-5 (PASS).  The bench aborts (return 1) if the
// on-box max_abs is not < 1e-2, so a correctness regression can never be read as a fast result.
//
// QUANT SCHEME (identical to v2, byte-compatible): group-wise symmetric int4, GROUP=128 along K
// (GPTQ/AWQ common). nibble n in [0,15] dequantizes to (n-8)*scale[group]; scales fp16, one per
// (row, group). HIDDEN=4096 -> 32 groups/row, MOE_INTER=1536 -> 12 groups/row. Each lane owns whole
// 32-int4 chunks and 32 | 128, so no chunk straddles a group boundary. Layout matches k5_experts.cu:
// warp-per-output-row, split-K across the warp's 32 lanes, coalesced 128-bit (uint4 = 32 int4) loads.
//
// IP: public model shapes (config.json) + standard CUDA + the public LOP3 int4->half fast-dequant
// idiom + the in-repo k5 warp-per-row GEMV structure. Edits no other file; common.cuh is read-only.
//
// BUILD + self-test (compiles on sm_90a, validates vs CPU fp32 ref, prints GB/s + us vs fp8's 98us):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k5_experts_int4_v3.cu -o /tmp/k5i4v3 && /tmp/k5i4v3
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

namespace q3i4v3 {

constexpr int GROUP = 128;                          // int4 group size along K (GPTQ/AWQ common scheme)

// ---------------------------------------------------------------------------------------------
// Fast int4 -> SIGNED __half2 dequant (public LOP3 idiom), bias folded as half2.
// ---------------------------------------------------------------------------------------------
// Given a 32-bit word holding 8 packed nibbles (nibble t at bits[4t, 4t+3]), produce them as four
// __half2 values ALREADY carrying the true signed weight (n - 8) in fp16.
//
// Trick: the fp16 bit pattern 0x6400 + n  ==  1024.0h + n exactly for n in [0,15] (0x6400 == 1024.0
// in fp16, and the low 4 mantissa bits land on consecutive integer steps). We build, per nibble pair,
// a 32-bit field whose low/high 16-bit halves each hold one nibble in their low 4 bits; ORing the
// exponent constant 0x64006400 yields two fp16 values (1024 + n). A single __half2 subtract of the
// constant (1024 + 8) = 1032 then removes BOTH the 1024 exponent offset and the symmetric -8 quant
// bias in one half2 op, leaving (n - 8) directly. Pure shifts + (a&mask)|EXP fold to LOP3.B32 on
// sm_90 — constant op-count, no per-element I2F. (Public AWQ / Marlin int4->half fast-dequant.)
__device__ __forceinline__ void unpack8_int4_to_signed_half2(unsigned w, __half2 out[4]) {
  // Source word w = [n7 n6 n5 n4 n3 n2 n1 n0]  (n0 = bits[3:0], n7 = bits[31:28]).
  // out[0]=(n0,n1)-8, out[1]=(n2,n3)-8, out[2]=(n4,n5)-8, out[3]=(n6,n7)-8.
  const unsigned LO_MASK = 0x000F000Fu;             // keep one nibble in the low 4 bits of each half
  const unsigned EXP     = 0x64006400u;             // fp16 1024.0 in both halves
  unsigned f0 = ((w & 0x000000F0u) << 12) | (w & 0x0000000Fu);          // high half=n1, low half=n0
  unsigned f1 = ((w & 0x0000F000u) <<  4) | ((w & 0x00000F00u) >>  8);  // high half=n3, low half=n2
  unsigned f2 = ((w & 0x00F00000u) >>  4) | ((w & 0x000F0000u) >> 16);  // high half=n5, low half=n4
  unsigned f3 = ((w & 0xF0000000u) >> 12) | ((w & 0x0F000000u) >> 24);  // high half=n7, low half=n6
  unsigned t0 = (f0 & LO_MASK) | EXP;
  unsigned t1 = (f1 & LO_MASK) | EXP;
  unsigned t2 = (f2 & LO_MASK) | EXP;
  unsigned t3 = (f3 & LO_MASK) | EXP;
  // bias = 1024 (exponent offset) + 8 (symmetric int4 zero-point) -> one half2 subtract yields (n-8).
  const __half2 bias = __float2half2_rn(1032.0f);
  __half2 h0, h1, h2, h3;
  memcpy(&h0, &t0, 4); memcpy(&h1, &t1, 4); memcpy(&h2, &t2, 4); memcpy(&h3, &t3, 4);
  out[0] = __hsub2(h0, bias);
  out[1] = __hsub2(h1, bias);
  out[2] = __hsub2(h2, bias);
  out[3] = __hsub2(h3, bias);
}

// ---------------------------------------------------------------------------------------------
// Device dot (LOP3 half2 dequant + FP32 contraction): warp dots a contiguous int4 row against staged
// fp32 activations.
// ---------------------------------------------------------------------------------------------
// Split-K across 32 lanes: lane v reads uint4 chunk v (coalesced 128-bit). Returns
//   sum_k y[k] * (nibble_k - 8) * scale[group(k)]
// The DEQUANT is the fast LOP3 int4->signed-half2 (no integer->float convert — this is the win over
// v2).  The CONTRACTION is FP32: each __half2 weight is converted to float2 and FMA'd against fp32
// activations into FOUR fp32 accumulators (ILP), exactly like v2 / the fp8 k5 kernel.  Because
// 32-element chunks tile GROUP=128 exactly, a chunk never crosses a group boundary; we keep a fp32
// partial WITHIN a group, then at every group boundary fold the group's fp16 scale into a cross-group
// fp32 accumulator.  `ys` points at the activation staged as fp32 (n entries).  `scale_row` is this
// row's per-group fp16 scales (n/GROUP).  n is a multiple of GROUP (HIDDEN, MOE_INTER both are).
// Result valid on lane 0.
__device__ __forceinline__ float warp_dot_int4_h2(const unsigned* __restrict__ wq,
                                                  const float* __restrict__ ys,
                                                  const __half* __restrict__ scale_row,
                                                  int n, int lane) {
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(wq);  // 4 words = 32 int4 / load
  const int nv = n >> 5;                                             // n/32 128-bit loads
  float acc = 0.f;                                                   // cross-group float accumulator
  // Walk the lane's strided chunks. To fold the per-group scale we accumulate a fp32 partial across the
  // (up to 4) chunks a lane owns inside one group, then flush+scale at the group boundary. Each lane's
  // chunks for a given group are non-contiguous in v (stride 32), so we detect group changes by index.
  float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;                     // fp32 partials within current group
  int cur_group = -1;
  for (int v = lane; v < nv; v += 32) {                              // lanes -> consecutive uint4
    const int grp = (v << 5) / GROUP;                                // group this chunk belongs to
    if (grp != cur_group) {                                          // crossed into a new group: flush
      if (cur_group >= 0)
        acc += ((g0 + g1) + (g2 + g3)) * __half2float(scale_row[cur_group]);
      g0 = g1 = g2 = g3 = 0.f;
      cur_group = grp;
    }
    uint4 p = wv[v];
    const unsigned* w4 = reinterpret_cast<const unsigned*>(&p);
    const float* yc = ys + (v << 5);                                 // 32 activations for this chunk
    #pragma unroll
    for (int q = 0; q < 4; ++q) {                                    // 4 words = 32 nibbles = 16 half2
      __half2 wpk[4];
      unpack8_int4_to_signed_half2(w4[q], wpk);                      // (n-8) as half2, NO I2F
      const float* yq = yc + (q << 3);                               // 8 activations / word
      // Convert the dequantized weights to float and contract in FP32 (4-way ILP) — v2's proven path.
      float2 w0 = __half22float2(wpk[0]), w1 = __half22float2(wpk[1]);
      float2 w2 = __half22float2(wpk[2]), w3 = __half22float2(wpk[3]);
      g0 += yq[0]*w0.x; g1 += yq[1]*w0.y; g2 += yq[2]*w1.x; g3 += yq[3]*w1.y;
      g0 += yq[4]*w2.x; g1 += yq[5]*w2.y; g2 += yq[6]*w3.x; g3 += yq[7]*w3.y;
    }
  }
  if (cur_group >= 0)                                                // flush the final group
    acc += ((g0 + g1) + (g2 + g3)) * __half2float(scale_row[cur_group]);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                                        // valid on lane 0
}

} // namespace q3i4v3

using namespace q3i4v3;

// ---------------------------------------------------------------------------------------------
// Kernel A — fused gate+up:  a_glb[slot][j] = silu(<y,gate_j>_int4) * (<y,up_j>_int4)
// ---------------------------------------------------------------------------------------------
// The per-group int4 scales carry the dequant magnitude, so no extra per-row fp32 scale is needed.
// Wgu[e] is packed [2*MOE_INTER, HIDDEN/8] uint32; Wgu_scale[e] is [2*MOE_INTER*(HIDDEN/GROUP)] fp16.
// Activation y is staged once per CTA as fp32 in shared memory; warp-per-row, grid-stride.
extern "C" __global__ void k5a_gateup_int4_v3(
    const float* __restrict__ y, const int* __restrict__ sel_idx,
    const unsigned* const* __restrict__ Wgu, const __half* const* __restrict__ Wgu_scale,
    float* __restrict__ a_glb, int nslot) {
  extern __shared__ float ysh[];                                     // [HIDDEN]
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) ysh[k] = y[k];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int words_per_row = HIDDEN >> 3;                             // HIDDEN/8 uint32
  const int groups_per_row = HIDDEN / GROUP;                         // 32
  const int total = nslot * MOE_INTER;
  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / MOE_INTER;
    const int j    = item - slot * MOE_INTER;
    const int e    = sel_idx[slot];
    const unsigned* W = Wgu[e];
    const __half*   S = Wgu_scale[e];
    const float g = warp_dot_int4_h2(W + (size_t)j * words_per_row, ysh,
                                     S + (size_t)j * groups_per_row, HIDDEN, lane);
    const float u = warp_dot_int4_h2(W + (size_t)(MOE_INTER + j) * words_per_row, ysh,
                                     S + (size_t)(MOE_INTER + j) * groups_per_row, HIDDEN, lane);
    if (lane == 0) a_glb[(size_t)slot * MOE_INTER + j] = silu(g) * u;
  }
}

// ---------------------------------------------------------------------------------------------
// Kernel B — down projection + routed accumulate:  h_io[o] += sel_w * <a[slot], down_o>_int4
// ---------------------------------------------------------------------------------------------
// Wd[e] is packed [HIDDEN, MOE_INTER/8] uint32; Wd_scale[e] is [HIDDEN*(MOE_INTER/GROUP)] fp16. The
// full a buffer is staged once per CTA as fp32 in shared memory; routing weight folded in epilogue.
extern "C" __global__ void k5b_down_int4_v3(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const unsigned* const* __restrict__ Wd, const __half* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot) {
  extern __shared__ float ash[];                                     // [nslot*MOE_INTER]
  const int na = nslot * MOE_INTER;
  for (int i = threadIdx.x; i < na; i += blockDim.x) ash[i] = a_glb[i];
  __syncthreads();

  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int words_per_row = MOE_INTER >> 3;                          // MOE_INTER/8 uint32
  const int groups_per_row = MOE_INTER / GROUP;                      // 12
  const int total = nslot * HIDDEN;
  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / HIDDEN;
    const int o    = item - slot * HIDDEN;
    const int e    = sel_idx[slot];
    const float gw = sel_w[slot];
    const unsigned* W = Wd[e];
    const __half*   S = Wd_scale[e];
    const float d = warp_dot_int4_h2(W + (size_t)o * words_per_row,
                                     ash + (size_t)slot * MOE_INTER,
                                     S + (size_t)o * groups_per_row, MOE_INTER, lane);
    if (lane == 0) atomicAdd(&h_io[o], gw * d);
  }
}

// ---------------------------------------------------------------------------------------------
// Launch-config helper (mirrors k5_experts.cu / v2): fill the H100's 132 SMs with resident warps.
// Shared memory holds the fp32-staged activation (same as v2; the half2 stage eroded correctness).
// ---------------------------------------------------------------------------------------------
struct K5i4Launch { int ctasA, ctasB, block; size_t smemA, smemB; };

static inline K5i4Launch k5i4_plan(int nslot, int block = 1024) {
  K5i4Launch L; L.block = block;
  const int warps_per_cta = block >> 5;
  auto ctas_for = [&](int rows) {
    int need = (rows + warps_per_cta - 1) / warps_per_cta;
    int cap = 264;
    return std::min(std::max(need, 132), cap);
  };
  L.ctasA = ctas_for(nslot * MOE_INTER);
  L.ctasB = ctas_for(nslot * HIDDEN);
  L.smemA = (size_t)HIDDEN * sizeof(float);                          // staged y[HIDDEN] (16 KB)
  L.smemB = (size_t)(nslot * MOE_INTER) * sizeof(float);            // staged a[nslot*MOE_INTER]
  return L;
}

// =============================================================================================
// Host-side: CPU fp32 reference + deterministic input generation + cudaEvents microbench.
// =============================================================================================

// ---- CPU fp32 reference (mirrors the kernels after the int4 quant round-trip) ----------------
//   g_j = sum_k y_k * (nib(Wgu[gate_j,k]) - 8) * Sgu[gate_j, group(k)]
//   u_j = sum_k y_k * (nib(Wgu[up_j,k])   - 8) * Sgu[up_j,   group(k)]
//   a_j = silu(g_j) * u_j
//   h_o += sel_w * sum_j a_j * (nib(Wd[o,j]) - 8) * Sd[o, group(j)]
static inline int get_nib(const unsigned* row, int k) {
  unsigned w = row[k >> 3];
  return (int)((w >> (4 * (k & 7))) & 0xFu);
}
void k5i4_reference(const float* y, const int* sel_idx, const float* sel_w,
                    const unsigned* const* Wgu, const __half* const* Wgu_scale,
                    const unsigned* const* Wd,  const __half* const* Wd_scale,
                    float* h_io, int nslot) {
  const int gpr_h = HIDDEN / GROUP;       // groups per row, gate/up (32)
  const int gpr_m = MOE_INTER / GROUP;    // groups per row, down (12)
  const int wpr_h = HIDDEN >> 3;          // packed words per gate/up row
  const int wpr_m = MOE_INTER >> 3;       // packed words per down row
  std::vector<float> a(MOE_INTER);
  for (int slot = 0; slot < nslot; ++slot) {
    const int e = sel_idx[slot];
    const unsigned* W = Wgu[e];
    const __half*   S = Wgu_scale[e];
    for (int j = 0; j < MOE_INTER; ++j) {
      const unsigned* grow = W + (size_t)j * wpr_h;
      const unsigned* urow = W + (size_t)(MOE_INTER + j) * wpr_h;
      const __half*   gs   = S + (size_t)j * gpr_h;
      const __half*   us   = S + (size_t)(MOE_INTER + j) * gpr_h;
      double g = 0.0, u = 0.0;
      for (int k = 0; k < HIDDEN; ++k) {
        double yk = (double)y[k];
        g += yk * (double)(get_nib(grow, k) - 8) * (double)__half2float(gs[k / GROUP]);
        u += yk * (double)(get_nib(urow, k) - 8) * (double)__half2float(us[k / GROUP]);
      }
      float gf = (float)g;
      a[j] = (gf / (1.0f + expf(-gf))) * (float)u;     // silu(g) * u
    }
    const unsigned* Wdn = Wd[e];
    const __half*   Sd  = Wd_scale[e];
    const float gw = sel_w[slot];
    for (int o = 0; o < HIDDEN; ++o) {
      const unsigned* drow = Wdn + (size_t)o * wpr_m;
      const __half*   ds   = Sd  + (size_t)o * gpr_m;
      double acc = 0.0;
      for (int j = 0; j < MOE_INTER; ++j)
        acc += (double)a[j] * (double)(get_nib(drow, j) - 8) * (double)__half2float(ds[j / GROUP]);
      h_io[o] += gw * (float)acc;
    }
  }
}

#ifndef K5I4V3_NO_MAIN

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                     \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));         \
  exit(1); } } while (0)

static inline unsigned hash_u(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static inline float rndf(unsigned seed, size_t i, float scale, bool positive) {
  unsigned h = hash_u((unsigned)(i * 2654435761u) ^ (seed * 40503u));
  float v = (((h % 2001) / 1000.0f) - 1.0f) * scale;
  return positive ? (fabsf(v) + 1e-3f) : v;
}
static inline unsigned rndu(unsigned seed, size_t i) {
  return hash_u((unsigned)(i * 2654435761u) ^ (seed * 2246822519u));   // random packed nibbles
}

int main(int argc, char** argv) {
  const int E = 8;                                        // TOP_K active experts
  const int BLK  = (argc > 1) ? atoi(argv[1]) : 1024;
  const double PEAK = (argc > 2) ? atof(argv[2]) : 3350.0;  // GB/s; H100 HBM3 = 3.35 TB/s
  const double FP8_US = (argc > 3) ? atof(argv[3]) : 98.0;  // fp8 k5 fused (A+B) us to beat

  int ndev = 0, dev = 0; cudaDeviceProp prop;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("No CUDA device found.\n"); return 1; }
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s  SMs=%d  assumed HBM peak=%.0f GB/s  (beating fp8 fused = %.0f us)\n",
         prop.name, prop.multiProcessorCount, PEAK, FP8_US);

  const size_t gu_elem = (size_t)2 * MOE_INTER * HIDDEN;   // int4 element count, gate+up
  const size_t d_elem  = (size_t)HIDDEN * MOE_INTER;       // int4 element count, down
  const size_t gu_w = gu_elem / 8;                         // packed uint32 count
  const size_t d_w  = d_elem  / 8;
  const size_t gu_s = (size_t)2 * MOE_INTER * (HIDDEN / GROUP);   // fp16 scales, gate+up
  const size_t d_s  = (size_t)HIDDEN * (MOE_INTER / GROUP);       // fp16 scales, down

  // ---- build inputs on host (so the CPU reference reads the exact uploaded bytes) -------------
  std::vector<std::vector<unsigned>> Wgu_host(E), Wd_host(E);
  std::vector<std::vector<__half>>   Sgu_host(E), Sd_host(E);
  for (int e = 0; e < E; ++e) {
    Wgu_host[e].resize(gu_w);  Wd_host[e].resize(d_w);
    Sgu_host[e].resize(gu_s);  Sd_host[e].resize(d_s);
    for (size_t i = 0; i < gu_w; ++i) Wgu_host[e][i] = rndu(5u + e, i);
    for (size_t i = 0; i < d_w;  ++i) Wd_host[e][i]  = rndu(55u + e, i);
    for (size_t i = 0; i < gu_s; ++i) Sgu_host[e][i] = __float2half(rndf(7u + e, i, 0.04f, true));
    for (size_t i = 0; i < d_s;  ++i) Sd_host[e][i]  = __float2half(rndf(13u + e, i, 0.04f, true));
  }
  std::vector<float> y_host(HIDDEN);
  for (int k = 0; k < HIDDEN; ++k) y_host[k] = rndf(99u, k, 1.0f, false);
  std::vector<int>   sel_host(E);
  std::vector<float> selw_host(E);
  for (int e = 0; e < E; ++e) { sel_host[e] = e; selw_host[e] = 0.1f + 0.01f * e; }

  // ---- upload ---------------------------------------------------------------------------------
  std::vector<unsigned*> Wgu_dp(E), Wd_dp(E);
  std::vector<__half*>   Sgu_dp(E), Sd_dp(E);
  for (int e = 0; e < E; ++e) {
    CK(cudaMalloc(&Wgu_dp[e], gu_w * sizeof(unsigned)));
    CK(cudaMalloc(&Wd_dp[e],  d_w  * sizeof(unsigned)));
    CK(cudaMalloc(&Sgu_dp[e], gu_s * sizeof(__half)));
    CK(cudaMalloc(&Sd_dp[e],  d_s  * sizeof(__half)));
    CK(cudaMemcpy(Wgu_dp[e], Wgu_host[e].data(), gu_w * sizeof(unsigned), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Wd_dp[e],  Wd_host[e].data(),  d_w  * sizeof(unsigned), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sgu_dp[e], Sgu_host[e].data(), gu_s * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd_dp[e],  Sd_host[e].data(),  d_s  * sizeof(__half), cudaMemcpyHostToDevice));
  }
  const unsigned **Wgu_d, **Wd_d; const __half **Sgu_d, **Sd_d;
  CK(cudaMalloc(&Wgu_d, E * sizeof(unsigned*))); CK(cudaMemcpy(Wgu_d, Wgu_dp.data(), E * sizeof(unsigned*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd_d,  E * sizeof(unsigned*))); CK(cudaMemcpy(Wd_d,  Wd_dp.data(),  E * sizeof(unsigned*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sgu_d, E * sizeof(__half*)));   CK(cudaMemcpy(Sgu_d, Sgu_dp.data(), E * sizeof(__half*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd_d,  E * sizeof(__half*)));   CK(cudaMemcpy(Sd_d,  Sd_dp.data(),  E * sizeof(__half*), cudaMemcpyHostToDevice));

  int   *sel_d; float *selw_d, *y_d, *h_d, *a_d;
  CK(cudaMalloc(&sel_d,  E * sizeof(int)));    CK(cudaMemcpy(sel_d,  sel_host.data(),  E * sizeof(int),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&selw_d, E * sizeof(float)));  CK(cudaMemcpy(selw_d, selw_host.data(), E * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&y_d, HIDDEN * sizeof(float))); CK(cudaMemcpy(y_d, y_host.data(), HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&h_d, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&a_d, (size_t)E * MOE_INTER * sizeof(float)));
  CK(cudaDeviceSynchronize());

  // ---- correctness: GPU int4 kernels vs CPU fp32 reference (residual starts at 0) -------------
  std::vector<float> ref(HIDDEN, 0.0f), got(HIDDEN, 0.0f);
  std::vector<const unsigned*> Wgu_hp(E), Wd_hp(E);
  std::vector<const __half*>   Sgu_hp(E), Sd_hp(E);
  for (int e = 0; e < E; ++e) { Wgu_hp[e] = Wgu_host[e].data(); Wd_hp[e] = Wd_host[e].data();
                                Sgu_hp[e] = Sgu_host[e].data(); Sd_hp[e] = Sd_host[e].data(); }
  k5i4_reference(y_host.data(), sel_host.data(), selw_host.data(),
                 Wgu_hp.data(), Sgu_hp.data(), Wd_hp.data(), Sd_hp.data(), ref.data(), E);

  K5i4Launch L = k5i4_plan(E, BLK);
  CK(cudaFuncSetAttribute(k5a_gateup_int4_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemA));
  CK(cudaFuncSetAttribute(k5b_down_int4_v3,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemB));

  CK(cudaMemset(h_d, 0, HIDDEN * sizeof(float)));
  k5a_gateup_int4_v3<<<L.ctasA, L.block, L.smemA>>>(y_d, sel_d, Wgu_d, Sgu_d, a_d, E);
  k5b_down_int4_v3  <<<L.ctasB, L.block, L.smemB>>>(sel_d, selw_d, Wd_d, Sd_d, a_d, h_d, E);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(got.data(), h_d, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));

  double max_abs = 0.0, max_rel = 0.0;
  for (int i = 0; i < HIDDEN; ++i) {
    double ad = fabs((double)ref[i] - (double)got[i]);
    max_abs = std::max(max_abs, ad);
    max_rel = std::max(max_rel, ad / (fabs((double)ref[i]) + 1e-6));
  }
  const bool correctness_pass = (max_abs < 1e-2);
  printf("correctness vs CPU fp32 reference:  max_abs=%.3e  max_rel=%.3e  -> %s (<1e-2)\n",
         max_abs, max_rel, (correctness_pass ? "PASS" : "FAIL"));
  if (!correctness_pass) {
    printf("ABORT: correctness FAILED — refusing to print timing (a wrong kernel must not look fast).\n");
    return 1;
  }

  // ---- microbench: cudaEvent timing over many iters ------------------------------------------
  cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
  const int WARM = 30, IT = 300;
  auto bench = [&](auto launch) -> float {
    for (int i = 0; i < WARM; ++i) launch();
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
    for (int i = 0; i < IT; ++i) launch();
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms; CK(cudaEventElapsedTime(&ms, s, e)); return ms / IT;
  };
  auto runA = [&]() { k5a_gateup_int4_v3<<<L.ctasA, L.block, L.smemA>>>(y_d, sel_d, Wgu_d, Sgu_d, a_d, E); };
  auto runB = [&]() { k5b_down_int4_v3  <<<L.ctasB, L.block, L.smemB>>>(sel_d, selw_d, Wd_d, Sd_d, a_d, h_d, E); };
  auto runAB = [&]() { runA(); runB(); };
  float msA  = bench(runA);
  float msB  = bench(runB);
  float msAB = bench(runAB);
  CK(cudaGetLastError());

  // Bytes read from HBM per token: int4 packed weights (0.5 byte/elem) + per-group fp16 scales.
  const double bytesA = (double)E * (gu_w * 4.0 + gu_s * 2.0);   // packed + scales
  const double bytesB = (double)E * (d_w  * 4.0 + d_s  * 2.0);
  const double bytesT = bytesA + bytesB;
  // fp8-equivalent bytes for an apples-to-apples "effective vs fp8" number.
  const double fp8_bytesT = (double)E * (gu_elem + d_elem);
  auto gbps = [](double bytes, float ms) { return bytes / 1e6 / ms; };
  const double usAB = msAB * 1e3;

  printf("\nper-token int4 weight read: %.1f MB  (gate+up %.1f MB + down %.1f MB, incl fp16 group scales)\n",
         bytesT / 1e6, bytesA / 1e6, bytesB / 1e6);
  printf("equivalent fp8 read would be %.1f MB (this int4 path moves %.2fx fewer bytes)\n",
         fp8_bytesT / 1e6, fp8_bytesT / bytesT);
  printf("launch: block=%d  CTAs(A)=%d  CTAs(B)=%d  GROUP=%d\n", L.block, L.ctasA, L.ctasB, GROUP);
  printf("  %-22s %10s %10s %10s\n", "stage", "us/tok", "GB/s", "%HBMpeak");
  printf("  %-22s %10.2f %10.1f %9.1f%%\n", "gate+up (A)", msA  * 1e3, gbps(bytesA, msA),  100.0 * gbps(bytesA, msA)  / PEAK);
  printf("  %-22s %10.2f %10.1f %9.1f%%\n", "down    (B)", msB  * 1e3, gbps(bytesB, msB),  100.0 * gbps(bytesB, msB)  / PEAK);
  printf("  %-22s %10.2f %10.1f %9.1f%%\n", "fused   (A+B)", usAB, gbps(bytesT, msAB), 100.0 * gbps(bytesT, msAB) / PEAK);
  printf("  %-22s %10.2f %10.1f %9.1f%%  <- fp8-equivalent bytes / int4 time (target >2000)\n",
         "effective (A+B)", usAB, gbps(fp8_bytesT, msAB), 100.0 * gbps(fp8_bytesT, msAB) / PEAK);
  printf("\nvs fp8 fused %.1f us:  int4 v3 = %.1f us  -> %.2fx  %s\n",
         FP8_US, usAB, FP8_US / usAB, (usAB < FP8_US ? "INT4 WINS" : "still slower than fp8"));
  printf("MoE-expert decode over %d layers: %.2f ms/token\n", N_LAYERS, msAB * N_LAYERS);

  for (int e2 = 0; e2 < E; ++e2) {
    cudaFree(Wgu_dp[e2]); cudaFree(Wd_dp[e2]); cudaFree(Sgu_dp[e2]); cudaFree(Sd_dp[e2]);
  }
  cudaFree(Wgu_d); cudaFree(Wd_d); cudaFree(Sgu_d); cudaFree(Sd_d);
  cudaFree(sel_d); cudaFree(selw_d); cudaFree(y_d); cudaFree(h_d); cudaFree(a_d);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return 0;
}
#endif // K5I4V3_NO_MAIN
