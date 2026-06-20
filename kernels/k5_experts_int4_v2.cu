// k5_experts_int4_v2.cu — INT4 (W4A16) fused MoE-expert GEMV for Qwen3-235B-A22B, B=1 DECODE.
//
// WHY THIS FILE: the routed experts are the decode bottleneck (~14.2B of ~21.6B active params/token).
// At B=1 every projection is a GEMV (M=1) -> HBM-bandwidth-bound. Storing the expert weights as 4-bit
// instead of fp8 HALVES the dominant byte term, so the bandwidth-bound ceiling is ~2x the fp8 kernel
// (the in-repo fast fp8 GEMV k5_experts.cu measures ~1530 GB/s / ~46% peak). The first int4 attempt
// (k5_experts_int4.cu) only reached ~0.57x of fp8 (435 GB/s): it was NOT bandwidth-bound but
// INSTRUCTION-ISSUE-bound, because its unpack did 8 scalar (shift, mask, sub-8, int->float) chains per
// 32-bit word -> ~32 scalar ALU ops + 32 I2F converts per uint4 load, swamping the load.
//
// THE FIX (this file): replace the scalar nibble loop with the well-known LOP3-based int4->half2
// fast-dequant idiom (FasterTransformer / AWQ / Marlin style, all public). Each 32-bit packed word
// holds 8 nibbles; we materialize them as 4 `half2` values using a constant number of LOP3.B32 +
// half2 FMA ops — the SAME structural trick the proven fp8 kernel uses (fp8x2->half2 hardware
// convert), but built from bit ops since there is no hardware int4->half path. The contraction then
// runs entirely on the half2 datapath, so the unpack cost per load is small and CONSTANT, and the
// kernel becomes bandwidth-bound. Layout is byte-identical to k5_experts.cu: warp-per-output-row,
// split-K across the warp's 32 lanes, coalesced 128-bit (uint4 = 32 int4) loads.
//
// QUANT SCHEME: group-wise symmetric int4, GROUP=128 along the contraction (K) dimension — the common
// GPTQ/AWQ scheme. nibble n in [0,15] dequantizes to (n - 8) * scale[group]. Scales are fp16, one per
// (row, group). HIDDEN=4096 -> 32 groups/row; MOE_INTER=1536 -> 12 groups/row. Because each lane owns
// whole 32-int4 chunks and 32 | 128, a lane never straddles a group boundary mid-chunk; we fold the
// group scale onto each chunk's partial as it is produced (still one scale-multiply per group, hoisted
// out of the per-element FMA exactly like the fp8 kernel folds its per-row scale).
//
// IP: public model shapes (config.json) + standard CUDA + the public LOP3 int4->half fast-dequant
// idiom. Reuses the in-repo proven k5 warp-per-row fp8 GEMV structure (cited above). Edits no other
// file; common.cuh is read-only.
//
// BUILD + self-test (compiles cleanly on sm_90a, validates vs CPU fp32 ref, prints GB/s + %peak):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k5_experts_int4_v2.cu -o /tmp/k5i4v2 && /tmp/k5i4v2
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

namespace q3i4v2 {

constexpr int GROUP = 128;                          // int4 group size along K (GPTQ/AWQ common scheme)

// ---------------------------------------------------------------------------------------------
// Fast int4 -> half2 dequant (public LOP3 idiom).
// ---------------------------------------------------------------------------------------------
// Given a 32-bit word holding 8 packed nibbles (nibble t at bits [4t, 4t+3]), produce them as four
// half2 values, each lane = (float)nibble interpreted directly (NO -8 bias here — the symmetric
// (n-8) shift is folded into the scaled reduction below, so this stays pure bit ops + one FMA).
//
// Trick: build an fp16 bit pattern 0x6400 + nibble  ==  1024.0h + n  (because 0x6400 == 1024.0 in
// fp16 and the low 4 mantissa bits land exactly on integer steps for n in [0,15]). LOP3 merges the
// mask, the constant OR, and leaves a half2 whose value is (1024 + n); subtracting 1024 yields n.
// Two output nibbles are packed per half2 (even/odd interleave by construction of the source word).
//
// We process the 8 nibbles of a word as: low byte halves -> h2a,h2b ; high -> h2c,h2d. PRMT spreads
// the four nibble-pairs into 4 half2 lanes; one LOP3 OR-masks in the 0x6400 exponent; one half2 FMA
// (×1, −1024) removes the bias. This is the same op-count idiom used by AWQ/Marlin int4 dequant.
__device__ __forceinline__ void unpack8_int4_to_half2(unsigned w, __half2 out[4]) {
  // Source word w = [n7 n6 n5 n4 n3 n2 n1 n0]  (n0 = bits[3:0], n7 = bits[31:28]).
  // We want 4 half2: out[0]=(n0,n1), out[1]=(n2,n3), out[2]=(n4,n5), out[3]=(n6,n7).
  //
  // For each pair we build a 32-bit field whose low 16-bit half holds the even nibble and whose high
  // 16-bit half holds the odd nibble, each nibble sitting in the low 4 bits of its half. ORing the
  // fp16 exponent 0x6400 (== 1024.0h) gives two half values (1024 + n); a single half2 subtract of
  // 1024 yields the integers n in [0,15]. The build is pure shifts + LOP3-style (a&mask)|EXP, which
  // the compiler folds to LOP3.B32 on sm_90 — constant op-count, no per-element I2F. (Public AWQ /
  // Marlin int4->half fast-dequant idiom; structurally the same as k5's fp8x2->half2 hardware path.)
  const unsigned LO_MASK = 0x000F000Fu;             // keep one nibble in the low 4 bits of each half
  const unsigned EXP     = 0x64006400u;             // fp16 1024.0 in both halves
  unsigned f0 = ((w & 0x000000F0u) << 12) | (w & 0x0000000Fu);        // high half=n1, low half=n0
  unsigned f1 = ((w & 0x0000F000u) <<  4) | ((w & 0x00000F00u) >>  8);  // high half=n3, low half=n2
  unsigned f2 = ((w & 0x00F00000u) >>  4) | ((w & 0x000F0000u) >> 16);  // high half=n5, low half=n4
  unsigned f3 = ((w & 0xF0000000u) >> 12) | ((w & 0x0F000000u) >> 24);  // high half=n7, low half=n6
  unsigned t0 = (f0 & LO_MASK) | EXP;
  unsigned t1 = (f1 & LO_MASK) | EXP;
  unsigned t2 = (f2 & LO_MASK) | EXP;
  unsigned t3 = (f3 & LO_MASK) | EXP;
  const __half2 bias = __float2half2_rn(1024.0f);
  __half2 h0, h1, h2, h3;
  memcpy(&h0, &t0, 4); memcpy(&h1, &t1, 4); memcpy(&h2, &t2, 4); memcpy(&h3, &t3, 4);
  out[0] = __hsub2(h0, bias);
  out[1] = __hsub2(h1, bias);
  out[2] = __hsub2(h2, bias);
  out[3] = __hsub2(h3, bias);
}

// ---------------------------------------------------------------------------------------------
// Device dot: warp dots a contiguous int4 row (n weights, 8/uint32, 32/uint4) against staged f32 ys.
// ---------------------------------------------------------------------------------------------
// Split-K across 32 lanes: lane v reads uint4 chunk v (coalesced 128-bit). Returns
//   sum_k ys[k] * (nibble_k - 8) * scale[group(k)]
// with the per-group symmetric scale folded as each 32-element chunk is reduced. n must be a multiple
// of 32 (HIDDEN=4096, MOE_INTER=1536 both are) and GROUP must be a multiple of 32 (128 is), so a chunk
// never crosses a group boundary. `scale_row` points at this row's per-group fp16 scales (n/GROUP of
// them). Result valid on lane 0.
__device__ __forceinline__ float warp_dot_int4_v2(const unsigned* __restrict__ wq,
                                                  const float* __restrict__ ys,
                                                  const __half* __restrict__ scale_row,
                                                  int n, int lane) {
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(wq);  // 4 words = 32 int4 / load
  const int nv = n >> 5;                                             // n/32 128-bit loads
  // (GROUP=128 is a multiple of the 32-element chunk, so a chunk never straddles a group boundary.)
  float acc = 0.f;
  for (int v = lane; v < nv; v += 32) {                              // lanes -> consecutive uint4
    uint4 p = wv[v];
    const unsigned* w4 = reinterpret_cast<const unsigned*>(&p);
    const float* yy = ys + (v << 5);
    // dot of this 32-element chunk (bias -8 NOT yet applied per element; we apply it on the chunk sum)
    float c0 = 0.f, c1 = 0.f;                                        // 2 accumulators -> ILP
    float chunk_y = 0.f;                                             // sum of ys over the chunk (for -8 bias)
    #pragma unroll
    for (int q = 0; q < 4; ++q) {                                    // 4 words = 32 nibbles
      __half2 h[4];
      unpack8_int4_to_half2(w4[q], h);                               // n in [0,15] as half2 (no -8 yet)
      const float* yq = yy + (q << 3);                              // 8 ys for this word
      float2 f0 = __half22float2(h[0]);  // (n0, n1)
      float2 f1 = __half22float2(h[1]);  // (n2, n3)
      float2 f2 = __half22float2(h[2]);  // (n4, n5)
      float2 f3 = __half22float2(h[3]);  // (n6, n7)
      c0 += yq[0]*f0.x; c1 += yq[1]*f0.y;
      c0 += yq[2]*f1.x; c1 += yq[3]*f1.y;
      c0 += yq[4]*f2.x; c1 += yq[5]*f2.y;
      c0 += yq[6]*f3.x; c1 += yq[7]*f3.y;
      chunk_y += yq[0]+yq[1]+yq[2]+yq[3]+yq[4]+yq[5]+yq[6]+yq[7];
    }
    // (n-8)*scale = (sum ys*n - 8*sum ys) * scale_group.  v->group: v*32 / GROUP.
    float sc = __half2float(scale_row[(v << 5) / GROUP]);
    acc += ((c0 + c1) - 8.0f * chunk_y) * sc;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                                        // valid on lane 0
}

} // namespace q3i4v2

using namespace q3i4v2;

// ---------------------------------------------------------------------------------------------
// Kernel A — fused gate+up:  a_glb[slot][j] = silu(<y,gate_j>_int4) * (<y,up_j>_int4)
// ---------------------------------------------------------------------------------------------
// The per-group int4 scales already carry the dequant magnitude, so no extra per-row fp32 scale is
// needed (unlike fp8's per-channel scale). Wgu[e] is packed [2*MOE_INTER, HIDDEN/8] uint32; the scale
// array Sgu[e] is [2*MOE_INTER * (HIDDEN/GROUP)] fp16. Warp-per-row, grid-stride over (slot, j).
extern "C" __global__ void k5a_gateup_int4_v2(
    const float* __restrict__ y, const int* __restrict__ sel_idx,
    const unsigned* const* __restrict__ Wgu, const __half* const* __restrict__ Wgu_scale,
    float* __restrict__ a_glb, int nslot) {
  extern __shared__ float ys[];                                      // [HIDDEN]
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) ys[k] = y[k];
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
    const float g = warp_dot_int4_v2(W + (size_t)j * words_per_row, ys,
                                     S + (size_t)j * groups_per_row, HIDDEN, lane);
    const float u = warp_dot_int4_v2(W + (size_t)(MOE_INTER + j) * words_per_row, ys,
                                     S + (size_t)(MOE_INTER + j) * groups_per_row, HIDDEN, lane);
    if (lane == 0) a_glb[(size_t)slot * MOE_INTER + j] = silu(g) * u;
  }
}

// ---------------------------------------------------------------------------------------------
// Kernel B — down projection + routed accumulate:  h_io[o] += sel_w * <a[slot], down_o>_int4
// ---------------------------------------------------------------------------------------------
// Wd[e] is packed [HIDDEN, MOE_INTER/8] uint32; Sd[e] is [HIDDEN * (MOE_INTER/GROUP)] fp16. The full a
// buffer is staged in shared memory once per CTA; routing weight folded into the atomic epilogue.
extern "C" __global__ void k5b_down_int4_v2(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const unsigned* const* __restrict__ Wd, const __half* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot) {
  extern __shared__ float as[];                                      // [nslot*MOE_INTER]
  const int na = nslot * MOE_INTER;
  for (int i = threadIdx.x; i < na; i += blockDim.x) as[i] = a_glb[i];
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
    const float d = warp_dot_int4_v2(W + (size_t)o * words_per_row, as + (size_t)slot * MOE_INTER,
                                     S + (size_t)o * groups_per_row, MOE_INTER, lane);
    if (lane == 0) atomicAdd(&h_io[o], gw * d);
  }
}

// ---------------------------------------------------------------------------------------------
// Launch-config helper (mirrors k5_experts.cu): fill the H100's 132 SMs with resident warps.
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
  L.smemA = (size_t)HIDDEN * sizeof(float);
  L.smemB = (size_t)nslot * MOE_INTER * sizeof(float);
  return L;
}

// =============================================================================================
// Host-side: CPU fp32 reference + deterministic input generation + cudaEvents microbench.
// =============================================================================================

// ---- CPU fp32 reference (mirrors the kernels after the int4 quant round-trip) ----------------
// For each active expert:
//   g_j = sum_k y_k * (nib(Wgu[gate_j,k]) - 8) * Sgu[gate_j, group(k)]
//   u_j = sum_k y_k * (nib(Wgu[up_j,k])   - 8) * Sgu[up_j,   group(k)]
//   a_j = silu(g_j) * u_j
//   h_o += sel_w * sum_j a_j * (nib(Wd[o,j]) - 8) * Sd[o, group(j)]
// nib(W[r,k]) reads nibble k of packed row r.
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

#ifndef K5I4V2_NO_MAIN

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

  int ndev = 0, dev = 0; cudaDeviceProp prop;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("No CUDA device found.\n"); return 1; }
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s  SMs=%d  assumed HBM peak=%.0f GB/s\n", prop.name, prop.multiProcessorCount, PEAK);

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
  CK(cudaFuncSetAttribute(k5a_gateup_int4_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemA));
  CK(cudaFuncSetAttribute(k5b_down_int4_v2,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)L.smemB));

  CK(cudaMemset(h_d, 0, HIDDEN * sizeof(float)));
  k5a_gateup_int4_v2<<<L.ctasA, L.block, L.smemA>>>(y_d, sel_d, Wgu_d, Sgu_d, a_d, E);
  k5b_down_int4_v2  <<<L.ctasB, L.block, L.smemB>>>(sel_d, selw_d, Wd_d, Sd_d, a_d, h_d, E);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(got.data(), h_d, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));

  double max_abs = 0.0, max_rel = 0.0;
  for (int i = 0; i < HIDDEN; ++i) {
    double ad = fabs((double)ref[i] - (double)got[i]);
    max_abs = std::max(max_abs, ad);
    max_rel = std::max(max_rel, ad / (fabs((double)ref[i]) + 1e-6));
  }
  printf("correctness vs CPU fp32 reference:  max_abs=%.3e  max_rel=%.3e  -> %s (<1e-2)\n",
         max_abs, max_rel, (max_abs < 1e-2 ? "PASS" : "FAIL"));

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
  auto runA = [&]() { k5a_gateup_int4_v2<<<L.ctasA, L.block, L.smemA>>>(y_d, sel_d, Wgu_d, Sgu_d, a_d, E); };
  auto runB = [&]() { k5b_down_int4_v2  <<<L.ctasB, L.block, L.smemB>>>(sel_d, selw_d, Wd_d, Sd_d, a_d, h_d, E); };
  auto runAB = [&]() { runA(); runB(); };
  float msA  = bench(runA);
  float msB  = bench(runB);
  float msAB = bench(runAB);
  CK(cudaGetLastError());

  // Bytes read from HBM per token: the int4 packed weights (0.5 byte/elem) + the per-group fp16 scales.
  const double bytesA = (double)E * (gu_w * 4.0 + gu_s * 2.0);   // packed + scales
  const double bytesB = (double)E * (d_w  * 4.0 + d_s  * 2.0);
  const double bytesT = bytesA + bytesB;
  // For an apples-to-apples "effective vs fp8" number: fp8 reads (gu_elem+d_elem) bytes/expert.
  const double fp8_bytesT = (double)E * (gu_elem + d_elem);
  auto gbps = [](double bytes, float ms) { return bytes / 1e6 / ms; };

  printf("\nper-token int4 weight read: %.1f MB  (gate+up %.1f MB + down %.1f MB, incl fp16 group scales)\n",
         bytesT / 1e6, bytesA / 1e6, bytesB / 1e6);
  printf("equivalent fp8 read would be %.1f MB (this int4 path moves %.2fx fewer bytes)\n",
         fp8_bytesT / 1e6, fp8_bytesT / bytesT);
  printf("launch: block=%d  CTAs(A)=%d  CTAs(B)=%d  GROUP=%d\n", L.block, L.ctasA, L.ctasB, GROUP);
  printf("  %-22s %10s %10s %10s\n", "stage", "us/tok", "GB/s", "%HBMpeak");
  printf("  %-22s %10.2f %10.1f %9.1f%%\n", "gate+up (A)", msA  * 1e3, gbps(bytesA, msA),  100.0 * gbps(bytesA, msA)  / PEAK);
  printf("  %-22s %10.2f %10.1f %9.1f%%\n", "down    (B)", msB  * 1e3, gbps(bytesB, msB),  100.0 * gbps(bytesB, msB)  / PEAK);
  printf("  %-22s %10.2f %10.1f %9.1f%%\n", "fused   (A+B)", msAB * 1e3, gbps(bytesT, msAB), 100.0 * gbps(bytesT, msAB) / PEAK);
  // "effective" GB/s = fp8-equivalent bytes / int4 time -> the >2x-fp8 target metric.
  printf("  %-22s %10.2f %10.1f %9.1f%%  <- fp8-equivalent bytes / int4 time (target >2000)\n",
         "effective (A+B)", msAB * 1e3, gbps(fp8_bytesT, msAB), 100.0 * gbps(fp8_bytesT, msAB) / PEAK);
  printf("\nMoE-expert decode over %d layers: %.2f ms/token\n", N_LAYERS, msAB * N_LAYERS);

  for (int e2 = 0; e2 < E; ++e2) {
    cudaFree(Wgu_dp[e2]); cudaFree(Wd_dp[e2]); cudaFree(Sgu_dp[e2]); cudaFree(Sd_dp[e2]);
  }
  cudaFree(Wgu_d); cudaFree(Wd_d); cudaFree(Sgu_d); cudaFree(Sd_d);
  cudaFree(sel_d); cudaFree(selw_d); cudaFree(y_d); cudaFree(h_d); cudaFree(a_d);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return 0;
}
#endif // K5I4V2_NO_MAIN
