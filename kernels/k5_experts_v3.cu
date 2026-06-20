// k5_experts_v3.cu — cp.async double-buffered fused fp8 MoE-expert GEMV for Qwen3-235B-A22B, B=1 DECODE.
//
// WHY THIS FILE: the routed experts are the single-token decode bottleneck (~14.2B of ~21.6B active
// params/token). At batch size 1 each projection is a GEMV (M=1), so the kernel is strictly
// HBM-bandwidth-bound, NOT compute-bound. The in-repo fast warp-per-output-row fp8 GEMV
// (k5_experts.cu) measures ~1530 GB/s = ~45.7% of the H100's 3.35 TB/s HBM3 peak. The limiter at
// M=1 is NOT ALU and NOT the load width — it is MEMORY-LEVEL PARALLELISM: too few weight loads are
// in flight at once to cover HBM latency, so the pipe sits ~half idle. ~54% MBU is left on the table.
//
// WHAT V3 ADDS (the levers in the task, all standard CUDA / public idioms):
//   (1) cp.async DOUBLE/TRIPLE-BUFFERED staging of the fp8 weight tiles global->shared. Each warp
//       issues `cp.async.cg.shared.global` for the NEXT contraction tile of every row it owns, then
//       commits the group and computes the dequant+FMA of the PREVIOUS tile out of shared memory.
//       This is a software pipeline (STAGES=2..3): the async-copy engine keeps many weight loads in
//       flight per warp while the math runs, which is exactly the in-flight-parallelism the M=1
//       GEMV was starving for. cp.async issues 16-byte (uint4 = 16 fp8) coalesced transactions — the
//       same 128-bit access the proven k5 kernel used, just decoupled from the consuming math.
//   (2) MULTIPLE OUTPUT ROWS PER WARP (ROWS_PER_WARP = R). One warp computes R adjacent output rows.
//       This (a) multiplies the independent in-flight loads per warp by R (more MLP for free),
//       (b) amortizes the staged-activation reads and the warp-shuffle reduce over R rows, and
//       (c) raises arithmetic-per-sync. R staged tiles per stage share the same `y` slice.
//   (3) WIDER ILP IN THE DEQUANT. Per uint4 (16 fp8) we keep 4 fp32 accumulators per row and unroll
//       the 4x(fp8x2->half2) hardware convert, so the FMA chain never serializes on a single accum.
//   (4) OCCUPANCY KNOBS. Block size, STAGES and the per-stage tile width are swept on the host so the
//       resident-warp count saturates HBM without spilling shared memory (the >48KB opt-in is set).
//
// Kept identical to k5_experts.cu so this is a drop-in faster path:
//   * Layout: K-major fp8 rows, warp split-K across 32 lanes (lane v owns uint4 chunk v of the tile).
//   * Fusion: Kernel A produces a[slot][j] = silu(s_g*<y,gate_j>) * (s_u*<y,up_j>) in one pass;
//             Kernel B does h[o] += sel_w * s_d * <a[slot], down_o> straight into the residual.
//   * Per-output-channel fp8 scale folded once onto the reduced dot.
//   * Hardware fp8x2->half2 dequant (8 vector converts/uint4, not 16 scalar casts).
//
// IP: public model shapes (config.json, via common.cuh) + standard CUDA + the public cp.async
// software-pipeline idiom (CUDA C++ Programming Guide). Reuses the in-repo k5 warp-per-row fp8 GEMV
// structure. Edits no other file; common.cuh is read-only.
//
// BUILD + self-test (compiles cleanly sm_90a, validates vs CPU fp32 ref <1e-2, sweeps knobs, GB/s):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k5_experts_v3.cu -o /tmp/k5v3 && /tmp/k5v3
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

namespace q3v3 {

// ---------------------------------------------------------------------------------------------
// cp.async primitives (inline PTX, sm_80+). 16-byte cache-global async copy + commit/wait group.
// ---------------------------------------------------------------------------------------------
// cp.async.cg copies 16 bytes global->shared bypassing L1 (.cg = cache-global; right for
// stream-once weights). We pass the shared destination as a generic pointer converted with
// __cvta_generic_to_shared so the .shared state-space form is used. Each call stages one uint4.
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_src));
}
// Mark the end of a copy group, then wait until all but the N most-recent groups have landed.
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ---------------------------------------------------------------------------------------------
// Dequant+FMA of ONE staged uint4 (16 fp8) chunk into a row's 4 accumulators (the v3 ILP core).
// ---------------------------------------------------------------------------------------------
// `p` is a uint4 of 16 fp8 weights already resident in shared memory; `yy` points at the matching 16
// staged activations. Uses the hardware fp8x2->half2 convert (4 per uint4) and 4 fp32 accumulators so
// the FMA chain has independent destinations -> full ILP. Identical numerics to k5_experts.cu.
__device__ __forceinline__ void fma_uint4_fp8(const uint4& p, const float* __restrict__ yy,
                                              float& a0, float& a1, float& a2, float& a3) {
  const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
  #pragma unroll
  for (int q = 0; q < 4; ++q) {                         // 4 x 32-bit words = 4 x (2 fp8 pairs)
    unsigned wq = wu[q];
    __nv_fp8x2_e4m3 lo, hi;
    lo.__x = (unsigned short)(wq & 0xffffu);
    hi.__x = (unsigned short)(wq >> 16);
    float2 fl = __half22float2((__half2)lo);
    float2 fh = __half22float2((__half2)hi);
    const float* yq = yy + (q << 2);
    a0 += yq[0] * fl.x;  a1 += yq[1] * fl.y;
    a2 += yq[2] * fh.x;  a3 += yq[3] * fh.y;
  }
}
__device__ __forceinline__ float warp_reduce(float acc) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                           // valid on lane 0
}

// ---------------------------------------------------------------------------------------------
// Core: warp computes ROWS dot-products  <w_row, ys>  for ROWS contiguous fp8 rows, cp.async pipelined.
// ---------------------------------------------------------------------------------------------
// W0 points at the first of ROWS K-major fp8 rows (each `n` long, row stride `n`); ys is the staged
// activation (`n` floats in shared). Split-K across the warp's 32 lanes: lane v consumes uint4 chunk v
// of each tile. We stream the contraction in tiles of (32 uint4 = 512 fp8) so all 32 lanes are busy
// per tile; STAGES buffers of the per-warp tile are kept in shared and filled ahead by cp.async.
//
// Shared layout for this warp: `wbuf` is [STAGES][ROWS][32] uint4 (one uint4 per lane per row per
// stage). `out[r]` receives the lane-0 reduced dot for row r. n must be a multiple of 16 (HIDDEN=4096,
// MOE_INTER=1536 both are). nv = n/16 uint4 per row; we process them 32 at a time (TILE_V = 32).
template <int ROWS, int STAGES>
__device__ __forceinline__ void warp_dot_rows_pipe(const fp8* __restrict__ W0, int n, int lane,
                                                   const float* __restrict__ ys,
                                                   uint4* __restrict__ wbuf, float out[ROWS]) {
  constexpr int TILE_V = 32;                            // uint4 per lane-sweep == warpSize
  const int nv = n >> 4;                                // uint4 per row
  const int ntile = (nv + TILE_V - 1) / TILE_V;         // contraction tiles
  // per-row weight base as uint4
  const uint4* __restrict__ Wv[ROWS];
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) Wv[r] = reinterpret_cast<const uint4*>(W0 + (size_t)r * n);

  // wbuf[stage][row][lane]; index helper
  auto slot = [&](int st, int r) -> uint4* { return wbuf + ((size_t)st * ROWS + r) * TILE_V; };

  // ---- prologue: kick off the first min(STAGES, ntile) tiles ----
  int fetch = 0;
  #pragma unroll 1
  for (; fetch < STAGES && fetch < ntile; ++fetch) {
    const int base = fetch * TILE_V;
    const int v    = base + lane;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      if (v < nv) cp_async_16(slot(fetch, r) + lane, Wv[r] + v);
    }
    cp_async_commit();
  }

  float acc[ROWS][4];
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) { acc[r][0]=acc[r][1]=acc[r][2]=acc[r][3]=0.f; }

  // ---- steady state: consume tile `t`, prefetch tile `t+STAGES` ----
  #pragma unroll 1
  for (int t = 0; t < ntile; ++t) {
    cp_async_wait<STAGES - 1>();                        // ensure tile t has landed
    __syncwarp();
    const int st   = t % STAGES;
    const int base = t * TILE_V;
    const int v    = base + lane;
    if (v < nv) {
      const float* yy = ys + (v << 4);
      #pragma unroll
      for (int r = 0; r < ROWS; ++r)
        fma_uint4_fp8(*(slot(st, r) + lane), yy, acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    }
    // prefetch the tile STAGES ahead into the slot we just drained
    const int nf = t + STAGES;
    if (nf < ntile) {
      const int fbase = nf * TILE_V;
      const int fv    = fbase + lane;
      __syncwarp();                                     // slot reuse: all lanes done reading st
      #pragma unroll
      for (int r = 0; r < ROWS; ++r)
        if (fv < nv) cp_async_16(slot(st, r) + lane, Wv[r] + fv);
    }
    cp_async_commit();
  }

  // Drain any cp.async groups still pending so this call is self-contained (the wbuf ring can be
  // reused immediately by a following call in the same warp, e.g. the gate-then-up pass in kernel A).
  cp_async_wait<0>();
  __syncwarp();

  #pragma unroll
  for (int r = 0; r < ROWS; ++r)
    out[r] = warp_reduce((acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]));
}

} // namespace q3v3
using namespace q3v3;

// =============================================================================================
// Kernel A — fused gate+up, cp.async pipelined, ROWS_PER_WARP rows/warp.
// =============================================================================================
// a_glb[slot][j] = silu(s_g * <y,gate_j>) * (s_u * <y,up_j>). gate is rows [0,MOE_INTER), up is
// [MOE_INTER,2*MOE_INTER) of the stacked Wgu[e] [2*MOE_INTER, HIDDEN]. We give each warp R adjacent j
// values; for each it dots BOTH its gate row and its up row (2R pipelined rows/warp). y is staged once
// per CTA into the front of dynamic smem; the per-warp cp.async ring buffers live after it.
//
// Dynamic smem = HIDDEN*4 (ys) + blockWarps * STAGES * R * 32 * sizeof(uint4).
template <int R, int STAGES>
__global__ void k5a_gateup_v3(
    const float* __restrict__ y, const int* __restrict__ sel_idx,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    float* __restrict__ a_glb, int nslot) {
  extern __shared__ char smem[];
  float* ys = reinterpret_cast<float*>(smem);                          // [HIDDEN]
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) ys[k] = y[k];
  __syncthreads();

  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // Per-warp ring buffer (16-byte aligned: ys is HIDDEN floats, a multiple of 4). The gate pass and
  // the up pass run sequentially and each needs only R rows of buffer, so the ring is STAGES*R*32
  // uint4 (the gate pass drains itself before the up pass reuses the same slots).
  uint4* wbuf_all = reinterpret_cast<uint4*>(ys + HIDDEN);
  uint4* wbuf = wbuf_all + (size_t)warp_in_block * STAGES * R * 32;

  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int njrow = (MOE_INTER + R - 1) / R;                           // row-groups of R along j
  const int total = nslot * njrow;

  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / njrow;
    const int jg   = item - slot * njrow;
    const int j0   = jg * R;
    const int e    = sel_idx[slot];
    const fp8*   W = Wgu[e];
    const float* S = Wgu_scale[e];

    // Build the 2R contiguous rows: [gate_{j0..}, up_{j0..}]. We dot gate rows then up rows as one
    // 2R-row pipelined pass; both halves share ys. Tail rows (j0+r >= MOE_INTER) read row 0 harmlessly
    // and are masked out at store time.
    const fp8* rows = W + (size_t)j0 * HIDDEN;        // gate rows start; up rows are MOE_INTER*HIDDEN later
    // Two separate pipelined groups (gate block, up block) reuse the same wbuf in sequence to keep the
    // ring small; each is an R-row pass.
    float g[R], u[R];
    warp_dot_rows_pipe<R, STAGES>(rows, HIDDEN, lane, ys, wbuf, g);
    warp_dot_rows_pipe<R, STAGES>(W + (size_t)(MOE_INTER + j0) * HIDDEN, HIDDEN, lane, ys, wbuf, u);

    if (lane == 0) {
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        const int j = j0 + r;
        if (j < MOE_INTER)
          a_glb[(size_t)slot * MOE_INTER + j] = silu(g[r] * S[j]) * (u[r] * S[MOE_INTER + j]);
      }
    }
  }
}

// =============================================================================================
// Kernel B — down projection + routed accumulate, cp.async pipelined, ROWS_PER_WARP rows/warp.
// =============================================================================================
// h_io[o] += sel_w * s_d * <a[slot], down_o>. The full a buffer (nslot*MOE_INTER floats) is staged
// into the front of dynamic smem once per CTA; the per-warp cp.async ring buffers follow it. Warp owns
// R adjacent output channels o; routing weight + per-channel down scale folded into the atomic.
//
// Dynamic smem = nslot*MOE_INTER*4 (as) + blockWarps * STAGES * R * 32 * sizeof(uint4).
template <int R, int STAGES>
__global__ void k5b_down_v3(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot) {
  extern __shared__ char smem[];
  float* as = reinterpret_cast<float*>(smem);                          // [nslot*MOE_INTER]
  const int na = nslot * MOE_INTER;
  for (int i = threadIdx.x; i < na; i += blockDim.x) as[i] = a_glb[i];
  __syncthreads();

  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // align the uint4 ring buffer to 16 bytes after `as` (na is nslot*1536, multiple of 4)
  uint4* wbuf_all = reinterpret_cast<uint4*>(as + na);
  uint4* wbuf = wbuf_all + (size_t)warp_in_block * STAGES * R * 32;

  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const int norow = (HIDDEN + R - 1) / R;                              // row-groups of R along o
  const int total = nslot * norow;

  for (int item = gwarp; item < total; item += nwarp) {
    const int slot = item / norow;
    const int og   = item - slot * norow;
    const int o0   = og * R;
    const int e    = sel_idx[slot];
    const float gw = sel_w[slot];
    const fp8*   W = Wd[e];
    const float* S = Wd_scale[e];
    const float* asl = as + (size_t)slot * MOE_INTER;

    float d[R];
    warp_dot_rows_pipe<R, STAGES>(W + (size_t)o0 * MOE_INTER, MOE_INTER, lane, asl, wbuf, d);

    if (lane == 0) {
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        const int o = o0 + r;
        if (o < HIDDEN) atomicAdd(&h_io[o], gw * d[r] * S[o]);
      }
    }
  }
}

// =============================================================================================
// Launch planning + dispatch over the swept knobs (R, STAGES, block).
// =============================================================================================
struct V3Cfg { int R, STAGES, block; };

// Per-warp ring-buffer bytes. Kernel A runs the gate pass then the up pass sequentially over the
// SAME ring (each an R-row pipelined pass), so both A and B need only STAGES * R * 32 uint4 per warp.
static inline size_t ringA(const V3Cfg& c) { return (size_t)c.STAGES * (c.R) * 32 * sizeof(uint4); }
static inline size_t ringB(const V3Cfg& c) { return (size_t)c.STAGES * (c.R) * 32 * sizeof(uint4); }
static inline size_t smemA(const V3Cfg& c) {
  const int warps = c.block >> 5;
  return (size_t)HIDDEN * sizeof(float) + (size_t)warps * ringA(c);
}
static inline size_t smemB(const V3Cfg& c, int nslot) {
  const int warps = c.block >> 5;
  return (size_t)nslot * MOE_INTER * sizeof(float) + (size_t)warps * ringB(c);
}

// CTA count: enough warps to cover the work with a few resident waves over the 132 SMs.
static inline int ctas_for(int rows, int R, int block) {
  const int warps_per_cta = block >> 5;
  const int rowgroups = (rows + R - 1) / R;
  int need = (rowgroups + warps_per_cta - 1) / warps_per_cta;
  return std::min(std::max(need, 132), 264);
}

// Typed launchers (template args must be compile-time, so we switch over the swept (R,STAGES)).
typedef void (*FnA)(const float*, const int*, const fp8* const*, const float* const*, float*, int);
typedef void (*FnB)(const int*, const float*, const fp8* const*, const float* const*, const float*, float*, int);

template <int R, int STAGES> static FnA getA() { return &k5a_gateup_v3<R, STAGES>; }
template <int R, int STAGES> static FnB getB() { return &k5b_down_v3<R, STAGES>; }

// =============================================================================================
// CPU fp32 reference (identical numerics to k5_experts.cu after the fp8 round-trip).
// =============================================================================================
void k5v3_reference(const float* y, const int* sel_idx, const float* sel_w,
                    const fp8* const* Wgu, const float* const* Wgu_scale,
                    const fp8* const* Wd,  const float* const* Wd_scale,
                    float* h_io, int nslot) {
  std::vector<float> a(MOE_INTER);
  for (int slot = 0; slot < nslot; ++slot) {
    const int e = sel_idx[slot];
    const fp8*   W  = Wgu[e];
    const float* Sg = Wgu_scale[e];
    for (int j = 0; j < MOE_INTER; ++j) {
      const fp8* grow = W + (size_t)j * HIDDEN;
      const fp8* urow = W + (size_t)(MOE_INTER + j) * HIDDEN;
      double g = 0.0, u = 0.0;
      for (int k = 0; k < HIDDEN; ++k) {
        g += (double)y[k] * (double)(float)grow[k];
        u += (double)y[k] * (double)(float)urow[k];
      }
      float gs = (float)g * Sg[j];
      float us = (float)u * Sg[MOE_INTER + j];
      a[j] = (gs / (1.0f + expf(-gs))) * us;
    }
    const fp8*   Wdn = Wd[e];
    const float* Sd  = Wd_scale[e];
    const float  gw  = sel_w[slot];
    for (int o = 0; o < HIDDEN; ++o) {
      const fp8* drow = Wdn + (size_t)o * MOE_INTER;
      double acc = 0.0;
      for (int j = 0; j < MOE_INTER; ++j) acc += (double)a[j] * (double)(float)drow[j];
      h_io[o] += gw * (float)acc * Sd[o];
    }
  }
}

#ifndef K5V3_NO_MAIN

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                     \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));         \
  exit(1); } } while (0)

static inline unsigned hash_u(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static inline float rnd(unsigned seed, size_t i, float scale, bool positive) {
  unsigned h = hash_u((unsigned)(i * 2654435761u) ^ (seed * 40503u));
  float v = (((h % 2001) / 1000.0f) - 1.0f) * scale;
  return positive ? (fabsf(v) + 1e-3f) : v;
}

// Dispatch one (R,STAGES) config's A and B launches. Returns false if the config is unrunnable
// (e.g. shared-memory request exceeds the device max) so the sweep can skip it gracefully.
struct Bufs {
  const float* y_d; const int* sel_d; const float* selw_d;
  const fp8* const* Wgu_d; const float* const* Sgu_d;
  const fp8* const* Wd_d;  const float* const* Sd_d;
  float* a_d; float* h_d; int E;
};

// Opt in to >48KB dynamic smem for this config's A/B kernels. Call ONCE per config before timing;
// keeping it out of the hot launch path avoids two redundant host-side setAttribute calls per iter
// that would otherwise serialize host issue and pessimize the swept us/tok numbers.
static bool set_cfg_smem(const V3Cfg& c, const Bufs& b, int maxSmem, FnA fa, FnB fb) {
  size_t sa = smemA(c), sb = smemB(c, b.E);
  if ((int)sa > maxSmem || (int)sb > maxSmem) return false;
  cudaFuncSetAttribute((const void*)fa, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sa);
  cudaFuncSetAttribute((const void*)fb, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sb);
  return true;
}

// Hot path: just the two launches (smem already opted in via set_cfg_smem).
static void launch_cfg(const V3Cfg& c, const Bufs& b, cudaStream_t s, FnA fa, FnB fb) {
  size_t sa = smemA(c), sb = smemB(c, b.E);
  int ctasA = ctas_for(b.E * MOE_INTER, c.R, c.block);
  int ctasB = ctas_for(b.E * HIDDEN,    c.R, c.block);
  fa<<<ctasA, c.block, sa, s>>>(b.y_d, b.sel_d, b.Wgu_d, b.Sgu_d, b.a_d, b.E);
  fb<<<ctasB, c.block, sb, s>>>(b.sel_d, b.selw_d, b.Wd_d, b.Sd_d, b.a_d, b.h_d, b.E);
}

int main(int argc, char** argv) {
  const int E = 8;                                        // TOP_K active experts
  const double PEAK = (argc > 1) ? atof(argv[1]) : 3350.0; // GB/s; H100 HBM3 = 3.35 TB/s

  int ndev = 0, dev = 0; cudaDeviceProp prop;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("No CUDA device found.\n"); return 1; }
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  int maxSmem = prop.sharedMemPerBlockOptin;
  printf("device: %s  SMs=%d  smemOptin=%d KB  assumed HBM peak=%.0f GB/s\n",
         prop.name, prop.multiProcessorCount, maxSmem >> 10, PEAK);

  const size_t gu_n = (size_t)2 * MOE_INTER * HIDDEN;     // 3072*4096 fp8 per expert
  const size_t d_n  = (size_t)HIDDEN * MOE_INTER;         // 4096*1536 fp8 per expert

  // ---- build inputs on host (so the CPU reference uses the exact uploaded fp8 bytes) ----------
  std::vector<std::vector<fp8>>   Wgu_host(E), Wd_host(E);
  std::vector<std::vector<float>> Sgu_host(E), Sd_host(E);
  for (int e = 0; e < E; ++e) {
    Wgu_host[e].resize(gu_n);  Wd_host[e].resize(d_n);
    Sgu_host[e].resize(2 * MOE_INTER);  Sd_host[e].resize(HIDDEN);
    for (size_t i = 0; i < gu_n; ++i) Wgu_host[e][i] = (fp8)rnd(1u + e, i, 0.25f, false);
    for (size_t i = 0; i < d_n;  ++i) Wd_host[e][i]  = (fp8)rnd(100u + e, i, 0.25f, false);
    for (int i = 0; i < 2 * MOE_INTER; ++i) Sgu_host[e][i] = rnd(7u + e, i, 0.02f, true);
    for (int i = 0; i < HIDDEN; ++i)        Sd_host[e][i]  = rnd(13u + e, i, 0.02f, true);
  }
  std::vector<float> y_host(HIDDEN);
  for (int k = 0; k < HIDDEN; ++k) y_host[k] = rnd(99u, k, 1.0f, false);
  std::vector<int>   sel_host(E);
  std::vector<float> selw_host(E);
  for (int e = 0; e < E; ++e) { sel_host[e] = e; selw_host[e] = 0.1f + 0.01f * e; }

  // ---- upload ---------------------------------------------------------------------------------
  std::vector<fp8*>   Wgu_dp(E), Wd_dp(E);
  std::vector<float*> Sgu_dp(E), Sd_dp(E);
  for (int e = 0; e < E; ++e) {
    CK(cudaMalloc(&Wgu_dp[e], gu_n * sizeof(fp8)));
    CK(cudaMalloc(&Wd_dp[e],  d_n  * sizeof(fp8)));
    CK(cudaMalloc(&Sgu_dp[e], 2 * MOE_INTER * sizeof(float)));
    CK(cudaMalloc(&Sd_dp[e],  HIDDEN * sizeof(float)));
    CK(cudaMemcpy(Wgu_dp[e], Wgu_host[e].data(), gu_n * sizeof(fp8), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Wd_dp[e],  Wd_host[e].data(),  d_n  * sizeof(fp8), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sgu_dp[e], Sgu_host[e].data(), 2 * MOE_INTER * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd_dp[e],  Sd_host[e].data(),  HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
  }
  const fp8 **Wgu_d, **Wd_d; const float **Sgu_d, **Sd_d;
  CK(cudaMalloc(&Wgu_d, E * sizeof(fp8*)));  CK(cudaMemcpy(Wgu_d, Wgu_dp.data(), E * sizeof(fp8*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd_d,  E * sizeof(fp8*)));  CK(cudaMemcpy(Wd_d,  Wd_dp.data(),  E * sizeof(fp8*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sgu_d, E * sizeof(float*))); CK(cudaMemcpy(Sgu_d, Sgu_dp.data(), E * sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd_d,  E * sizeof(float*))); CK(cudaMemcpy(Sd_d,  Sd_dp.data(),  E * sizeof(float*), cudaMemcpyHostToDevice));

  int   *sel_d; float *selw_d, *y_d, *h_d, *a_d;
  CK(cudaMalloc(&sel_d,  E * sizeof(int)));    CK(cudaMemcpy(sel_d,  sel_host.data(),  E * sizeof(int),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&selw_d, E * sizeof(float)));  CK(cudaMemcpy(selw_d, selw_host.data(), E * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&y_d, HIDDEN * sizeof(float))); CK(cudaMemcpy(y_d, y_host.data(), HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&h_d, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&a_d, (size_t)E * MOE_INTER * sizeof(float)));
  CK(cudaDeviceSynchronize());

  Bufs b{ y_d, sel_d, selw_d, Wgu_d, Sgu_d, Wd_d, Sd_d, a_d, h_d, E };

  // ---- CPU reference (residual starts at 0) ---------------------------------------------------
  std::vector<float> ref(HIDDEN, 0.0f);
  std::vector<const fp8*> Wgu_hp(E), Wd_hp(E); std::vector<const float*> Sgu_hp(E), Sd_hp(E);
  for (int e = 0; e < E; ++e) { Wgu_hp[e] = Wgu_host[e].data(); Wd_hp[e] = Wd_host[e].data();
                                Sgu_hp[e] = Sgu_host[e].data(); Sd_hp[e] = Sd_host[e].data(); }
  k5v3_reference(y_host.data(), sel_host.data(), selw_host.data(),
                 Wgu_hp.data(), Sgu_hp.data(), Wd_hp.data(), Sd_hp.data(), ref.data(), E);

  // Bytes that MUST be read from HBM per token (the fp8 weights; everything else is <1 MB).
  const double bytesA = (double)E * gu_n;
  const double bytesB = (double)E * d_n;
  const double bytesT = bytesA + bytesB;
  auto gbps = [](double bytes, float ms) { return bytes / 1e6 / ms; };

  cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
  const int WARM = 30, IT = 300;
  auto bench = [&](auto fn) -> float {
    for (int i = 0; i < WARM; ++i) fn();
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
    for (int i = 0; i < IT; ++i) fn();
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms; CK(cudaEventElapsedTime(&ms, s, e)); return ms / IT;
  };

  // The (R, STAGES) configs we can dispatch (compile-time template instantiations).
  struct Entry { V3Cfg c; FnA fa; FnB fb; };
  std::vector<Entry> entries;
  auto add = [&](int R, int STAGES, int block, FnA fa, FnB fb) {
    entries.push_back({ V3Cfg{R, STAGES, block}, fa, fb });
  };
  // R in {1,2,4}, STAGES in {2,3}, block in {256,512}. (Compile-time instantiated below.)
  for (int blk : {256, 512}) {
    add(1, 2, blk, getA<1,2>(), getB<1,2>());
    add(1, 3, blk, getA<1,3>(), getB<1,3>());
    add(2, 2, blk, getA<2,2>(), getB<2,2>());
    add(2, 3, blk, getA<2,3>(), getB<2,3>());
    add(4, 2, blk, getA<4,2>(), getB<4,2>());
    add(4, 3, blk, getA<4,3>(), getB<4,3>());
  }

  printf("\nper-token expert weight read: %.1f MB  (gate+up %.1f MB + down %.1f MB)\n",
         bytesT / 1e6, bytesA / 1e6, bytesB / 1e6);
  printf("baseline to beat: k5_experts.cu = 1530 GB/s = 45.7%% MBU\n\n");
  printf("  %-4s %-7s %-6s %9s %9s %9s %9s %10s %8s\n",
         "R", "STAGES", "block", "smemA", "smemB", "us/tok", "GB/s", "%HBMpeak", "valid");

  double best_gbps = 0.0; V3Cfg best{};
  for (auto& en : entries) {
    const V3Cfg& c = en.c;
    size_t sa = smemA(c), sb = smemB(c, E);
    if ((int)sa > maxSmem || (int)sb > maxSmem) {
      printf("  %-4d %-7d %-6d %9zu %9zu %9s %9s %9s %8s\n",
             c.R, c.STAGES, c.block, sa, sb, "-", "-", "-", "skip(smem)");
      continue;
    }
    // opt in to this config's smem ONCE (kept out of the timed loop below)
    if (!set_cfg_smem(c, b, maxSmem, en.fa, en.fb)) continue;
    // correctness for THIS config
    CK(cudaMemset(h_d, 0, HIDDEN * sizeof(float)));
    launch_cfg(c, b, 0, en.fa, en.fb);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> got(HIDDEN, 0.0f);
    CK(cudaMemcpy(got.data(), h_d, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs = 0.0;
    for (int i = 0; i < HIDDEN; ++i) max_abs = std::max(max_abs, fabs((double)ref[i] - (double)got[i]));
    bool ok = max_abs < 1e-2;

    auto runAB = [&]() { launch_cfg(c, b, 0, en.fa, en.fb); };
    float msAB = bench(runAB);
    CK(cudaGetLastError());
    double gb = gbps(bytesT, msAB);
    printf("  %-4d %-7d %-6d %9zu %9zu %9.2f %9.1f %8.1f%% %8s\n",
           c.R, c.STAGES, c.block, sa, sb, msAB * 1e3, gb, 100.0 * gb / PEAK,
           ok ? "PASS" : "FAIL");
    if (ok && gb > best_gbps) { best_gbps = gb; best = c; }
  }

  // ---- best-config detailed report (A / B / A+B) ----------------------------------------------
  if (best_gbps > 0.0) {
    // find the matching entry to get the function pointers
    FnA fa = nullptr; FnB fb = nullptr;
    for (auto& en : entries)
      if (en.c.R == best.R && en.c.STAGES == best.STAGES && en.c.block == best.block) { fa = en.fa; fb = en.fb; }
    size_t sa = smemA(best), sb = smemB(best, E);
    cudaFuncSetAttribute((const void*)fa, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sa);
    cudaFuncSetAttribute((const void*)fb, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sb);
    int ctasA = ctas_for(E * MOE_INTER, best.R, best.block);
    int ctasB = ctas_for(E * HIDDEN,    best.R, best.block);
    auto runA = [&]() { fa<<<ctasA, best.block, sa>>>(y_d, sel_d, Wgu_d, Sgu_d, a_d, E); };
    auto runB = [&]() { fb<<<ctasB, best.block, sb>>>(sel_d, selw_d, Wd_d, Sd_d, a_d, h_d, E); };
    auto runAB = [&]() { runA(); runB(); };
    float msA  = bench(runA);
    float msB  = bench(runB);
    float msAB = bench(runAB);
    CK(cudaGetLastError());
    printf("\nBEST: R=%d STAGES=%d block=%d  CTAs(A)=%d CTAs(B)=%d\n",
           best.R, best.STAGES, best.block, ctasA, ctasB);
    printf("  %-14s %10s %10s %10s\n", "stage", "us/tok", "GB/s", "%HBMpeak");
    printf("  %-14s %10.2f %10.1f %9.1f%%\n", "gate+up (A)", msA*1e3,  gbps(bytesA, msA),  100.0*gbps(bytesA, msA)/PEAK);
    printf("  %-14s %10.2f %10.1f %9.1f%%\n", "down    (B)", msB*1e3,  gbps(bytesB, msB),  100.0*gbps(bytesB, msB)/PEAK);
    printf("  %-14s %10.2f %10.1f %9.1f%%\n", "fused   (A+B)", msAB*1e3, gbps(bytesT, msAB), 100.0*gbps(bytesT, msAB)/PEAK);
    printf("\nMoE-expert decode over %d layers: %.2f ms/token\n", N_LAYERS, msAB * N_LAYERS);
    double improve = 100.0 * (best_gbps - 1530.0) / 1530.0;
    printf("vs k5 baseline (1530 GB/s): %+.1f%%  -> %s\n", improve,
           best_gbps > 1530.0 ? "FASTER" : "slower");
  } else {
    printf("\nno config passed correctness\n");
  }

  for (int e2 = 0; e2 < E; ++e2) {
    cudaFree(Wgu_dp[e2]); cudaFree(Wd_dp[e2]); cudaFree(Sgu_dp[e2]); cudaFree(Sd_dp[e2]);
  }
  cudaFree(Wgu_d); cudaFree(Wd_d); cudaFree(Sgu_d); cudaFree(Sd_d);
  cudaFree(sel_d); cudaFree(selw_d); cudaFree(y_d); cudaFree(h_d); cudaFree(a_d);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return 0;
}
#endif // K5V3_NO_MAIN
