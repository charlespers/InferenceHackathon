// prefill_wgmma.cu — Qwen3-235B-A22B PREFILL GEMMs on H100 fp8 TENSOR CORES (sm_90a).
//
// WHY: the existing prefill path (prefill_attn.cu / prefill_moe.cu) runs the big
// projection/MoE GEMMs as SIMT fp32-accumulate kernels. On H100 that tops out near
// the ~67 TFLOP/s CUDA-core fp32 peak — i.e. <1% of the ~1.98 PFLOP/s fp8
// tensor-core peak. Prefill (prompt processing, M = seq = 512) is a *compute*-bound
// GEMM workload, so the single biggest MFU lever is to run it on the fp8 tensor
// cores. This file rewrites the prefill GEMMs to use the fp8 e4m3 tensor-core
// instruction `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, with
// cp.async-staged shared-memory tiles and fp32 accumulation. It also FIXES the MoE
// to evaluate only the routed TOP_K=8 experts per token (the old prefill_moe
// microbench looped over all 128).
//
// Scope (the two GEMMs that dominate prefill FLOPs):
//   (a) projection GEMMs  Y[M,N] = X[M,K] @ deq(W[N,K])^T   — used for Wqkv and Wo.
//   (b) MoE expert GEMMs   gate/up [HIDDEN->MOE_INTER] + SwiGLU, down [MOE_INTER->HIDDEN],
//       run per routed expert over only its gathered tokens, scatter-added with the
//       routing weight into the residual.
// Attention score/softmax/V (the M x M flash part) is left to prefill_attn.cu; this
// file targets the dense weight GEMMs, which carry the overwhelming majority of the
// prefill FLOPs and ALL of the fp8 tensor-core opportunity.
//
// TENSOR-CORE OPERANDS. The fp8 mma needs BOTH operands in fp8 e4m3. Weights are
// already fp8. Activations are fp32 in the reference pipeline, so we quantize X to
// fp8 e4m3 in-kernel using a per-row absmax activation scale (preserves dynamic
// range; cost is O(M*K), negligible vs the GEMM). The mma accumulates the raw fp8 x
// fp8 products in fp32; the epilogue rescales by act_scale[m] * weight_scale[n].
// This mirrors the standard fp8 GEMM recipe (per-tensor/per-row act scale,
// per-output-channel weight scale) and matches common.cuh's weight convention:
//   logical W is [OUT, IN] row-major (IN contiguous), scale[OUT] per output channel,
//   Y[m,n] = sum_k X[m,k] * deq(W[n,k], scale[n]).
//
// LAYOUT. m16n8k32 takes A row-major [M,K] and B col-major [N,K]. Our weight storage
// [N,K] row-major IS exactly "B col-major" for this instruction, so weights feed the
// tensor core with no transpose. Activations are quantized into a row-major [M,K]
// fp8 buffer (A row-major). Both operands are 32-wide in K (k32), matching e4m3.
//
// Build (kernels-only TU):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -c kernels/prefill_wgmma.cu
// Build the self-test + microbench executable:
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/prefill_wgmma.cu -o /tmp/pw
//   /tmp/pw            # CPU-vs-GPU validation (small M) then the M=512 microbench
//
// Standard CUDA + public PTX ISA (mma.sync, cp.async) only. References shapes from
// common.cuh; all helpers are local so this file never edits common.cuh.
#include "common.cuh"
using namespace q3;

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// ===========================================================================
//  fp8 mma primitive: m16n8k32, e4m3 x e4m3 -> f32, A row-major, B col-major.
//
//  PTX: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//    A fragment : 4 x .b32 regs (16 fp8 lanes-worth packed; per-thread 16 fp8 = 4 u32)
//    B fragment : 2 x .b32 regs (per-thread 8 fp8 = 2 u32)
//    C/D frag   : 4 x .f32     (per-thread 2 of the 16x8 = 4 f32 over the tile)
//  Operand register packing follows the standard m16nNk32 fp8 thread-fragment map
//  (PTX ISA "Warp-level matrix fragments for mma.m16n8k32"). We stage the tiles into
//  shared memory in the natural layouts and have each thread gather its fragment with
//  the documented lane->element mapping, then issue the mma.
// ===========================================================================

// One m16n8k32 fp8 mma: D[16x8] += A[16x32] * B[8x32]^T(col-major B = [N,K]).
// a[4], b[2] hold this thread's packed fp8 fragments; c[4]/d[4] the f32 accumulators.
static __device__ __forceinline__ void mma_m16n8k32_e4m3(
    const unsigned (&a)[4], const unsigned (&b)[2], float (&d)[4], const float (&c)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
  // Host / non-TC path: keep the symbol valid (never executed on device < sm_89).
  d[0]=c[0]; d[1]=c[1]; d[2]=c[2]; d[3]=c[3];
  (void)a; (void)b;
#endif
}

// ===========================================================================
//  cp.async helpers (16-byte cg copies global->shared) + commit/wait.
// ===========================================================================
static __device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
#else
  *reinterpret_cast<uint4*>(smem) = *reinterpret_cast<const uint4*>(gmem);
#endif
}
static __device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
static __device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// ===========================================================================
//  Tile geometry.  CTA computes a BM x BN output tile; one warp per (warp_m, warp_n)
//  sub-tile of WM x WN = 16 x 64.  K is stepped in BK=32 chunks (one k32 per mma).
//
//  BM = 64 (4 warp rows of 16), BN = 64 (1 warp col of 64 = 8 n8 tiles).
//  WARPS = (BM/16) * (BN/WN) = 4 * 1 = 4 warps  -> 128 threads/CTA.
//  Each warp: 1 m16 row x 8 n8 cols  -> 8 mma.m16n8k32 per K-step, 8x f32[4] accs.
// ===========================================================================
#define BM   64
#define BN   64
#define BK   32
#define WARP_M 16          // rows per warp
#define WARP_N 64          // cols per warp
#define N8   (WARP_N/8)    // 8 n8 sub-tiles per warp along N
#define NWARPS ((BM/WARP_M)*(BN/WARP_N))   // 4
#define NTHREADS (NWARPS*32)               // 128

// Shared tile bytes: As[BM*BK] + Bs[BN*BK] fp8, double-buffered.
#define ASMEM (BM*BK)      // 64*32 = 2048 fp8
#define BSMEM (BN*BK)      // 64*32 = 2048 fp8

// ---------------------------------------------------------------------------
//  Load this thread's A fragment (4xu32 = 16 fp8) for warp row `warpRow` from the
//  shared A tile As[BM][BK] (row-major).  Official PTX m16n8k32 8-bit A-fragment map
//  (groupID = laneid>>2 in 0..7, tig = laneid&3 in 0..3; each .b32 = 4 consecutive
//   fp8 along K):
//      a0 : row = groupID,     K = tig*4 + {0..3}        (K-half 0, rows 0..7)
//      a1 : row = groupID + 8, K = tig*4 + {0..3}        (K-half 0, rows 8..15)
//      a2 : row = groupID,     K = tig*4 + 16 + {0..3}   (K-half 1, rows 0..7)
//      a3 : row = groupID + 8, K = tig*4 + 16 + {0..3}   (K-half 1, rows 8..15)
//  (REGISTER ORDER MATTERS: a1 is the +8 row at K-half-0, a2 is the same row as a0 at
//   K-half-1. Getting this order wrong silently corrupts the result.)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_A_frag(
    const fp8* __restrict__ As, int warpRow, int lane, unsigned (&a)[4]) {
  const int groupID = lane >> 2;              // 0..7
  const int kq      = (lane & 3) * 4;         // 0,4,8,12 -> start fp8 in a 16-wide half
  // As is [BM][BK]; this warp covers rows [warpRow*16 .. +16).
  const int r0 = warpRow * 16 + groupID;          // rows 0..7 of the 16-row tile
  const int r1 = warpRow * 16 + 8 + groupID;      // rows 8..15
  auto pack = [&](int row, int kbase) -> unsigned {
    const fp8* p = As + row * BK + kbase;             // 4 consecutive fp8 along K
    return *reinterpret_cast<const unsigned*>(p);
  };
  a[0] = pack(r0, kq);          // a0: row group0, K-half 0
  a[1] = pack(r1, kq);          // a1: row group1, K-half 0
  a[2] = pack(r0, kq + 16);     // a2: row group0, K-half 1
  a[3] = pack(r1, kq + 16);     // a3: row group1, K-half 1
}

// ---------------------------------------------------------------------------
//  Load this thread's B fragment (2xu32 = 8 fp8) for n8 sub-tile `n8idx` from the
//  shared B tile Bs[BN][BK] (row-major, i.e. col-major operand: each "column" n is a
//  contiguous K row).  m16n8k32 B-fragment lane map (8 cols x 32 K):
//    col   = L >> 2            (0..7)
//    kq    = (L & 3) * 4
//    b[0]=K[0..3] of col, b[1]=K[16..19] of col.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void load_B_frag(
    const fp8* __restrict__ Bs, int n8idx, int lane, unsigned (&b)[2]) {
  const int col = (n8idx * 8) + (lane >> 2);   // 0..BN-1
  const int kq  = (lane & 3) * 4;
  const fp8* p = Bs + col * BK + kq;           // Bs[col][k], K contiguous
  b[0] = *reinterpret_cast<const unsigned*>(p);
  b[1] = *reinterpret_cast<const unsigned*>(p + 16);
}

// ---------------------------------------------------------------------------
//  Store this thread's accumulators for one n8 sub-tile into a row-major C tile.
//  m16n8k32 C/D-fragment lane map (16 rows x 8 cols), 4 f32 per thread:
//    d[0],d[1] : row = (L>>2),       col = (L&3)*2 + {0,1}
//    d[2],d[3] : row = (L>>2)+8,     col = (L&3)*2 + {0,1}
//  cb(row,col) is invoked with tile-local row in [0,16) and col in [0,8).
// ---------------------------------------------------------------------------
template <class CB>
static __device__ __forceinline__ void foreach_C(int lane, const float (&d)[4], CB cb) {
  const int rowA = lane >> 2;
  const int rowB = rowA + 8;
  const int col0 = (lane & 3) * 2;
  cb(rowA, col0,   d[0]);
  cb(rowA, col0+1, d[1]);
  cb(rowB, col0,   d[2]);
  cb(rowB, col0+1, d[3]);
}

// ===========================================================================
//  Quantize fp32 activations X[M,K] -> fp8 e4m3 Xq[M,K] with per-row absmax scale.
//  act_scale[m] = absmax(X[m,:]) / 448  (e4m3 max representable magnitude).
//  Xq[m,k] = round_to_e4m3( X[m,k] / act_scale[m] ).  Dequant later: *act_scale[m].
//  One block per row, blockDim threads stride the row. Two passes (max, then store).
// ===========================================================================
__global__ void quantize_rows_e4m3(
    const float* __restrict__ X, fp8* __restrict__ Xq, float* __restrict__ act_scale,
    int M, int K) {
  int m = blockIdx.x;
  if (m >= M) return;
  const float* xr = X + (size_t)m * K;
  fp8* qr = Xq + (size_t)m * K;

  __shared__ float red[256];
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(xr[k]));
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float E4M3_MAX = 448.0f;
  float scale = (red[0] > 0.f) ? (red[0] / E4M3_MAX) : 1.f;
  float inv = 1.f / scale;
  if (threadIdx.x == 0) act_scale[m] = scale;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    qr[k] = fp8(xr[k] * inv);
}

// ===========================================================================
//  Core fp8 tensor-core GEMM:  Y[M,N] = (Xq[M,K] @ Wq[N,K]^T) scaled.
//  Xq fp8 (row-major), Wq fp8 (row-major [N,K] = B col-major), act_scale[M] per row,
//  Wscale[N] per output channel.  Y = raw_mma * act_scale[m] * Wscale[n].
//
//  Optional fused SwiGLU + second-matrix mode: if `Wq2`!=nullptr we compute a SECOND
//  GEMM with the same Xq into the same tile (the "up" projection) and write
//  Y[m,n] = silu(g) * u where g = gemm(Wq), u = gemm(Wq2).  This fuses gate+up like
//  k5/prefill_moe so Xq is read once.  When Wq2==nullptr it is a plain scaled GEMM.
//
//  Epilogue write modes:
//    epi == 0 : Y[m,n] = result                       (overwrite)
//    epi == 1 : atomicAdd(&Y[row_map[m], n], rweight[m]*result)   (scatter-add)
//  row_map / rweight may be nullptr for epi==0.
// ===========================================================================
__global__ void __launch_bounds__(NTHREADS)
gemm_fp8_tc(
    const fp8* __restrict__ Xq, const float* __restrict__ act_scale,  // [M,K], [M]
    const fp8* __restrict__ Wq, const float* __restrict__ Wscale,     // [N,K], [N]
    const fp8* __restrict__ Wq2, const float* __restrict__ Wscale2,   // [N,K], [N] or null
    float* __restrict__ Y,            // [M,N]  (or residual [*,N] when scattering)
    const int* __restrict__ row_map,  // [M] global row for scatter (epi==1)
    const float* __restrict__ rweight,// [M] routing weight   (epi==1)
    int M, int N, int K, int epi) {

  __shared__ fp8 As[2][ASMEM];
  __shared__ fp8 Bs[2][BSMEM];
  __shared__ fp8 Bs2[2][BSMEM];     // only used if Wq2

  const int tid    = threadIdx.x;
  const int warp   = tid >> 5;
  const int lane   = tid & 31;
  const int warpRow= warp;                 // NWARPS==4 == BM/16, one warp per 16-row band
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;
  const bool dual = (Wq2 != nullptr);

  // accumulators: per warp, N8 sub-tiles, 4 f32 each (gate). up uses a second set.
  float accG[N8][4];
  float accU[N8][4];
  #pragma unroll
  for (int t = 0; t < N8; ++t)
    #pragma unroll
    for (int q = 0; q < 4; ++q) { accG[t][q] = 0.f; accU[t][q] = 0.f; }

  // ---- staged tile loaders (cp.async, 16B = 16 fp8 per thread) ----
  // As tile: BM*BK = 2048 fp8 -> 128 uint4 -> 128 threads, 1 each.
  // Bs tile: BN*BK = 2048 fp8 -> 128 uint4 -> 128 threads, 1 each.
  auto load_A = [&](int buf, int k0) {
    int idx = tid;                          // one 16B chunk per thread
    int r = (idx * 16) / BK;                // row within tile (16 fp8 along K fit in BK=32 -> 2 per row)
    int c = (idx * 16) % BK;                // k offset within row
    int gm = blockRow + r;
    int gk = k0 + c;
    fp8* dst = &As[buf][r * BK + c];
    if (gm < M) {
      cp_async_16(dst, Xq + (size_t)gm * K + gk);
    } else {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0,0,0,0);
    }
  };
  auto load_B = [&](const fp8* __restrict__ W, fp8 (*Bbuf)[BSMEM], int buf, int k0) {
    int idx = tid;
    int r = (idx * 16) / BK;                // col index within N tile
    int c = (idx * 16) % BK;
    int gn = blockCol + r;
    int gk = k0 + c;
    fp8* dst = &Bbuf[buf][r * BK + c];
    if (gn < N) {
      cp_async_16(dst, W + (size_t)gn * K + gk);
    } else {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0,0,0,0);
    }
  };

  const int nK = (K + BK - 1) / BK;
  int buf = 0;
  // prologue: stage k0=0
  load_A(buf, 0);
  load_B(Wq, Bs, buf, 0);
  if (dual) load_B(Wq2, Bs2, buf, 0);
  cp_async_commit();

  for (int kt = 0; kt < nK; ++kt) {
    int nbuf = buf ^ 1;
    int k1 = (kt + 1) * BK;
    if (kt + 1 < nK) {
      load_A(nbuf, k1);
      load_B(Wq, Bs, nbuf, k1);
      if (dual) load_B(Wq2, Bs2, nbuf, k1);
      cp_async_commit();
    }
    cp_async_wait_all();
    __syncthreads();

    // ---- mma over this K=32 chunk ----
    unsigned af[4];
    load_A_frag(As[buf], warpRow, lane, af);
    #pragma unroll
    for (int t = 0; t < N8; ++t) {
      unsigned bf[2];
      load_B_frag(Bs[buf], t, lane, bf);
      float dd[4]; const float cc[4] = {accG[t][0],accG[t][1],accG[t][2],accG[t][3]};
      mma_m16n8k32_e4m3(af, bf, dd, cc);
      accG[t][0]=dd[0]; accG[t][1]=dd[1]; accG[t][2]=dd[2]; accG[t][3]=dd[3];
      if (dual) {
        unsigned bf2[2];
        load_B_frag(Bs2[buf], t, lane, bf2);
        float du[4]; const float cu[4] = {accU[t][0],accU[t][1],accU[t][2],accU[t][3]};
        mma_m16n8k32_e4m3(af, bf2, du, cu);
        accU[t][0]=du[0]; accU[t][1]=du[1]; accU[t][2]=du[2]; accU[t][3]=du[3];
      }
    }
    __syncthreads();
    buf = nbuf;
  }

  // ---- epilogue: rescale + write ----
  #pragma unroll
  for (int t = 0; t < N8; ++t) {
    foreach_C(lane, accG[t], [&](int rr, int cc, float gval) {
      int gm = blockRow + warpRow * 16 + rr;
      int gn = blockCol + t * 8 + cc;
      if (gm >= M || gn >= N) return;
      float as = act_scale[gm];
      float val;
      if (dual) {
        // need the matching up accumulator element; recompute from accU with same lane map.
        float uval;
        // foreach_C is structured so d[0..3] map identically; fetch via index.
        // Reconstruct which d-index (rr,cc) corresponds to:
        int ridx = (rr < 8) ? 0 : 1;            // 0 -> d0/d1, 1 -> d2/d3
        int cidx = cc - (lane & 3) * 2;          // 0 or 1
        int di = ridx * 2 + cidx;
        uval = accU[t][di] * as * Wscale2[gn];
        float g = gval * as * Wscale[gn];
        val = silu(g) * uval;
      } else {
        val = gval * as * Wscale[gn];
      }
      if (epi == 0) {
        Y[(size_t)gm * N + gn] = val;
      } else {
        int trow = row_map[gm];
        float w = rweight[gm];
        atomicAdd(&Y[(size_t)trow * N + gn], w * val);
      }
    });
  }
}

// ===========================================================================
//  Gather routed token rows X[M,HIDDEN] -> Xe[T,HIDDEN] by token_id[T].
// ===========================================================================
__global__ void gather_rows(
    const float* __restrict__ X, const int* __restrict__ token_id,
    float* __restrict__ Xe, int T, int K) {
  int t = blockIdx.x;
  if (t >= T) return;
  const float* src = X + (size_t)token_id[t] * K;
  float* dst = Xe + (size_t)t * K;
  for (int i = threadIdx.x; i < K; i += blockDim.x) dst[i] = src[i];
}

// ===========================================================================
//  Host launch helpers.
// ===========================================================================
namespace prefill_wgmma {

inline dim3 grid2d(int M, int N) { return dim3((N + BN - 1) / BN, (M + BM - 1) / BM); }

// Plain scaled fp8-TC projection: Y[M,N] = X[M,K] @ deq(W[N,K])^T.
// d_Xq/d_act are scratch [M*K]/[M] for the in-kernel activation quant.
void project(const float* d_X, fp8* d_Xq, float* d_act,
             const fp8* d_W, const float* d_Wscale,
             float* d_Y, int M, int N, int K, cudaStream_t s = 0) {
  quantize_rows_e4m3<<<M, 256, 0, s>>>(d_X, d_Xq, d_act, M, K);
  gemm_fp8_tc<<<grid2d(M, N), NTHREADS, 0, s>>>(
      d_Xq, d_act, d_W, d_Wscale, nullptr, nullptr, d_Y, nullptr, nullptr, M, N, K, 0);
}

// One routed expert over its T gathered tokens (top-8 path):
//   H = silu(Xe@Wgate^T) * (Xe@Wup^T)   [T, MOE_INTER]
//   residual[token_id] += rweight * (H @ Wdown^T)   [*, HIDDEN]
void run_expert(const float* d_X, const int* d_token_id, const float* d_rweight,
                const fp8* d_Wgate, const float* d_Sg,
                const fp8* d_Wup,   const float* d_Su,
                const fp8* d_Wdown, const float* d_Sd,
                float* d_Xe, fp8* d_XeQ, float* d_XeAct,
                float* d_H,  fp8* d_HQ,  float* d_HAct,
                float* d_residual, int T, cudaStream_t s = 0) {
  if (T <= 0) return;
  // gather rows for this expert
  gather_rows<<<T, 256, 0, s>>>(d_X, d_token_id, d_Xe, T, HIDDEN);
  // gate+up fused -> H[T, MOE_INTER]
  quantize_rows_e4m3<<<T, 256, 0, s>>>(d_Xe, d_XeQ, d_XeAct, T, HIDDEN);
  gemm_fp8_tc<<<grid2d(T, MOE_INTER), NTHREADS, 0, s>>>(
      d_XeQ, d_XeAct, d_Wgate, d_Sg, d_Wup, d_Su, d_H, nullptr, nullptr,
      T, MOE_INTER, HIDDEN, 0);
  // down + routed scatter-add into residual[M, HIDDEN]
  quantize_rows_e4m3<<<T, 256, 0, s>>>(d_H, d_HQ, d_HAct, T, MOE_INTER);
  gemm_fp8_tc<<<grid2d(T, HIDDEN), NTHREADS, 0, s>>>(
      d_HQ, d_HAct, d_Wdown, d_Sd, nullptr, nullptr, d_residual,
      d_token_id, d_rweight, T, HIDDEN, MOE_INTER, 1);
}

} // namespace prefill_wgmma

// ===========================================================================
//  CPU fp32 reference + GPU validation + microbench
// ===========================================================================
#ifndef PREFILL_WGMMA_NO_MAIN

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
  exit(1); } } while(0)

static fp8 to_fp8(float x) { return fp8(x); }
static float from_fp8(fp8 x) { return float(x); }

template <class T>
static T* dev(const std::vector<T>& h) {
  T* p; CUDA_CHECK(cudaMalloc(&p, h.size()*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(p, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

// CPU reference: scaled GEMM Y[m,n] = sum_k X[m,k]*deq(W[n,k]).  Models the SAME fp8
// activation quantization the kernel does, so the comparison is fair (both quantize X).
static void cpu_project_ref(const std::vector<float>& X,
                            const std::vector<fp8>& W, const std::vector<float>& S,
                            std::vector<float>& Y, int M, int N, int K) {
  const float E4M3_MAX = 448.0f;
  Y.assign((size_t)M * N, 0.f);
  for (int m = 0; m < M; ++m) {
    float mx = 0.f;
    for (int k = 0; k < K; ++k) mx = std::max(mx, std::fabs(X[(size_t)m*K+k]));
    float as = (mx > 0.f) ? mx / E4M3_MAX : 1.f;
    std::vector<float> xq(K);
    for (int k = 0; k < K; ++k) xq[k] = from_fp8(to_fp8(X[(size_t)m*K+k] / as)); // quant round-trip
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (double)xq[k] * (double)from_fp8(W[(size_t)n*K+k]);
      Y[(size_t)m*N+n] = (float)(acc * (double)as * (double)S[n]);
    }
  }
}

static void cpu_expert_ref(const std::vector<float>& X, const std::vector<int>& tid,
                           const std::vector<float>& rw,
                           const std::vector<fp8>& Wg, const std::vector<float>& Sg,
                           const std::vector<fp8>& Wu, const std::vector<float>& Su,
                           const std::vector<fp8>& Wd, const std::vector<float>& Sd,
                           std::vector<float>& residual, int T, int hidden, int inter) {
  // gather, quantize x, gate/up, swiglu, quantize h, down, scatter-add.
  const float E4M3_MAX = 448.0f;
  for (int t = 0; t < T; ++t) {
    int tok = tid[t];
    const float* xr = &X[(size_t)tok*hidden];
    float mx = 0.f; for (int k=0;k<hidden;++k) mx=std::max(mx,std::fabs(xr[k]));
    float xs = (mx>0.f)? mx/E4M3_MAX : 1.f;
    std::vector<float> xq(hidden);
    for (int k=0;k<hidden;++k) xq[k]=from_fp8(to_fp8(xr[k]/xs));
    std::vector<float> h(inter);
    for (int n=0;n<inter;++n){
      double g=0,u=0;
      for (int k=0;k<hidden;++k){
        g += (double)xq[k]*(double)from_fp8(Wg[(size_t)n*hidden+k]);
        u += (double)xq[k]*(double)from_fp8(Wu[(size_t)n*hidden+k]);
      }
      float gg=(float)(g*(double)xs*(double)Sg[n]);
      float uu=(float)(u*(double)xs*(double)Su[n]);
      h[n]= (gg/(1.f+std::exp(-gg)))*uu;
    }
    float hm=0.f; for (int k=0;k<inter;++k) hm=std::max(hm,std::fabs(h[k]));
    float hs=(hm>0.f)? hm/E4M3_MAX : 1.f;
    std::vector<float> hq(inter);
    for (int k=0;k<inter;++k) hq[k]=from_fp8(to_fp8(h[k]/hs));
    for (int n=0;n<hidden;++n){
      double d=0;
      for (int k=0;k<inter;++k) d += (double)hq[k]*(double)from_fp8(Wd[(size_t)n*inter+k]);
      residual[(size_t)tok*hidden+n] += rw[t]*(float)(d*(double)hs*(double)Sd[n]);
    }
  }
}

static double compare(const std::vector<float>& got, const std::vector<float>& ref,
                      double* out_rel) {
  double maxerr=0,maxrel=0;
  for (size_t i=0;i<got.size();++i){
    double e=std::fabs((double)got[i]-(double)ref[i]);
    if (e>maxerr) maxerr=e;
    double den=std::fabs((double)ref[i])+1e-3;
    if (e/den>maxrel) maxrel=e/den;
  }
  if (out_rel) *out_rel=maxrel;
  return maxerr;
}

static void validate_project() {
  printf("=== prefill_wgmma: projection GEMM validation ===\n");
  // small-but-real-shaped: M small, K=HIDDEN, N spans a few BN tiles.
  const int M = 20, K = HIDDEN, N = 192;   // N=192 = 3*BN, M=20 crosses BM tiles
  std::mt19937 rng(99);
  std::normal_distribution<float> nd(0.f,0.5f);
  std::uniform_real_distribution<float> ud(0.4f,1.2f);
  std::vector<float> X((size_t)M*K); for (auto& v:X) v=nd(rng);
  std::vector<fp8> W((size_t)N*K); std::vector<float> S(N);
  for (int n=0;n<N;++n){ float s=0.02f+0.01f*ud(rng); S[n]=s;
    for (int k=0;k<K;++k) W[(size_t)n*K+k]=to_fp8(nd(rng)*0.05f/s); }

  std::vector<float> ref; cpu_project_ref(X,W,S,ref,M,N,K);

  float* dX=dev(X); fp8* dW=dev(W); float* dS=dev(S);
  fp8* dXq; CUDA_CHECK(cudaMalloc(&dXq,(size_t)M*K*sizeof(fp8)));
  float* dAct; CUDA_CHECK(cudaMalloc(&dAct,(size_t)M*sizeof(float)));
  float* dY; CUDA_CHECK(cudaMalloc(&dY,(size_t)M*N*sizeof(float)));
  prefill_wgmma::project(dX,dXq,dAct,dW,dS,dY,M,N,K);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got((size_t)M*N);
  CUDA_CHECK(cudaMemcpy(got.data(),dY,got.size()*sizeof(float),cudaMemcpyDeviceToHost));

  double rel; double err=compare(got,ref,&rel);
  printf("  M=%d N=%d K=%d  max_abs_err=%.3e  max_rel_err=%.3e -> %s (threshold 1e-2)\n",
         M,N,K,err,rel,(err<1e-2)?"PASS":"FAIL");
  cudaFree(dX);cudaFree(dW);cudaFree(dS);cudaFree(dXq);cudaFree(dAct);cudaFree(dY);
}

static void validate_moe() {
  printf("=== prefill_wgmma: MoE (top-8, routed-only) validation ===\n");
  const int M = 8, hidden = HIDDEN, inter = MOE_INTER, ne = 3;
  std::mt19937 rng(321);
  std::normal_distribution<float> nd(0.f,0.5f);
  std::uniform_real_distribution<float> ud(0.4f,1.2f), wd(0.1f,0.9f);
  std::vector<float> X((size_t)M*hidden); for (auto& v:X) v=nd(rng)*0.3f;
  auto mk=[&](int out,int in,std::vector<fp8>&W,std::vector<float>&S){
    W.resize((size_t)out*in); S.resize(out);
    for (int o=0;o<out;++o){ float s=0.02f+0.01f*ud(rng); S[o]=s;
      for (int k=0;k<in;++k) W[(size_t)o*in+k]=to_fp8(nd(rng)*0.05f/s); } };
  std::vector<std::vector<fp8>> Wg(ne),Wu(ne),Wd(ne);
  std::vector<std::vector<float>> Sg(ne),Su(ne),Sd(ne);
  for (int e=0;e<ne;++e){ mk(inter,hidden,Wg[e],Sg[e]); mk(inter,hidden,Wu[e],Su[e]); mk(hidden,inter,Wd[e],Sd[e]); }

  std::vector<std::vector<int>> tid(ne); std::vector<std::vector<float>> rw(ne);
  for (int m=0;m<M;++m){ int e0=m%ne,e1=(m+1)%ne;
    tid[e0].push_back(m); rw[e0].push_back(wd(rng));
    tid[e1].push_back(m); rw[e1].push_back(wd(rng)); }

  std::vector<float> ref((size_t)M*hidden,0.f);
  for (int e=0;e<ne;++e)
    cpu_expert_ref(X,tid[e],rw[e],Wg[e],Sg[e],Wu[e],Su[e],Wd[e],Sd[e],ref,(int)tid[e].size(),hidden,inter);

  float* dX=dev(X);
  float* dres; CUDA_CHECK(cudaMalloc(&dres,(size_t)M*hidden*sizeof(float)));
  CUDA_CHECK(cudaMemset(dres,0,(size_t)M*hidden*sizeof(float)));
  int maxT=0; for (int e=0;e<ne;++e) maxT=std::max(maxT,(int)tid[e].size());
  float *dXe,*dH,*dXeAct,*dHAct; fp8 *dXeQ,*dHQ;
  CUDA_CHECK(cudaMalloc(&dXe,(size_t)maxT*hidden*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dH ,(size_t)maxT*inter *sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dXeQ,(size_t)maxT*hidden*sizeof(fp8)));
  CUDA_CHECK(cudaMalloc(&dHQ ,(size_t)maxT*inter *sizeof(fp8)));
  CUDA_CHECK(cudaMalloc(&dXeAct,(size_t)maxT*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dHAct ,(size_t)maxT*sizeof(float)));

  for (int e=0;e<ne;++e){
    int T=(int)tid[e].size();
    int* dtid=dev(tid[e]); float* drw=dev(rw[e]);
    fp8 *dWg=dev(Wg[e]),*dWu=dev(Wu[e]),*dWd=dev(Wd[e]);
    float *dSg=dev(Sg[e]),*dSu=dev(Su[e]),*dSd=dev(Sd[e]);
    prefill_wgmma::run_expert(dX,dtid,drw,dWg,dSg,dWu,dSu,dWd,dSd,
                              dXe,dXeQ,dXeAct,dH,dHQ,dHAct,dres,T);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dtid);cudaFree(drw);cudaFree(dWg);cudaFree(dWu);cudaFree(dWd);
    cudaFree(dSg);cudaFree(dSu);cudaFree(dSd);
  }
  std::vector<float> got((size_t)M*hidden);
  CUDA_CHECK(cudaMemcpy(got.data(),dres,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
  double rel; double err=compare(got,ref,&rel);
  printf("  M=%d experts=%d (routed top-k only)  max_abs_err=%.3e  max_rel_err=%.3e -> %s\n",
         M,ne,err,rel,(err<1e-2)?"PASS":"FAIL");
  cudaFree(dX);cudaFree(dres);cudaFree(dXe);cudaFree(dH);cudaFree(dXeQ);cudaFree(dHQ);
  cudaFree(dXeAct);cudaFree(dHAct);
}

static void microbench_proj(int M) {
  printf("=== prefill_wgmma: projection microbench (M=%d) ===\n", M);
  // Stand-in for the QKV (N=QKV_OUT,K=HIDDEN) + O (N=HIDDEN,K=Q_DIM) projections.
  struct Job { const char* name; int N, K; };
  Job jobs[] = { {"Wqkv", QKV_OUT, HIDDEN}, {"Wo", HIDDEN, Q_DIM} };
  const double H100_FP8_TC = 1979.0;
  double total_ms = 0, total_flops = 0;
  for (auto& j : jobs) {
    int N=j.N, K=j.K;
    std::vector<fp8> W((size_t)N*K, fp8(0.02f)); std::vector<float> S(N,0.02f);
    std::vector<float> X((size_t)M*K,0.01f);
    float* dX=dev(X); fp8* dW=dev(W); float* dS=dev(S);
    fp8* dXq; CUDA_CHECK(cudaMalloc(&dXq,(size_t)M*K*sizeof(fp8)));
    float* dAct; CUDA_CHECK(cudaMalloc(&dAct,(size_t)M*sizeof(float)));
    float* dY; CUDA_CHECK(cudaMalloc(&dY,(size_t)M*N*sizeof(float)));
    for (int i=0;i<3;++i) prefill_wgmma::project(dX,dXq,dAct,dW,dS,dY,M,N,K);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    const int iters=30; CUDA_CHECK(cudaEventRecord(a));
    for (int i=0;i<iters;++i) prefill_wgmma::project(dX,dXq,dAct,dW,dS,dY,M,N,K);
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); ms/=iters;
    double flops = 2.0*(double)M*N*K;
    double tf = flops/(ms*1e-3)/1e12;
    printf("  %-5s M=%d N=%d K=%d : %.4f ms  %.1f TFLOP/s  (%.1f%% fp8-TC peak)\n",
           j.name,M,N,K,ms,tf,100.0*tf/H100_FP8_TC);
    total_ms += ms; total_flops += flops;
    cudaEventDestroy(a);cudaEventDestroy(b);
    cudaFree(dX);cudaFree(dW);cudaFree(dS);cudaFree(dXq);cudaFree(dAct);cudaFree(dY);
  }
  double tf_all = total_flops/(total_ms*1e-3)/1e12;
  printf("  proj total/layer  : %.4f ms  %.1f TFLOP/s  (%.1f%% fp8-TC peak)\n",
         total_ms, tf_all, 100.0*tf_all/H100_FP8_TC);
}

static void microbench_moe(int M) {
  printf("=== prefill_wgmma: MoE microbench (M=%d, routed top-8 only) ===\n", M);
  const int hidden=HIDDEN, inter=MOE_INTER;
  // top-8 routing: M*TOP_K expert-token rows total. Balanced across experts that are
  // hit -> per-expert T. We bench the per-expert GEMMs (gate/up + down) which is the
  // tensor-core workload; gather/quant are O(rows) and amortized.
  int total_rows = M * TOP_K;
  int active = std::min(N_EXPERTS, total_rows);          // experts actually used
  int per_expert = std::max(1, total_rows / active);

  std::vector<fp8> Wg((size_t)inter*hidden, fp8(0.02f)); std::vector<float> Sg(inter,0.02f);
  std::vector<fp8> Wu((size_t)inter*hidden, fp8(0.02f)); std::vector<float> Su(inter,0.02f);
  std::vector<fp8> Wd((size_t)hidden*inter, fp8(0.02f)); std::vector<float> Sd(hidden,0.02f);
  std::vector<float> X((size_t)M*hidden,0.01f);
  std::vector<int> tid(per_expert); for (int i=0;i<per_expert;++i) tid[i]=i%M;
  std::vector<float> rw(per_expert,0.125f);

  float* dX=dev(X);
  float* dres; CUDA_CHECK(cudaMalloc(&dres,(size_t)M*hidden*sizeof(float)));
  CUDA_CHECK(cudaMemset(dres,0,(size_t)M*hidden*sizeof(float)));
  int* dtid=dev(tid); float* drw=dev(rw);
  fp8 *dWg=dev(Wg),*dWu=dev(Wu),*dWd=dev(Wd);
  float *dSg=dev(Sg),*dSu=dev(Su),*dSd=dev(Sd);
  float *dXe,*dH,*dXeAct,*dHAct; fp8 *dXeQ,*dHQ;
  CUDA_CHECK(cudaMalloc(&dXe,(size_t)per_expert*hidden*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dH ,(size_t)per_expert*inter *sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dXeQ,(size_t)per_expert*hidden*sizeof(fp8)));
  CUDA_CHECK(cudaMalloc(&dHQ ,(size_t)per_expert*inter *sizeof(fp8)));
  CUDA_CHECK(cudaMalloc(&dXeAct,(size_t)per_expert*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dHAct ,(size_t)per_expert*sizeof(float)));

  auto run_layer=[&](){
    for (int e=0;e<active;++e)
      prefill_wgmma::run_expert(dX,dtid,drw,dWg,dSg,dWu,dSu,dWd,dSd,
                                dXe,dXeQ,dXeAct,dH,dHQ,dHAct,dres,per_expert);
  };
  for (int i=0;i<2;++i) run_layer();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  const int iters=5; CUDA_CHECK(cudaEventRecord(a));
  for (int i=0;i<iters;++i) run_layer();
  CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); ms/=iters;

  double flops = (double)total_rows * 6.0 * (double)hidden * (double)inter; // gate+up+down
  double tf = flops/(ms*1e-3)/1e12;
  const double H100_FP8_TC = 1979.0;
  printf("  active experts=%d rows/expert=%d total_rows=%d (vs OLD path: ran all %d)\n",
         active, per_expert, total_rows, N_EXPERTS);
  printf("  time/MoE-layer    : %.4f ms\n", ms);
  printf("  achieved          : %.1f TFLOP/s  (%.1f%% fp8-TC peak)\n", tf, 100.0*tf/H100_FP8_TC);
  printf("  x%d layers         : %.2f ms\n", N_LAYERS, ms*N_LAYERS);
  cudaEventDestroy(a);cudaEventDestroy(b);
  cudaFree(dX);cudaFree(dres);cudaFree(dtid);cudaFree(drw);
  cudaFree(dWg);cudaFree(dWu);cudaFree(dWd);cudaFree(dSg);cudaFree(dSu);cudaFree(dSd);
  cudaFree(dXe);cudaFree(dH);cudaFree(dXeQ);cudaFree(dHQ);cudaFree(dXeAct);cudaFree(dHAct);
}

int main(int argc, char** argv) {
  int dev_count=0;
  if (cudaGetDeviceCount(&dev_count)!=cudaSuccess || dev_count==0) {
    fprintf(stderr,"No CUDA device available; this binary needs an H100 (sm_90a).\n");
    return 0;
  }
  validate_project();
  validate_moe();
  int M = (argc>1)?atoi(argv[1]):512;
  microbench_proj(M);
  microbench_moe(M);
  return 0;
}

#endif // PREFILL_WGMMA_NO_MAIN
