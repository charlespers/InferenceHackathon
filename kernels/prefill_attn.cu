// prefill_attn.cu — Qwen3-235B-A22B PREFILL attention path (sm_90a / H100).
//
// PREFILL (prompt processing): M = sequence length (e.g. 512). Unlike decode
// (GEMV, M=1, bandwidth-bound), this is a GEMM (M x K @ K x N) and is
// compute-heavy, so we use shared-memory tiling + register accumulation that can
// actually exploit H100 throughput.
//
// Pipeline (per transformer layer, single sequence, no batch dim):
//   1) input RMSNorm over HIDDEN, fused into ...
//   2) fused QKV GEMM:  Xn[M,HIDDEN] @ Wqkv[HIDDEN, QKV_OUT] -> QKV[M, QKV_OUT]
//        epilogue: per-head QK-norm (RMSNorm over HEAD_DIM) on Q and K, then RoPE
//        (theta = 1e6) on Q and K. V is passed through. (no bias)
//   3) causal flash-attention over the full M x M score matrix with online softmax,
//        GQA broadcast (64 Q heads share 4 KV heads, 16:1).
//   4) O-projection GEMM: AttnOut[M, Q_DIM] @ Wo[Q_DIM, HIDDEN] -> Out[M, HIDDEN]
//        accumulated into the residual stream.
//
// fp8 e4m3 weights with per-output-channel scales (dequant -> bf16/fp32, then GEMM).
// Tensor-core / wgmma is intentionally NOT used here: priority is
// compiles-cleanly > numerically-correct > fast. We use a portable, well-tiled
// fp32-accumulate SIMT GEMM. We report % of bf16 (non-tensor-core) FMA peak.
//
// Build (kernels only, as a TU):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -c kernels/prefill_attn.cu
// Build the self-test + microbench executable:
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/prefill_attn.cu -o /tmp/kp
//   /tmp/kp            # runs CPU-vs-GPU validation then the M=512 microbench
//
#include "common.cuh"
using namespace q3;

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

// ===========================================================================
//  Weight layout convention (matches common.cuh::Fp8Weight)
//  A logical weight W of shape [OUT, IN] is stored ROW-MAJOR with IN contiguous,
//  i.e. element (o, k) lives at w[(size_t)o*IN + k]. Per-output-channel scale is
//  scale[o]. Dequant: float(w) * scale[o].
//
//  For the GEMM  Y[M,N] = X[M,K] @ W^T   where W is [N, K] (out=N, in=K):
//      Y[m,n] = sum_k X[m,k] * deq(W[n,k], scale[n])
//  This is exactly the projection convention (each output channel n is a row of W).
// ===========================================================================

// ---------------------------------------------------------------------------
//  Tiled fp8-weight GEMM:  Y[M,N] = X[M,K] @ deq(W[N,K])^T
//  X is fp32 (activations). W is fp8 e4m3, row-major [N,K], scale[N] per output.
//  Classic 64x64 output tile, BK=16 contraction step, 8x8 register micro-tile.
//  Block = 16x16 threads (256), each thread owns an 8x8 fragment of the C tile.
// ---------------------------------------------------------------------------
#define BM 64
#define BN 64
#define BK 16
#define TM 8
#define TN 8
// threads per block = (BM/TM)*(BN/TN) = 8*8 = 64

__device__ __forceinline__ float dequant(fp8 v, float s) {
  return static_cast<float>(v) * s;
}

// Generic tiled GEMM. `apply_scale_in_smem == false`: we keep W's fp8 bits in
// registers as float (already dequant'd) — simplest + correct. Scales are
// per-output-channel (per-N), applied after the K reduction.
__global__ void gemm_xwT_fp8(
    const float* __restrict__ X,   // [M, K]
    const fp8*   __restrict__ W,   // [N, K] row-major
    const float* __restrict__ Wscale, // [N]
    float*       __restrict__ Y,   // [M, N]
    int M, int N, int K) {
  __shared__ float As[BK][BM];   // transposed tile of X  (k-major for broadcast)
  __shared__ float Bs[BK][BN];   // tile of W (dequant'd), n-major

  const int tid = threadIdx.x;            // 0..63
  const int threadRow = tid / (BN / TN);  // 0..7
  const int threadCol = tid % (BN / TN);  // 0..7

  const int blockRow = blockIdx.y * BM;   // first row (m) of this output tile
  const int blockCol = blockIdx.x * BN;   // first col (n) of this output tile

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  // Each thread loads (BM*BK)/64 = 16 elements of A and (BN*BK)/64 = 16 of B per step.
  // Use a simple strided load over the whole tile with all 64 threads.
  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- load X tile [BM x BK] into As[BK][BM] (transposed) ----
    // tile has BM*BK = 1024 elements, 64 threads -> 16 each.
    #pragma unroll
    for (int e = 0; e < (BM * BK) / 64; ++e) {
      int idx = tid + e * 64;        // 0..1023
      int r = idx / BK;              // 0..BM-1  (m within tile)
      int c = idx % BK;              // 0..BK-1  (k within step)
      int gm = blockRow + r;
      int gk = k0 + c;
      float v = (gm < M && gk < K) ? X[(size_t)gm * K + gk] : 0.f;
      As[c][r] = v;
    }
    // ---- load W tile [BN x BK] into Bs[BK][BN] (dequant'd, transposed) ----
    #pragma unroll
    for (int e = 0; e < (BN * BK) / 64; ++e) {
      int idx = tid + e * 64;        // 0..1023
      int r = idx / BK;              // 0..BN-1  (n within tile)
      int c = idx % BK;              // 0..BK-1  (k within step)
      int gn = blockCol + r;
      int gk = k0 + c;
      float v = 0.f;
      if (gn < N && gk < K) {
        v = dequant(W[(size_t)gn * K + gk], Wscale[gn]);
      }
      Bs[c][r] = v;
    }
    __syncthreads();

    // ---- compute: outer product accumulate over BK ----
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float aReg[TM], bReg[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) aReg[i] = As[kk][threadRow * TM + i];
      #pragma unroll
      for (int j = 0; j < TN; ++j) bReg[j] = Bs[kk][threadCol * TN + j];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += aReg[i] * bReg[j];
    }
    __syncthreads();
  }

  // ---- write back ----
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    int gm = blockRow + threadRow * TM + i;
    if (gm >= M) continue;
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      int gn = blockCol + threadCol * TN + j;
      if (gn < N) Y[(size_t)gm * N + gn] = acc[i][j];
    }
  }
}

// ---------------------------------------------------------------------------
//  Fused RMSNorm prologue:  Xn[m,:] = X[m,:] * rsqrt(mean(X[m,:]^2)+eps) * w[:]
//  One block per row m, blockDim = 256, block reduction in shared memory.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_rows(
    const float* __restrict__ X,        // [M, HIDDEN]
    const float* __restrict__ w,        // [HIDDEN]
    float* __restrict__ Xn,             // [M, HIDDEN]
    int M, int hidden) {
  int m = blockIdx.x;
  if (m >= M) return;
  const float* xr = X + (size_t)m * hidden;
  float* or_ = Xn + (size_t)m * hidden;

  __shared__ float red[256];
  float ss = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = xr[i];
    ss += v * v;
  }
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / hidden + RMS_EPS);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    or_[i] = xr[i] * inv * w[i];
  }
}

// ---------------------------------------------------------------------------
//  QK-norm + RoPE epilogue applied to the QKV GEMM output.
//  QKV layout per token (row of length QKV_OUT = 9216):
//      [ Q (Q_DIM=8192) | K (KV_DIM=512) | V (KV_DIM=512) ]
//  Q has N_Q_HEADS=64 heads of HEAD_DIM=128; K has N_KV_HEADS=4 heads of 128.
//  QK-norm = per-head RMSNorm over the 128-dim, with weights q_norm[128]/k_norm[128].
//  RoPE (theta=1e6): rotate pairs (d, d + HEAD_DIM/2) using position = m (token index).
//
//  One block per (token m, head). We dispatch enough blocks for all Q heads and
//  all K heads. V is left untouched (already correct in the buffer).
//
//  Layout for RoPE pairing: Qwen uses the "GPT-NeoX" rotate-half convention:
//      out[d]            = x[d]*cos - x[d+H/2]*sin
//      out[d+H/2]        = x[d+H/2]*cos + x[d]*sin
//  with freq_i = theta^(-2i/HEAD_DIM), angle = pos * freq_i, i in [0, H/2).
// ---------------------------------------------------------------------------
__global__ void qknorm_rope(
    float* __restrict__ QKV,            // [M, QKV_OUT] in place
    const float* __restrict__ q_norm,   // [HEAD_DIM]
    const float* __restrict__ k_norm,   // [HEAD_DIM]
    int M) {
  // grid.x = M, grid.y = N_Q_HEADS + N_KV_HEADS, block = HEAD_DIM threads (128)
  const int m = blockIdx.x;
  const int head = blockIdx.y;         // 0..(N_Q_HEADS+N_KV_HEADS-1)
  const int d = threadIdx.x;           // 0..HEAD_DIM-1
  if (m >= M) return;

  const bool isQ = head < N_Q_HEADS;
  const int local_head = isQ ? head : (head - N_Q_HEADS);
  const float* normw = isQ ? q_norm : k_norm;

  // base offset of this head's 128-dim vector inside the token row
  size_t base;
  if (isQ) base = (size_t)m * QKV_OUT + (size_t)local_head * HEAD_DIM;
  else     base = (size_t)m * QKV_OUT + Q_DIM + (size_t)local_head * HEAD_DIM;

  float* vec = QKV + base;

  // ---- per-head RMSNorm over HEAD_DIM ----
  __shared__ float red[HEAD_DIM];
  float x = vec[d];
  red[d] = x * x;
  __syncthreads();
  // tree reduce over 128 lanes
  for (int s = HEAD_DIM / 2; s > 0; s >>= 1) {
    if (d < s) red[d] += red[d + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / HEAD_DIM + RMS_EPS);
  float xn = x * inv * normw[d];
  __syncthreads();
  // stash normalized vector so we can read the rotate-half partner
  red[d] = xn;
  __syncthreads();

  // ---- RoPE ----
  const int half = HEAD_DIM / 2;
  int i = d % half;                    // freq index 0..half-1
  bool lower = d < half;               // d in [0,half) is the "real" part
  float freq = powf(ROPE_THETA, -2.0f * (float)i / (float)HEAD_DIM);
  float angle = (float)m * freq;
  float cs = cosf(angle), sn = sinf(angle);

  float self = red[d];
  float partner = lower ? red[d + half] : red[d - half];
  float out;
  if (lower) out = self * cs - partner * sn;     // out[i]      = x[i]*cos - x[i+H/2]*sin
  else       out = self * cs + partner * sn;     // out[i+H/2]  = x[i+H/2]*cos + x[i]*sin
  vec[d] = out;
}

// ---------------------------------------------------------------------------
//  Causal flash-attention (prefill), GQA broadcast, online softmax.
//  Q,K,V live inside the QKV buffer (per token row, see layout above).
//  Output: AttnOut[M, Q_DIM] (per Q head, 128-dim), to be consumed by O-proj.
//
//  One block per (query-tile, q_head). We tile the query rows (BR rows) and
//  stream key/value tiles (BC cols) with online softmax. Causal mask: a query at
//  global row qm attends only to keys with km <= qm.
//
//  This is a straightforward, correct flash implementation: each thread owns one
//  query row of the BR-tile and keeps its 128-dim accumulator + running max/denom
//  in registers/shared. BR == blockDim.x. We pick BR=64, BC=64.
// ---------------------------------------------------------------------------
#define BR 64
#define BC 64

__global__ void flash_attn_causal(
    const float* __restrict__ QKV,      // [M, QKV_OUT]
    float* __restrict__ AttnOut,        // [M, Q_DIM]
    int M, float scale) {
  const int qtile = blockIdx.x;         // query tile index
  const int qhead = blockIdx.y;         // 0..N_Q_HEADS-1
  const int tr = threadIdx.x;           // 0..BR-1  (one query row per thread)
  const int qrow = qtile * BR + tr;     // global query row

  const int kvhead = qhead / GQA_GROUP; // GQA broadcast: 16 Q heads -> 1 KV head

  // shared key/value tile for the BC keys currently being processed
  __shared__ float Ks[BC][HEAD_DIM];
  __shared__ float Vs[BC][HEAD_DIM];

  // per-thread query vector + accumulator (HEAD_DIM=128 floats each)
  float q[HEAD_DIM];
  float o[HEAD_DIM];
  float m_run = -1e30f;   // running max
  float l_run = 0.f;      // running denom

  const bool active = (qrow < M);
  // load query (normed+roped) for this row/head
  if (active) {
    const float* qptr = QKV + (size_t)qrow * QKV_OUT + (size_t)qhead * HEAD_DIM;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) { q[d] = qptr[d]; o[d] = 0.f; }
  } else {
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) { q[d] = 0.f; o[d] = 0.f; }
  }

  // causal: only need key columns up to (qtile*BR + BR-1)
  int kmax = min(M, qtile * BR + BR);   // exclusive upper bound of useful keys

  for (int k0 = 0; k0 < kmax; k0 += BC) {
    // cooperatively load K,V tile [BC x HEAD_DIM] for this kv head
    for (int idx = tr; idx < BC * HEAD_DIM; idx += BR) {
      int kc = idx / HEAD_DIM;          // 0..BC-1
      int d  = idx % HEAD_DIM;
      int km = k0 + kc;
      float kval = 0.f, vval = 0.f;
      if (km < M) {
        const float* kptr = QKV + (size_t)km * QKV_OUT + Q_DIM + (size_t)kvhead * HEAD_DIM;
        const float* vptr = QKV + (size_t)km * QKV_OUT + Q_DIM + KV_DIM + (size_t)kvhead * HEAD_DIM;
        kval = kptr[d];
        vval = vptr[d];
      }
      Ks[kc][d] = kval;
      Vs[kc][d] = vval;
    }
    __syncthreads();

    if (active) {
      #pragma unroll 1
      for (int kc = 0; kc < BC; ++kc) {
        int km = k0 + kc;
        if (km >= M) break;
        if (km > qrow) break;           // causal mask (keys are in order)
        // score = scale * dot(q, K[kc])
        float s = 0.f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) s += q[d] * Ks[kc][d];
        s *= scale;
        // online softmax update
        float m_new = fmaxf(m_run, s);
        float corr = __expf(m_run - m_new);
        float p = __expf(s - m_new);
        l_run = l_run * corr + p;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) o[d] = o[d] * corr + p * Vs[kc][d];
        m_run = m_new;
      }
    }
    __syncthreads();
  }

  if (active) {
    float invl = (l_run > 0.f) ? (1.f / l_run) : 0.f;
    float* outp = AttnOut + (size_t)qrow * Q_DIM + (size_t)qhead * HEAD_DIM;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) outp[d] = o[d] * invl;
  }
}

// ===========================================================================
//  Launch helpers (M as a runtime parameter)
// ===========================================================================
namespace prefill_attn {

inline dim3 gemm_grid(int M, int N) {
  return dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
}

// Full prefill attention block. All device pointers. tmp buffers must be sized:
//   d_Xn   : M*HIDDEN
//   d_qkv  : M*QKV_OUT
//   d_attn : M*Q_DIM
//   d_out  : M*HIDDEN   (this is the attention output added to residual by caller)
void run(const float* d_X,             // [M, HIDDEN] residual input
         const float* d_in_norm,       // [HIDDEN]
         const fp8* d_Wqkv, const float* d_Wqkv_scale,   // [QKV_OUT, HIDDEN]
         const float* d_q_norm, const float* d_k_norm,   // [HEAD_DIM]
         const fp8* d_Wo, const float* d_Wo_scale,       // [HIDDEN, Q_DIM]
         float* d_Xn, float* d_qkv, float* d_attn, float* d_out,
         int M, cudaStream_t stream = 0) {
  // 1) RMSNorm
  rmsnorm_rows<<<M, 256, 0, stream>>>(d_X, d_in_norm, d_Xn, M, HIDDEN);
  // 2) QKV GEMM: Xn[M,HIDDEN] @ Wqkv[QKV_OUT,HIDDEN]^T -> qkv[M,QKV_OUT]
  gemm_xwT_fp8<<<gemm_grid(M, QKV_OUT), 64, 0, stream>>>(
      d_Xn, d_Wqkv, d_Wqkv_scale, d_qkv, M, QKV_OUT, HIDDEN);
  // 2b) QK-norm + RoPE epilogue
  {
    dim3 g(M, N_Q_HEADS + N_KV_HEADS);
    qknorm_rope<<<g, HEAD_DIM, 0, stream>>>(d_qkv, d_q_norm, d_k_norm, M);
  }
  // 3) flash attention
  {
    dim3 g((M + BR - 1) / BR, N_Q_HEADS);
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    flash_attn_causal<<<g, BR, 0, stream>>>(d_qkv, d_attn, M, scale);
  }
  // 4) O-projection: attn[M,Q_DIM] @ Wo[HIDDEN,Q_DIM]^T -> out[M,HIDDEN]
  gemm_xwT_fp8<<<gemm_grid(M, HIDDEN), 64, 0, stream>>>(
      d_attn, d_Wo, d_Wo_scale, d_out, M, HIDDEN, Q_DIM);
}

} // namespace prefill_attn

// ===========================================================================
//  CPU fp32 reference + GPU validation + microbench
//  Compiled only when PREFILL_ATTN_MAIN is defined (default for the executable).
// ===========================================================================
#ifndef PREFILL_ATTN_NO_MAIN

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
  exit(1); } } while(0)

// ---- host-side fp8 e4m3 emulation for building reference weights ----
// We quantize a host fp32 weight to e4m3 by round-trip through __nv_fp8_e4m3 on
// host (the type is usable on host). The CPU reference dequantizes the SAME bits,
// so quantization error is shared by both paths and the comparison is fair.
static fp8 to_fp8(float x) { return fp8(x); }
static float from_fp8(fp8 x) { return float(x); }

// CPU reference of the whole attention block (fp32, dequantized fp8 weights).
static void cpu_attn_ref(
    const std::vector<float>& X,        // [M, HIDDEN]
    const std::vector<float>& in_norm,  // [HIDDEN]
    const std::vector<fp8>& Wqkv, const std::vector<float>& Wqkv_scale,
    const std::vector<float>& q_norm, const std::vector<float>& k_norm,
    const std::vector<fp8>& Wo, const std::vector<float>& Wo_scale,
    std::vector<float>& Out,            // [M, HIDDEN]
    int M, int hidden, int head_dim, int n_q, int n_kv, int qkv_out, int q_dim,
    int kv_dim, float rope_theta, float eps) {
  // 1) RMSNorm
  std::vector<float> Xn((size_t)M * hidden);
  for (int m = 0; m < M; ++m) {
    double ss = 0;
    for (int i = 0; i < hidden; ++i) { float v = X[(size_t)m*hidden+i]; ss += (double)v*v; }
    float inv = 1.0f / std::sqrt((float)(ss / hidden) + eps);
    for (int i = 0; i < hidden; ++i)
      Xn[(size_t)m*hidden+i] = X[(size_t)m*hidden+i] * inv * in_norm[i];
  }
  // 2) QKV GEMM (Xn @ Wqkv^T)
  std::vector<float> qkv((size_t)M * qkv_out);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < qkv_out; ++n) {
      double acc = 0;
      for (int k = 0; k < hidden; ++k)
        acc += (double)Xn[(size_t)m*hidden+k] * (double)(from_fp8(Wqkv[(size_t)n*hidden+k]) * Wqkv_scale[n]);
      qkv[(size_t)m*qkv_out+n] = (float)acc;
    }
  // 2b) QK-norm + RoPE on Q and K
  int half = head_dim / 2;
  auto apply = [&](int m, int head_base, const std::vector<float>& nw, int local_head) {
    size_t base = (size_t)m*qkv_out + head_base + (size_t)local_head*head_dim;
    // rmsnorm over head_dim
    double ss = 0;
    for (int d = 0; d < head_dim; ++d) { float v = qkv[base+d]; ss += (double)v*v; }
    float inv = 1.0f / std::sqrt((float)(ss / head_dim) + eps);
    std::vector<float> xn(head_dim);
    for (int d = 0; d < head_dim; ++d) xn[d] = qkv[base+d] * inv * nw[d];
    // rope
    for (int i = 0; i < half; ++i) {
      float freq = std::pow(rope_theta, -2.0f * (float)i / (float)head_dim);
      float ang = (float)m * freq;
      float cs = std::cos(ang), sn = std::sin(ang);
      float a = xn[i], b = xn[i+half];
      qkv[base+i]      = a*cs - b*sn;
      qkv[base+i+half] = b*cs + a*sn;
    }
  };
  for (int m = 0; m < M; ++m) {
    for (int h = 0; h < n_q; ++h)  apply(m, 0, q_norm, h);
    for (int h = 0; h < n_kv; ++h) apply(m, q_dim, k_norm, h);
  }
  // 3) causal attention, GQA broadcast
  int gqa = n_q / n_kv;
  float scale = 1.0f / std::sqrt((float)head_dim);
  std::vector<float> attn((size_t)M * q_dim, 0.f);
  for (int h = 0; h < n_q; ++h) {
    int kvh = h / gqa;
    for (int qm = 0; qm < M; ++qm) {
      const float* qv = &qkv[(size_t)qm*qkv_out + (size_t)h*head_dim];
      // softmax over keys 0..qm
      std::vector<float> sc(qm+1);
      float mx = -1e30f;
      for (int km = 0; km <= qm; ++km) {
        const float* kv = &qkv[(size_t)km*qkv_out + q_dim + (size_t)kvh*head_dim];
        float s = 0; for (int d = 0; d < head_dim; ++d) s += qv[d]*kv[d];
        s *= scale; sc[km] = s; if (s > mx) mx = s;
      }
      float den = 0; for (int km = 0; km <= qm; ++km) { sc[km] = std::exp(sc[km]-mx); den += sc[km]; }
      float* op = &attn[(size_t)qm*q_dim + (size_t)h*head_dim];
      for (int km = 0; km <= qm; ++km) {
        const float* vv = &qkv[(size_t)km*qkv_out + q_dim + kv_dim + (size_t)kvh*head_dim];
        float w = sc[km] / den;
        for (int d = 0; d < head_dim; ++d) op[d] += w * vv[d];
      }
    }
  }
  // 4) O-projection (attn @ Wo^T), Wo is [hidden, q_dim]
  Out.assign((size_t)M * hidden, 0.f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < hidden; ++n) {
      double acc = 0;
      for (int k = 0; k < q_dim; ++k)
        acc += (double)attn[(size_t)m*q_dim+k] * (double)(from_fp8(Wo[(size_t)n*q_dim+k]) * Wo_scale[n]);
      Out[(size_t)m*hidden+n] = (float)acc;
    }
}

template <class T>
static T* dev(const std::vector<T>& h) {
  T* p; CUDA_CHECK(cudaMalloc(&p, h.size()*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(p, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

static void validate() {
  printf("=== prefill_attn validation (reduced dims) ===\n");
  // Reduced dims to keep CPU ref cheap, but exercise all the structure:
  //   keep HEAD_DIM=128 (RoPE/QK-norm depend on it) and the real head counts.
  // Use the real model shapes; M small.
  const int M = 8;
  const int hidden = HIDDEN, head_dim = HEAD_DIM;
  const int n_q = N_Q_HEADS, n_kv = N_KV_HEADS;
  const int q_dim = Q_DIM, kv_dim = KV_DIM, qkv_out = QKV_OUT;

  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 0.5f);
  std::uniform_real_distribution<float> ud(0.4f, 1.2f);

  std::vector<float> X((size_t)M*hidden);
  for (auto& v : X) v = nd(rng);
  std::vector<float> in_norm(hidden); for (auto& v : in_norm) v = ud(rng);
  std::vector<float> q_norm(head_dim); for (auto& v : q_norm) v = ud(rng);
  std::vector<float> k_norm(head_dim); for (auto& v : k_norm) v = ud(rng);

  // Wqkv [qkv_out, hidden], per-out-channel scale.
  std::vector<fp8> Wqkv((size_t)qkv_out*hidden);
  std::vector<float> Wqkv_scale(qkv_out);
  for (int o = 0; o < qkv_out; ++o) {
    float s = 0.02f + 0.01f*ud(rng);   // small scale so quantized values are sane
    Wqkv_scale[o] = s;
    for (int k = 0; k < hidden; ++k)
      Wqkv[(size_t)o*hidden+k] = to_fp8(nd(rng) * 0.05f / s); // encode raw/scale into fp8
  }
  // Wo [hidden, q_dim]
  std::vector<fp8> Wo((size_t)hidden*q_dim);
  std::vector<float> Wo_scale(hidden);
  for (int o = 0; o < hidden; ++o) {
    float s = 0.02f + 0.01f*ud(rng);
    Wo_scale[o] = s;
    for (int k = 0; k < q_dim; ++k)
      Wo[(size_t)o*q_dim+k] = to_fp8(nd(rng) * 0.05f / s);
  }

  // ---- CPU reference ----
  std::vector<float> ref;
  cpu_attn_ref(X, in_norm, Wqkv, Wqkv_scale, q_norm, k_norm, Wo, Wo_scale, ref,
               M, hidden, head_dim, n_q, n_kv, qkv_out, q_dim, kv_dim, ROPE_THETA, RMS_EPS);

  // ---- GPU ----
  float *dX = dev(X), *dN = dev(in_norm), *dQn = dev(q_norm), *dKn = dev(k_norm);
  fp8 *dWqkv = dev(Wqkv), *dWo = dev(Wo);
  float *dWqkvS = dev(Wqkv_scale), *dWoS = dev(Wo_scale);
  float *dXn, *dqkv, *dattn, *dout;
  CUDA_CHECK(cudaMalloc(&dXn,   (size_t)M*hidden*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dqkv,  (size_t)M*qkv_out*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dattn, (size_t)M*q_dim*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout,  (size_t)M*hidden*sizeof(float)));

  prefill_attn::run(dX, dN, dWqkv, dWqkvS, dQn, dKn, dWo, dWoS,
                    dXn, dqkv, dattn, dout, M);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> got((size_t)M*hidden);
  CUDA_CHECK(cudaMemcpy(got.data(), dout, got.size()*sizeof(float), cudaMemcpyDeviceToHost));

  double maxerr = 0, maxrel = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    double e = std::fabs((double)got[i] - (double)ref[i]);
    if (e > maxerr) maxerr = e;
    double denom = std::fabs((double)ref[i]) + 1e-3;
    if (e/denom > maxrel) maxrel = e/denom;
  }
  printf("  M=%d  max_abs_err=%.3e  max_rel_err=%.3e  -> %s (threshold 1e-2)\n",
         M, maxerr, maxrel, (maxerr < 1e-2) ? "PASS" : "FAIL");

  cudaFree(dX);cudaFree(dN);cudaFree(dQn);cudaFree(dKn);cudaFree(dWqkv);cudaFree(dWo);
  cudaFree(dWqkvS);cudaFree(dWoS);cudaFree(dXn);cudaFree(dqkv);cudaFree(dattn);cudaFree(dout);
}

static void microbench(int M) {
  printf("=== prefill_attn microbench (M=%d) ===\n", M);
  const int hidden = HIDDEN, q_dim = Q_DIM, qkv_out = QKV_OUT;
  // random device buffers (content irrelevant for timing)
  std::vector<fp8> Wqkv((size_t)qkv_out*hidden, fp8(0.01f));
  std::vector<float> Wqkv_scale(qkv_out, 0.02f);
  std::vector<fp8> Wo((size_t)hidden*q_dim, fp8(0.01f));
  std::vector<float> Wo_scale(hidden, 0.02f);
  std::vector<float> X((size_t)M*hidden, 0.01f);
  std::vector<float> in_norm(hidden,1.f), q_norm(HEAD_DIM,1.f), k_norm(HEAD_DIM,1.f);

  float *dX=dev(X), *dN=dev(in_norm), *dQn=dev(q_norm), *dKn=dev(k_norm);
  fp8 *dWqkv=dev(Wqkv), *dWo=dev(Wo);
  float *dWqkvS=dev(Wqkv_scale), *dWoS=dev(Wo_scale);
  float *dXn,*dqkv,*dattn,*dout;
  CUDA_CHECK(cudaMalloc(&dXn,(size_t)M*hidden*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dqkv,(size_t)M*qkv_out*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dattn,(size_t)M*q_dim*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout,(size_t)M*hidden*sizeof(float)));

  // warmup
  for (int i=0;i<3;++i)
    prefill_attn::run(dX,dN,dWqkv,dWqkvS,dQn,dKn,dWo,dWoS,dXn,dqkv,dattn,dout,M);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  const int iters = 20;
  CUDA_CHECK(cudaEventRecord(a));
  for (int i=0;i<iters;++i)
    prefill_attn::run(dX,dN,dWqkv,dWqkvS,dQn,dKn,dWo,dWoS,dXn,dqkv,dattn,dout,M);
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); ms/=iters;

  // FLOPs: 2 GEMMs + attention. (2*M*N*K each, 2 for MAC.)
  double f_qkv = 2.0*M*qkv_out*hidden;
  double f_o   = 2.0*M*hidden*q_dim;
  // attention: QK^T (M*M/2 *head_dim) + softmax*V (M*M/2 *head_dim) over n_q heads, *2 MAC
  double f_attn = 2.0 * ( (double)N_Q_HEADS * ((double)M*(M+1)/2.0) * HEAD_DIM * 2.0 );
  double flops = f_qkv + f_o + f_attn;
  double tflops = flops / (ms*1e-3) / 1e12;

  // H100 SXM bf16 (non-tensor-core CUDA-core FMA) peak ~ 67 TFLOP/s fp32 FMA *? .
  // We use a conservative bf16 CUDA-core reference. H100 fp32 FMA peak ~ 67 TFLOP/s.
  // (Tensor-core fp8 peak ~1.98 PFLOP/s; we are SIMT so report vs CUDA-core fp32.)
  const double H100_CUDACORE_FP32 = 67.0;     // TFLOP/s, SXM, boost
  const double H100_FP8_TC        = 1979.0;   // TFLOP/s tensor-core fp8 (dense)
  printf("  time/layer        : %.3f ms\n", ms);
  printf("  achieved          : %.2f TFLOP/s\n", tflops);
  printf("  %% of CUDA-core fp32 peak (~67 TF) : %.1f%%\n", 100.0*tflops/H100_CUDACORE_FP32);
  printf("  %% of fp8 TC peak  (~1.98 PF)      : %.2f%%\n", 100.0*tflops/H100_FP8_TC);
  printf("  TTFT contribution : %.3f ms (this layer) ; x%d layers = %.1f ms\n",
         ms, N_LAYERS, ms*N_LAYERS);

  cudaEventDestroy(a);cudaEventDestroy(b);
  cudaFree(dX);cudaFree(dN);cudaFree(dQn);cudaFree(dKn);cudaFree(dWqkv);cudaFree(dWo);
  cudaFree(dWqkvS);cudaFree(dWoS);cudaFree(dXn);cudaFree(dqkv);cudaFree(dattn);cudaFree(dout);
}

int main(int argc, char** argv) {
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
    fprintf(stderr, "No CUDA device available; this binary needs an H100 (sm_90a).\n");
    return 0;
  }
  validate();
  int M = (argc > 1) ? atoi(argv[1]) : 512;
  microbench(M);
  return 0;
}

#endif // PREFILL_ATTN_NO_MAIN
