// spec_verify_forward_bench.cu — PROVE (or DISPROVE) verify(k) ~= decode(1) on the H100, the right way.
//
// Target: Qwen3-235B-A22B, B=1 decode, H100 (sm_90a), single GPU. This is the companion to
// spec_verify_bench.cu and it exists to settle ONE question with the best available kernel:
//
//     Does processing M = (gamma+1) draft positions through the SAME weights cost ~the SAME
//     wall-clock as M=1 (the core spec assumption), or does it scale with M?
//
// WHY A SECOND BENCH:
//   spec_verify_bench.cu reuses the K5 *GEMV* idiom (warp-per-row, split-K-32-lanes, fp32 CUDA-core
//   FMA) and adds an M axis. On-box that path measured NEAR-LINEAR M scaling (M=3 -> 2.66x, M=5 ->
//   4.41x, M=8 -> 7.14x of the M=1 time): the weight tile is staged once, but the per-row arithmetic
//   runs on CUDA cores and is NOT hidden under the (already sub-roofline ~24% MBU) weight read, so it
//   serializes. That is the "per-row scaling" the eagle3-results-playbook flagged as a kernel artifact.
//   CONCLUSION FROM THAT BENCH: the B=1 GEMV kernel is the WRONG kernel for the verify.
//
//   why-spec-wins.md says exactly this: "the verify path wants the *batched* MoE kernel, not the B=1
//   GEMV K5 -- they're different kernels. K5 optimizes the draft and the no-spec fallback; the verify
//   uses the efficient grouped-GEMM path." This bench proves the verify-as-GEMM hits the amortization
//   the GEMV cannot, using cuBLAS (well-tuned tensor-core GEMM) as the achievable-kernel proxy.
//
// THE EXPERIMENT:
//   The verify forward streams a fixed set of weights from HBM ONCE and applies them to M activation
//   columns. We model the per-token weight volume of ONE decode forward as a single big GEMM:
//       Y[N, M] = W[N, K] @ X[K, M]
//   with W sized to the active per-token weight read of Qwen3-235B and X holding M verify columns.
//   On tensor cores the GEMM is HBM-bound for these tall-skinny shapes (K=4096, M<=8), so:
//       * if verify is truly weight-read-bound, time(M) ~= time(1)  (FLAT -> spec assumption HOLDS)
//       * if it is compute/per-column-bound, time(M) scales with M  (the GEMV failure mode)
//   We measure the curve directly. bf16 in/out, tensor-core math (cuBLAS picks the wgmma kernel).
//
// WEIGHT VOLUME modeled (one decode forward, single-GPU full model, bf16):
//   * MoE experts:  TOP_K=8 experts x (gate+up [2*MOE_INTER, HIDDEN] + down [HIDDEN, MOE_INTER]).
//                   We fold the whole active expert set into ONE [N_exp, HIDDEN] GEMM (the union the
//                   verify reads once). N_exp = TOP_K * (2*MOE_INTER + ... ) folded to a tall matrix.
//   * lm_head:      [VOCAB, HIDDEN].
//   Both are the M-independent reads; we time each as its own GEMM and also the attention QKV/O GEMMs.
//
// Build:
//   nvcc -arch=sm_90a -O3 --use_fast_math -I /root/e2e spec_verify_forward_bench.cu \
//        -lcublas -o /tmp/specfwd && CUDA_VISIBLE_DEVICES=0 /tmp/specfwd
//
// IP: public model shapes (common.cuh) + standard cuBLAS. Writes its own file; edits nothing else.
// ============================================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "common.cuh"
using namespace q3;

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  printf("cuBLAS err %s:%d: %d\n",__FILE__,__LINE__,(int)s_); exit(1);} } while(0)

// One GEMM weight panel: Y[N,M] = W[N,K] @ X[K,M], W stored bf16 row-major [N,K].
// Times the GEMM for M in {1,2,4,8} and reports time / GB/s / ratio-vs-M=1.
struct Panel { const char* name; int N; int K; };

int main(int argc, char** argv) {
  const double PEAK = (argc > 1) ? atof(argv[1]) : 3350.0;   // H100 HBM3 GB/s
  int dev=0; cudaDeviceProp prop;
  CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s  SMs=%d  HBM peak=%.0f GB/s  (bf16 tensor-core GEMM via cuBLAS)\n",
         prop.name, prop.multiProcessorCount, PEAK);
  printf("QUESTION: does a verify of M draft columns cost ~= M=1 (FLAT) on the RIGHT (GEMM) kernel?\n\n");

  cublasHandle_t h; CB(cublasCreate(&h));
  CB(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));

  // Per-token weight panels of ONE Qwen3-235B decode forward (single-GPU, bf16).
  // Expert union the verify reads once: TOP_K active experts, each gate+up [2*MOE_INTER, HIDDEN]
  // and down [HIDDEN, MOE_INTER]. We fold gate+up of all TOP_K experts into one tall [N,K] panel
  // (K=HIDDEN) and down into one [N,K] panel (K=MOE_INTER). lm_head is its own panel.
  std::vector<Panel> panels = {
    { "experts gate+up (8x[2*1536,4096])", TOP_K * 2 * MOE_INTER, HIDDEN },     // 24576 x 4096
    { "experts down    (8x[4096,1536])",   TOP_K * HIDDEN,        MOE_INTER },   // 32768 x 1536
    { "attn QKV        ([9216,4096])",     QKV_OUT,               HIDDEN },      //  9216 x 4096
    { "attn O          ([4096,8192])",     HIDDEN,                Q_DIM },       //  4096 x 8192
    { "lm_head         ([151936,4096])",   VOCAB,                 HIDDEN },      // big, ~1.2GB bf16
  };

  const int Ms[] = {1, 2, 4, 8};
  const int NM = 4;
  const int WARM = 15, IT = 60;
  cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
  const __nv_bfloat16 alpha = __float2bfloat16(1.0f), beta = __float2bfloat16(0.0f);

  // Accumulate a full per-forward step time at each M (sum of all panels; experts/attn x N_LAYERS + lm_head).
  double step_ms[16] = {0};
  printf("%-38s %3s %10s %10s %9s\n", "panel", "M", "us", "GB/s", "t/t(M=1)");

  for (auto& p : panels) {
    // alloc W[N,K] bf16, X[K,Mmax] bf16, Y[N,Mmax] bf16
    size_t wsz=(size_t)p.N*p.K, xsz=(size_t)p.K*8, ysz=(size_t)p.N*8;
    __nv_bfloat16 *W,*X,*Y;
    CK(cudaMalloc(&W, wsz*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&X, xsz*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&Y, ysz*sizeof(__nv_bfloat16)));
    CK(cudaMemset(W, 1, wsz*sizeof(__nv_bfloat16)));   // nonzero bytes; values irrelevant for timing
    CK(cudaMemset(X, 1, xsz*sizeof(__nv_bfloat16)));

    double t_m1 = 0;
    for (int mi=0; mi<NM; ++mi) {
      int M = Ms[mi];
      // Column-major cuBLAS: compute Y[N,M] = W[N,K] @ X[K,M].
      // We store W,X row-major; treat as col-major transposed: Y^T[M,N] = X^T[M,K] @ W^T[K,N].
      // Simplest: use cublasGemmEx with op_N on col-major interpretations that map to our row-major.
      // C[N,M] (col-major) = A[N,K] * B[K,M] with A=W (lda=N? ) -- to avoid confusion we compute
      // C = W * X where W is [N,K] row-major == [K,N] col-major (so opA=T gives [N,K]).
      auto run = [&](){
        // col-major: C(m=N rows? ) We want C[N,M]. Use: C(N,M) = op(A)[N,K] * op(B)[K,M].
        // A=W row-major[N,K] -> col-major[K,N], opA=CUBLAS_OP_T -> [N,K]. lda=K.
        // B=X row-major[K,M] -> col-major[M,K], opB=CUBLAS_OP_T -> [K,M]. ldb=M.
        // C col-major[N,M], ldc=N.
        CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_T,
                        p.N, M, p.K,
                        &alpha,
                        W, CUDA_R_16BF, p.K,
                        X, CUDA_R_16BF, M,
                        &beta,
                        Y, CUDA_R_16BF, p.N,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      };
      for (int i=0;i<WARM;i++) run();
      CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
      for (int i=0;i<IT;i++) run();
      CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
      float ms; CK(cudaEventElapsedTime(&ms,s,e)); ms/=IT;
      CK(cudaGetLastError());
      double bytes = (double)wsz*sizeof(__nv_bfloat16);   // weight read is the M-independent term
      if (M==1) t_m1 = ms;
      printf("%-38s %3d %10.2f %10.1f %9.3f\n", p.name, M, ms*1e3, bytes/1e6/ms, ms/t_m1);

      // build per-forward step time: experts+attn panels x N_LAYERS, lm_head x 1
      double mult = (p.name[0]=='l') ? 1.0 : (double)N_LAYERS;  // lm_head once, rest per layer
      step_ms[mi] += ms * mult;
    }
    printf("\n");
    cudaFree(W); cudaFree(X); cudaFree(Y);
  }

  // ===== projected single-GPU forward + spec multiplier =====
  printf("== modeled single-GPU forward/token (sum of panels x N_LAYERS + lm_head) ==\n");
  printf("%-6s %12s %12s %12s\n", "M", "fwd ms", "tok/s(plain)", "fwd/fwd(M=1)");
  for (int mi=0; mi<NM; ++mi)
    printf("%-6d %12.3f %12.1f %12.3f\n", Ms[mi], step_ms[mi], 1000.0/step_ms[mi], step_ms[mi]/step_ms[0]);

  printf("\n== verify(k) ~= decode(1) check ==\n");
  printf("   If fwd/fwd(M=1) stays ~1.0 as M grows, the verify is weight-read-bound -> spec assumption HOLDS.\n");
  printf("   E[accepted] = (1 - a^(g+1))/(1 - a); multiplier = E[acc] / (fwd(M)/fwd(M=1)).\n\n");
  const double ALPHAS[] = {0.7, 0.8};
  printf("   %-5s %-6s %-4s %10s %12s %12s\n", "a","gamma","M","E[acc]","fwd ratio","spec mult");
  for (double a : ALPHAS) {
    for (int mi=1; mi<NM; ++mi) {     // skip M=1 (no spec)
      int M = Ms[mi], gamma = M-1;
      double ea = (a>=1.0)? gamma+1.0 : (1.0-pow(a,gamma+1))/(1.0-a);
      double ratio = step_ms[mi]/step_ms[0];
      printf("   %-5.2f %-6d %-4d %10.3f %12.3f %12.3f\n", a, gamma, M, ea, ratio, ea/ratio);
    }
  }
  cublasDestroy(h);
  cudaEventDestroy(s); cudaEventDestroy(e);
  return 0;
}
