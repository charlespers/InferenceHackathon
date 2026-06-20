// prefill_moe.cu — Qwen3-235B-A22B PREFILL MoE path (sm_90a / H100).
//
// PREFILL: M tokens (e.g. 512) each routed to TOP_K=8 of N_EXPERTS=128 experts.
// This is a GEMM workload (per expert: [tokens_e x HIDDEN] @ weights), compute
// heavy, so we use shared-memory tiling + register accumulation.
//
// Per expert e, for the set of tokens routed to it:
//   gate = Xe[t,:] @ Wgate[e]^T   ([n_tok, HIDDEN] @ [HIDDEN, MOE_INTER])
//   up   = Xe[t,:] @ Wup[e]^T
//   h    = SiLU(gate) * up                                   (SwiGLU, [n_tok, MOE_INTER])
//   down = h @ Wdown[e]^T          ([n_tok, MOE_INTER] @ [MOE_INTER, HIDDEN])
//   residual[token] += routing_weight[token,e] * down        (scatter-add)
//
// Weight storage convention (matches common.cuh::Fp8Weight): logical [OUT, IN]
// row-major with IN contiguous, per-output-channel scale[OUT]. So:
//   Wgate[e],Wup[e] : [MOE_INTER, HIDDEN]   (gate|up stacked: [2*MOE_INTER, HIDDEN])
//   Wdown[e]        : [HIDDEN, MOE_INTER]
// and Y[t,n] = sum_k X[t,k] * deq(W[n,k], scale[n]).
//
// fp8 e4m3 weights, per-output-channel scales, dequant -> fp32 GEMM.
// Tensor-core/wgmma intentionally NOT used (portable SIMT). We report % of bf16
// CUDA-core peak.
//
// Build:
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -c kernels/prefill_moe.cu
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/prefill_moe.cu -o /tmp/kp
//
#include "common.cuh"
using namespace q3;

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

__device__ __forceinline__ float deqf(fp8 v, float s) {
  return static_cast<float>(v) * s;
}

// ---------------------------------------------------------------------------
//  Tile params (same family as prefill_attn).
//  64x64 output tile, BK=16, 8x8 register micro-tile, 64 threads/block.
// ---------------------------------------------------------------------------
#define MBM 64
#define MBN 64
#define MBK 16
#define MTM 8
#define MTN 8

// ---------------------------------------------------------------------------
//  Fused gate+up GEMM with SiLU:  H[T, MOE_INTER] for ONE expert.
//  Xe : [T, HIDDEN] gathered tokens for this expert.
//  Wgate,Wup : [MOE_INTER, HIDDEN] row-major, scales [MOE_INTER].
//  H[t,n] = SiLU( sum_k Xe[t,k]*deq(Wgate[n,k]) ) * ( sum_k Xe[t,k]*deq(Wup[n,k]) )
//  Output tile is over (T rows, MOE_INTER cols). We compute gate and up in the
//  same K loop using two accumulators, then fuse the SiLU*mul in the epilogue.
// ---------------------------------------------------------------------------
__global__ void moe_gate_up_silu(
    const float* __restrict__ Xe,        // [T, HIDDEN]
    const fp8*   __restrict__ Wgate, const float* __restrict__ Sg,  // [MOE_INTER, HIDDEN]
    const fp8*   __restrict__ Wup,   const float* __restrict__ Su,  // [MOE_INTER, HIDDEN]
    float*       __restrict__ H,         // [T, MOE_INTER]
    int T) {
  const int K = HIDDEN, N = MOE_INTER;
  __shared__ float As[MBK][MBM];
  __shared__ float Bg[MBK][MBN];
  __shared__ float Bu[MBK][MBN];

  const int tid = threadIdx.x;
  const int threadRow = tid / (MBN / MTN);   // 0..7
  const int threadCol = tid % (MBN / MTN);   // 0..7
  const int blockRow = blockIdx.y * MBM;
  const int blockCol = blockIdx.x * MBN;

  float accG[MTM][MTN], accU[MTM][MTN];
  #pragma unroll
  for (int i=0;i<MTM;++i)
    #pragma unroll
    for (int j=0;j<MTN;++j){ accG[i][j]=0.f; accU[i][j]=0.f; }

  for (int k0 = 0; k0 < K; k0 += MBK) {
    #pragma unroll
    for (int e=0;e<(MBM*MBK)/64;++e){
      int idx=tid+e*64; int r=idx/MBK, c=idx%MBK;
      int gm=blockRow+r, gk=k0+c;
      As[c][r] = (gm<T && gk<K) ? Xe[(size_t)gm*K+gk] : 0.f;
    }
    #pragma unroll
    for (int e=0;e<(MBN*MBK)/64;++e){
      int idx=tid+e*64; int r=idx/MBK, c=idx%MBK;
      int gn=blockCol+r, gk=k0+c;
      if (gn<N && gk<K){
        Bg[c][r]=deqf(Wgate[(size_t)gn*K+gk], Sg[gn]);
        Bu[c][r]=deqf(Wup[(size_t)gn*K+gk],   Su[gn]);
      } else { Bg[c][r]=0.f; Bu[c][r]=0.f; }
    }
    __syncthreads();
    #pragma unroll
    for (int kk=0; kk<MBK; ++kk){
      float aR[MTM], gR[MTN], uR[MTN];
      #pragma unroll
      for (int i=0;i<MTM;++i) aR[i]=As[kk][threadRow*MTM+i];
      #pragma unroll
      for (int j=0;j<MTN;++j){ gR[j]=Bg[kk][threadCol*MTN+j]; uR[j]=Bu[kk][threadCol*MTN+j]; }
      #pragma unroll
      for (int i=0;i<MTM;++i)
        #pragma unroll
        for (int j=0;j<MTN;++j){ accG[i][j]+=aR[i]*gR[j]; accU[i][j]+=aR[i]*uR[j]; }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i=0;i<MTM;++i){
    int gm=blockRow+threadRow*MTM+i; if (gm>=T) continue;
    #pragma unroll
    for (int j=0;j<MTN;++j){
      int gn=blockCol+threadCol*MTN+j; if (gn>=N) continue;
      H[(size_t)gm*N+gn] = silu(accG[i][j]) * accU[i][j];
    }
  }
}

// ---------------------------------------------------------------------------
//  Down GEMM with routing-weight scatter-add into the residual.
//  H : [T, MOE_INTER] (silu(gate)*up), Wdown : [HIDDEN, MOE_INTER], scale [HIDDEN].
//  out[t,n] = sum_k H[t,k]*deq(Wdown[n,k]).
//  We then scatter-add  residual[token_id[t]] += rweight[t] * out[t,:].
//    token_id : [T]  -> global token row in the residual buffer
//    rweight  : [T]  -> routing weight (gate prob) for that (token,expert)
//  atomicAdd handles the case where multiple experts (across separate launches)
//  write the same token row.
// ---------------------------------------------------------------------------
__global__ void moe_down_scatter(
    const float* __restrict__ H,         // [T, MOE_INTER]
    const fp8*   __restrict__ Wdown, const float* __restrict__ Sd, // [HIDDEN, MOE_INTER]
    const int*   __restrict__ token_id,  // [T]
    const float* __restrict__ rweight,   // [T]
    float*       __restrict__ residual,  // [M, HIDDEN]
    int T) {
  const int K = MOE_INTER, N = HIDDEN;
  __shared__ float As[MBK][MBM];
  __shared__ float Bs[MBK][MBN];

  const int tid = threadIdx.x;
  const int threadRow = tid / (MBN / MTN);
  const int threadCol = tid % (MBN / MTN);
  const int blockRow = blockIdx.y * MBM;
  const int blockCol = blockIdx.x * MBN;

  float acc[MTM][MTN];
  #pragma unroll
  for (int i=0;i<MTM;++i)
    #pragma unroll
    for (int j=0;j<MTN;++j) acc[i][j]=0.f;

  for (int k0=0;k0<K;k0+=MBK){
    #pragma unroll
    for (int e=0;e<(MBM*MBK)/64;++e){
      int idx=tid+e*64; int r=idx/MBK, c=idx%MBK;
      int gm=blockRow+r, gk=k0+c;
      As[c][r]=(gm<T && gk<K)? H[(size_t)gm*K+gk] : 0.f;
    }
    #pragma unroll
    for (int e=0;e<(MBN*MBK)/64;++e){
      int idx=tid+e*64; int r=idx/MBK, c=idx%MBK;
      int gn=blockCol+r, gk=k0+c;
      Bs[c][r]=(gn<N && gk<K)? deqf(Wdown[(size_t)gn*K+gk], Sd[gn]) : 0.f;
    }
    __syncthreads();
    #pragma unroll
    for (int kk=0;kk<MBK;++kk){
      float aR[MTM], bR[MTN];
      #pragma unroll
      for (int i=0;i<MTM;++i) aR[i]=As[kk][threadRow*MTM+i];
      #pragma unroll
      for (int j=0;j<MTN;++j) bR[j]=Bs[kk][threadCol*MTN+j];
      #pragma unroll
      for (int i=0;i<MTM;++i)
        #pragma unroll
        for (int j=0;j<MTN;++j) acc[i][j]+=aR[i]*bR[j];
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i=0;i<MTM;++i){
    int gm=blockRow+threadRow*MTM+i; if (gm>=T) continue;
    int tok = token_id[gm];
    float w = rweight[gm];
    float* rrow = residual + (size_t)tok * N;
    #pragma unroll
    for (int j=0;j<MTN;++j){
      int gn=blockCol+threadCol*MTN+j; if (gn>=N) continue;
      atomicAdd(&rrow[gn], w * acc[i][j]);
    }
  }
}

// ---------------------------------------------------------------------------
//  Gather kernel: build Xe[T,HIDDEN] from the global activation X[M,HIDDEN]
//  using token_id[T]. (Routing/sorting is done host-side or by a separate
//  router kernel; here we just gather the rows for one expert's batch.)
// ---------------------------------------------------------------------------
__global__ void moe_gather(
    const float* __restrict__ X,         // [M, HIDDEN]
    const int*   __restrict__ token_id,  // [T]
    float*       __restrict__ Xe,        // [T, HIDDEN]
    int T) {
  int t = blockIdx.x;
  if (t >= T) return;
  int tok = token_id[t];
  const float* src = X + (size_t)tok * HIDDEN;
  float* dst = Xe + (size_t)t * HIDDEN;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) dst[i] = src[i];
}

// ===========================================================================
//  Launch helper for ONE expert's token batch.
//  Caller provides, per expert e: the gathered/contiguous token rows.
//  Buffers: Xe[T*HIDDEN], H[T*MOE_INTER] scratch.
// ===========================================================================
namespace prefill_moe {

inline dim3 grid2d(int M, int N){ return dim3((N+MBN-1)/MBN, (M+MBM-1)/MBM); }

// Run one expert over its T tokens. token_id/rweight are [T] device arrays mapping
// local row -> global token + routing weight. residual is [M, HIDDEN].
void run_expert(
    const float* d_X,                    // [M, HIDDEN] activations to gather from
    const int* d_token_id,               // [T]
    const float* d_rweight,              // [T]
    const fp8* d_Wgate, const float* d_Sg,   // [MOE_INTER, HIDDEN]
    const fp8* d_Wup,   const float* d_Su,   // [MOE_INTER, HIDDEN]
    const fp8* d_Wdown, const float* d_Sd,   // [HIDDEN, MOE_INTER]
    float* d_Xe,                         // [T, HIDDEN] scratch
    float* d_H,                          // [T, MOE_INTER] scratch
    float* d_residual,                   // [M, HIDDEN] accumulate target
    int T, cudaStream_t stream = 0) {
  if (T <= 0) return;
  moe_gather<<<T, 256, 0, stream>>>(d_X, d_token_id, d_Xe, T);
  moe_gate_up_silu<<<grid2d(T, MOE_INTER), 64, 0, stream>>>(
      d_Xe, d_Wgate, d_Sg, d_Wup, d_Su, d_H, T);
  moe_down_scatter<<<grid2d(T, HIDDEN), 64, 0, stream>>>(
      d_H, d_Wdown, d_Sd, d_token_id, d_rweight, d_residual, T);
}

} // namespace prefill_moe

// ===========================================================================
//  CPU fp32 reference + GPU validation + microbench
// ===========================================================================
#ifndef PREFILL_MOE_NO_MAIN

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
  exit(1); } } while(0)

static fp8 to_fp8(float x){ return fp8(x); }
static float from_fp8(fp8 x){ return float(x); }

template <class T>
static T* dev(const std::vector<T>& h){
  T* p; CUDA_CHECK(cudaMalloc(&p,h.size()*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));
  return p;
}

// CPU reference: process one expert's token batch into the residual.
static void cpu_expert_ref(
    const std::vector<float>& X, const std::vector<int>& token_id,
    const std::vector<float>& rweight,
    const std::vector<fp8>& Wg, const std::vector<float>& Sg,
    const std::vector<fp8>& Wu, const std::vector<float>& Su,
    const std::vector<fp8>& Wd, const std::vector<float>& Sd,
    std::vector<float>& residual,        // [M, HIDDEN] accumulate
    int T, int hidden, int inter) {
  for (int t=0;t<T;++t){
    int tok = token_id[t];
    const float* xr = &X[(size_t)tok*hidden];
    std::vector<float> h(inter);
    for (int n=0;n<inter;++n){
      double g=0,u=0;
      for (int k=0;k<hidden;++k){
        g += (double)xr[k]*(double)(from_fp8(Wg[(size_t)n*hidden+k])*Sg[n]);
        u += (double)xr[k]*(double)(from_fp8(Wu[(size_t)n*hidden+k])*Su[n]);
      }
      float gg=(float)g; float s=gg/(1.f+std::exp(-gg));
      h[n]=s*(float)u;
    }
    for (int n=0;n<hidden;++n){
      double d=0;
      for (int k=0;k<inter;++k) d += (double)h[k]*(double)(from_fp8(Wd[(size_t)n*inter+k])*Sd[n]);
      residual[(size_t)tok*hidden+n] += rweight[t]*(float)d;
    }
  }
}

static void validate() {
  printf("=== prefill_moe validation (reduced dims) ===\n");
  // Use real HIDDEN/MOE_INTER, small M and a couple of experts to keep CPU cheap.
  const int M = 8, hidden = HIDDEN, inter = MOE_INTER;
  const int n_test_experts = 3;

  std::mt19937 rng(777);
  std::normal_distribution<float> nd(0.f,0.5f);
  std::uniform_real_distribution<float> ud(0.4f,1.2f);

  std::vector<float> X((size_t)M*hidden);
  for (auto& v:X) v=nd(rng)*0.3f;

  // Per expert: gate/up [inter,hidden], down [hidden,inter]
  auto mk_w = [&](int out,int in, std::vector<fp8>& W, std::vector<float>& S){
    W.resize((size_t)out*in); S.resize(out);
    for (int o=0;o<out;++o){ float s=0.02f+0.01f*ud(rng); S[o]=s;
      for (int k=0;k<in;++k) W[(size_t)o*in+k]=to_fp8(nd(rng)*0.05f/s); }
  };
  std::vector<std::vector<fp8>> Wg(n_test_experts),Wu(n_test_experts),Wd(n_test_experts);
  std::vector<std::vector<float>> Sg(n_test_experts),Su(n_test_experts),Sd(n_test_experts);
  for (int e=0;e<n_test_experts;++e){
    mk_w(inter,hidden,Wg[e],Sg[e]); mk_w(inter,hidden,Wu[e],Su[e]); mk_w(hidden,inter,Wd[e],Sd[e]);
  }

  // Build a simple routing: each token -> 2 of the test experts with random weights.
  // We accumulate everything into one residual, both CPU and GPU.
  // token_id/rweight per expert.
  std::vector<std::vector<int>> tid(n_test_experts);
  std::vector<std::vector<float>> rw(n_test_experts);
  std::uniform_real_distribution<float> wd(0.1f,0.9f);
  for (int m=0;m<M;++m){
    int e0=m%n_test_experts, e1=(m+1)%n_test_experts;
    tid[e0].push_back(m); rw[e0].push_back(wd(rng));
    tid[e1].push_back(m); rw[e1].push_back(wd(rng));
  }

  // ---- CPU ref ----
  std::vector<float> ref((size_t)M*hidden, 0.f);
  for (int e=0;e<n_test_experts;++e)
    cpu_expert_ref(X,tid[e],rw[e],Wg[e],Sg[e],Wu[e],Su[e],Wd[e],Sd[e],ref,
                   (int)tid[e].size(),hidden,inter);

  // ---- GPU ----
  float* dX=dev(X);
  float* dres; CUDA_CHECK(cudaMalloc(&dres,(size_t)M*hidden*sizeof(float)));
  CUDA_CHECK(cudaMemset(dres,0,(size_t)M*hidden*sizeof(float)));
  // scratch sized to the max T across experts
  int maxT=0; for (int e=0;e<n_test_experts;++e) maxT=std::max(maxT,(int)tid[e].size());
  float* dXe; CUDA_CHECK(cudaMalloc(&dXe,(size_t)maxT*hidden*sizeof(float)));
  float* dH;  CUDA_CHECK(cudaMalloc(&dH,(size_t)maxT*inter*sizeof(float)));

  for (int e=0;e<n_test_experts;++e){
    int T=(int)tid[e].size();
    int* dtid=dev(tid[e]); float* drw=dev(rw[e]);
    fp8 *dWg=dev(Wg[e]),*dWu=dev(Wu[e]),*dWd=dev(Wd[e]);
    float *dSg=dev(Sg[e]),*dSu=dev(Su[e]),*dSd=dev(Sd[e]);
    prefill_moe::run_expert(dX,dtid,drw,dWg,dSg,dWu,dSu,dWd,dSd,dXe,dH,dres,T);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dtid);cudaFree(drw);cudaFree(dWg);cudaFree(dWu);cudaFree(dWd);
    cudaFree(dSg);cudaFree(dSu);cudaFree(dSd);
  }

  std::vector<float> got((size_t)M*hidden);
  CUDA_CHECK(cudaMemcpy(got.data(),dres,got.size()*sizeof(float),cudaMemcpyDeviceToHost));

  double maxerr=0,maxrel=0;
  for (size_t i=0;i<got.size();++i){
    double e=std::fabs((double)got[i]-(double)ref[i]);
    if (e>maxerr) maxerr=e;
    double den=std::fabs((double)ref[i])+1e-3;
    if (e/den>maxrel) maxrel=e/den;
  }
  printf("  M=%d experts=%d  max_abs_err=%.3e  max_rel_err=%.3e  -> %s (threshold 1e-2)\n",
         M,n_test_experts,maxerr,maxrel,(maxerr<1e-2)?"PASS":"FAIL");

  cudaFree(dX);cudaFree(dres);cudaFree(dXe);cudaFree(dH);
}

static void microbench(int M) {
  printf("=== prefill_moe microbench (M=%d, full MoE layer) ===\n", M);
  const int hidden=HIDDEN, inter=MOE_INTER;
  // Full MoE layer = M tokens * TOP_K=8 expert evaluations. With balanced routing,
  // total expert-token rows = M*TOP_K, spread over N_EXPERTS=128 experts. For timing
  // we just simulate a balanced load: each expert gets T = M*TOP_K/N_EXPERTS rows.
  int total_rows = M * TOP_K;
  int per_expert = std::max(1, total_rows / N_EXPERTS);
  int active_experts = N_EXPERTS;        // assume all experts hit at M=512

  // single set of weights reused for all experts (timing only)
  std::vector<fp8> Wg((size_t)inter*hidden, fp8(0.01f));
  std::vector<float> Sg(inter,0.02f);
  std::vector<fp8> Wu((size_t)inter*hidden, fp8(0.01f));
  std::vector<float> Su(inter,0.02f);
  std::vector<fp8> Wd((size_t)hidden*inter, fp8(0.01f));
  std::vector<float> Sd(hidden,0.02f);
  std::vector<float> X((size_t)M*hidden,0.01f);
  std::vector<int> tid(per_expert); for (int i=0;i<per_expert;++i) tid[i]=i%M;
  std::vector<float> rw(per_expert,0.125f);

  float* dX=dev(X);
  float* dres; CUDA_CHECK(cudaMalloc(&dres,(size_t)M*hidden*sizeof(float)));
  CUDA_CHECK(cudaMemset(dres,0,(size_t)M*hidden*sizeof(float)));
  int* dtid=dev(tid); float* drw=dev(rw);
  fp8 *dWg=dev(Wg),*dWu=dev(Wu),*dWd=dev(Wd);
  float *dSg=dev(Sg),*dSu=dev(Su),*dSd=dev(Sd);
  float* dXe; CUDA_CHECK(cudaMalloc(&dXe,(size_t)per_expert*hidden*sizeof(float)));
  float* dH;  CUDA_CHECK(cudaMalloc(&dH,(size_t)per_expert*inter*sizeof(float)));

  auto run_layer=[&](){
    for (int e=0;e<active_experts;++e)
      prefill_moe::run_expert(dX,dtid,drw,dWg,dSg,dWu,dSu,dWd,dSd,dXe,dH,dres,per_expert);
  };
  for (int i=0;i<2;++i) run_layer();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  const int iters=5;
  CUDA_CHECK(cudaEventRecord(a));
  for (int i=0;i<iters;++i) run_layer();
  CUDA_CHECK(cudaEventRecord(b));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); ms/=iters;

  // FLOPs per token (top-8): gate+up = 2 * 2*HIDDEN*MOE_INTER, down = 2*MOE_INTER*HIDDEN.
  // total per (token,expert) = 2*(2*H*I) + 2*(I*H) = 6*H*I  (2 for MAC counted: gate 2*H*I*2? )
  // Be explicit: gate GEMM 2*H*I flops, up GEMM 2*H*I, down GEMM 2*I*H -> 6*H*I per row.
  double flops = (double)total_rows * 6.0 * (double)hidden * (double)inter;
  double tflops = flops/(ms*1e-3)/1e12;

  const double H100_CUDACORE_FP32 = 67.0;
  const double H100_FP8_TC        = 1979.0;
  printf("  active experts    : %d, rows/expert : %d, total rows : %d\n",
         active_experts, per_expert, total_rows);
  printf("  time/MoE-layer    : %.3f ms\n", ms);
  printf("  achieved          : %.2f TFLOP/s\n", tflops);
  printf("  %% of CUDA-core fp32 peak (~67 TF) : %.1f%%\n", 100.0*tflops/H100_CUDACORE_FP32);
  printf("  %% of fp8 TC peak  (~1.98 PF)      : %.2f%%\n", 100.0*tflops/H100_FP8_TC);
  printf("  TTFT contribution : %.3f ms (this layer) ; x%d layers = %.1f ms\n",
         ms, N_LAYERS, ms*N_LAYERS);

  cudaEventDestroy(a);cudaEventDestroy(b);
  cudaFree(dX);cudaFree(dres);cudaFree(dtid);cudaFree(drw);
  cudaFree(dWg);cudaFree(dWu);cudaFree(dWd);cudaFree(dSg);cudaFree(dSu);cudaFree(dSd);
  cudaFree(dXe);cudaFree(dH);
}

int main(int argc, char** argv){
  int dev_count=0;
  if (cudaGetDeviceCount(&dev_count)!=cudaSuccess || dev_count==0){
    fprintf(stderr,"No CUDA device available; this binary needs an H100 (sm_90a).\n");
    return 0;
  }
  validate();
  int M=(argc>1)?atoi(argv[1]):512;
  microbench(M);
  return 0;
}

#endif // PREFILL_MOE_NO_MAIN
