// tc_attn_probe.cu (v2) — does a TENSOR-CORE attention (QK^T + P.V as cuBLAS GEMMs) stay FLAT in draft
// width M, where warp-shuffle K2 scales ~4x@M=8? Proxy for a wgmma flash-decode verify. fp16 TC.
// Per head (batched over H): all matrices column-major.
//   GEMM1 scores S[M x ctx] = Q^T(M x d) * K(d x ctx)   [Q stored d x M, K stored d x ctx]
//   GEMM2 out    O[d x M]   = V(d x ctx) * S^T(ctx x M)  [V stored d x ctx]
// Softmax (cheap/~flat) omitted: isolate the GEMM M-scaling. H independent K/V (no GQA broadcast) ->
// absolutes overstate HBM but ratio_vs_M1 is clean (identical K/V read at every M).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "common.cuh"
using namespace q3;
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS err %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)

int main(int argc,char**argv){
  int ctx=(argc>1)?atoi(argv[1]):4096; int iters=(argc>2)?atoi(argv[2]):300;
  const int H=N_Q_HEADS, d=HEAD_DIM, MMAX=16;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("== TC attention (cuBLAS fp16 TC) flatness probe ==  %s SMs=%d  ctx=%d H=%d d=%d\n",p.name,p.multiProcessorCount,ctx,H,d);
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH));
  auto mk=[&](size_t n){ __half* dp; CK(cudaMalloc(&dp,n*sizeof(__half))); std::vector<__half> hb(n);
    for(size_t i=0;i<n;i++){ unsigned s=(unsigned)(i*2654435761u); hb[i]=__float2half(((int)((s>>10)&0x3ff)-512)/512.f*0.3f);} 
    CK(cudaMemcpy(dp,hb.data(),n*sizeof(__half),cudaMemcpyHostToDevice)); return dp; };
  __half *Q=mk((size_t)H*MMAX*d), *K=mk((size_t)H*ctx*d), *V=mk((size_t)H*ctx*d);
  __half *S=mk((size_t)H*MMAX*ctx), *O=mk((size_t)H*d*MMAX);
  const float a=1.f,b=0.f;
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  printf("\n  M    GEMM1+2 us   us/query   ratio_vs_M1\n  ------------------------------------------------\n");
  int Ms[]={1,4,8,16}; double us1=0;
  for(int M:Ms){
    auto run=[&](){
      // GEMM1: C[M x ctx] = op_T(Q[d x M]) * op_N(K[d x ctx]); m=M,n=ctx,k=d; lda=d,ldb=d,ldc=M
      CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N, M,ctx,d, &a,
        Q,CUDA_R_16F,d,(long long)MMAX*d, K,CUDA_R_16F,d,(long long)ctx*d, &b,
        S,CUDA_R_16F,M,(long long)MMAX*ctx, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      // GEMM2: C[d x M] = op_N(V[d x ctx]) * op_T(S[M x ctx]); m=d,n=M,k=ctx; lda=d,ldb=M,ldc=d
      CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_T, d,M,ctx, &a,
        V,CUDA_R_16F,d,(long long)ctx*d, S,CUDA_R_16F,M,(long long)MMAX*ctx, &b,
        O,CUDA_R_16F,d,(long long)d*MMAX, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    for(int w=0;w<30;w++) run(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0)); for(int i=0;i<iters;i++) run(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); double us=ms*1000.0/iters; if(M==1)us1=us;
    printf("  %-4d %10.2f   %8.2f   %.3f%s\n",M,us,us/M,us/us1,(M>1&&us/us1<1.5)?"  <- FLATTER than warp-shuffle(4x)":"");
  }
  printf("\n  ratio(M=8) << 4.0 (warp-shuffle K2) => TC flash-decode verify is worth building.\n");
  return 0;
}
