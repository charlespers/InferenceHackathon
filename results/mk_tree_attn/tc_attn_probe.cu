// tc_attn_probe.cu — FEASIBILITY: does a TENSOR-CORE attention (QK^T + P.V as cuBLAS GEMMs) stay FLAT
// in draft width M, where the warp-shuffle K2 scales ~4x@M=8? Proxy for a wgmma flash-decode verify.
// Models ONE layer's attention as 2 batched GEMMs over H heads, ctx KV rows, head_dim d. bf16 TC =
// the achievable-kernel proxy (same proxy Charles's spec_verify_forward_bench uses for weight panels).
//   GEMM1 (QK^T): per head [ctx x d]·[d x M] -> [ctx x M]   (K read once, M = free output cols)
//   GEMM2 (P.V) : per head [d x ctx]·[ctx x M] -> [d x M]
// Caveat: uses H independent K/V (no GQA broadcast) so ABSOLUTES overstate HBM, but the RATIO vs M=1
// is clean (same K/V read at every M). Softmax (cheap, ~flat) omitted — we isolate the GEMM M-scaling.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "common.cuh"
using namespace q3;
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS err %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)

int main(int argc,char**argv){
  int ctx=(argc>1)?atoi(argv[1]):4096; int iters=(argc>2)?atoi(argv[2]):300;
  const int H=N_Q_HEADS, d=HEAD_DIM, MMAX=16;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("== TC attention (cuBLAS bf16) flatness probe ==  %s SMs=%d  ctx=%d H=%d d=%d\n",p.name,p.multiProcessorCount,ctx,H,d);
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH));
  // buffers (bf16): K[H*ctx*d], V[H*ctx*d], Q[H*MMAX*d], S[H*ctx*MMAX], O[H*d*MMAX]
  auto mk=[&](size_t n){ __nv_bfloat16* p; CK(cudaMalloc(&p,n*sizeof(__nv_bfloat16))); std::vector<__nv_bfloat16> hb(n);
    for(size_t i=0;i<n;i++){ unsigned s=(unsigned)(i*2654435761u); hb[i]=__float2bfloat16(((int)((s>>10)&0x3ff)-512)/512.f*0.3f);} 
    CK(cudaMemcpy(p,hb.data(),n*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice)); return p; };
  __nv_bfloat16 *K=mk((size_t)H*ctx*d), *V=mk((size_t)H*ctx*d), *Q=mk((size_t)H*MMAX*d);
  __nv_bfloat16 *S=mk((size_t)H*ctx*MMAX), *O=mk((size_t)H*d*MMAX);
  const float a=1.f,b=0.f;
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  printf("\n  M    GEMM1+2 us   us/query   ratio_vs_M1\n  ------------------------------------------------\n");
  int Ms[]={1,4,8,16}; double us1=0;
  for(int M:Ms){
    auto run=[&](){
      // GEMM1: S[ctx x M] = K[ctx x d] * Q[M x d]^T  per head.  (col-major: m=ctx,n=M,k=d; A=K (ctx x d, lda=ctx? )
      // Use op: C(ctx x M) = K(ctx x d) * Qt(d x M). Store K col-major as [d x ctx] so K^T=(ctx x d). Simpler:
      // treat all as col-major raw; compute C = A^T * B with A=[d x ctx](=K rows), B=[d x M](=Q rows).
      CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N, ctx,M,d, &a,
        K,CUDA_R_16BF,d,(long long)ctx*d, Q,CUDA_R_16BF,d,(long long)MMAX*d, &b,
        S,CUDA_R_16BF,ctx,(long long)ctx*MMAX, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      // GEMM2: O[d x M] = V[d x ctx] * S[ctx x M].  A=V as [ctx x d] -> op_T gives [d x ctx]; B=S[ctx x M].
      CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N, d,M,ctx, &a,
        V,CUDA_R_16BF,d,(long long)ctx*d, S,CUDA_R_16BF,ctx,(long long)ctx*MMAX, &b,
        O,CUDA_R_16BF,d,(long long)d*MMAX, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    for(int w=0;w<30;w++) run(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0)); for(int i=0;i<iters;i++) run(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); double us=ms*1000.0/iters; if(M==1)us1=us;
    printf("  %-4d %10.2f   %8.2f   %.3f%s\n",M,us,us/M,us/us1,(M>1&&us/us1<1.5)?"  <- FLATTER than warp-shuffle(4x)":"");
  }
  printf("\n  If ratio(M=8) << 4.0 (the warp-shuffle K2), a TC flash-decode verify is worth building.\n");
  return 0;
}
