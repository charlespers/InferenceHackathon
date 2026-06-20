// tc_attn_full.cu — FULL TC attention (QK^T -> scaled softmax -> P.V), validated vs CPU fp32, M-flatness.
// Removes the "softmax omitted" caveat of tc_attn_probe. cuBLAS fp16 GEMMs + a custom warp softmax over ctx.
// Per head (batched H), column-major:
//   GEMM1 S[M x ctx] = (1/sqrt(d)) * Q^T(M x d) * K(d x ctx)   [Q stored d x M, K stored d x ctx], ldc=M
//   SOFTMAX over ctx (dim n) per (head,row m): element (m,t) at base_h + t*M + m, stride M
//   GEMM2 O[d x M] = V(d x ctx) * P^T(ctx x M)                  [V stored d x ctx], ldb=M, ldc=d
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "common.cuh"
using namespace q3;
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS err %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)

// one warp per (head,row m): softmax over ctx at stride M within per-head block (stride MMAX*ctx).
__global__ void softmax_cols(__half* S, int H, int M, int MMAX, int ctx){
  int gw = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (gw >= H*M) return;
  int h = gw / M, m = gw % M;
  size_t base = (size_t)h*MMAX*ctx + m;
  float mx=-FLT_MAX;
  for (int t=lane; t<ctx; t+=32) mx=fmaxf(mx,(float)S[base+(size_t)t*M]);
  #pragma unroll
  for (int o=16;o>0;o>>=1) mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,o));
  float sm=0.f;
  for (int t=lane; t<ctx; t+=32){ float e=__expf((float)S[base+(size_t)t*M]-mx); S[base+(size_t)t*M]=__float2half(e); sm+=e; }
  #pragma unroll
  for (int o=16;o>0;o>>=1) sm+=__shfl_xor_sync(0xffffffffu,sm,o);
  float inv = sm>0.f?1.f/sm:0.f;
  for (int t=lane; t<ctx; t+=32) S[base+(size_t)t*M]=__float2half((float)S[base+(size_t)t*M]*inv);
}

int main(int argc,char**argv){
  int ctx=(argc>1)?atoi(argv[1]):4096; int iters=(argc>2)?atoi(argv[2]):300;
  const int H=N_Q_HEADS, d=HEAD_DIM, MMAX=16;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("== FULL TC attention (QK^T->softmax->P.V) ==  %s ctx=%d H=%d d=%d\n",p.name,ctx,H,d);
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH));
  std::vector<float> rQ((size_t)H*MMAX*d), rK((size_t)H*ctx*d), rV((size_t)H*ctx*d); // fp32 ref copies
  auto mk=[&](size_t n, std::vector<float>* ref, unsigned seed){ __half* dp; CK(cudaMalloc(&dp,n*sizeof(__half)));
    std::vector<__half> hb(n); for(size_t i=0;i<n;i++){ seed=seed*1664525u+1013904223u; float v=((int)((seed>>9)&0x3ff)-512)/512.f*0.3f; hb[i]=__float2half(v); if(ref)(*ref)[i]=(float)hb[i]; }
    CK(cudaMemcpy(dp,hb.data(),n*sizeof(__half),cudaMemcpyHostToDevice)); return dp; };
  __half *Q=mk((size_t)H*MMAX*d,&rQ,1), *K=mk((size_t)H*ctx*d,&rK,7), *V=mk((size_t)H*ctx*d,&rV,13);
  __half *S=mk((size_t)H*MMAX*ctx,nullptr,3), *O=mk((size_t)H*d*MMAX,nullptr,5);
  const float scale=1.f/sqrtf((float)d), one=1.f, zero=0.f;
  auto run=[&](int M){
    CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N, M,ctx,d, &scale,
      Q,CUDA_R_16F,d,(long long)MMAX*d, K,CUDA_R_16F,d,(long long)ctx*d, &zero,
      S,CUDA_R_16F,M,(long long)MMAX*ctx, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    int warps=H*M, blk=128, grid=(warps*32+blk-1)/blk;
    softmax_cols<<<grid,blk>>>(S,H,M,MMAX,ctx);
    CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_T, d,M,ctx, &one,
      V,CUDA_R_16F,d,(long long)ctx*d, S,CUDA_R_16F,M,(long long)MMAX*ctx, &zero,
      O,CUDA_R_16F,d,(long long)d*MMAX, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  // correctness vs CPU fp32 (M=1, 3 heads)
  run(1); CK(cudaDeviceSynchronize());
  std::vector<__half> hO((size_t)H*d*MMAX); CK(cudaMemcpy(hO.data(),O,hO.size()*sizeof(__half),cudaMemcpyDeviceToHost));
  double maxerr=0,maxref=0;
  for(int qh:{0,H/2,H-1}){
    std::vector<float> lg(ctx); float mx=-FLT_MAX;
    for(int t=0;t<ctx;t++){ double dp=0; for(int c=0;c<d;c++) dp+=rQ[(size_t)qh*MMAX*d + 0*d + c]*rK[((size_t)qh*ctx+t)*d+c]; lg[t]=(float)dp*scale; mx=fmaxf(mx,lg[t]); }
    double den=0; std::vector<double> w(ctx); for(int t=0;t<ctx;t++){ w[t]=exp((double)(lg[t]-mx)); den+=w[t]; }
    for(int c=0;c<d;c++){ double o=0; for(int t=0;t<ctx;t++) o+=w[t]*rV[((size_t)qh*ctx+t)*d+c]; o/=den;
      double got=(float)hO[(size_t)qh*d*MMAX + 0*d + c]; maxerr=fmax(maxerr,fabs(o-got)); maxref=fmax(maxref,fabs(o)); } }
  printf("correctness M=1 (3 heads): max_abs_err=%.3e  global_rel=%.2f%%  -> %s\n", maxerr, 100.0*maxerr/maxref, (maxerr/maxref<0.05)?"PASS":"FAIL");
  // flatness
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  printf("\n  M    full-attn us   us/query   ratio_vs_M1\n  ------------------------------------------------\n");
  int Ms[]={1,4,8,16}; double us1=0;
  for(int M:Ms){ for(int w=0;w<30;w++) run(M); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0)); for(int i=0;i<iters;i++) run(M); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); double us=ms*1000.0/iters; if(M==1)us1=us;
    printf("  %-4d %10.2f   %8.2f   %.3f%s\n",M,us,us/M,us/us1,(M>1&&us/us1<1.5)?"  <- FLAT":""); }
  printf("\n  (full attention incl. softmax — vs warp-shuffle k2 ~4x@M=8)\n");
  return 0;
}
