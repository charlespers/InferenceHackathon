// tc_verify_attn.cu — COMPLETE M=k spec-verify attention, validated + M-flatness.
// Decomposition (proven): per draft query m attends [shared context KV [0,ctx)] U [draft KV 0..m] (CHAIN/causal).
//   (A) CONTEXT: cuBLAS TC GEMM (flat in M) -> normalized O_ctx + per-(h,m) softmax stats (mx_ctx, sm_ctx).
//   (B) DRAFT-SELF: tiny warp kernel, query m over draft K/V 0..m (causal) -> normalized O_d + (mx_d, sm_d).
//   MERGE: online-softmax combine of the two partials -> O[h][m][d].
// Validated vs CPU fp32 (full masked attention). M=1 reduces to context-only (no preceding drafts).
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
#define CB(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)
constexpr int VPL = HEAD_DIM/32;  // 4 channels per lane

// softmax over ctx (contiguous, scores [ctx x M] per head); emit P (in place) + mx,sm per (h,m).
__global__ void softmax_ctx(__half* S, float* mxo, float* smo, int H,int M,int MMAX,int ctx){
  int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(gw>=H*M)return;
  int hh=gw/M,m=gw%M; size_t base=(size_t)hh*MMAX*ctx+(size_t)m*ctx;
  float mx=-FLT_MAX; for(int t=lane;t<ctx;t+=32) mx=fmaxf(mx,(float)S[base+t]);
  #pragma unroll
  for(int o=16;o>0;o>>=1) mx=fmaxf(mx,__shfl_xor_sync(~0u,mx,o));
  float sm=0; for(int t=lane;t<ctx;t+=32){float e=__expf((float)S[base+t]-mx);S[base+t]=__float2half(e);sm+=e;}
  #pragma unroll
  for(int o=16;o>0;o>>=1) sm+=__shfl_xor_sync(~0u,sm,o);
  float inv=sm>0?1.f/sm:0; for(int t=lane;t<ctx;t+=32) S[base+t]=__float2half((float)S[base+t]*inv);
  if(lane==0){ mxo[(size_t)hh*MMAX+m]=mx; smo[(size_t)hh*MMAX+m]=sm; }
}

// draft-self: one warp per (h,m); query m attends draftK/V[h][0..m] (causal). normalized O_d + mx_d,sm_d.
__global__ void draft_self(const __half* Q,const __half* dK,const __half* dV, __half* Od,float* mxo,float* smo,
                           int H,int M,int MMAX,int d){
  int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(gw>=H*M)return;
  int hh=gw/M,m=gw%M; float scale=rsqrtf((float)d);
  const __half* q=Q+((size_t)hh*MMAX+m)*d;
  float qr[VPL];
  #pragma unroll
  for(int c=0;c<VPL;c++) qr[c]=(float)q[lane*VPL+c];
  float mx=-FLT_MAX,sm=0,acc[VPL];
  #pragma unroll
  for(int c=0;c<VPL;c++) acc[c]=0;
  for(int j=0;j<=m;j++){
    const __half* k=dK+((size_t)hh*MMAX+j)*d; float p=0;
    #pragma unroll
    for(int c=0;c<VPL;c++) p+=qr[c]*(float)k[lane*VPL+c];
    #pragma unroll
    for(int o=16;o>0;o>>=1) p+=__shfl_xor_sync(~0u,p,o);
    float s=p*scale; const __half* v=dV+((size_t)hh*MMAX+j)*d;
    float mn=fmaxf(mx,s),corr=__expf(mx-mn),pe=__expf(s-mn); sm=sm*corr+pe;
    #pragma unroll
    for(int c=0;c<VPL;c++) acc[c]=acc[c]*corr+pe*(float)v[lane*VPL+c];
    mx=mn;
  }
  float inv=sm>0?1.f/sm:0; __half* o=Od+((size_t)hh*MMAX+m)*d;
  #pragma unroll
  for(int c=0;c<VPL;c++) o[lane*VPL+c]=__float2half(acc[c]*inv);
  if(lane==0){ mxo[(size_t)hh*MMAX+m]=mx; smo[(size_t)hh*MMAX+m]=sm; }
}

// merge two normalized partials via their (mx,sm): O = (Oc*wc + Od*wd)/(wc+wd), w=sm*exp(mx-mxg).
__global__ void merge(const __half* Oc,const float* mxc,const float* smc,
                      const __half* Od,const float* mxd,const float* smd, __half* O,int H,int M,int MMAX,int d){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; int total=H*M*d; if(idx>=total) return;
  int c=idx%d, hm=idx/d, m=hm%M, hh=hm/M; size_t st=(size_t)hh*MMAX+m;
  float mc=mxc[st],sc=smc[st],md=mxd[st],sd=smd[st];
  float mg=fmaxf(mc,md); float wc=sc*__expf(mc-mg), wd=sd*__expf(md-mg); float den=wc+wd;
  float oc=(float)Oc[((size_t)hh*MMAX+m)*d+c], od=(float)Od[((size_t)hh*MMAX+m)*d+c];
  O[((size_t)hh*MMAX+m)*d+c]=__float2half(den>0?(oc*wc+od*wd)/den:0.f);
}

int main(int argc,char**argv){
  int ctx=(argc>1)?atoi(argv[1]):4096, iters=(argc>2)?atoi(argv[2]):300;
  const int H=N_Q_HEADS,d=HEAD_DIM,MMAX=16; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("== TC spec-VERIFY attention (context TC + masked draft-self + merge) == %s ctx=%d H=%d d=%d\n",p.name,ctx,H,d);
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH));
  std::vector<float> rQ((size_t)H*MMAX*d),rK((size_t)H*ctx*d),rV((size_t)H*ctx*d),rdK((size_t)H*MMAX*d),rdV((size_t)H*MMAX*d);
  auto mk=[&](size_t n,std::vector<float>*ref,unsigned sd){__half*dp;CK(cudaMalloc(&dp,n*2));std::vector<__half>hb(n);
    for(size_t i=0;i<n;i++){sd=sd*1664525u+1013904223u;float v=((int)((sd>>9)&0x3ff)-512)/512.f*0.3f;hb[i]=__float2half(v);if(ref)(*ref)[i]=(float)hb[i];}
    CK(cudaMemcpy(dp,hb.data(),n*2,cudaMemcpyHostToDevice));return dp;};
  __half *Q=mk((size_t)H*MMAX*d,&rQ,1),*K=mk((size_t)H*ctx*d,&rK,7),*V=mk((size_t)H*ctx*d,&rV,13);
  __half *dK=mk((size_t)H*MMAX*d,&rdK,21),*dV=mk((size_t)H*MMAX*d,&rdV,29);
  __half *S=mk((size_t)H*MMAX*ctx,nullptr,3),*Oc=mk((size_t)H*MMAX*d,nullptr,5),*Od=mk((size_t)H*MMAX*d,nullptr,9),*O=mk((size_t)H*MMAX*d,nullptr,11);
  float *mxc,*smc,*mxd,*smd; for(float**pp:{&mxc,&smc,&mxd,&smd}) CK(cudaMalloc(pp,(size_t)H*MMAX*4));
  const float scale=1.f/sqrtf((float)d),zero=0.f;
  auto run=[&](int M){
    CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N, ctx,M,d, &scale, K,CUDA_R_16F,d,(long long)ctx*d,
      Q,CUDA_R_16F,d,(long long)MMAX*d, &zero, S,CUDA_R_16F,ctx,(long long)MMAX*ctx, H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    int w=H*M,blk=128; softmax_ctx<<<(w*32+blk-1)/blk,blk>>>(S,mxc,smc,H,M,MMAX,ctx);
    const float one=1.f;
    CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_N, d,M,ctx, &one, V,CUDA_R_16F,d,(long long)ctx*d,
      S,CUDA_R_16F,ctx,(long long)MMAX*ctx, &zero, Oc,CUDA_R_16F,d,(long long)MMAX*d, H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    draft_self<<<(w*32+blk-1)/blk,blk>>>(Q,dK,dV,Od,mxd,smd,H,M,MMAX,d);
    int tot=H*M*d; merge<<<(tot+255)/256,256>>>(Oc,mxc,smc,Od,mxd,smd,O,H,M,MMAX,d);
  };
  // correctness vs CPU fp32 (full masked attn), M=8, 3 heads, a few m
  int Mv=8; run(Mv); CK(cudaDeviceSynchronize());
  std::vector<__half> hO((size_t)H*MMAX*d); CK(cudaMemcpy(hO.data(),O,hO.size()*2,cudaMemcpyDeviceToHost));
  double maxerr=0,maxref=0;
  for(int qh:{0,H/2,H-1}) for(int m:{0,3,7}){
    int N=ctx+m+1; std::vector<double> lg(N); double mx=-1e30;
    for(int t=0;t<ctx;t++){double dpr=0;for(int c=0;c<d;c++)dpr+=rQ[((size_t)qh*MMAX+m)*d+c]*rK[((size_t)qh*ctx+t)*d+c];lg[t]=dpr*scale;mx=fmax(mx,lg[t]);}
    for(int j=0;j<=m;j++){double dpr=0;for(int c=0;c<d;c++)dpr+=rQ[((size_t)qh*MMAX+m)*d+c]*rdK[((size_t)qh*MMAX+j)*d+c];lg[ctx+j]=dpr*scale;mx=fmax(mx,lg[ctx+j]);}
    double den=0;std::vector<double> wv(N);for(int t=0;t<N;t++){wv[t]=exp(lg[t]-mx);den+=wv[t];}
    for(int c=0;c<d;c++){double o=0;for(int t=0;t<ctx;t++)o+=wv[t]*rV[((size_t)qh*ctx+t)*d+c];for(int j=0;j<=m;j++)o+=wv[ctx+j]*rdV[((size_t)qh*MMAX+j)*d+c];o/=den;
      double got=(float)hO[((size_t)qh*MMAX+m)*d+c];maxerr=fmax(maxerr,fabs(o-got));maxref=fmax(maxref,fabs(o));}}
  printf("correctness (M=8, masked full verify, 3 heads x m={0,3,7}): max_abs_err=%.3e rel=%.2f%% -> %s\n",maxerr,100*maxerr/maxref,(maxerr/maxref<0.05)?"PASS":"FAIL");
  cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
  printf("\n  M    verify-attn us   us/query   ratio_vs_M1\n  ------------------------------------------------\n");
  int Ms[]={1,4,8,16};double u1=0;
  for(int M:Ms){for(int w=0;w<30;w++)run(M);CK(cudaDeviceSynchronize());CK(cudaEventRecord(e0));for(int i=0;i<iters;i++)run(M);CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));
    float ms;CK(cudaEventElapsedTime(&ms,e0,e1));double us=ms*1000.0/iters;if(M==1)u1=us;
    printf("  %-4d %12.2f   %8.2f   %.3f%s\n",M,us,us/M,us/u1,(M>1&&us/u1<1.3)?"  <- ~FLAT":"");}
  return 0;
}
