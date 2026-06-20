// tc_verify_fp8.cu — tree-masked TC spec-verify attention with FP8 e4m3 context KV (Charles's cache dtype).
// Context K/V stored fp8 e4m3 + per-channel scale (real = q8*scale, matching k2_load4). Dequant-on-load to
// fp16 is M-INDEPENDENT (done once per call regardless of draft width) -> flatness preserved; the real fused
// engine kernel loads fp8 inline (no fp16 spill). This proxy confirms fp8 ACCURACY + FLATNESS end-to-end.
// NOTE: 64 independent heads (no GQA) -> KV HBM overstated, but flatness (M = free GEMM cols) is GQA-indep.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include "common.cuh"
using namespace q3;
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)
constexpr int VPL = HEAD_DIM/32;
typedef __nv_fp8_e4m3 fp8e;

// dequant fp8 [N rows x d] with per-channel scale[d] -> fp16. M-independent (whole KV cache, once per call).
__global__ void dequant(const fp8e* in,const float* sc,__half* out,size_t rows,int d){
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=rows*d) return;
  out[idx]=__float2half((float)in[idx]*sc[idx%d]);
}
__global__ void softmax_ctx(__half* S,float* mxo,float* smo,int H,int M,int MMAX,int ctx){
  int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(gw>=H*M)return;
  int hh=gw/M,m=gw%M; size_t base=(size_t)hh*MMAX*ctx+(size_t)m*ctx; float mx=-FLT_MAX;
  for(int t=lane;t<ctx;t+=32) mx=fmaxf(mx,(float)S[base+t]);
  #pragma unroll
  for(int o=16;o>0;o>>=1) mx=fmaxf(mx,__shfl_xor_sync(~0u,mx,o));
  float sm=0; for(int t=lane;t<ctx;t+=32){float e=__expf((float)S[base+t]-mx);S[base+t]=__float2half(e);sm+=e;}
  #pragma unroll
  for(int o=16;o>0;o>>=1) sm+=__shfl_xor_sync(~0u,sm,o);
  float inv=sm>0?1.f/sm:0; for(int t=lane;t<ctx;t+=32) S[base+t]=__float2half((float)S[base+t]*inv);
  if(lane==0){mxo[(size_t)hh*MMAX+m]=mx;smo[(size_t)hh*MMAX+m]=sm;}
}
__global__ void draft_self_tree(const __half* Q,const __half* dK,const __half* dV,const int* parent,
                                __half* Od,float* mxo,float* smo,int H,int M,int MMAX,int d){
  int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(gw>=H*M)return;
  int hh=gw/M,m=gw%M; float scale=rsqrtf((float)d); const __half* q=Q+((size_t)hh*MMAX+m)*d; float qr[VPL];
  #pragma unroll
  for(int c=0;c<VPL;c++) qr[c]=(float)q[lane*VPL+c];
  float mx=-FLT_MAX,sm=0,acc[VPL];
  #pragma unroll
  for(int c=0;c<VPL;c++) acc[c]=0;
  for(int j=m;j>=0;j=parent[j]){
    const __half* k=dK+((size_t)hh*MMAX+j)*d; float p=0;
    #pragma unroll
    for(int c=0;c<VPL;c++) p+=qr[c]*(float)k[lane*VPL+c];
    #pragma unroll
    for(int o=16;o>0;o>>=1) p+=__shfl_xor_sync(~0u,p,o);
    float s=p*scale; const __half* v=dV+((size_t)hh*MMAX+j)*d;
    float mn=fmaxf(mx,s),corr=__expf(mx-mn),pe=__expf(s-mn); sm=sm*corr+pe;
    #pragma unroll
    for(int c=0;c<VPL;c++) acc[c]=acc[c]*corr+pe*(float)v[lane*VPL+c];
    mx=mn; if(parent[j]<0) break;
  }
  float inv=sm>0?1.f/sm:0; __half* o=Od+((size_t)hh*MMAX+m)*d;
  #pragma unroll
  for(int c=0;c<VPL;c++) o[lane*VPL+c]=__float2half(acc[c]*inv);
  if(lane==0){mxo[(size_t)hh*MMAX+m]=mx;smo[(size_t)hh*MMAX+m]=sm;}
}
__global__ void merge(const __half* Oc,const float* mxc,const float* smc,const __half* Od,const float* mxd,
                      const float* smd,__half* O,int H,int M,int MMAX,int d){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=H*M*d)return;
  int c=idx%d,hm=idx/d,m=hm%M,hh=hm/M; size_t st=(size_t)hh*MMAX+m;
  float mc=mxc[st],sc=smc[st],md=mxd[st],sd=smd[st],mg=fmaxf(mc,md);
  float wc=sc*__expf(mc-mg),wd=sd*__expf(md-mg),den=wc+wd;
  float oc=(float)Oc[((size_t)hh*MMAX+m)*d+c],od=(float)Od[((size_t)hh*MMAX+m)*d+c];
  O[((size_t)hh*MMAX+m)*d+c]=__float2half(den>0?(oc*wc+od*wd)/den:0.f);
}

int main(int argc,char**argv){
  int ctx=(argc>1)?atoi(argv[1]):4096,iters=(argc>2)?atoi(argv[2]):300;
  const int H=N_Q_HEADS,d=HEAD_DIM,MMAX=16; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("== FP8 tree-masked TC spec-verify attention == %s ctx=%d H=%d d=%d (fp8 e4m3 KV + per-channel scale)\n",p.name,ctx,H,d);
  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetMathMode(h,CUBLAS_TENSOR_OP_MATH));
  std::vector<int> par(MMAX); for(int m=0;m<MMAX;m++) par[m]=(m==0)?-1:(m-1)/2;
  int* dpar; CK(cudaMalloc(&dpar,MMAX*4)); CK(cudaMemcpy(dpar,par.data(),MMAX*4,cudaMemcpyHostToDevice));
  // per-channel scales
  std::vector<float> ksc(d),vsc(d); for(int c=0;c<d;c++){ksc[c]=0.4f+0.3f*((c*37)%11)/11.f; vsc[c]=0.4f+0.3f*((c*53)%13)/13.f;}
  float *dKsc,*dVsc; CK(cudaMalloc(&dKsc,d*4)); CK(cudaMalloc(&dVsc,d*4));
  CK(cudaMemcpy(dKsc,ksc.data(),d*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dVsc,vsc.data(),d*4,cudaMemcpyHostToDevice));
  std::vector<float> rQ((size_t)H*MMAX*d),rK((size_t)H*ctx*d),rV((size_t)H*ctx*d),rdK((size_t)H*MMAX*d),rdV((size_t)H*MMAX*d);
  // fp16 maker (Q, draft K/V)
  auto mk16=[&](size_t n,std::vector<float>*ref,unsigned sd){__half*dp;CK(cudaMalloc(&dp,n*2));std::vector<__half>hb(n);
    for(size_t i=0;i<n;i++){sd=sd*1664525u+1013904223u;float v=((int)((sd>>9)&0x3ff)-512)/512.f*0.3f;hb[i]=__float2half(v);if(ref)(*ref)[i]=(float)hb[i];}
    CK(cudaMemcpy(dp,hb.data(),n*2,cudaMemcpyHostToDevice));return dp;};
  // fp8 maker for context K/V: store q8=quant(v/scale[c]); ref = (float)q8 * scale[c]
  auto mk8=[&](size_t rows,std::vector<float>&ref,const std::vector<float>&sc,unsigned sd){
    size_t n=rows*d; fp8e*dp;CK(cudaMalloc(&dp,n));std::vector<fp8e>hb(n);
    for(size_t i=0;i<n;i++){sd=sd*1664525u+1013904223u;float v=((int)((sd>>9)&0x3ff)-512)/512.f*0.3f;int c=i%d;
      fp8e q((float)(v/sc[c])); hb[i]=q; ref[i]=(float)q*sc[c];}
    CK(cudaMemcpy(dp,hb.data(),n,cudaMemcpyHostToDevice));return dp;};
  __half *Q=mk16((size_t)H*MMAX*d,&rQ,1);
  fp8e *K8=mk8((size_t)H*ctx,rK,ksc,7), *V8=mk8((size_t)H*ctx,rV,vsc,13);
  __half *dK=mk16((size_t)H*MMAX*d,&rdK,21),*dV=mk16((size_t)H*MMAX*d,&rdV,29);
  __half *Kf=mk16((size_t)H*ctx*d,nullptr,2),*Vf=mk16((size_t)H*ctx*d,nullptr,4);
  __half *S=mk16((size_t)H*MMAX*ctx,nullptr,3),*Oc=mk16((size_t)H*MMAX*d,nullptr,5),*Od=mk16((size_t)H*MMAX*d,nullptr,9),*O=mk16((size_t)H*MMAX*d,nullptr,11);
  float *mxc,*smc,*mxd,*smd; for(float**pp:{&mxc,&smc,&mxd,&smd}) CK(cudaMalloc(pp,(size_t)H*MMAX*4));
  const float scale=1.f/sqrtf((float)d),zero=0.f;
  auto run=[&](int M){
    size_t rows=(size_t)H*ctx; int blk=256;
    dequant<<<(rows*d+blk-1)/blk,blk>>>(K8,dKsc,Kf,rows,d);   // M-INDEPENDENT
    dequant<<<(rows*d+blk-1)/blk,blk>>>(V8,dVsc,Vf,rows,d);
    CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N, ctx,M,d, &scale, Kf,CUDA_R_16F,d,(long long)ctx*d,
      Q,CUDA_R_16F,d,(long long)MMAX*d, &zero, S,CUDA_R_16F,ctx,(long long)MMAX*ctx, H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    int w=H*M,b2=128; softmax_ctx<<<(w*32+b2-1)/b2,b2>>>(S,mxc,smc,H,M,MMAX,ctx);
    const float one=1.f;
    CB(cublasGemmStridedBatchedEx(h,CUBLAS_OP_N,CUBLAS_OP_N, d,M,ctx, &one, Vf,CUDA_R_16F,d,(long long)ctx*d,
      S,CUDA_R_16F,ctx,(long long)MMAX*ctx, &zero, Oc,CUDA_R_16F,d,(long long)MMAX*d, H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    draft_self_tree<<<(w*32+b2-1)/b2,b2>>>(Q,dK,dV,dpar,Od,mxd,smd,H,M,MMAX,d);
    int tot=H*M*d; merge<<<(tot+255)/256,256>>>(Oc,mxc,smc,Od,mxd,smd,O,H,M,MMAX,d);
  };
  run(16); CK(cudaDeviceSynchronize());
  std::vector<__half> hO((size_t)H*MMAX*d); CK(cudaMemcpy(hO.data(),O,hO.size()*2,cudaMemcpyDeviceToHost));
  double maxerr=0,maxref=0;
  for(int qh:{0,H/2,H-1}) for(int m:{0,5,15}){
    std::vector<int> anc; for(int j=m;j>=0;j=par[j]){anc.push_back(j); if(par[j]<0)break;}
    int A=anc.size(),N=ctx+A; std::vector<double> lg(N); double mx=-1e30;
    for(int t=0;t<ctx;t++){double dpr=0;for(int c=0;c<d;c++)dpr+=rQ[((size_t)qh*MMAX+m)*d+c]*rK[((size_t)qh*ctx+t)*d+c];lg[t]=dpr*scale;mx=fmax(mx,lg[t]);}
    for(int a=0;a<A;a++){int j=anc[a];double dpr=0;for(int c=0;c<d;c++)dpr+=rQ[((size_t)qh*MMAX+m)*d+c]*rdK[((size_t)qh*MMAX+j)*d+c];lg[ctx+a]=dpr*scale;mx=fmax(mx,lg[ctx+a]);}
    double den=0;std::vector<double> wv(N);for(int t=0;t<N;t++){wv[t]=exp(lg[t]-mx);den+=wv[t];}
    for(int c=0;c<d;c++){double o=0;for(int t=0;t<ctx;t++)o+=wv[t]*rV[((size_t)qh*ctx+t)*d+c];for(int a=0;a<A;a++)o+=wv[ctx+a]*rdV[((size_t)qh*MMAX+anc[a])*d+c];o/=den;
      double got=(float)hO[((size_t)qh*MMAX+m)*d+c];maxerr=fmax(maxerr,fabs(o-got));maxref=fmax(maxref,fabs(o));}}
  printf("correctness (M=16 tree, fp8 KV vs fp32-of-fp8): max_abs_err=%.3e rel=%.2f%% -> %s (fp8 tol ~5%%)\n",maxerr,100*maxerr/maxref,(maxerr/maxref<0.05)?"PASS":"FAIL");
  cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
  printf("\n  M(nodes)  verify-attn us   ratio_vs_M1\n  ---------------------------------------\n");
  int Ms[]={1,4,8,16}; double u1=0;
  for(int M:Ms){for(int w=0;w<30;w++)run(M);CK(cudaDeviceSynchronize());CK(cudaEventRecord(e0));for(int i=0;i<iters;i++)run(M);CK(cudaEventRecord(e1));CK(cudaEventSynchronize(e1));
    float ms;CK(cudaEventElapsedTime(&ms,e0,e1));double us=ms*1000.0/iters;if(M==1)u1=us;
    printf("  %-8d %12.2f   %.3f%s\n",M,us,us/u1,(M>1&&us/u1<1.3)?"  <- ~FLAT":"");}
  return 0;
}
