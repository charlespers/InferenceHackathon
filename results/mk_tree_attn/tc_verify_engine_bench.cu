// tc_verify_engine_bench.cu — validates engine/native/tc_verify_attn.cuh at the ENGINE shape:
//   H = Q_HEADS_RANK (=8 at TP8, the OPEN flatness risk — measured at 64 before), with the engine's I/O:
//   float q_mq in (-> fp16), fp8 e4m3 KV + per-channel scale (-> fp16 dequant-on-load, M-independent),
//   float attn_out_mq out. CHAIN causal mask (parent[m]=m-1). fp32 gate + flatness M=1..16.
// STATUS: compile-checked off-box; GPU-validation pending (run when the box frees). Build:
//   nvcc -arch=sm_90a -O3 --use_fast_math -I <engine/native> -I <kernels> tc_verify_engine_bench.cu -lcublas
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include "tc_verify_attn.cuh"   // the drop-in under test (pulls in common.cuh)
using namespace q3;
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
typedef __nv_fp8_e4m3 fp8e;

__global__ void f2h(const float* in, __half* out, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=__float2half(in[i]); }
__global__ void h2f(const __half* in, float* out, size_t n){ size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=(float)in[i]; }
// dequant fp8[rows x d] * per-channel scale[d] -> fp16  (M-independent: whole KV, once per call)
__global__ void deq(const fp8e* in, const float* sc, __half* out, size_t rows, int d){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<rows*(size_t)d) out[i]=__float2half((float)in[i]*sc[i%d]);
}

int main(int argc,char**argv){
  int ctx=(argc>1)?atoi(argv[1]):4096, iters=(argc>2)?atoi(argv[2]):200, H=(argc>3)?atoi(argv[3]):8; // H=Q_HEADS_RANK
  const int d=HEAD_DIM, MMAX=16; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("== ENGINE-shape verify (drop-in tc_verify_attn.cuh) == %s ctx=%d H=%d d=%d (float q, fp8 KV->fp16, CHAIN)\n",p.name,ctx,H,d);
  cublasHandle_t cb; cublasCreate(&cb); cublasSetMathMode(cb,CUBLAS_TENSOR_OP_MATH);
  std::vector<int> par(MMAX); for(int m=0;m<MMAX;m++) par[m]=m-1;          // CHAIN
  int* dpar; CK(cudaMalloc(&dpar,MMAX*4)); CK(cudaMemcpy(dpar,par.data(),MMAX*4,cudaMemcpyHostToDevice));
  std::vector<float> ksc(d),vsc(d); for(int c=0;c<d;c++){ksc[c]=0.4f+0.3f*((c*37)%11)/11.f; vsc[c]=0.4f+0.3f*((c*53)%13)/13.f;}
  float *dKsc,*dVsc; CK(cudaMalloc(&dKsc,d*4)); CK(cudaMalloc(&dVsc,d*4));
  CK(cudaMemcpy(dKsc,ksc.data(),d*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dVsc,vsc.data(),d*4,cudaMemcpyHostToDevice));
  // refs (fp32) for the gate
  std::vector<float> rQ((size_t)MMAX*H*d), rK((size_t)H*ctx*d), rV((size_t)H*ctx*d), rdK((size_t)MMAX*H*d), rdV((size_t)MMAX*H*d);
  // ENGINE-source buffers: float q, fp8 KV. (draft K/V fp16 here — fresh from the verify forward.)
  auto mkf=[&](size_t n,std::vector<float>&ref,unsigned s){ float* dp; CK(cudaMalloc(&dp,n*4)); std::vector<float> h(n);
    for(size_t i=0;i<n;i++){s=s*1664525u+1013904223u; h[i]=((int)((s>>9)&0x3ff)-512)/512.f*0.3f; ref[i]=h[i];}
    CK(cudaMemcpy(dp,h.data(),n*4,cudaMemcpyHostToDevice)); return dp; };
  auto mk8=[&](size_t rows,std::vector<float>&ref,const std::vector<float>&sc,unsigned s){ size_t n=rows*(size_t)d; fp8e*dp;CK(cudaMalloc(&dp,n));
    std::vector<fp8e> h(n); for(size_t i=0;i<n;i++){s=s*1664525u+1013904223u; float v=((int)((s>>9)&0x3ff)-512)/512.f*0.3f; int c=i%d; fp8e q((float)(v/sc[c])); h[i]=q; ref[i]=(float)q*sc[c];}
    CK(cudaMemcpy(dp,h.data(),n,cudaMemcpyHostToDevice)); return dp; };
  float* q_f   = mkf((size_t)MMAX*H*d, rQ, 1);
  fp8e*  K8    = mk8((size_t)H*ctx, rK, ksc, 7), *V8 = mk8((size_t)H*ctx, rV, vsc, 13);
  float* dK_f  = mkf((size_t)MMAX*H*d, rdK, 21), *dV_f = mkf((size_t)MMAX*H*d, rdV, 29);
  // fp16 working buffers (what the drop-in consumes)
  auto h16=[&](size_t n){ __half* dp; CK(cudaMalloc(&dp,n*2)); return dp; };
  __half *Q=h16((size_t)MMAX*H*d), *Kf=h16((size_t)H*ctx*d), *Vf=h16((size_t)H*ctx*d), *dK=h16((size_t)MMAX*H*d), *dV=h16((size_t)MMAX*H*d);
  __half *S=h16((size_t)H*MMAX*ctx), *Oc=h16((size_t)H*MMAX*d), *Od=h16((size_t)H*MMAX*d), *out=h16((size_t)MMAX*H*d);
  float *mxc,*smc,*mxd,*smd; for(float**pp:{&mxc,&smc,&mxd,&smd}) CK(cudaMalloc(pp,(size_t)H*MMAX*4));
  float* out_f; CK(cudaMalloc(&out_f,(size_t)MMAX*H*d*4));
  int blk=256; auto NB=[&](size_t n){ return (int)((n+blk-1)/blk); };
  f2h<<<NB((size_t)MMAX*H*d),blk>>>(q_f,Q,(size_t)MMAX*H*d);
  f2h<<<NB((size_t)MMAX*H*d),blk>>>(dK_f,dK,(size_t)MMAX*H*d); f2h<<<NB((size_t)MMAX*H*d),blk>>>(dV_f,dV,(size_t)MMAX*H*d);
  auto run=[&](int M){
    deq<<<NB((size_t)H*ctx*d),blk>>>(K8,dKsc,Kf,(size_t)H*ctx,d);   // M-INDEPENDENT
    deq<<<NB((size_t)H*ctx*d),blk>>>(V8,dVsc,Vf,(size_t)H*ctx,d);
    tcv::verify_attn(cb, Q,Kf,Vf, dK,dV, dpar, ctx, M, MMAX, H, S,Oc,Od, mxc,smc,mxd,smd, out, 0);
    h2f<<<NB((size_t)M*H*d),blk>>>(out, out_f, (size_t)M*H*d);
  };
  // correctness vs CPU fp32 (M=8, chain), heads {0, H/2, H-1}, m {0,3,7}
  int Mv=8; run(Mv); CK(cudaDeviceSynchronize());
  std::vector<float> hO((size_t)MMAX*H*d); CK(cudaMemcpy(hO.data(),out_f,(size_t)Mv*H*d*4,cudaMemcpyDeviceToHost));
  double me=0,mr=0; const float scale=1.f/sqrtf((float)d);
  for(int qh:{0,H/2,H-1}) for(int m:{0,3,7}){
    int N=ctx+m+1; std::vector<double> lg(N); double mx=-1e30;
    for(int t=0;t<ctx;t++){double dp=0; for(int c=0;c<d;c++) dp+=rQ[((size_t)m*H+qh)*d+c]*rK[((size_t)qh*ctx+t)*d+c]; lg[t]=dp*scale; mx=fmax(mx,lg[t]);}
    for(int j=0;j<=m;j++){double dp=0; for(int c=0;c<d;c++) dp+=rQ[((size_t)m*H+qh)*d+c]*rdK[((size_t)j*H+qh)*d+c]; lg[ctx+j]=dp*scale; mx=fmax(mx,lg[ctx+j]);}
    double den=0; std::vector<double> w(N); for(int t=0;t<N;t++){w[t]=exp(lg[t]-mx); den+=w[t];}
    for(int c=0;c<d;c++){double o=0; for(int t=0;t<ctx;t++) o+=w[t]*rV[((size_t)qh*ctx+t)*d+c]; for(int j=0;j<=m;j++) o+=w[ctx+j]*rdV[((size_t)j*H+qh)*d+c]; o/=den;
      double got=hO[((size_t)m*H+qh)*d+c]; me=fmax(me,fabs(o-got)); mr=fmax(mr,fabs(o));}}
  printf("correctness (H=%d M=8 chain, fp8 KV): max_abs_err=%.3e rel=%.2f%% -> %s\n",H,me,100*me/mr,(me/mr<0.05)?"PASS":"FAIL");
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  printf("\n  M    verify us   ratio_vs_M1  (H=%d per-rank — the open flatness question)\n  --------------------------------------------------\n",H);
  int Ms[]={1,4,8,16}; double u1=0;
  for(int M:Ms){ for(int w=0;w<30;w++) run(M); CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0)); for(int i=0;i<iters;i++) run(M); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); double us=ms*1000.0/iters; if(M==1)u1=us;
    printf("  %-4d %9.2f   %.3f%s\n",M,us,us/u1,(M>1&&us/u1<1.3)?"  <- ~FLAT":""); }
  printf("\n  (NOTE abs includes M-independent fp8->fp16 dequant materialization; native-fp8 GEMM removes it.)\n");
  return 0;
}
