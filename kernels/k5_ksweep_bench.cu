// K5 k-sweep — does dropping experts (adaptive top-k) convert to WALL-CLOCK on the TUNED kernel?
//
// djamoils' tools/moe_kernel_microbench.py answers this on torch/cuBLAS, which has its OWN B=1 GEMV launch
// floor that can hide the slope. The tuned K5 winner (k5_experts_warp.cu, grid.x = nslot, NO moe_align
// padding floor) is exactly where the byte saving SHOULD become time. This times the warp winner for
// k = 2,4,6,8 active experts and reports whether time(k) ≈ floor + k·slope with slope dominant
// (bandwidth-bound → adaptive-k pays) — the structural claim from overhead-attribution.md / results-reaction-02.md.
//
//   build: /usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math kernels/k5_ksweep_bench.cu -I kernels -o ksweep
//   run:   CUDA_VISIBLE_DEVICES=0 ./ksweep [peak_GBps=3350]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;
#include "k5_experts_warp.cu"   // k5a_gateup_warp / k5b_down_warp (the measured winner)

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)
__global__ void fill_fp8(fp8* w,size_t n,unsigned s){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2654435761u)+s*40503u; w[i]=fp8((((h%2000)/1000.0f)-1.0f)*0.25f);} }
__global__ void fill_f32(float* a,size_t n,unsigned s,float sc,int pos){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2246822519u)+s*40503u; float v=(((h%2000)/1000.0f)-1.0f)*sc; a[i]=pos?(fabsf(v)+1e-3f):v;} }

int main(int argc,char**argv){
  const int EMAX=8, CTAS=264, BLK=1024; const double PEAK=(argc>1)?atof(argv[1]):3350.0;
  const size_t gu_n=(size_t)2*MOE_INTER*HIDDEN, d_n=(size_t)HIDDEN*MOE_INTER, per=gu_n+d_n; // bytes/expert (fp8)
  fp8 *Wgu[EMAX],*Wd[EMAX]; float *Sgu[EMAX],*Sd[EMAX];
  for(int e=0;e<EMAX;e++){ CK(cudaMalloc(&Wgu[e],gu_n));CK(cudaMalloc(&Wd[e],d_n));
    CK(cudaMalloc(&Sgu[e],(size_t)2*MOE_INTER*4));CK(cudaMalloc(&Sd[e],(size_t)HIDDEN*4));
    fill_fp8<<<512,256>>>(Wgu[e],gu_n,1u+e);fill_fp8<<<512,256>>>(Wd[e],d_n,100u+e);
    fill_f32<<<64,256>>>(Sgu[e],2*MOE_INTER,7u+e,0.02f,1);fill_f32<<<64,256>>>(Sd[e],HIDDEN,13u+e,0.02f,1);}
  const fp8 **Wgu_d,**Wd_d; const float **Sgu_d,**Sd_d;
  CK(cudaMalloc(&Wgu_d,EMAX*sizeof(fp8*)));CK(cudaMemcpy(Wgu_d,Wgu,EMAX*sizeof(fp8*),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd_d,EMAX*sizeof(fp8*)));CK(cudaMemcpy(Wd_d,Wd,EMAX*sizeof(fp8*),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sgu_d,EMAX*sizeof(float*)));CK(cudaMemcpy(Sgu_d,Sgu,EMAX*sizeof(float*),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd_d,EMAX*sizeof(float*)));CK(cudaMemcpy(Sd_d,Sd,EMAX*sizeof(float*),cudaMemcpyHostToDevice));
  int sel_h[EMAX]; float selw_h[EMAX]; for(int e=0;e<EMAX;e++){sel_h[e]=e;selw_h[e]=0.1f+0.01f*e;}
  int *sel_d; float *selw_d,*y_d,*a_d,*h_d;
  CK(cudaMalloc(&sel_d,EMAX*4));CK(cudaMemcpy(sel_d,sel_h,EMAX*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&selw_d,EMAX*4));CK(cudaMemcpy(selw_d,selw_h,EMAX*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&y_d,HIDDEN*4));fill_f32<<<16,256>>>(y_d,HIDDEN,99u,1.0f,0);
  CK(cudaMalloc(&a_d,(size_t)EMAX*MOE_INTER*4));CK(cudaMalloc(&h_d,HIDDEN*4));CK(cudaDeviceSynchronize());
  const size_t smemA=(size_t)HIDDEN*4, smemB=(size_t)EMAX*MOE_INTER*4;
  CK(cudaFuncSetAttribute(k5a_gateup_warp,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemA));
  CK(cudaFuncSetAttribute(k5b_down_warp, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemB));
  cudaEvent_t s,e; CK(cudaEventCreate(&s));CK(cudaEventCreate(&e)); const int WARM=20,IT=400;

  printf("K5 tuned warp winner, k-sweep (k active experts). peak %.0f GB/s\n", PEAK);
  printf("  %3s %10s %9s %7s %14s\n","k","ms/call","GB/s","e","marginal ms/2k");
  double prev_ms=0; int prev_k=0;
  for(int k=2;k<=EMAX;k+=2){
    dim3 g(CTAS);
    auto run=[&](){ k5a_gateup_warp<<<g,BLK,smemA>>>(y_d,sel_d,Wgu_d,Sgu_d,a_d,k);
                    k5b_down_warp <<<g,BLK,smemB>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,k); };
    for(int i=0;i<WARM;i++) run(); CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
    for(int i=0;i<IT;i++) run(); CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms; CK(cudaEventElapsedTime(&ms,s,e)); ms/=IT;
    double bytes=(double)k*per, gb=bytes/1e6/ms, ee=gb/PEAK;
    double marg = prev_k? (ms-prev_ms)/((k-prev_k)) : 0.0;   // ms per 1 expert (per-2k step /2)
    printf("  %3d %10.4f %9.1f %7.3f %14s\n", k, ms, gb, ee, prev_k? "" : "(base)");
    if(prev_k) printf("      -> marginal %.4f ms/expert (slope); intercept-free if ~const\n", marg);
    prev_ms=ms; prev_k=k;
  }
  printf("\nRead: if ms(k) is ~linear in k with near-zero intercept -> bandwidth-bound, adaptive-k converts\n");
  printf("to wall-clock on the tuned kernel (unlike cuBLAS/fused_moe's padding floor). Combine with\n");
  printf("router_mass.py (how often k drops) for the honest e2e expert-term speedup.\n");
  return 0;
}
