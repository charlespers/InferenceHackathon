// Isolated down-proj benchmark: winner k5b_down_warp (48KB all-a smem) vs k5b_down_warp_v2 (per-slot 6KB).
// Confirms whether the occupancy fix lifts the measured down-proj e (was 0.405). Public facts + std CUDA.
//   build: /usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math kernels/k5_downproj_bench.cu -I kernels -o dbench
//   run:   CUDA_VISIBLE_DEVICES=0 ./dbench [peak_GBps=3350]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;
#include "k5_experts_warp.cu"    // k5b_down_warp (winner, all-a smem) + warp_dot
#include "k5_experts_warp2.cu"   // k5b_down_warp_v2 (per-slot smem) + warp_dot_v2

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)
__global__ void fill_fp8(fp8* w,size_t n,unsigned s){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2654435761u)+s*40503u; w[i]=fp8((((h%2000)/1000.0f)-1.0f)*0.25f);} }
__global__ void fill_f32(float* a,size_t n,unsigned s,float sc,int pos){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2246822519u)+s*40503u; float v=(((h%2000)/1000.0f)-1.0f)*sc; a[i]=pos?(fabsf(v)+1e-3f):v;} }

int main(int argc,char**argv){
  const int E=8; const double PEAK=(argc>1)?atof(argv[1]):3350.0;
  const size_t d_n=(size_t)HIDDEN*MOE_INTER;
  fp8 *Wd_h[E]; float *Sd_h[E];
  for(int e=0;e<E;e++){ CK(cudaMalloc(&Wd_h[e],d_n*sizeof(fp8)));CK(cudaMalloc(&Sd_h[e],(size_t)HIDDEN*sizeof(float)));
    fill_fp8<<<512,256>>>(Wd_h[e],d_n,100u+e); fill_f32<<<64,256>>>(Sd_h[e],HIDDEN,13u+e,0.02f,1);}
  const fp8 **Wd_d; const float **Sd_d;
  CK(cudaMalloc(&Wd_d,E*sizeof(fp8*)));CK(cudaMemcpy(Wd_d,Wd_h,E*sizeof(fp8*),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd_d,E*sizeof(float*)));CK(cudaMemcpy(Sd_d,Sd_h,E*sizeof(float*),cudaMemcpyHostToDevice));
  int sel_h[E]; float selw_h[E]; for(int e=0;e<E;e++){sel_h[e]=e;selw_h[e]=0.1f+0.01f*e;}
  int *sel_d; float *selw_d,*a_d,*h_d;
  CK(cudaMalloc(&sel_d,E*sizeof(int)));CK(cudaMemcpy(sel_d,sel_h,E*sizeof(int),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&selw_d,E*sizeof(float)));CK(cudaMemcpy(selw_d,selw_h,E*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&a_d,(size_t)E*MOE_INTER*sizeof(float))); fill_f32<<<64,256>>>(a_d,(size_t)E*MOE_INTER,55u,1.0f,0);
  CK(cudaMalloc(&h_d,HIDDEN*sizeof(float))); CK(cudaDeviceSynchronize());

  const double bytes=(double)E*d_n;                              // 50.33 MB (down weights, fp8)
  const size_t smemAll=(size_t)E*MOE_INTER*sizeof(float), smemSlot=(size_t)MOE_INTER*sizeof(float);
  CK(cudaFuncSetAttribute(k5b_down_warp,   cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemAll));
  CK(cudaFuncSetAttribute(k5b_down_warp_v2,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemSlot));
  cudaEvent_t s,e; CK(cudaEventCreate(&s));CK(cudaEventCreate(&e)); const int WARM=20,IT=400;
  float *ref=(float*)malloc(HIDDEN*sizeof(float)),*got=(float*)malloc(HIDDEN*sizeof(float));

  // --- winner (all-a smem), best config 264x1024 ---
  CK(cudaMemset(h_d,0,HIDDEN*sizeof(float)));
  k5b_down_warp<<<264,1024,smemAll>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E);
  CK(cudaDeviceSynchronize());CK(cudaMemcpy(ref,h_d,HIDDEN*sizeof(float),cudaMemcpyDeviceToHost));
  for(int i=0;i<WARM;i++)k5b_down_warp<<<264,1024,smemAll>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E);
  CK(cudaDeviceSynchronize());CK(cudaEventRecord(s));
  for(int i=0;i<IT;i++)k5b_down_warp<<<264,1024,smemAll>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E);
  CK(cudaEventRecord(e));CK(cudaEventSynchronize(e)); float msW;CK(cudaEventElapsedTime(&msW,s,e));msW/=IT;
  printf("  %-26s %8s %9s %6s %8s\n","down kernel","ms","GB/s","e","maxrel");
  printf("  %-26s %8.4f %9.1f %6.3f %8s\n","winner k5b (48KB all-a)",msW,bytes/1e6/msW,bytes/1e6/msW/PEAK,"(ref)");

  // --- v2 (per-slot 6KB), sweep tiles x block ---
  int cfg[][2]={{16,512},{33,512},{66,512},{33,1024},{66,1024},{132,512},{16,1024},{66,256}};
  int ncfg=sizeof(cfg)/sizeof(cfg[0]); double best=0; int bn=0,bb=0;
  for(int c=0;c<ncfg;c++){ int NT=cfg[c][0],blk=cfg[c][1]; dim3 g(NT,E);
    CK(cudaMemset(h_d,0,HIDDEN*sizeof(float)));
    k5b_down_warp_v2<<<g,blk,smemSlot>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E);
    CK(cudaDeviceSynchronize());CK(cudaMemcpy(got,h_d,HIDDEN*sizeof(float),cudaMemcpyDeviceToHost));
    double mr=0; for(int i=0;i<HIDDEN;i++){double ad=fabs((double)ref[i]-got[i]); mr=std::max(mr,ad/(fabs((double)ref[i])+1e-6));}
    for(int i=0;i<WARM;i++)k5b_down_warp_v2<<<g,blk,smemSlot>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E);
    CK(cudaDeviceSynchronize());CK(cudaEventRecord(s));
    for(int i=0;i<IT;i++)k5b_down_warp_v2<<<g,blk,smemSlot>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E);
    CK(cudaEventRecord(e));CK(cudaEventSynchronize(e)); float ms;CK(cudaEventElapsedTime(&ms,s,e));ms/=IT;
    double ee=bytes/1e6/ms/PEAK;
    char name[48]; snprintf(name,sizeof(name),"v2 per-slot (%dx%d)",NT,blk);
    printf("  %-26s %8.4f %9.1f %6.3f %8.1e\n",name,ms,bytes/1e6/ms,ee,mr);
    if(ee>best){best=ee;bn=NT;bb=blk;}
  }
  printf("\n  down-proj: winner e=%.3f -> v2 best e=%.3f (%dx%d).  %s\n",
         bytes/1e6/msW/PEAK,best,bn,bb, best>bytes/1e6/msW/PEAK ? "v2 WINS" : "no improvement (k5b likely DRAM-bound; try sub-warp split-K)");
  return 0;
}
