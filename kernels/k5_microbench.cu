// k5_microbench.cu — correctness + HBM-bandwidth benchmark for the K5 MoE expert kernel on H100/H200.
// Compares the scalar reference (k5_experts.cu) against the measured-best warp-per-row kernel
// (k5_experts_warp.cu), with the gate/up vs down split broken out. Public model facts + standard CUDA.
//   build:  /usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math kernels/k5_microbench.cu -I kernels -o k5bench
//   run:    CUDA_VISIBLE_DEVICES=0 ./k5bench
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;
#define K5_NO_MAIN              // pull in kernels + reference, but not k5_experts.cu's own main()
#include "k5_experts.cu"        // reference k5_experts_fused (+ k5a_gateup/k5b_down/k5_reference)
#include "k5_experts_warp.cu"   // k5a_gateup_warp, k5b_down_warp (the winner)

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)
__global__ void fill_fp8(fp8* w,size_t n,unsigned seed){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2654435761u)+seed*40503u; w[i]=fp8((((h%2000)/1000.0f)-1.0f)*0.25f);} }
__global__ void fill_f32(float* a,size_t n,unsigned seed,float sc,int pos){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2246822519u)+seed*40503u; float v=(((h%2000)/1000.0f)-1.0f)*sc; a[i]=pos?(fabsf(v)+1e-3f):v;} }

int main(int argc,char**argv){
  const int E=8;
  const int CTAS=(argc>1)?atoi(argv[1]):264, BLK=(argc>2)?atoi(argv[2]):1024;
  const double PEAK=(argc>3)?atof(argv[3]):3350.0;   // 3350 H100, 4800 H200
  const size_t gu_n=(size_t)2*MOE_INTER*HIDDEN, d_n=(size_t)HIDDEN*MOE_INTER;
  fp8 *Wgu_h[E],*Wd_h[E]; float *Sgu_h[E],*Sd_h[E];
  for(int e=0;e<E;e++){ CK(cudaMalloc(&Wgu_h[e],gu_n*sizeof(fp8)));CK(cudaMalloc(&Wd_h[e],d_n*sizeof(fp8)));
    CK(cudaMalloc(&Sgu_h[e],(size_t)2*MOE_INTER*sizeof(float)));CK(cudaMalloc(&Sd_h[e],(size_t)HIDDEN*sizeof(float)));
    fill_fp8<<<512,256>>>(Wgu_h[e],gu_n,1u+e);fill_fp8<<<512,256>>>(Wd_h[e],d_n,100u+e);
    fill_f32<<<64,256>>>(Sgu_h[e],2*MOE_INTER,7u+e,0.02f,1);fill_f32<<<64,256>>>(Sd_h[e],HIDDEN,13u+e,0.02f,1);}
  const fp8 **Wgu_d,**Wd_d; const float **Sgu_d,**Sd_d;
  CK(cudaMalloc(&Wgu_d,E*sizeof(fp8*)));CK(cudaMemcpy(Wgu_d,Wgu_h,E*sizeof(fp8*),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd_d,E*sizeof(fp8*)));CK(cudaMemcpy(Wd_d,Wd_h,E*sizeof(fp8*),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sgu_d,E*sizeof(float*)));CK(cudaMemcpy(Sgu_d,Sgu_h,E*sizeof(float*),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd_d,E*sizeof(float*)));CK(cudaMemcpy(Sd_d,Sd_h,E*sizeof(float*),cudaMemcpyHostToDevice));
  int sel_h[E]; float selw_h[E]; for(int e=0;e<E;e++){sel_h[e]=e;selw_h[e]=0.1f+0.01f*e;}
  int *sel_d; float *selw_d,*y_d,*h_d,*a_d;
  CK(cudaMalloc(&sel_d,E*sizeof(int)));CK(cudaMemcpy(sel_d,sel_h,E*sizeof(int),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&selw_d,E*sizeof(float)));CK(cudaMemcpy(selw_d,selw_h,E*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&y_d,HIDDEN*sizeof(float)));fill_f32<<<16,256>>>(y_d,HIDDEN,99u,1.0f,0);
  CK(cudaMalloc(&h_d,HIDDEN*sizeof(float)));CK(cudaMalloc(&a_d,(size_t)E*MOE_INTER*sizeof(float)));
  CK(cudaDeviceSynchronize());

  float *ref=(float*)malloc(HIDDEN*sizeof(float)),*got=(float*)malloc(HIDDEN*sizeof(float));
  CK(cudaMemset(h_d,0,HIDDEN*sizeof(float)));
  k5_experts_fused<<<E,256>>>(y_d,sel_d,selw_d,Wgu_d,Sgu_d,Wd_d,Sd_d,h_d);
  CK(cudaDeviceSynchronize());CK(cudaMemcpy(ref,h_d,HIDDEN*sizeof(float),cudaMemcpyDeviceToHost));

  const size_t smemA=(size_t)HIDDEN*sizeof(float), smemB=(size_t)E*MOE_INTER*sizeof(float);
  CK(cudaFuncSetAttribute(k5a_gateup_warp,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemA));
  CK(cudaFuncSetAttribute(k5b_down_warp, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemB));
  dim3 gA(CTAS), gB(CTAS);

  // correctness
  CK(cudaMemset(h_d,0,HIDDEN*sizeof(float)));
  k5a_gateup_warp<<<gA,BLK,smemA>>>(y_d,sel_d,Wgu_d,Sgu_d,a_d,E);
  k5b_down_warp <<<gB,BLK,smemB>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E);
  CK(cudaDeviceSynchronize());CK(cudaMemcpy(got,h_d,HIDDEN*sizeof(float),cudaMemcpyDeviceToHost));
  double mr=0; for(int i=0;i<HIDDEN;i++){double ad=fabs((double)ref[i]-got[i]); mr=std::max(mr,ad/(fabs((double)ref[i])+1e-6));}

  cudaEvent_t s,e; CK(cudaEventCreate(&s));CK(cudaEventCreate(&e)); const int WARM=20,IT=300;
  auto tA=[&](){ for(int i=0;i<WARM;i++) k5a_gateup_warp<<<gA,BLK,smemA>>>(y_d,sel_d,Wgu_d,Sgu_d,a_d,E);
    CK(cudaDeviceSynchronize());CK(cudaEventRecord(s));
    for(int i=0;i<IT;i++) k5a_gateup_warp<<<gA,BLK,smemA>>>(y_d,sel_d,Wgu_d,Sgu_d,a_d,E);
    CK(cudaEventRecord(e));CK(cudaEventSynchronize(e)); float ms;CK(cudaEventElapsedTime(&ms,s,e));return ms/IT; };
  auto tB=[&](){ for(int i=0;i<WARM;i++) k5b_down_warp<<<gB,BLK,smemB>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E);
    CK(cudaDeviceSynchronize());CK(cudaEventRecord(s));
    for(int i=0;i<IT;i++) k5b_down_warp<<<gB,BLK,smemB>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E);
    CK(cudaEventRecord(e));CK(cudaEventSynchronize(e)); float ms;CK(cudaEventElapsedTime(&ms,s,e));return ms/IT; };
  auto tRef=[&](){ for(int i=0;i<5;i++) k5_experts_fused<<<E,256>>>(y_d,sel_d,selw_d,Wgu_d,Sgu_d,Wd_d,Sd_d,h_d);
    CK(cudaDeviceSynchronize());CK(cudaEventRecord(s));
    for(int i=0;i<30;i++) k5_experts_fused<<<E,256>>>(y_d,sel_d,selw_d,Wgu_d,Sgu_d,Wd_d,Sd_d,h_d);
    CK(cudaEventRecord(e));CK(cudaEventSynchronize(e)); float ms;CK(cudaEventElapsedTime(&ms,s,e));return ms/30; };

  float msR=tRef(), msA=tA(), msB=tB(); float msW=msA+msB;
  double bA=(double)E*gu_n, bB=(double)E*d_n, bT=bA+bB;
  printf("device peak assumed %.0f GB/s   launch (CTAs=%d, block=%d)   correctness max_rel=%.2e\n",PEAK,CTAS,BLK,mr);
  printf("  %-26s %9s %9s %7s\n","stage","ms/call","GB/s","e");
  printf("  %-26s %9.4f %9.1f %7.3f\n","reference (8 CTAs)",       msR, bT/1e6/msR, bT/1e6/msR/PEAK);
  printf("  %-26s %9.4f %9.1f %7.3f\n","warp gate+up (A, 101MB)",  msA, bA/1e6/msA, bA/1e6/msA/PEAK);
  printf("  %-26s %9.4f %9.1f %7.3f\n","warp down   (B,  50MB)",   msB, bB/1e6/msB, bB/1e6/msB/PEAK);
  printf("  %-26s %9.4f %9.1f %7.3f\n","warp total  (A+B,151MB)",  msW, bT/1e6/msW, bT/1e6/msW/PEAK);
  printf("  speedup vs reference: %.1fx   |   MoE-only decode x94 layers: ref %.0f ms -> warp %.2f ms\n",
         msR/msW, msR*94, msW*94);
  return 0;
}
