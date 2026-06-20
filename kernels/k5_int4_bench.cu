// fp8 (winner, e=0.46) vs INT4 expert GEMV throughput on H100. Answers: does halving expert bytes to
// 4-bit deliver ~2x, or does the in-register nibble unpack become issue-bound? Plus a CPU check that the
// int4 down-proj unpack is correct.
//   build: /usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math kernels/k5_int4_bench.cu -I kernels -o i4bench
//   run:   CUDA_VISIBLE_DEVICES=0 ./i4bench [peak_GBps=3350]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;
#include "k5_experts_warp.cu"   // fp8 winner: k5a_gateup_warp / k5b_down_warp / warp_dot
#include "k5_experts_int4.cu"   // int4: k5a_gateup_int4 / k5b_down_int4 / warp_dot_int4

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)
__global__ void fill_fp8(fp8* w,size_t n,unsigned s){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2654435761u)+s*40503u; w[i]=fp8((((h%2000)/1000.0f)-1.0f)*0.25f);} }
__global__ void fill_u32(unsigned* w,size_t n,unsigned s){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){ w[i]=(unsigned)(i*2654435761u)+s*40503u; } }  // random nibbles
__global__ void fill_f32(float* a,size_t n,unsigned s,float sc,int pos){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2246822519u)+s*40503u; float v=(((h%2000)/1000.0f)-1.0f)*sc; a[i]=pos?(fabsf(v)+1e-3f):v;} }

int main(int argc,char**argv){
  const int E=8, CTAS=264, BLK=1024; const double PEAK=(argc>1)?atof(argv[1]):3350.0;
  const size_t gu=(size_t)2*MOE_INTER*HIDDEN, dn=(size_t)HIDDEN*MOE_INTER;   // element counts
  const size_t gu_w=gu/8, dn_w=dn/8;                                          // int4 packed uint32 counts
  // fp8 experts
  fp8 *Wgu8[E],*Wd8[E]; float *Sgu[E],*Sd[E];
  // int4 experts (packed uint32)
  unsigned *Wgu4[E],*Wd4[E];
  for(int e=0;e<E;e++){
    CK(cudaMalloc(&Wgu8[e],gu*sizeof(fp8)));CK(cudaMalloc(&Wd8[e],dn*sizeof(fp8)));
    CK(cudaMalloc(&Sgu[e],(size_t)2*MOE_INTER*sizeof(float)));CK(cudaMalloc(&Sd[e],(size_t)HIDDEN*sizeof(float)));
    CK(cudaMalloc(&Wgu4[e],gu_w*sizeof(unsigned)));CK(cudaMalloc(&Wd4[e],dn_w*sizeof(unsigned)));
    fill_fp8<<<512,256>>>(Wgu8[e],gu,1u+e);fill_fp8<<<512,256>>>(Wd8[e],dn,100u+e);
    fill_f32<<<64,256>>>(Sgu[e],2*MOE_INTER,7u+e,0.02f,1);fill_f32<<<64,256>>>(Sd[e],HIDDEN,13u+e,0.02f,1);
    fill_u32<<<512,256>>>(Wgu4[e],gu_w,5u+e);fill_u32<<<512,256>>>(Wd4[e],dn_w,55u+e);
  }
  auto mkptr=[&](void**h,size_t n){ void**d; CK(cudaMalloc(&d,E*n)); CK(cudaMemcpy(d,h,E*n,cudaMemcpyHostToDevice)); return d; };
  const fp8 **Wgu8d=(const fp8**)mkptr((void**)Wgu8,sizeof(fp8*)), **Wd8d=(const fp8**)mkptr((void**)Wd8,sizeof(fp8*));
  const unsigned **Wgu4d=(const unsigned**)mkptr((void**)Wgu4,sizeof(unsigned*)), **Wd4d=(const unsigned**)mkptr((void**)Wd4,sizeof(unsigned*));
  const float **Sgud=(const float**)mkptr((void**)Sgu,sizeof(float*)), **Sdd=(const float**)mkptr((void**)Sd,sizeof(float*));
  int sel_h[E]; float selw_h[E]; for(int e=0;e<E;e++){sel_h[e]=e;selw_h[e]=0.1f+0.01f*e;}
  int *sel; float *selw,*y,*a,*h;
  CK(cudaMalloc(&sel,E*sizeof(int)));CK(cudaMemcpy(sel,sel_h,E*sizeof(int),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&selw,E*sizeof(float)));CK(cudaMemcpy(selw,selw_h,E*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&y,HIDDEN*sizeof(float)));fill_f32<<<16,256>>>(y,HIDDEN,99u,1.0f,0);
  CK(cudaMalloc(&a,(size_t)E*MOE_INTER*sizeof(float)));CK(cudaMalloc(&h,HIDDEN*sizeof(float)));
  CK(cudaDeviceSynchronize());
  const size_t smemA=(size_t)HIDDEN*4, smemB=(size_t)E*MOE_INTER*4;
  CK(cudaFuncSetAttribute(k5a_gateup_warp,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemA));
  CK(cudaFuncSetAttribute(k5b_down_warp, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemB));
  CK(cudaFuncSetAttribute(k5a_gateup_int4,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemA));
  CK(cudaFuncSetAttribute(k5b_down_int4, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemB));
  cudaEvent_t s,e; CK(cudaEventCreate(&s));CK(cudaEventCreate(&e)); const int WARM=20,IT=300;

  auto t_fp8=[&](){ for(int i=0;i<WARM;i++){k5a_gateup_warp<<<CTAS,BLK,smemA>>>(y,sel,Wgu8d,Sgud,a,E);k5b_down_warp<<<CTAS,BLK,smemB>>>(sel,selw,Wd8d,Sdd,a,h,E);}
    CK(cudaDeviceSynchronize());CK(cudaEventRecord(s)); for(int i=0;i<IT;i++){k5a_gateup_warp<<<CTAS,BLK,smemA>>>(y,sel,Wgu8d,Sgud,a,E);k5b_down_warp<<<CTAS,BLK,smemB>>>(sel,selw,Wd8d,Sdd,a,h,E);}
    CK(cudaEventRecord(e));CK(cudaEventSynchronize(e)); float ms;CK(cudaEventElapsedTime(&ms,s,e));return ms/IT; };
  auto t_int4=[&](){ for(int i=0;i<WARM;i++){k5a_gateup_int4<<<CTAS,BLK,smemA>>>(y,sel,Wgu4d,Sgud,a,E);k5b_down_int4<<<CTAS,BLK,smemB>>>(sel,selw,Wd4d,Sdd,a,h,E);}
    CK(cudaDeviceSynchronize());CK(cudaEventRecord(s)); for(int i=0;i<IT;i++){k5a_gateup_int4<<<CTAS,BLK,smemA>>>(y,sel,Wgu4d,Sgud,a,E);k5b_down_int4<<<CTAS,BLK,smemB>>>(sel,selw,Wd4d,Sdd,a,h,E);}
    CK(cudaEventRecord(e));CK(cudaEventSynchronize(e)); float ms;CK(cudaEventElapsedTime(&ms,s,e));return ms/IT; };

  // --- int4 unpack correctness: CPU-reference a few down outputs ---
  CK(cudaMemset(h,0,HIDDEN*sizeof(float)));
  k5a_gateup_int4<<<CTAS,BLK,smemA>>>(y,sel,Wgu4d,Sgud,a,E);
  k5b_down_int4<<<CTAS,BLK,smemB>>>(sel,selw,Wd4d,Sdd,a,h,E);
  CK(cudaDeviceSynchronize());
  float *ah=(float*)malloc((size_t)E*MOE_INTER*4); CK(cudaMemcpy(ah,a,(size_t)E*MOE_INTER*4,cudaMemcpyDeviceToHost));
  float *hh=(float*)malloc(HIDDEN*4); CK(cudaMemcpy(hh,h,HIDDEN*4,cudaMemcpyDeviceToHost));
  unsigned *wdh=(unsigned*)malloc(dn_w*4); float *sdh=(float*)malloc((size_t)HIDDEN*4);
  double maxrel=0; const int wpr=MOE_INTER>>3;
  for(int oi=0; oi<3; oi++){ int o=(oi==0?0:oi==1?100:1000); double ref=0;
    for(int slot=0; slot<E; slot++){ int ee=sel_h[slot];
      CK(cudaMemcpy(wdh,Wd4[ee],dn_w*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(sdh,Sd[ee],HIDDEN*4,cudaMemcpyDeviceToHost));
      double acc=0; for(int j=0;j<MOE_INTER;j++){ unsigned w=wdh[(size_t)o*wpr + (j>>3)]; int nib=(int)((w>>(4*(j&7)))&0xF)-8; acc += (double)ah[(size_t)slot*MOE_INTER+j]*nib; }
      ref += selw_h[slot]*acc*sdh[o]; }
    double rel=fabs(ref-hh[o])/(fabs(ref)+1e-6); maxrel=std::max(maxrel,rel);
    printf("  int4 down o=%-4d  kernel=%.4f  cpuref=%.4f  rel=%.2e\n",o,hh[o],ref,rel);
  }

  double fp8_bytes=(double)E*(gu+dn), int4_bytes=fp8_bytes/2.0;   // fp8: 1B/elem; int4: 0.5B/elem
  float msf=t_fp8(), msi=t_int4();
  printf("\n  %-12s %9s %10s %7s\n","precision","ms/call","GB/s","e");
  printf("  %-12s %9.4f %10.1f %7.3f\n","fp8 (winner)", msf, fp8_bytes/1e6/msf, fp8_bytes/1e6/msf/PEAK);
  printf("  %-12s %9.4f %10.1f %7.3f\n","int4", msi, int4_bytes/1e6/msi, int4_bytes/1e6/msi/PEAK);
  printf("  int4 speedup vs fp8: %.2fx  (2.0x = bandwidth-bound ideal; <2.0x = unpack issue-bound)\n", msf/msi);
  printf("  int4 unpack correctness max_rel=%.2e (vs CPU ref; should be ~0)\n", maxrel);
  return 0;
}
