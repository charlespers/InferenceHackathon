// k4_tp8_micro.cu — sweep KSPLIT for tp8_k4_gate_gemv + test a precomputed-RMSNorm variant.
// Build: nvcc -arch=sm_90a -O3 --use_fast_math -I . k4_tp8_micro.cu -o /tmp/k4mt
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
static constexpr float RMS_EPS_L = 1e-6f;

// gate_gemv parameterized by KSPLIT (template), redundant per-CTA RMSNorm.
template<int KS>
__global__ void gate_gemv(const float* h,const float* wpn,const fp8* Wg,float* gl){
  extern __shared__ float ys[];
  float part=0.f; for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x){float v=h[i];part+=v*v;}
  #pragma unroll
  for(int o=16;o>0;o>>=1)part+=__shfl_down_sync(0xffffffffu,part,o);
  __shared__ float wss[32]; const int lane=threadIdx.x&31,wid=threadIdx.x>>5;
  if(lane==0)wss[wid]=part; __syncthreads();
  __shared__ float ri; if(threadIdx.x==0){float ss=0;int nw=(blockDim.x+31)>>5;for(int i=0;i<nw;i++)ss+=wss[i];ri=rsqrtf(ss/HIDDEN+RMS_EPS_L);}
  __syncthreads(); const float rinv=ri;
  for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x)ys[i]=h[i]*rinv*wpn[i];
  __syncthreads();
  const int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nw=(gridDim.x*blockDim.x)>>5;
  const int total=N_EXPERTS*KS, nv=HIDDEN>>4, chunk=(nv+KS-1)/KS;
  for(int it=gw;it<total;it+=nw){ int e=it/KS,ks=it-e*KS,v0=ks*chunk,v1=min(v0+chunk,nv);
    const uint4* wv=(const uint4*)(Wg+(size_t)e*HIDDEN); float a0=0,a1=0;
    for(int v=v0+lane;v<v1;v+=32){ uint4 p=wv[v]; const unsigned* wu=(const unsigned*)&p; const float* yy=ys+(v<<4);
      #pragma unroll
      for(int q=0;q<4;++q){unsigned wq=wu[q];__nv_fp8x2_e4m3 lo,hi;lo.__x=(unsigned short)(wq&0xffffu);hi.__x=(unsigned short)(wq>>16);
        float2 fl=__half22float2((__half2)lo),fh=__half22float2((__half2)hi);const float* yq=yy+(q<<2);
        a0+=yq[0]*fl.x;a1+=yq[1]*fl.y;a0+=yq[2]*fh.x;a1+=yq[3]*fh.y;}}
    float acc=a0+a1;
    #pragma unroll
    for(int o=16;o>0;o>>=1)acc+=__shfl_down_sync(0xffffffffu,acc,o);
    if(lane==0)atomicAdd(&gl[e],acc); }
}

// precompute y = RMSNorm(h)*wpn once, then a pure gemv (no redundant norm).
__global__ void rmsnorm_y(const float* h,const float* wpn,float* y){
  __shared__ float wss[32]; float part=0.f; for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x){float v=h[i];part+=v*v;}
  #pragma unroll
  for(int o=16;o>0;o>>=1)part+=__shfl_down_sync(0xffffffffu,part,o);
  const int lane=threadIdx.x&31,wid=threadIdx.x>>5; if(lane==0)wss[wid]=part; __syncthreads();
  __shared__ float ri; if(threadIdx.x==0){float ss=0;int nw=(blockDim.x+31)>>5;for(int i=0;i<nw;i++)ss+=wss[i];ri=rsqrtf(ss/HIDDEN+RMS_EPS_L);}
  __syncthreads(); for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x)y[i]=h[i]*ri*wpn[i];
}
template<int KS>
__global__ void gate_gemv_pre(const float* y,const fp8* Wg,float* gl){
  const int lane=threadIdx.x&31;
  const int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nw=(gridDim.x*blockDim.x)>>5;
  const int total=N_EXPERTS*KS, nv=HIDDEN>>4, chunk=(nv+KS-1)/KS;
  for(int it=gw;it<total;it+=nw){ int e=it/KS,ks=it-e*KS,v0=ks*chunk,v1=min(v0+chunk,nv);
    const uint4* wv=(const uint4*)(Wg+(size_t)e*HIDDEN); float a0=0,a1=0;
    for(int v=v0+lane;v<v1;v+=32){ uint4 p=wv[v]; const unsigned* wu=(const unsigned*)&p; const float* yy=y+(v<<4);
      #pragma unroll
      for(int q=0;q<4;++q){unsigned wq=wu[q];__nv_fp8x2_e4m3 lo,hi;lo.__x=(unsigned short)(wq&0xffffu);hi.__x=(unsigned short)(wq>>16);
        float2 fl=__half22float2((__half2)lo),fh=__half22float2((__half2)hi);const float* yq=yy+(q<<2);
        a0+=yq[0]*fl.x;a1+=yq[1]*fl.y;a0+=yq[2]*fh.x;a1+=yq[3]*fh.y;}}
    float acc=a0+a1;
    #pragma unroll
    for(int o=16;o>0;o>>=1)acc+=__shfl_down_sync(0xffffffffu,acc,o);
    if(lane==0)atomicAdd(&gl[e],acc); }
}

static float frnd(unsigned s,size_t i){unsigned h=s*2654435761u+(unsigned)i*40503u;h^=h>>13;h*=2246822519u;h^=h>>16;return ((h%2001)/1000.f)-1.f;}

template<int KS>
double bench_gemv(const char* nm,const float* h,const float* wpn,const fp8* Wg,float* gl,cudaStream_t s,int reps){
  const int block=128,warps=block>>5,need=N_EXPERTS*KS,ctas=(need+warps-1)/warps;
  const size_t smem=(size_t)HIDDEN*4;
  auto fn=[&]{CK(cudaMemsetAsync(gl,0,N_EXPERTS*4,s)); gate_gemv<KS><<<ctas,block,smem,s>>>(h,wpn,Wg,gl);};
  for(int i=0;i<50;i++)fn(); CK(cudaStreamSynchronize(s));
  cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0,s));for(int i=0;i<reps;i++)fn();CK(cudaEventRecord(e1,s));CK(cudaEventSynchronize(e1));
  float ms=0;CK(cudaEventElapsedTime(&ms,e0,e1));double us=ms*1e3/reps;printf("  %-26s %8.3f us\n",nm,us);
  CK(cudaEventDestroy(e0));CK(cudaEventDestroy(e1));return us;
}
template<int KS>
double bench_pre(const char* nm,const float* h,const float* wpn,const fp8* Wg,float* y,float* gl,cudaStream_t s,int reps){
  const int block=128,warps=block>>5,need=N_EXPERTS*KS,ctas=(need+warps-1)/warps;
  auto fn=[&]{CK(cudaMemsetAsync(gl,0,N_EXPERTS*4,s)); rmsnorm_y<<<1,256,0,s>>>(h,wpn,y); gate_gemv_pre<KS><<<ctas,block,0,s>>>(y,Wg,gl);};
  for(int i=0;i<50;i++)fn(); CK(cudaStreamSynchronize(s));
  cudaEvent_t e0,e1;CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0,s));for(int i=0;i<reps;i++)fn();CK(cudaEventRecord(e1,s));CK(cudaEventSynchronize(e1));
  float ms=0;CK(cudaEventElapsedTime(&ms,e0,e1));double us=ms*1e3/reps;printf("  %-26s %8.3f us\n",nm,us);
  CK(cudaEventDestroy(e0));CK(cudaEventDestroy(e1));return us;
}

int main(int argc,char**argv){
  int reps=argc>1?atoi(argv[1]):4000;
  std::vector<float> hh(HIDDEN),hw(HIDDEN); for(int i=0;i<HIDDEN;i++){hh[i]=frnd(99u,i);hw[i]=fabsf(frnd(40u,i)*0.5f)+1e-3f;}
  std::vector<fp8> hg((size_t)N_EXPERTS*HIDDEN); for(size_t i=0;i<hg.size();i++)hg[i]=(fp8)(frnd(41u,i)*0.25f);
  float *h,*wpn,*gl,*y; fp8* Wg;
  CK(cudaMalloc(&h,HIDDEN*4));CK(cudaMemcpy(h,hh.data(),HIDDEN*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&wpn,HIDDEN*4));CK(cudaMemcpy(wpn,hw.data(),HIDDEN*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wg,hg.size()));CK(cudaMemcpy(Wg,hg.data(),hg.size(),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&gl,N_EXPERTS*4));CK(cudaMalloc(&y,HIDDEN*4));
  cudaStream_t s;CK(cudaStreamCreate(&s));
  printf("== K4 gate_gemv KSPLIT sweep (reps=%d) ==\n",reps);
  printf("-- redundant-RMSNorm (current structure) --\n");
  bench_gemv<4>("KSPLIT=4 (current)",h,wpn,Wg,gl,s,reps);
  bench_gemv<8>("KSPLIT=8",h,wpn,Wg,gl,s,reps);
  bench_gemv<16>("KSPLIT=16",h,wpn,Wg,gl,s,reps);
  bench_gemv<32>("KSPLIT=32",h,wpn,Wg,gl,s,reps);
  printf("-- precomputed RMSNorm (separate tiny kernel + pure gemv) --\n");
  bench_pre<4>("pre KSPLIT=4",h,wpn,Wg,y,gl,s,reps);
  bench_pre<8>("pre KSPLIT=8",h,wpn,Wg,y,gl,s,reps);
  bench_pre<16>("pre KSPLIT=16",h,wpn,Wg,y,gl,s,reps);
  bench_pre<32>("pre KSPLIT=32",h,wpn,Wg,y,gl,s,reps);
  // correctness: current vs precompute
  std::vector<float> g_cur(N_EXPERTS),g_pre(N_EXPERTS);
  CK(cudaMemsetAsync(gl,0,N_EXPERTS*4,s)); gate_gemv<8><<<(N_EXPERTS*8+3)/4,128,(size_t)HIDDEN*4,s>>>(h,wpn,Wg,gl);
  CK(cudaStreamSynchronize(s)); CK(cudaMemcpy(g_cur.data(),gl,N_EXPERTS*4,cudaMemcpyDeviceToHost));
  CK(cudaMemsetAsync(gl,0,N_EXPERTS*4,s)); rmsnorm_y<<<1,256,0,s>>>(h,wpn,y); gate_gemv_pre<8><<<(N_EXPERTS*8+3)/4,128,0,s>>>(y,Wg,gl);
  CK(cudaStreamSynchronize(s)); CK(cudaMemcpy(g_pre.data(),gl,N_EXPERTS*4,cudaMemcpyDeviceToHost));
  float me=0,mr=0; for(int i=0;i<N_EXPERTS;i++){me=fmaxf(me,fabsf(g_cur[i]-g_pre[i]));mr=fmaxf(mr,fabsf(g_cur[i]));}
  printf("  max|cur-pre|=%.3e  (ref max=%.3e)\n",me,mr);
  return 0;
}
