// k4_router_fast.cu — occupancy-fixed MoE router microbench: multi-block gate GEMV + parallel top-8,
// vs the 1-CTA k4_router (110us) and the GEMM-gate+select path (21.7us). The gate reads only
// Wgate[128,4096] fp8 = 0.5 MB -> ideal ~0.15us HBM; the 1-CTA version starves the GPU (1 block / 132 SMs).
//
// FAST DESIGN:
//   kernel1 (gate): grid = N_EXPERTS blocks (one CTA per expert), block = 256 threads. Each block
//     RMSNorm-stages y once (cheap, 4096 floats) and dots ITS expert row against y (256-way split-K),
//     writes one logit. 128 CTAs fill the GPU -> the 0.5 MB read is bandwidth-bound, not 1-block-bound.
//   kernel2 (select): 1 block, 128 threads -> block-parallel softmax + top-8 (iterative argmax with a
//     shared reduction), renormalize. O(128*8) but parallel across 128 lanes, not single-threaded.
//
// Build: nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k4_router_fast.cu -o /tmp/k4fast
#include "common.cuh"
#include <cfloat>
#include <cstdio>
#include <vector>
using namespace q3;
#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ printf("ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)

// ---- kernel1: one CTA per expert. RMSNorm(h) -> y in smem, dot Wgate[e] -> logits[e]. ----
extern "C" __global__ void k4f_gate(const float* __restrict__ h, const float* __restrict__ wn,
                                    const fp8* __restrict__ Wgate, const float* __restrict__ Wscale,
                                    float* __restrict__ logits) {
  const int e = blockIdx.x;                       // this CTA owns expert e
  extern __shared__ float ys[];                   // [HIDDEN]
  // RMSNorm (block-wide) — cheap relative to the gate read, computed per-CTA to avoid a global y pass.
  float part = 0.f;
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) { float v = h[i]; part += v*v; }
  for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(~0u, part, o);
  __shared__ float wss[32]; const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) wss[wid] = part; __syncthreads();
  __shared__ float rinv;
  if (threadIdx.x == 0) { float ss=0; int nw=(blockDim.x+31)>>5; for(int i=0;i<nw;i++) ss+=wss[i];
                          rinv = rsqrtf(ss/HIDDEN + RMS_EPS); }
  __syncthreads();
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) ys[i] = h[i]*rinv*wn[i];
  __syncthreads();
  // gate dot: 256-way split-K over HIDDEN, uint4 (16 fp8) per step.
  const uint4* wv = reinterpret_cast<const uint4*>(Wgate + (size_t)e*HIDDEN);
  const int nv = HIDDEN >> 4;
  float acc = 0.f;
  for (int v = threadIdx.x; v < nv; v += blockDim.x) {
    uint4 p = wv[v]; const unsigned* wu = (const unsigned*)&p; const float* yy = ys + (v<<4);
    #pragma unroll
    for (int q=0;q<4;++q){ unsigned wq=wu[q]; __nv_fp8x2_e4m3 lo,hi; lo.__x=(unsigned short)(wq&0xffff); hi.__x=(unsigned short)(wq>>16);
      float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi); const float* yq=yy+(q<<2);
      acc += yq[0]*fl.x + yq[1]*fl.y + yq[2]*fh.x + yq[3]*fh.y; }
  }
  for (int o=16;o>0;o>>=1) acc += __shfl_down_sync(~0u, acc, o);
  __shared__ float blk[32]; if (lane==0) blk[wid]=acc; __syncthreads();
  if (threadIdx.x==0){ float s=0; int nw=(blockDim.x+31)>>5; for(int i=0;i<nw;i++) s+=blk[i]; logits[e]=s*Wscale[e]; }
}

// ---- kernel2: 1 block, parallel softmax + top-8 over 128 logits. ----
extern "C" __global__ void k4f_select(const float* __restrict__ logits, int* __restrict__ sel_idx, float* __restrict__ sel_w) {
  const int t = threadIdx.x;                       // 128 threads, one per expert
  __shared__ float lg[N_EXPERTS]; __shared__ int taken[N_EXPERTS];
  lg[t] = logits[t]; taken[t] = 0; __syncthreads();
  // max + sumexp via shared reductions (simple, 128-wide)
  __shared__ float red[N_EXPERTS];
  red[t]=lg[t]; __syncthreads();
  for (int s=64;s>0;s>>=1){ if(t<s) red[t]=fmaxf(red[t],red[t+s]); __syncthreads(); }
  float mx=red[0]; __syncthreads();
  red[t]=__expf(lg[t]-mx); __syncthreads();
  for (int s=64;s>0;s>>=1){ if(t<s) red[t]+=red[t+s]; __syncthreads(); }
  float inv=1.f/red[0]; __syncthreads();
  // iterative top-8 argmax (8 passes; each pass: parallel argmax over un-taken)
  __shared__ int   bi_sh; __shared__ float bv_sh; __shared__ float chosen;
  if (t==0) chosen=0.f; __syncthreads();
  for (int s=0;s<TOP_K;++s){
    red[t] = taken[t] ? -1.f : __expf(lg[t]-mx)*inv;   // prob, -1 if taken
    __shared__ int idx[N_EXPERTS]; idx[t]=t; __syncthreads();
    for (int k=64;k>0;k>>=1){ if(t<k){ if(red[t+k]>red[t]){ red[t]=red[t+k]; idx[t]=idx[t+k]; } } __syncthreads(); }
    if (t==0){ bi_sh=idx[0]; bv_sh=red[0]; taken[idx[0]]=1; sel_idx[s]=idx[0]; sel_w[s]=red[0]; chosen+=red[0]; }
    __syncthreads();
  }
  if (t==0){ float ic=1.f/chosen; for(int s=0;s<TOP_K;++s) sel_w[s]*=ic; }
}

// ---- the OLD 1-CTA router (reference), copied minimal for the head-to-head ----
extern "C" __global__ void k4_old(const float* __restrict__ h, const float* __restrict__ wn,
                                  const fp8* __restrict__ Wgate, const float* __restrict__ Wscale,
                                  int* __restrict__ sel_idx, float* __restrict__ sel_w) {
  extern __shared__ float smem[]; float* ys=smem; __shared__ float logits[N_EXPERTS];
  float part=0.f; for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x){float v=h[i];part+=v*v;}
  for(int o=16;o>0;o>>=1) part+=__shfl_down_sync(~0u,part,o);
  __shared__ float wss[32]; int lane=threadIdx.x&31,wid=threadIdx.x>>5; if(lane==0)wss[wid]=part; __syncthreads();
  __shared__ float rinv; if(threadIdx.x==0){float ss=0;int nw=(blockDim.x+31)>>5;for(int i=0;i<nw;i++)ss+=wss[i];rinv=rsqrtf(ss/HIDDEN+RMS_EPS);} __syncthreads();
  for(int i=threadIdx.x;i<HIDDEN;i+=blockDim.x) ys[i]=h[i]*rinv*wn[i]; __syncthreads();
  int gw=threadIdx.x>>5,nw=blockDim.x>>5;
  for(int e=gw;e<N_EXPERTS;e+=nw){ const uint4* wv=(const uint4*)(Wgate+(size_t)e*HIDDEN); int nv=HIDDEN>>4; float a=0;
    for(int v=lane;v<nv;v+=32){ uint4 p=wv[v]; const unsigned* wu=(const unsigned*)&p; const float* yy=ys+(v<<4);
      #pragma unroll
      for(int q=0;q<4;++q){unsigned wq=wu[q];__nv_fp8x2_e4m3 lo,hi;lo.__x=(unsigned short)(wq&0xffff);hi.__x=(unsigned short)(wq>>16);
        float2 fl=__half22float2((__half2)lo),fh=__half22float2((__half2)hi);const float* yq=yy+(q<<2);
        a+=yq[0]*fl.x+yq[1]*fl.y+yq[2]*fh.x+yq[3]*fh.y;}}
    for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(~0u,a,o); if(lane==0)logits[e]=a*Wscale[e]; }
  __syncthreads();
  if(threadIdx.x==0){ float mx=-FLT_MAX; for(int e=0;e<N_EXPERTS;++e)mx=fmaxf(mx,logits[e]); float sum=0; for(int e=0;e<N_EXPERTS;++e)sum+=__expf(logits[e]-mx); float is=1.f/sum,ch=0;
    for(int s=0;s<TOP_K;++s){int bi=-1;float bv=-1;for(int e=0;e<N_EXPERTS;++e){bool tk=false;for(int j=0;j<s;++j)if(sel_idx[j]==e){tk=true;break;}if(tk)continue;float p=__expf(logits[e]-mx)*is;if(p>bv){bv=p;bi=e;}}sel_idx[s]=bi;sel_w[s]=bv;ch+=bv;}
    float ic=1.f/ch; for(int s=0;s<TOP_K;++s)sel_w[s]*=ic; }
}

__global__ void fillf(float* p,int n,unsigned s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned x=(i*2654435761u)^s;p[i]=((x>>9)*1.1920929e-7f-0.5f)*0.1f;}}
__global__ void fill8(fp8* p,size_t n,unsigned s){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned x=((unsigned)i*2246822519u)^s;p[i]=(fp8)(((x>>9)*1.1920929e-7f-0.5f)*0.2f);}}

int main(){
  float *h,*wn,*Wscale,*logits,*sel_w; fp8* Wgate; int* sel_idx;
  CK(cudaMalloc(&h,HIDDEN*4)); CK(cudaMalloc(&wn,HIDDEN*4)); CK(cudaMalloc(&Wscale,N_EXPERTS*4));
  CK(cudaMalloc(&Wgate,(size_t)N_EXPERTS*HIDDEN)); CK(cudaMalloc(&logits,N_EXPERTS*4));
  CK(cudaMalloc(&sel_idx,TOP_K*4)); CK(cudaMalloc(&sel_w,TOP_K*4));
  fillf<<<(HIDDEN+255)/256,256>>>(h,HIDDEN,1); fillf<<<(HIDDEN+255)/256,256>>>(wn,HIDDEN,2);
  fillf<<<1,N_EXPERTS>>>(Wscale,N_EXPERTS,3); fill8<<<(N_EXPERTS*HIDDEN+255)/256,256>>>(Wgate,(size_t)N_EXPERTS*HIDDEN,4);
  CK(cudaDeviceSynchronize());
  size_t smem=(size_t)HIDDEN*4; cudaFuncSetAttribute(k4f_gate,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  cudaFuncSetAttribute(k4_old,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  auto fast=[&](){ k4f_gate<<<N_EXPERTS,256,smem>>>(h,wn,Wgate,Wscale,logits); k4f_select<<<1,N_EXPERTS>>>(logits,sel_idx,sel_w); };
  auto old =[&](){ k4_old<<<1,256,smem>>>(h,wn,Wgate,Wscale,sel_idx,sel_w); };
  cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e)); const int IT=5000;
  for(int i=0;i<200;i++) old(); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(s)); for(int i=0;i<IT;i++) old(); CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
  float mo; CK(cudaEventElapsedTime(&mo,s,e));
  for(int i=0;i<200;i++) fast(); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(s)); for(int i=0;i<IT;i++) fast(); CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
  float mf; CK(cudaEventElapsedTime(&mf,s,e));
  printf("router gate+select us/token:\n  OLD 1-CTA : %.2f us\n  FAST mblk : %.2f us  (%.2fx)\n",
         mo*1e3/IT, mf*1e3/IT, (mo/mf));
  printf("  x94 layers: OLD %.0f us  FAST %.0f us  -> saves %.0f us/token on the forward\n",
         mo*1e3/IT*94, mf*1e3/IT*94, (mo-mf)*1e3/IT*94);
  return 0;
}
