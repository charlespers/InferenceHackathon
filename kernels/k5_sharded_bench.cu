// k5_sharded_bench.cu — does the TP=8-SHARDED K5 hold ~45% MBU, or starve?
//
// THE QUESTION (from the integrated 4.1% MBU finding): at TP=8 each rank holds only 1/8 of every
// expert's intermediate columns (MOE_INTER 1536 -> 192/rank).  This changes the two MoE GEMVs in
// OPPOSITE ways:
//   Kernel A (gate+up): inner contraction is HIDDEN=4096 (UNCHANGED); only the row count shrinks
//                       1536 -> 192/expert.  Same per-warp work -> should hold.
//   Kernel B (down):    inner contraction is MOE_INTER -> 192/rank.  192/16 = 12 uint4 across 32
//                       lanes -> ~20 idle lanes, almost no per-lane work to hide HBM latency.  This
//                       is the starvation suspect.
//
// We measure BOTH geometries (full 1536, sharded 192) for BOTH the current warp-per-row kernel and a
// cp.async pipelined v3 variant, identical numerics, validated vs CPU fp32 ref (<1e-2).  The sharded
// kernels are byte-for-byte the geometry decode_step_tp8.cu launches (tp8_k5a_gateup/tp8_k5b_down).
//
// build: nvcc -arch=sm_90a -O3 --use_fast_math -I kernels kernels/k5_sharded_bench.cu -o /tmp/k5sh
// run:   CUDA_VISIBLE_DEVICES=7 /tmp/k5sh [block] [PEAK_GBps]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)

// =================================================================================================
// warp_dot — identical to k5_experts.cu / tp8_warp_dot (split-K coalesced fp8, hw fp8x2->half2).
// =================================================================================================
static __device__ __forceinline__ float warp_dot(const fp8* __restrict__ w,
                                                 const float* __restrict__ ys, int n, int lane){
  float a0=0.f, a1=0.f;
  const uint4* __restrict__ wv=reinterpret_cast<const uint4*>(w);
  const int nv=n>>4;
  for(int v=lane; v<nv; v+=32){
    uint4 p=wv[v];
    const unsigned* wu=reinterpret_cast<const unsigned*>(&p);
    const float* yy=ys+(v<<4);
    #pragma unroll
    for(int q=0;q<4;q++){
      unsigned wq=wu[q];
      __nv_fp8x2_e4m3 lo,hi; lo.__x=(unsigned short)(wq&0xffffu); hi.__x=(unsigned short)(wq>>16);
      float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi);
      const float* yq=yy+(q<<2);
      a0+=yq[0]*fl.x; a1+=yq[1]*fl.y; a0+=yq[2]*fh.x; a1+=yq[3]*fh.y;
    }
  }
  float acc=a0+a1;
  #pragma unroll
  for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,o);
  return acc;
}

// =================================================================================================
// Parameterized warp-per-row kernels (INTER is a runtime arg so we run full=1536 and shard=192).
// Wgu layout [2*INTER, HIDDEN] (rows [0,INTER) gate, [INTER,2*INTER) up). Wd layout [HIDDEN, INTER].
// =================================================================================================
extern "C" __global__ void kA_warp(const float* __restrict__ y, const int* __restrict__ sel,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Sgu,
    float* __restrict__ a_glb, int nslot, int INTER){
  extern __shared__ float ys[];                         // [HIDDEN]
  for(int k=threadIdx.x;k<HIDDEN;k+=blockDim.x) ys[k]=y[k];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int total=nslot*INTER;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/INTER, j=item-slot*INTER; const int e=sel[slot];
    const fp8* W=Wgu[e]; const float* S=Sgu[e];
    const float g=warp_dot(W+(size_t)j*HIDDEN,          ys, HIDDEN, lane);
    const float u=warp_dot(W+(size_t)(INTER+j)*HIDDEN,  ys, HIDDEN, lane);
    if(lane==0) a_glb[(size_t)slot*INTER+j]=silu(g*S[j])*(u*S[INTER+j]);
  }
}
extern "C" __global__ void kB_warp(const int* __restrict__ sel, const float* __restrict__ selw,
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Sd,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot, int INTER){
  extern __shared__ float as[];                         // [nslot*INTER]
  const int na=nslot*INTER;
  for(int i=threadIdx.x;i<na;i+=blockDim.x) as[i]=a_glb[i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int total=nslot*HIDDEN;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/HIDDEN, o=item-slot*HIDDEN;
    const int e=sel[slot]; const float gw=selw[slot];
    const fp8* W=Wd[e]; const float* S=Sd[e];
    const float d=warp_dot(W+(size_t)o*INTER, as+(size_t)slot*INTER, INTER, lane);
    if(lane==0) atomicAdd(&h_io[o], gw*d*S[o]);
  }
}

// =================================================================================================
// V2 kernel B for the SHARD: ROWS_PER_WARP output channels per warp.  At INTER=192 the contraction
// is only 12 uint4 (<32 lanes), so a single row barely occupies a warp and issues ~no MLP.  Giving
// one warp R adjacent output rows multiplies the in-flight independent loads R-fold WITHOUT widening
// the (fixed, tiny) contraction — the right lever when the inner dim, not the row count, is the
// starvation cause.  Each lane loads R uint4 per tile (R independent loads in flight) and keeps R
// running accumulators.  Same coalescing: lane v of row r reads uint4 v of that row.
template<int R>
__global__ void kB_warpR(const int* __restrict__ sel, const float* __restrict__ selw,
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Sd,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot, int INTER){
  extern __shared__ float as[];
  const int na=nslot*INTER;
  for(int i=threadIdx.x;i<na;i+=blockDim.x) as[i]=a_glb[i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int norow=(HIDDEN+R-1)/R;
  const int total=nslot*norow;
  const int nv=INTER>>4;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/norow, og=item-slot*norow; const int o0=og*R;
    const int e=sel[slot]; const float gw=selw[slot];
    const fp8* W=Wd[e]; const float* S=Sd[e];
    const float* asl=as+(size_t)slot*INTER;
    const uint4* wv[R];
    #pragma unroll
    for(int r=0;r<R;r++) wv[r]=reinterpret_cast<const uint4*>(W+(size_t)(o0+r)*INTER);
    float a0[R],a1[R];
    #pragma unroll
    for(int r=0;r<R;r++){a0[r]=0.f;a1[r]=0.f;}
    for(int v=lane; v<nv; v+=32){
      const float* yy=asl+(v<<4);
      uint4 p[R];
      #pragma unroll
      for(int r=0;r<R;r++) p[r]=wv[r][v];               // R independent loads in flight per lane
      #pragma unroll
      for(int r=0;r<R;r++){
        const unsigned* wu=reinterpret_cast<const unsigned*>(&p[r]);
        #pragma unroll
        for(int q=0;q<4;q++){
          unsigned wq=wu[q];
          __nv_fp8x2_e4m3 lo,hi; lo.__x=(unsigned short)(wq&0xffffu); hi.__x=(unsigned short)(wq>>16);
          float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi);
          const float* yq=yy+(q<<2);
          a0[r]+=yq[0]*fl.x; a1[r]+=yq[1]*fl.y; a0[r]+=yq[2]*fh.x; a1[r]+=yq[3]*fh.y;
        }
      }
    }
    #pragma unroll
    for(int r=0;r<R;r++){
      float acc=a0[r]+a1[r];
      #pragma unroll
      for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,o);
      if(lane==0){ const int o=o0+r; if(o<HIDDEN) atomicAdd(&h_io[o], gw*acc*S[o]); }
    }
  }
}

// V2 kernel A: R output channels (j) per warp.  Inner dim is HIDDEN=4096 (already fine), but at the
// shard A has only 192 rows/expert -> 1536 total rows for 264*8=2112 warps: UNDER-subscribed, the
// grid empties before HBM latency is hidden.  R rows/warp keeps every warp busy longer AND raises
// MLP.  Dots both gate and up rows for each of R channels (2R rows/warp).
template<int R>
__global__ void kA_warpR(const float* __restrict__ y, const int* __restrict__ sel,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Sgu,
    float* __restrict__ a_glb, int nslot, int INTER){
  extern __shared__ float ys[];
  for(int k=threadIdx.x;k<HIDDEN;k+=blockDim.x) ys[k]=y[k];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int njrow=(INTER+R-1)/R;
  const int total=nslot*njrow;
  const int nv=HIDDEN>>4;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/njrow, jg=item-slot*njrow; const int j0=jg*R;
    const int e=sel[slot]; const fp8* W=Wgu[e]; const float* S=Sgu[e];
    // 2R rows: gate[j0..j0+R), up[j0..j0+R)
    const uint4* gv[R]; const uint4* uv[R];
    #pragma unroll
    for(int r=0;r<R;r++){ gv[r]=reinterpret_cast<const uint4*>(W+(size_t)(j0+r)*HIDDEN);
                          uv[r]=reinterpret_cast<const uint4*>(W+(size_t)(INTER+j0+r)*HIDDEN); }
    float g0[R],g1[R],u0[R],u1[R];
    #pragma unroll
    for(int r=0;r<R;r++){g0[r]=g1[r]=u0[r]=u1[r]=0.f;}
    for(int v=lane; v<nv; v+=32){
      const float* yy=ys+(v<<4);
      uint4 gp[R],up[R];
      #pragma unroll
      for(int r=0;r<R;r++){ gp[r]=gv[r][v]; up[r]=uv[r][v]; }   // 2R independent loads in flight
      #pragma unroll
      for(int r=0;r<R;r++){
        const unsigned* gu=reinterpret_cast<const unsigned*>(&gp[r]);
        const unsigned* uu=reinterpret_cast<const unsigned*>(&up[r]);
        #pragma unroll
        for(int q=0;q<4;q++){
          const float* yq=yy+(q<<2);
          unsigned gq=gu[q]; __nv_fp8x2_e4m3 gl,gh; gl.__x=(unsigned short)(gq&0xffffu); gh.__x=(unsigned short)(gq>>16);
          float2 gfl=__half22float2((__half2)gl), gfh=__half22float2((__half2)gh);
          g0[r]+=yq[0]*gfl.x; g1[r]+=yq[1]*gfl.y; g0[r]+=yq[2]*gfh.x; g1[r]+=yq[3]*gfh.y;
          unsigned uq=uu[q]; __nv_fp8x2_e4m3 ul,uh; ul.__x=(unsigned short)(uq&0xffffu); uh.__x=(unsigned short)(uq>>16);
          float2 ufl=__half22float2((__half2)ul), ufh=__half22float2((__half2)uh);
          u0[r]+=yq[0]*ufl.x; u1[r]+=yq[1]*ufl.y; u0[r]+=yq[2]*ufh.x; u1[r]+=yq[3]*ufh.y;
        }
      }
    }
    #pragma unroll
    for(int r=0;r<R;r++){
      float gacc=g0[r]+g1[r], uacc=u0[r]+u1[r];
      #pragma unroll
      for(int o=16;o>0;o>>=1){ gacc+=__shfl_down_sync(0xffffffffu,gacc,o); uacc+=__shfl_down_sync(0xffffffffu,uacc,o); }
      if(lane==0){ const int j=j0+r; if(j<INTER) a_glb[(size_t)slot*INTER+j]=silu(gacc*S[j])*(uacc*S[INTER+j]); }
    }
  }
}

// =================================================================================================
// CPU fp32 reference (parameterized on INTER).
// =================================================================================================
static void reference(const float* y, const int* sel, const float* selw,
    const fp8* const* Wgu, const float* const* Sgu, const fp8* const* Wd, const float* const* Sd,
    float* h, int nslot, int INTER){
  std::vector<float> a(INTER);
  for(int slot=0;slot<nslot;slot++){
    const int e=sel[slot]; const fp8* W=Wgu[e]; const float* Sg=Sgu[e];
    for(int j=0;j<INTER;j++){
      const fp8* gr=W+(size_t)j*HIDDEN; const fp8* ur=W+(size_t)(INTER+j)*HIDDEN;
      double g=0,u=0; for(int k=0;k<HIDDEN;k++){ g+=(double)y[k]*(double)(float)gr[k]; u+=(double)y[k]*(double)(float)ur[k]; }
      float gs=(float)g*Sg[j], us=(float)u*Sg[INTER+j];
      a[j]=(gs/(1.0f+expf(-gs)))*us;
    }
    const fp8* Wdn=Wd[e]; const float* Sdn=Sd[e]; const float gw=selw[slot];
    for(int o=0;o<HIDDEN;o++){
      const fp8* dr=Wdn+(size_t)o*INTER; double acc=0;
      for(int j=0;j<INTER;j++) acc+=(double)a[j]*(double)(float)dr[j];
      h[o]+=gw*(float)acc*Sdn[o];
    }
  }
}

static inline unsigned hash_u(unsigned x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
static inline float rnd(unsigned seed,size_t i,float sc,bool pos){ unsigned h=hash_u((unsigned)(i*2654435761u)^(seed*40503u));
  float v=(((h%2001)/1000.0f)-1.0f)*sc; return pos?(fabsf(v)+1e-3f):v; }

// CTA count: same heuristic as the repo (fill 132 SMs, cap 264), but row-group aware.
static inline int ctas_for(long rows, int R, int block){
  int wpc=block>>5; long rg=(rows+R-1)/R; int need=(int)((rg+wpc-1)/wpc);
  return std::min(std::max(need,132),264);
}

int main(int argc,char**argv){
  const int E=8;
  const int BLK=(argc>1)?atoi(argv[1]):256;
  const double PEAK=(argc>2)?atof(argv[2]):3350.0;
  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  printf("device: %s  SMs=%d  smemOptin=%d KB  PEAK=%.0f GB/s  block=%d\n",
         prop.name, prop.multiProcessorCount, prop.sharedMemPerBlockOptin>>10, PEAK, BLK);

  // Run two geometries: full (INTER=1536) and TP=8 shard (INTER=192).
  struct Geo { const char* name; int INTER; };
  const int TP8 = 8;   // tensor-parallel ranks (decode_step_tp8.cu: MOE_INTER_RANK = 1536/8 = 192)
  Geo geos[2] = { {"FULL  (INTER=1536)", MOE_INTER}, {"SHARD (INTER=192, TP=8)", MOE_INTER/TP8} };

  cudaEvent_t ev0,ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
  const int WARM=30, IT=300;

  for(int gi=0; gi<2; ++gi){
    const int INTER=geos[gi].INTER;
    printf("\n================= %s =================\n", geos[gi].name);
    const size_t gu_n=(size_t)2*INTER*HIDDEN, d_n=(size_t)HIDDEN*INTER;

    // host weights (so the CPU ref sees identical fp8 bytes)
    std::vector<std::vector<fp8>> Wgu_h(E),Wd_h(E); std::vector<std::vector<float>> Sgu_h(E),Sd_h(E);
    for(int e=0;e<E;e++){ Wgu_h[e].resize(gu_n); Wd_h[e].resize(d_n); Sgu_h[e].resize(2*INTER); Sd_h[e].resize(HIDDEN);
      for(size_t i=0;i<gu_n;i++) Wgu_h[e][i]=(fp8)rnd(1u+e,i,0.25f,false);
      for(size_t i=0;i<d_n;i++)  Wd_h[e][i]=(fp8)rnd(100u+e,i,0.25f,false);
      for(int i=0;i<2*INTER;i++) Sgu_h[e][i]=rnd(7u+e,i,0.02f,true);
      for(int i=0;i<HIDDEN;i++)  Sd_h[e][i]=rnd(13u+e,i,0.02f,true); }
    std::vector<float> y_h(HIDDEN); for(int k=0;k<HIDDEN;k++) y_h[k]=rnd(99u,k,1.0f,false);
    std::vector<int> sel_h(E); std::vector<float> selw_h(E);
    for(int e=0;e<E;e++){ sel_h[e]=e; selw_h[e]=0.1f+0.01f*e; }

    // upload
    std::vector<fp8*> Wgu_dp(E),Wd_dp(E); std::vector<float*> Sgu_dp(E),Sd_dp(E);
    for(int e=0;e<E;e++){
      CK(cudaMalloc(&Wgu_dp[e],gu_n*sizeof(fp8))); CK(cudaMalloc(&Wd_dp[e],d_n*sizeof(fp8)));
      CK(cudaMalloc(&Sgu_dp[e],2*INTER*sizeof(float))); CK(cudaMalloc(&Sd_dp[e],HIDDEN*sizeof(float)));
      CK(cudaMemcpy(Wgu_dp[e],Wgu_h[e].data(),gu_n*sizeof(fp8),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(Wd_dp[e],Wd_h[e].data(),d_n*sizeof(fp8),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(Sgu_dp[e],Sgu_h[e].data(),2*INTER*sizeof(float),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(Sd_dp[e],Sd_h[e].data(),HIDDEN*sizeof(float),cudaMemcpyHostToDevice)); }
    const fp8 **Wgu_d,**Wd_d; const float **Sgu_d,**Sd_d;
    CK(cudaMalloc(&Wgu_d,E*sizeof(fp8*))); CK(cudaMemcpy(Wgu_d,Wgu_dp.data(),E*sizeof(fp8*),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Wd_d,E*sizeof(fp8*)));  CK(cudaMemcpy(Wd_d,Wd_dp.data(),E*sizeof(fp8*),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Sgu_d,E*sizeof(float*))); CK(cudaMemcpy(Sgu_d,Sgu_dp.data(),E*sizeof(float*),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&Sd_d,E*sizeof(float*)));  CK(cudaMemcpy(Sd_d,Sd_dp.data(),E*sizeof(float*),cudaMemcpyHostToDevice));
    int *sel_d; float *selw_d,*y_d,*h_d,*a_d;
    CK(cudaMalloc(&sel_d,E*sizeof(int))); CK(cudaMemcpy(sel_d,sel_h.data(),E*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&selw_d,E*sizeof(float))); CK(cudaMemcpy(selw_d,selw_h.data(),E*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&y_d,HIDDEN*sizeof(float))); CK(cudaMemcpy(y_d,y_h.data(),HIDDEN*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&h_d,HIDDEN*sizeof(float)));
    CK(cudaMalloc(&a_d,(size_t)E*INTER*sizeof(float)));
    CK(cudaDeviceSynchronize());

    // reference
    std::vector<float> ref(HIDDEN,0.f);
    std::vector<const fp8*> Wgu_hp(E),Wd_hp(E); std::vector<const float*> Sgu_hp(E),Sd_hp(E);
    for(int e=0;e<E;e++){ Wgu_hp[e]=Wgu_h[e].data(); Wd_hp[e]=Wd_h[e].data(); Sgu_hp[e]=Sgu_h[e].data(); Sd_hp[e]=Sd_h[e].data(); }
    reference(y_h.data(),sel_h.data(),selw_h.data(),Wgu_hp.data(),Sgu_hp.data(),Wd_hp.data(),Sd_hp.data(),ref.data(),E,INTER);

    const size_t smemA=(size_t)HIDDEN*sizeof(float), smemB=(size_t)E*INTER*sizeof(float);
    CK(cudaFuncSetAttribute(kA_warp,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemA));
    CK(cudaFuncSetAttribute(kB_warp,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemB));
    CK(cudaFuncSetAttribute(kA_warpR<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemA));
    CK(cudaFuncSetAttribute(kA_warpR<4>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemA));
    CK(cudaFuncSetAttribute(kB_warpR<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemB));
    CK(cudaFuncSetAttribute(kB_warpR<4>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemB));
    CK(cudaFuncSetAttribute(kB_warpR<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemB));
    CK(cudaFuncSetAttribute(kB_warpR<16>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemB));
    CK(cudaFuncSetAttribute(kA_warpR<3>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemA));

    const double bA=(double)E*gu_n, bB=(double)E*d_n;
    auto gbps=[&](double by,float ms){ return by/1e6/ms; };

    // ---- verify each variant, then time it ----
    auto verifyB=[&](auto launchAB)->double{
      CK(cudaMemset(h_d,0,HIDDEN*sizeof(float)));
      launchAB();
      CK(cudaDeviceSynchronize());
      std::vector<float> got(HIDDEN); CK(cudaMemcpy(got.data(),h_d,HIDDEN*sizeof(float),cudaMemcpyDeviceToHost));
      double mr=0; for(int i=0;i<HIDDEN;i++){ double ad=fabs((double)ref[i]-got[i]); mr=std::max(mr,ad/(fabs((double)ref[i])+1e-6)); }
      return mr;
    };
    auto timeit=[&](auto fn)->float{
      for(int i=0;i<WARM;i++) fn();
      CK(cudaDeviceSynchronize()); CK(cudaEventRecord(ev0));
      for(int i=0;i<IT;i++) fn();
      CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
      float ms; CK(cudaEventElapsedTime(&ms,ev0,ev1)); return ms/IT;
    };

    // baseline warp (R=1) — A and B separately so we see which starves
    int cAw=ctas_for((long)E*INTER,1,BLK), cBw=ctas_for((long)E*HIDDEN,1,BLK);
    auto Aw =[&](){ kA_warp<<<cAw,BLK,smemA>>>(y_d,sel_d,Wgu_d,Sgu_d,a_d,E,INTER); };
    auto Bw =[&](){ kB_warp<<<cBw,BLK,smemB>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E,INTER); };
    double mrW=verifyB([&](){ Aw(); Bw(); });
    float msAw=timeit(Aw), msBw=timeit(Bw);

    // R=2 and R=4 variants for A and B
    int cA2=ctas_for((long)E*INTER,2,BLK), cB2=ctas_for((long)E*HIDDEN,2,BLK);
    int cA4=ctas_for((long)E*INTER,4,BLK), cB4=ctas_for((long)E*HIDDEN,4,BLK);
    auto A2=[&](){ kA_warpR<2><<<cA2,BLK,smemA>>>(y_d,sel_d,Wgu_d,Sgu_d,a_d,E,INTER); };
    auto A4=[&](){ kA_warpR<4><<<cA4,BLK,smemA>>>(y_d,sel_d,Wgu_d,Sgu_d,a_d,E,INTER); };
    int cB8=ctas_for((long)E*HIDDEN,8,BLK), cB16=ctas_for((long)E*HIDDEN,16,BLK);
    int cA3=ctas_for((long)E*INTER,3,BLK);
    auto A3=[&](){ kA_warpR<3><<<cA3,BLK,smemA>>>(y_d,sel_d,Wgu_d,Sgu_d,a_d,E,INTER); };
    auto B2=[&](){ kB_warpR<2><<<cB2,BLK,smemB>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E,INTER); };
    auto B4=[&](){ kB_warpR<4><<<cB4,BLK,smemB>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E,INTER); };
    auto B8=[&](){ kB_warpR<8><<<cB8,BLK,smemB>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E,INTER); };
    auto B16=[&](){ kB_warpR<16><<<cB16,BLK,smemB>>>(sel_d,selw_d,Wd_d,Sd_d,a_d,h_d,E,INTER); };
    double mrA2=verifyB([&](){ A2(); Bw(); });
    double mrA3=verifyB([&](){ A3(); Bw(); });
    double mrA4=verifyB([&](){ A4(); Bw(); });
    double mrB2=verifyB([&](){ Aw(); B2(); });
    double mrB4=verifyB([&](){ Aw(); B4(); });
    double mrB8=verifyB([&](){ Aw(); B8(); });
    double mrB16=verifyB([&](){ Aw(); B16(); });
    float msA2=timeit(A2), msA3=timeit(A3), msA4=timeit(A4);
    float msB2=timeit(B2), msB4=timeit(B4), msB8=timeit(B8), msB16=timeit(B16);
    CK(cudaGetLastError());

    auto row=[&](const char* nm,double by,float ms,double mr){
      double gb=gbps(by,ms);
      printf("  %-22s %9.2f us  %9.1f GB/s  %6.1f%%  max_rel=%.1e %s\n",
             nm, ms*1e3, gb, 100.0*gb/PEAK, mr, mr<1e-2?"PASS":"FAIL"); };
    printf("  per-token weight read: A(gate+up)=%.1f MB  B(down)=%.1f MB\n", bA/1e6, bB/1e6);
    printf("  --- kernel A (gate+up), inner contraction = HIDDEN=4096 (shard changes ROWS only) ---\n");
    row("A warp R=1",   bA,msAw, mrW);
    row("A warpR R=2",  bA,msA2, mrA2);
    row("A warpR R=3",  bA,msA3, mrA3);
    row("A warpR R=4",  bA,msA4, mrA4);
    printf("  --- kernel B (down), inner contraction = INTER (shard SHRINKS this 1536->192) ---\n");
    row("B warp R=1",   bB,msBw, mrW);
    row("B warpR R=2",  bB,msB2, mrB2);
    row("B warpR R=4",  bB,msB4, mrB4);
    row("B warpR R=8",  bB,msB8, mrB8);
    row("B warpR R=16", bB,msB16,mrB16);

    // best A + best B fused number
    float bestA=std::min({msAw,msA2,msA3,msA4}), bestB=std::min({msBw,msB2,msB4,msB8,msB16});
    const char* nA = (bestA==msAw)?"R=1":(bestA==msA2)?"R=2":(bestA==msA3)?"R=3":"R=4";
    const char* nB = (bestB==msBw)?"R=1":(bestB==msB2)?"R=2":(bestB==msB4)?"R=4":(bestB==msB8)?"R=8":"R=16";
    double bT=bA+bB; float msT=bestA+bestB;
    printf("  --- best fused A+B ---\n");
    printf("  best A=%s (%.2fus) + best B=%s (%.2fus) = %.2f us/tok  %.1f GB/s  %.1f%% MBU\n",
           nA,bestA*1e3, nB,bestB*1e3, msT*1e3, gbps(bT,msT), 100.0*gbps(bT,msT)/PEAK);
    printf("  baseline warp R=1 fused: %.2f us/tok  %.1f%% MBU\n",
           (msAw+msBw)*1e3, 100.0*gbps(bT,msAw+msBw)/PEAK);

    for(int e=0;e<E;e++){ cudaFree(Wgu_dp[e]); cudaFree(Wd_dp[e]); cudaFree(Sgu_dp[e]); cudaFree(Sd_dp[e]); }
    cudaFree(Wgu_d);cudaFree(Wd_d);cudaFree(Sgu_d);cudaFree(Sd_d);
    cudaFree(sel_d);cudaFree(selw_d);cudaFree(y_d);cudaFree(h_d);cudaFree(a_d);
  }
  return 0;
}
