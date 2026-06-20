// k5_int4_shard_bench.cu — INT4 (W4A16, group-quant) vs fp8 expert GEMV at the TP=8 SHARDED dims.
// =================================================================================================
// WHY THIS FILE (vs the dead k5_experts_int4_v{1,2,3}.cu): those benched int4 against the FULL
// INTER=1536 geometry and hard-coded "fp8 = 98 us", concluding int4 = 0.58x (unpack-bound).  But the
// ENGINE runs the SHARDED geometry (MOE_INTER_RANK = 1536/8 = 192) with the MULTI-ROW-PER-WARP kernels
// (gate+up R=2, down R=16), NOT the warp-per-row GEMV the old benches used.  At the shard:
//   * down's inner contraction is only 192 (12 uint4) — the warp is STARVED on loads, so dequant ALU
//     is hidden behind the load-latency stall the engine's fp8 kernel ALREADY eats (22% MBU).
//   * R rows/warp amortize the per-load dequant constant cost over R independent rows.
// So the unpack-bound verdict must be RE-MEASURED at the geometry+kernel the engine actually uses, head
// to head against the SAME fp8 multi-row kernel.  This bench does exactly that, at M=1 (decode) and
// M=8 (spec-verify, where more activation columns amortize dequant further).
//
// DEQUANT (the v3 LOP3 half2 idiom — the only int4 path that was ever close): unpack 8 nibbles/word to
// four signed __half2 via (a&mask)|0x6400 then a single __hsub2(1032) folding the 1024 fp16 exponent
// offset AND the symmetric -8 zero-point.  No per-element integer->float convert.  Contract in fp32
// (the half2 weights -> float2 -> FMA), fold the per-GROUP fp16 scale at each group boundary.
//
// QUANT SCHEME: group-wise symmetric int4.  gate+up: GROUP_GU=128 along HIDDEN (4096 = 32 groups/row,
// AWQ/GPTQ standard).  down: GROUP_D=64 along the 192-wide shard (192 = 3 groups; 128 does NOT divide
// 192, so the down shard uses group-64 — slightly MORE scale bytes but byte-faithful and tiles the
// shard exactly).  Each lane owns whole uint4 (32 int4) chunks; 32 | GROUP so no chunk straddles a
// group boundary.
//
// build: nvcc -arch=sm_90a -O3 --use_fast_math -I kernels kernels/k5_int4_shard_bench.cu -o /tmp/k5i4sh
// run:   CUDA_VISIBLE_DEVICES=0 /tmp/k5i4sh [block=256] [PEAK_GBps=3350] [M=1]
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

constexpr int GROUP_GU = 128;   // gate+up int4 group along HIDDEN (4096/128 = 32 groups/row)
constexpr int GROUP_D  = 64;    // down    int4 group along the 192 shard (192/64 = 3 groups/row)

// -------------------------------------------------------------------------------------------------
// fp8 reference kernels — BYTE-IDENTICAL to k5_sharded_bench.cu / decode_step_tp8.cu (kA_warpR/kB_warpR).
// These are what the engine runs today; int4 must beat them at the shard.  Generalized to M activation
// columns (M=1 decode, M=8 spec-verify): the weight is read ONCE per warp and dotted against all M cols.
// -------------------------------------------------------------------------------------------------
template<int R, int M>
__global__ void fp8_kA(const float* __restrict__ y, const int* __restrict__ sel,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Sgu,
    float* __restrict__ a_glb, int nslot, int INTER){
  extern __shared__ float ysA[];                                   // [M*HIDDEN]
  for(int i=threadIdx.x;i<M*HIDDEN;i+=blockDim.x) ysA[i]=y[i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int njrow=(INTER+R-1)/R; const int total=nslot*njrow; const int nv=HIDDEN>>4;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/njrow, jg=item-slot*njrow; const int j0=jg*R;
    const int e=sel[slot]; const fp8* W=Wgu[e]; const float* S=Sgu[e];
    const uint4* gv[R]; const uint4* uv[R];
    #pragma unroll
    for(int r=0;r<R;r++){ gv[r]=reinterpret_cast<const uint4*>(W+(size_t)(j0+r)*HIDDEN);
                          uv[r]=reinterpret_cast<const uint4*>(W+(size_t)(INTER+j0+r)*HIDDEN); }
    float g[R*M], u[R*M];
    #pragma unroll
    for(int i=0;i<R*M;i++){ g[i]=0.f; u[i]=0.f; }
    for(int v=lane; v<nv; v+=32){
      uint4 gp[R],up[R];
      #pragma unroll
      for(int r=0;r<R;r++){ gp[r]=gv[r][v]; up[r]=uv[r][v]; }
      #pragma unroll
      for(int r=0;r<R;r++){
        const unsigned* gu=reinterpret_cast<const unsigned*>(&gp[r]);
        const unsigned* uu=reinterpret_cast<const unsigned*>(&up[r]);
        #pragma unroll
        for(int q=0;q<4;q++){
          unsigned gq=gu[q]; __nv_fp8x2_e4m3 gl,gh; gl.__x=(unsigned short)(gq&0xffffu); gh.__x=(unsigned short)(gq>>16);
          float2 gfl=__half22float2((__half2)gl), gfh=__half22float2((__half2)gh);
          unsigned uq=uu[q]; __nv_fp8x2_e4m3 ul,uh; ul.__x=(unsigned short)(uq&0xffffu); uh.__x=(unsigned short)(uq>>16);
          float2 ufl=__half22float2((__half2)ul), ufh=__half22float2((__half2)uh);
          #pragma unroll
          for(int m=0;m<M;m++){
            const float* yq=ysA+(size_t)m*HIDDEN+(v<<4)+(q<<2);
            g[r*M+m]+=yq[0]*gfl.x+yq[1]*gfl.y+yq[2]*gfh.x+yq[3]*gfh.y;
            u[r*M+m]+=yq[0]*ufl.x+yq[1]*ufl.y+yq[2]*ufh.x+yq[3]*ufh.y;
          }
        }
      }
    }
    #pragma unroll
    for(int r=0;r<R;r++) for(int m=0;m<M;m++){
      float gacc=g[r*M+m], uacc=u[r*M+m];
      #pragma unroll
      for(int o=16;o>0;o>>=1){ gacc+=__shfl_down_sync(0xffffffffu,gacc,o); uacc+=__shfl_down_sync(0xffffffffu,uacc,o); }
      if(lane==0){ const int j=j0+r; if(j<INTER)
        a_glb[((size_t)m*nslot+slot)*INTER+j]=silu(gacc*S[j])*(uacc*S[INTER+j]); }
    }
  }
}

template<int R, int M>
__global__ void fp8_kB(const int* __restrict__ sel, const float* __restrict__ selw,
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Sd,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot, int INTER){
  extern __shared__ float asB[];                                   // [M*nslot*INTER]
  const int na=M*nslot*INTER;
  for(int i=threadIdx.x;i<na;i+=blockDim.x) asB[i]=a_glb[i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int norow=(HIDDEN+R-1)/R; const int total=nslot*norow; const int nv=INTER>>4;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/norow, og=item-slot*norow; const int o0=og*R;
    const int e=sel[slot]; const float gw=selw[slot];
    const fp8* W=Wd[e]; const float* S=Sd[e];
    const uint4* wv[R];
    #pragma unroll
    for(int r=0;r<R;r++) wv[r]=reinterpret_cast<const uint4*>(W+(size_t)(o0+r)*INTER);
    float acc[R*M];
    #pragma unroll
    for(int i=0;i<R*M;i++) acc[i]=0.f;
    for(int v=lane; v<nv; v+=32){
      uint4 p[R];
      #pragma unroll
      for(int r=0;r<R;r++) p[r]=wv[r][v];
      #pragma unroll
      for(int r=0;r<R;r++){
        const unsigned* wu=reinterpret_cast<const unsigned*>(&p[r]);
        #pragma unroll
        for(int q=0;q<4;q++){
          unsigned wq=wu[q]; __nv_fp8x2_e4m3 lo,hi; lo.__x=(unsigned short)(wq&0xffffu); hi.__x=(unsigned short)(wq>>16);
          float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi);
          #pragma unroll
          for(int m=0;m<M;m++){
            const float* yq=asB+((size_t)m*nslot+slot)*INTER+(v<<4)+(q<<2);
            acc[r*M+m]+=yq[0]*fl.x+yq[1]*fl.y+yq[2]*fh.x+yq[3]*fh.y;
          }
        }
      }
    }
    #pragma unroll
    for(int r=0;r<R;r++) for(int m=0;m<M;m++){
      float a=acc[r*M+m];
      #pragma unroll
      for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffffu,a,o);
      if(lane==0){ const int o=o0+r; if(o<HIDDEN) atomicAdd(&h_io[(size_t)m*HIDDEN+o], gw*a*S[o]); }
    }
  }
}

// -------------------------------------------------------------------------------------------------
// INT4 fast dequant: 8 packed nibbles -> four signed __half2 (n-8), LOP3 idiom (v3), no I2F.
// -------------------------------------------------------------------------------------------------
__device__ __forceinline__ void unpack8(unsigned w, __half2 out[4]){
  const unsigned LO=0x000F000Fu, EXP=0x64006400u;
  unsigned f0=((w&0x000000F0u)<<12)|(w&0x0000000Fu);
  unsigned f1=((w&0x0000F000u)<< 4)|((w&0x00000F00u)>> 8);
  unsigned f2=((w&0x00F00000u)>> 4)|((w&0x000F0000u)>>16);
  unsigned f3=((w&0xF0000000u)>>12)|((w&0x0F000000u)>>24);
  unsigned t0=(f0&LO)|EXP, t1=(f1&LO)|EXP, t2=(f2&LO)|EXP, t3=(f3&LO)|EXP;
  const __half2 bias=__float2half2_rn(1032.0f);
  __half2 h0,h1,h2,h3; memcpy(&h0,&t0,4); memcpy(&h1,&t1,4); memcpy(&h2,&t2,4); memcpy(&h3,&t3,4);
  out[0]=__hsub2(h0,bias); out[1]=__hsub2(h1,bias); out[2]=__hsub2(h2,bias); out[3]=__hsub2(h3,bias);
}

// -------------------------------------------------------------------------------------------------
// INT4 kernel A (gate+up): R rows/warp, M activation cols.  Packed Wgu [2*INTER, HIDDEN/8] uint32;
// per-group fp16 scales [2*INTER, HIDDEN/GROUP_GU].  Inner dim HIDDEN tiles GROUP_GU=128 exactly.
// A lane walks uint4 chunks (32 int4); chunk v covers ints [v*32, v*32+32) -> group (v*32)/128 = v/4.
// We keep a per-(r,m) fp32 partial WITHIN a group; flush*scale at group boundary.
// -------------------------------------------------------------------------------------------------
template<int R, int M>
__global__ void i4_kA(const float* __restrict__ y, const int* __restrict__ sel,
    const unsigned* const* __restrict__ Wgu, const __half* const* __restrict__ Sgu,
    float* __restrict__ a_glb, int nslot, int INTER){
  extern __shared__ float ysA[];                                   // [M*HIDDEN]
  for(int i=threadIdx.x;i<M*HIDDEN;i+=blockDim.x) ysA[i]=y[i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int njrow=(INTER+R-1)/R; const int total=nslot*njrow;
  const int wpr=HIDDEN>>3;                                         // uint32/row
  const int gpr=HIDDEN/GROUP_GU;                                   // 32 groups/row
  const int nv=HIDDEN>>5;                                          // uint4 (32-int4) chunks/row
  const int chunks_per_group=GROUP_GU>>5;                          // 4 uint4 per group
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/njrow, jg=item-slot*njrow; const int j0=jg*R;
    const int e=sel[slot]; const unsigned* W=Wgu[e]; const __half* S=Sgu[e];
    const uint4* gv[R]; const uint4* uv[R]; const __half* gs[R]; const __half* us[R];
    #pragma unroll
    for(int r=0;r<R;r++){
      gv[r]=reinterpret_cast<const uint4*>(W+(size_t)(j0+r)*wpr);
      uv[r]=reinterpret_cast<const uint4*>(W+(size_t)(INTER+j0+r)*wpr);
      gs[r]=S+(size_t)(j0+r)*gpr; us[r]=S+(size_t)(INTER+j0+r)*gpr;
    }
    float gacc[R*M], uacc[R*M];                                    // cross-group fp32 accumulators
    float gpart[R*M], upart[R*M];                                  // within-group fp32 partials
    #pragma unroll
    for(int i=0;i<R*M;i++){ gacc[i]=0.f; uacc[i]=0.f; gpart[i]=0.f; upart[i]=0.f; }
    int cur_group=-1;
    for(int v=lane; v<nv; v+=32){
      const int grp=(v<<5)/GROUP_GU;
      if(grp!=cur_group){
        if(cur_group>=0){
          #pragma unroll
          for(int r=0;r<R;r++) for(int m=0;m<M;m++){
            gacc[r*M+m]+=gpart[r*M+m]*__half2float(gs[r][cur_group]);
            uacc[r*M+m]+=upart[r*M+m]*__half2float(us[r][cur_group]);
            gpart[r*M+m]=0.f; upart[r*M+m]=0.f;
          }
        }
        cur_group=grp;
      }
      uint4 gp[R],up[R];
      #pragma unroll
      for(int r=0;r<R;r++){ gp[r]=gv[r][v]; up[r]=uv[r][v]; }
      #pragma unroll
      for(int r=0;r<R;r++){
        const unsigned* gw=reinterpret_cast<const unsigned*>(&gp[r]);
        const unsigned* uw=reinterpret_cast<const unsigned*>(&up[r]);
        #pragma unroll
        for(int q=0;q<4;q++){
          __half2 gpk[4],upk[4]; unpack8(gw[q],gpk); unpack8(uw[q],upk);
          float2 g0=__half22float2(gpk[0]),g1=__half22float2(gpk[1]),g2=__half22float2(gpk[2]),g3=__half22float2(gpk[3]);
          float2 u0=__half22float2(upk[0]),u1=__half22float2(upk[1]),u2=__half22float2(upk[2]),u3=__half22float2(upk[3]);
          #pragma unroll
          for(int m=0;m<M;m++){
            const float* yq=ysA+(size_t)m*HIDDEN+(v<<5)+(q<<3);
            gpart[r*M+m]+=yq[0]*g0.x+yq[1]*g0.y+yq[2]*g1.x+yq[3]*g1.y+yq[4]*g2.x+yq[5]*g2.y+yq[6]*g3.x+yq[7]*g3.y;
            upart[r*M+m]+=yq[0]*u0.x+yq[1]*u0.y+yq[2]*u1.x+yq[3]*u1.y+yq[4]*u2.x+yq[5]*u2.y+yq[6]*u3.x+yq[7]*u3.y;
          }
        }
      }
    }
    #pragma unroll
    for(int r=0;r<R;r++) for(int m=0;m<M;m++){
      if(cur_group>=0){ gacc[r*M+m]+=gpart[r*M+m]*__half2float(gs[r][cur_group]);
                        uacc[r*M+m]+=upart[r*M+m]*__half2float(us[r][cur_group]); }
      float ga=gacc[r*M+m], ua=uacc[r*M+m];
      #pragma unroll
      for(int o=16;o>0;o>>=1){ ga+=__shfl_down_sync(0xffffffffu,ga,o); ua+=__shfl_down_sync(0xffffffffu,ua,o); }
      if(lane==0){ const int j=j0+r; if(j<INTER) a_glb[((size_t)m*nslot+slot)*INTER+j]=silu(ga)*ua; }
    }
    (void)chunks_per_group;
  }
}

// -------------------------------------------------------------------------------------------------
// INT4 kernel B (down): R rows/warp, M cols.  Packed Wd [HIDDEN, INTER/8] uint32; per-group fp16
// scales [HIDDEN, INTER/GROUP_D].  Inner dim INTER=192 tiles GROUP_D=64 (3 groups; 192/32=6 uint4/row,
// 2 uint4 per group).  At the shard the warp processes 6 uint4 split over 32 lanes -> lanes 0..5 work,
// 6..31 idle (the SAME starvation the fp8 kernel eats); R rows/warp amortize the dequant.
// -------------------------------------------------------------------------------------------------
template<int R, int M>
__global__ void i4_kB(const int* __restrict__ sel, const float* __restrict__ selw,
    const unsigned* const* __restrict__ Wd, const __half* const* __restrict__ Sd,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot, int INTER){
  extern __shared__ float asB[];                                   // [M*nslot*INTER]
  const int na=M*nslot*INTER;
  for(int i=threadIdx.x;i<na;i+=blockDim.x) asB[i]=a_glb[i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int norow=(HIDDEN+R-1)/R; const int total=nslot*norow;
  const int wpr=INTER>>3;                                          // uint32/row (24 at 192)
  const int gpr=INTER/GROUP_D;                                     // 3 groups/row at 192
  const int nv=INTER>>5;                                           // uint4 chunks/row (6 at 192)
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/norow, og=item-slot*norow; const int o0=og*R;
    const int e=sel[slot]; const float gw=selw[slot];
    const unsigned* W=Wd[e]; const __half* S=Sd[e];
    const uint4* wv[R]; const __half* ds[R];
    #pragma unroll
    for(int r=0;r<R;r++){ wv[r]=reinterpret_cast<const uint4*>(W+(size_t)(o0+r)*wpr); ds[r]=S+(size_t)(o0+r)*gpr; }
    float dacc[R*M], dpart[R*M];
    #pragma unroll
    for(int i=0;i<R*M;i++){ dacc[i]=0.f; dpart[i]=0.f; }
    int cur_group=-1;
    for(int v=lane; v<nv; v+=32){
      const int grp=(v<<5)/GROUP_D;
      if(grp!=cur_group){
        if(cur_group>=0){
          #pragma unroll
          for(int r=0;r<R;r++) for(int m=0;m<M;m++){
            dacc[r*M+m]+=dpart[r*M+m]*__half2float(ds[r][cur_group]); dpart[r*M+m]=0.f;
          }
        }
        cur_group=grp;
      }
      uint4 p[R];
      #pragma unroll
      for(int r=0;r<R;r++) p[r]=wv[r][v];
      #pragma unroll
      for(int r=0;r<R;r++){
        const unsigned* wu=reinterpret_cast<const unsigned*>(&p[r]);
        #pragma unroll
        for(int q=0;q<4;q++){
          __half2 wpk[4]; unpack8(wu[q],wpk);
          float2 w0=__half22float2(wpk[0]),w1=__half22float2(wpk[1]),w2=__half22float2(wpk[2]),w3=__half22float2(wpk[3]);
          #pragma unroll
          for(int m=0;m<M;m++){
            const float* yq=asB+((size_t)m*nslot+slot)*INTER+(v<<5)+(q<<3);
            dpart[r*M+m]+=yq[0]*w0.x+yq[1]*w0.y+yq[2]*w1.x+yq[3]*w1.y+yq[4]*w2.x+yq[5]*w2.y+yq[6]*w3.x+yq[7]*w3.y;
          }
        }
      }
    }
    #pragma unroll
    for(int r=0;r<R;r++) for(int m=0;m<M;m++){
      if(cur_group>=0) dacc[r*M+m]+=dpart[r*M+m]*__half2float(ds[r][cur_group]);
      float a=dacc[r*M+m];
      #pragma unroll
      for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffffu,a,o);
      // down per-group scale already folded into dacc above; epilogue just applies the routing weight.
      if(lane==0){ const int o=o0+r; if(o<HIDDEN) atomicAdd(&h_io[(size_t)m*HIDDEN+o], gw*a); }
    }
  }
}

// -------------------------------------------------------------------------------------------------
// SPEED-CEILING probe (kernel A only, M=1): the LEANEST possible int4 dequant+contract.  Contract in
// half2 with __hfma2 (NO half2->float2 convert — the cheapest datapath), stage y as __half2 in smem.
// This is NOT correctness-faithful (fp16 accumulate erodes the abs bar per the v3 note); it exists ONLY
// to answer "is the dequant ALU itself the wall?".  If even THIS can't beat fp8, int4 is dead here.
// -------------------------------------------------------------------------------------------------
template<int R>
__global__ void i4_kA_half(const __half2* __restrict__ y2, const int* __restrict__ sel,
    const unsigned* const* __restrict__ Wgu, const __half* const* __restrict__ Sgu,
    float* __restrict__ a_glb, int nslot, int INTER){
  extern __shared__ __half2 ysh2[];                                // [HIDDEN/2]
  for(int i=threadIdx.x;i<HIDDEN/2;i+=blockDim.x) ysh2[i]=y2[i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int njrow=(INTER+R-1)/R; const int total=nslot*njrow;
  const int wpr=HIDDEN>>3; const int gpr=HIDDEN/GROUP_GU; const int nv=HIDDEN>>5;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/njrow, jg=item-slot*njrow; const int j0=jg*R;
    const int e=sel[slot]; const unsigned* W=Wgu[e]; const __half* S=Sgu[e];
    const uint4* gv[R]; const uint4* uv[R]; const __half* gs[R]; const __half* us[R];
    #pragma unroll
    for(int r=0;r<R;r++){ gv[r]=reinterpret_cast<const uint4*>(W+(size_t)(j0+r)*wpr);
      uv[r]=reinterpret_cast<const uint4*>(W+(size_t)(INTER+j0+r)*wpr);
      gs[r]=S+(size_t)(j0+r)*gpr; us[r]=S+(size_t)(INTER+j0+r)*gpr; }
    float gacc[R],uacc[R]; __half2 gp2[R],up2[R];
    #pragma unroll
    for(int r=0;r<R;r++){ gacc[r]=0.f; uacc[r]=0.f; gp2[r]=__float2half2_rn(0.f); up2[r]=__float2half2_rn(0.f); }
    int cur_group=-1;
    for(int v=lane; v<nv; v+=32){
      const int grp=(v<<5)/GROUP_GU;
      if(grp!=cur_group){
        if(cur_group>=0){
          #pragma unroll
          for(int r=0;r<R;r++){ float2 gg=__half22float2(gp2[r]),uu=__half22float2(up2[r]);
            gacc[r]+=(gg.x+gg.y)*__half2float(gs[r][cur_group]); uacc[r]+=(uu.x+uu.y)*__half2float(us[r][cur_group]);
            gp2[r]=__float2half2_rn(0.f); up2[r]=__float2half2_rn(0.f); }
        }
        cur_group=grp;
      }
      uint4 gpk[R],upk[R];
      #pragma unroll
      for(int r=0;r<R;r++){ gpk[r]=gv[r][v]; upk[r]=uv[r][v]; }
      const __half2* yy=ysh2+((v<<5)>>1);                          // 16 half2 over this 32-int4 chunk
      #pragma unroll
      for(int r=0;r<R;r++){
        const unsigned* gw=reinterpret_cast<const unsigned*>(&gpk[r]);
        const unsigned* uw=reinterpret_cast<const unsigned*>(&upk[r]);
        #pragma unroll
        for(int q=0;q<4;q++){
          __half2 g4[4],u4[4]; unpack8(gw[q],g4); unpack8(uw[q],u4);
          const __half2* yq=yy+(q<<2);
          #pragma unroll
          for(int t=0;t<4;t++){ gp2[r]=__hfma2(g4[t],yq[t],gp2[r]); up2[r]=__hfma2(u4[t],yq[t],up2[r]); }
        }
      }
    }
    #pragma unroll
    for(int r=0;r<R;r++){ if(cur_group>=0){ float2 gg=__half22float2(gp2[r]),uu=__half22float2(up2[r]);
        gacc[r]+=(gg.x+gg.y)*__half2float(gs[r][cur_group]); uacc[r]+=(uu.x+uu.y)*__half2float(us[r][cur_group]); }
      float ga=gacc[r],ua=uacc[r];
      #pragma unroll
      for(int o=16;o>0;o>>=1){ ga+=__shfl_down_sync(0xffffffffu,ga,o); ua+=__shfl_down_sync(0xffffffffu,ua,o); }
      if(lane==0){ const int j=j0+r; if(j<INTER) a_glb[(size_t)slot*INTER+j]=silu(ga)*ua; } }
  }
}

// ---- host helpers ----
static inline unsigned hash_u(unsigned x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }
static inline float rnd(unsigned seed,size_t i,float sc,bool pos){ unsigned h=hash_u((unsigned)(i*2654435761u)^(seed*40503u));
  float v=(((h%2001)/1000.0f)-1.0f)*sc; return pos?(fabsf(v)+1e-3f):v; }
static inline unsigned rndu(unsigned seed,size_t i){ return hash_u((unsigned)(i*2654435761u)^(seed*2246822519u)); }
static inline int ctas_for(long rows,int R,int block){ int wpc=block>>5; long rg=(rows+R-1)/R;
  int need=(int)((rg+wpc-1)/wpc); return std::min(std::max(need,132),264); }
static inline int get_nib(const unsigned* row,int k){ unsigned w=row[k>>3]; return (int)((w>>(4*(k&7)))&0xFu); }

// Double-precision CPU int4 reference (the gold standard for the KERNEL correctness gate): mirrors the
// i4_kA/i4_kB datapath exactly (group-scaled signed-nibble dots, SiLU, routed accumulate) in fp64.
static void i4_reference(const float* y, const int* sel, const float* selw,
    const unsigned* const* Wgu, const __half* const* Sgu, const unsigned* const* Wd, const __half* const* Sd,
    float* h, int nslot, int INTER, int M){
  const int gpr_gu=HIDDEN/GROUP_GU, gpr_d=INTER/GROUP_D, wpr_gu=HIDDEN/8, wpr_d=INTER/8;
  std::vector<double> a(INTER);
  for(int m=0;m<M;m++){
    const float* ym=y+(size_t)m*HIDDEN;
    for(int slot=0;slot<nslot;slot++){
      const int e=sel[slot]; const unsigned* W=Wgu[e]; const __half* S=Sgu[e];
      for(int j=0;j<INTER;j++){
        const unsigned* gr=W+(size_t)j*wpr_gu; const unsigned* ur=W+(size_t)(INTER+j)*wpr_gu;
        const __half* gs=S+(size_t)j*gpr_gu;   const __half* us=S+(size_t)(INTER+j)*gpr_gu;
        double g=0,u=0;
        for(int k=0;k<HIDDEN;k++){ double yk=ym[k];
          g+=yk*(double)(get_nib(gr,k)-8)*(double)__half2float(gs[k/GROUP_GU]);
          u+=yk*(double)(get_nib(ur,k)-8)*(double)__half2float(us[k/GROUP_GU]); }
        double gf=g; a[j]=(gf/(1.0+exp(-gf)))*u;
      }
      const unsigned* Wdn=Wd[e]; const __half* Sd_=Sd[e]; const double gw=selw[slot];
      for(int o=0;o<HIDDEN;o++){
        const unsigned* dr=Wdn+(size_t)o*wpr_d; const __half* ds=Sd_+(size_t)o*gpr_d;
        double acc=0; for(int j=0;j<INTER;j++) acc+=a[j]*(double)(get_nib(dr,j)-8)*(double)__half2float(ds[j/GROUP_D]);
        h[(size_t)m*HIDDEN+o]+=(float)(gw*acc);
      }
    }
  }
}

int main(int argc,char**argv){
  const int E=8;
  const int BLK=(argc>1)?atoi(argv[1]):256;
  const double PEAK=(argc>2)?atof(argv[2]):3350.0;
  const int M=(argc>3)?atoi(argv[3]):1;
  const int INTER=MOE_INTER/8;                                     // 192 (TP=8 shard)
  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  printf("device: %s  SMs=%d  PEAK=%.0f GB/s  block=%d  M=%d  INTER(shard)=%d  GROUP_GU=%d GROUP_D=%d\n",
         prop.name, prop.multiProcessorCount, PEAK, BLK, M, INTER, GROUP_GU, GROUP_D);

  const size_t gu_n=(size_t)2*INTER*HIDDEN, d_n=(size_t)HIDDEN*INTER;     // elem counts
  const size_t gu_w=gu_n/8, d_w=d_n/8;                                    // packed uint32 counts
  const size_t gu_s=(size_t)2*INTER*(HIDDEN/GROUP_GU);                    // gate+up group scales
  const size_t d_s =(size_t)HIDDEN*(INTER/GROUP_D);                       // down group scales

  // ---- build inputs on host (so the reference reads the exact uploaded bytes) ----
  // fp8 weights derived from the SAME int4 round-trip so fp8 ref and int4 ref are the SAME math up to
  // fp8 rounding -> the int4-vs-fp8 max_rel is the genuine quant-error delta (the lossy-lever metric).
  std::vector<std::vector<unsigned>> Wgu4(E),Wd4(E);
  std::vector<std::vector<__half>>   Sgu4(E),Sd4(E);
  std::vector<std::vector<fp8>>      Wgu8(E),Wd8(E);
  std::vector<std::vector<float>>    Sgu8(E),Sd8(E);
  for(int e=0;e<E;e++){
    Wgu4[e].resize(gu_w); Wd4[e].resize(d_w); Sgu4[e].resize(gu_s); Sd4[e].resize(d_s);
    Wgu8[e].resize(gu_n); Wd8[e].resize(d_n); Sgu8[e].resize(2*INTER); Sd8[e].resize(HIDDEN);
    for(size_t i=0;i<gu_w;i++) Wgu4[e][i]=rndu(5u+e,i);
    for(size_t i=0;i<d_w;i++)  Wd4[e][i]=rndu(55u+e,i);
    for(size_t i=0;i<gu_s;i++) Sgu4[e][i]=__float2half(rnd(7u+e,i,0.02f,true));
    for(size_t i=0;i<d_s;i++)  Sd4[e][i]=__float2half(rnd(13u+e,i,0.02f,true));
    // fp8 reference weights = dequantized int4 (so both paths compute the same ideal dot up to format).
    // gate+up: row*HIDDEN+k -> nib*(n-8)*scale[group]; per-output fp8 scale folded as Sgu8 = 1 (the
    // dequant magnitude lives in the fp8 value itself here, matching k5_sharded_bench's per-channel scale).
    for(int row=0; row<2*INTER; row++){
      for(int k=0;k<HIDDEN;k++){
        int nib=get_nib(&Wgu4[e][(size_t)row*(HIDDEN/8)],k);
        float val=(float)(nib-8)*__half2float(Sgu4[e][(size_t)row*(HIDDEN/GROUP_GU)+k/GROUP_GU]);
        Wgu8[e][(size_t)row*HIDDEN+k]=(fp8)val;
      }
      Sgu8[e][row]=1.0f;
    }
    for(int o=0;o<HIDDEN;o++){
      for(int j=0;j<INTER;j++){
        int nib=get_nib(&Wd4[e][(size_t)o*(INTER/8)],j);
        float val=(float)(nib-8)*__half2float(Sd4[e][(size_t)o*(INTER/GROUP_D)+j/GROUP_D]);
        Wd8[e][(size_t)o*INTER+j]=(fp8)val;
      }
      Sd8[e][o]=1.0f;
    }
  }
  std::vector<float> y_h((size_t)M*HIDDEN);
  for(size_t i=0;i<(size_t)M*HIDDEN;i++) y_h[i]=rnd(99u+(unsigned)(i/HIDDEN),i%HIDDEN,1.0f,false);
  std::vector<int> sel_h(E); std::vector<float> selw_h(E);
  for(int e=0;e<E;e++){ sel_h[e]=e; selw_h[e]=0.1f+0.01f*e; }

  // ---- upload ----
  auto up=[&](auto& host, size_t bytes){ void* d; CK(cudaMalloc(&d,bytes)); CK(cudaMemcpy(d,host.data(),bytes,cudaMemcpyHostToDevice)); return d; };
  std::vector<unsigned*> Wgu4d(E),Wd4d(E); std::vector<__half*> Sgu4d(E),Sd4d(E);
  std::vector<fp8*> Wgu8d(E),Wd8d(E); std::vector<float*> Sgu8d(E),Sd8d(E);
  for(int e=0;e<E;e++){
    Wgu4d[e]=(unsigned*)up(Wgu4[e],gu_w*4); Wd4d[e]=(unsigned*)up(Wd4[e],d_w*4);
    Sgu4d[e]=(__half*)up(Sgu4[e],gu_s*2);   Sd4d[e]=(__half*)up(Sd4[e],d_s*2);
    Wgu8d[e]=(fp8*)up(Wgu8[e],gu_n);        Wd8d[e]=(fp8*)up(Wd8[e],d_n);
    Sgu8d[e]=(float*)up(Sgu8[e],2*INTER*4); Sd8d[e]=(float*)up(Sd8[e],HIDDEN*4);
  }
  auto mkpp=[&](void** h){ void** d; CK(cudaMalloc(&d,E*sizeof(void*))); CK(cudaMemcpy(d,h,E*sizeof(void*),cudaMemcpyHostToDevice)); return d; };
  const unsigned** Wgu4dd=(const unsigned**)mkpp((void**)Wgu4d.data());
  const unsigned** Wd4dd =(const unsigned**)mkpp((void**)Wd4d.data());
  const __half**   Sgu4dd=(const __half**)mkpp((void**)Sgu4d.data());
  const __half**   Sd4dd =(const __half**)mkpp((void**)Sd4d.data());
  const fp8**      Wgu8dd=(const fp8**)mkpp((void**)Wgu8d.data());
  const fp8**      Wd8dd =(const fp8**)mkpp((void**)Wd8d.data());
  const float**    Sgu8dd=(const float**)mkpp((void**)Sgu8d.data());
  const float**    Sd8dd =(const float**)mkpp((void**)Sd8d.data());
  int* sel_d=(int*)up(sel_h,E*4); float* selw_d=(float*)up(selw_h,E*4);
  float* y_d=(float*)up(y_h,(size_t)M*HIDDEN*4);
  float *a_d,*h_d; CK(cudaMalloc(&a_d,(size_t)M*E*INTER*4)); CK(cudaMalloc(&h_d,(size_t)M*HIDDEN*4));
  CK(cudaDeviceSynchronize());

  // ---- launch config (engine: gate+up R=2, down R=16) ----
  const int RA=2, RB=16;
  const size_t smemA=(size_t)M*HIDDEN*4, smemB=(size_t)M*E*INTER*4;
  int cA=ctas_for((long)E*INTER,RA,BLK), cB=ctas_for((long)E*HIDDEN,RB,BLK);
  #define SETSMEM(K) CK(cudaFuncSetAttribute(K,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)((smemA>smemB?smemA:smemB))))
  SETSMEM((fp8_kA<RA,1>)); SETSMEM((fp8_kB<RB,1>)); SETSMEM((i4_kA<RA,1>)); SETSMEM((i4_kB<RB,1>));
  SETSMEM((fp8_kA<RA,8>)); SETSMEM((fp8_kB<RB,8>)); SETSMEM((i4_kA<RA,8>)); SETSMEM((i4_kB<RB,8>));

  // ---- run once for correctness (int4 vs fp8-of-the-same-int4) at this M ----
  auto runA_i4=[&](){ if(M==1) i4_kA<RA,1><<<cA,BLK,smemA>>>(y_d,sel_d,Wgu4dd,Sgu4dd,a_d,E,INTER);
                      else      i4_kA<RA,8><<<cA,BLK,smemA>>>(y_d,sel_d,Wgu4dd,Sgu4dd,a_d,E,INTER); };
  auto runB_i4=[&](){ if(M==1) i4_kB<RB,1><<<cB,BLK,smemB>>>(sel_d,selw_d,Wd4dd,Sd4dd,a_d,h_d,E,INTER);
                      else      i4_kB<RB,8><<<cB,BLK,smemB>>>(sel_d,selw_d,Wd4dd,Sd4dd,a_d,h_d,E,INTER); };
  auto runA_8 =[&](){ if(M==1) fp8_kA<RA,1><<<cA,BLK,smemA>>>(y_d,sel_d,Wgu8dd,Sgu8dd,a_d,E,INTER);
                      else      fp8_kA<RA,8><<<cA,BLK,smemA>>>(y_d,sel_d,Wgu8dd,Sgu8dd,a_d,E,INTER); };
  auto runB_8 =[&](){ if(M==1) fp8_kB<RB,1><<<cB,BLK,smemB>>>(sel_d,selw_d,Wd8dd,Sd8dd,a_d,h_d,E,INTER);
                      else      fp8_kB<RB,8><<<cB,BLK,smemB>>>(sel_d,selw_d,Wd8dd,Sd8dd,a_d,h_d,E,INTER); };

  std::vector<float> h_i4((size_t)M*HIDDEN), h_f8((size_t)M*HIDDEN);
  CK(cudaMemset(h_d,0,(size_t)M*HIDDEN*4)); runA_i4(); runB_i4(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h_i4.data(),h_d,(size_t)M*HIDDEN*4,cudaMemcpyDeviceToHost));
  CK(cudaMemset(h_d,0,(size_t)M*HIDDEN*4)); runA_8(); runB_8(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h_f8.data(),h_d,(size_t)M*HIDDEN*4,cudaMemcpyDeviceToHost));

  // (1) KERNEL correctness gate: int4-GPU vs DOUBLE-precision int4 CPU reference (catches bugs; should ~1e-5).
  std::vector<const unsigned*> Wgu4hp(E),Wd4hp(E); std::vector<const __half*> Sgu4hp(E),Sd4hp(E);
  for(int e=0;e<E;e++){ Wgu4hp[e]=Wgu4[e].data(); Wd4hp[e]=Wd4[e].data(); Sgu4hp[e]=Sgu4[e].data(); Sd4hp[e]=Sd4[e].data(); }
  std::vector<float> ref((size_t)M*HIDDEN,0.f);
  i4_reference(y_h.data(),sel_h.data(),selw_h.data(),Wgu4hp.data(),Sgu4hp.data(),Wd4hp.data(),Sd4hp.data(),ref.data(),E,INTER,M);
  double k_abs=0,ref_max=0; for(size_t i=0;i<(size_t)M*HIDDEN;i++){ k_abs=std::max(k_abs,fabs((double)h_i4[i]-(double)ref[i])); ref_max=std::max(ref_max,fabs((double)ref[i])); }
  const bool kpass = k_abs < 1e-2;
  printf("correctness int4-GPU vs fp64 int4 CPU ref: max_abs=%.3e  (ref max|.|=%.3e)  -> %s (<1e-2 KERNEL gate)\n",
         k_abs,ref_max,kpass?"PASS":"FAIL");
  if(!kpass){ printf("ABORT: int4 kernel incorrect — not printing timing.\n"); return 1; }

  // (2) LOSSY-LEVER quality delta: int4 vs fp8-of-the-same-dequantized-weights, normalized by the
  // reference output magnitude (pointwise relative explodes on near-zero dots that cancel — not meaningful).
  double d_abs=0,sse=0,sref=0; for(size_t i=0;i<(size_t)M*HIDDEN;i++){ double d=(double)h_i4[i]-(double)h_f8[i];
    d_abs=std::max(d_abs,fabs(d)); sse+=d*d; sref+=(double)h_f8[i]*(double)h_f8[i]; }
  printf("int4 vs fp8 quant delta: max_abs=%.3e (=%.2f%% of ref-max)  rel-RMS=%.3e  (informational; the lossy lever)\n",
         d_abs, 100.0*d_abs/ref_max, sqrt(sse/(sref+1e-12)));

  // ---- bench ----
  cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e)); const int WARM=30,IT=300;
  auto bench=[&](auto fn)->float{ for(int i=0;i<WARM;i++) fn(); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(s)); for(int i=0;i<IT;i++) fn(); CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms; CK(cudaEventElapsedTime(&ms,s,e)); return ms/IT; };
  float aA8=bench(runA_8)*1e3, aB8=bench(runB_8)*1e3, aAi=bench(runA_i4)*1e3, aBi=bench(runB_i4)*1e3;

  // bytes read per token (one full expert sweep over E=8 routed): fp8 1B/elem; int4 0.5B + fp16 scales.
  const double f8A=(double)E*gu_n, f8B=(double)E*d_n, f8T=f8A+f8B;
  const double i4A=(double)E*(gu_w*4.0+gu_s*2.0), i4B=(double)E*(d_w*4.0+d_s*2.0), i4T=i4A+i4B;
  auto gbps=[&](double b,float us){ return b/1e3/us; };
  printf("\n  per-token expert read:  fp8 A=%.2f MB B=%.2f MB (T=%.2f) | int4 A=%.2f MB B=%.2f MB (T=%.2f)  -> %.2fx fewer bytes\n",
         f8A/1e6,f8B/1e6,f8T/1e6,i4A/1e6,i4B/1e6,i4T/1e6, f8T/i4T);
  printf("  %-16s %10s %10s %9s\n","stage(M=1..)","us","GB/s","%MBU");
  printf("  fp8  gate+up(A) %10.2f %10.1f %8.1f%%\n", aA8, gbps(f8A,aA8), 100*gbps(f8A,aA8)/PEAK);
  printf("  fp8  down   (B) %10.2f %10.1f %8.1f%%\n", aB8, gbps(f8B,aB8), 100*gbps(f8B,aB8)/PEAK);
  printf("  fp8  fused  A+B %10.2f %10.1f %8.1f%%\n", aA8+aB8, gbps(f8T,aA8+aB8), 100*gbps(f8T,aA8+aB8)/PEAK);
  printf("  int4 gate+up(A) %10.2f %10.1f %8.1f%%\n", aAi, gbps(i4A,aAi), 100*gbps(i4A,aAi)/PEAK);
  printf("  int4 down   (B) %10.2f %10.1f %8.1f%%\n", aBi, gbps(i4B,aBi), 100*gbps(i4B,aBi)/PEAK);
  printf("  int4 fused  A+B %10.2f %10.1f %8.1f%%\n", aAi+aBi, gbps(i4T,aAi+aBi), 100*gbps(i4T,aAi+aBi)/PEAK);
  printf("  int4 fused (fp8-equiv bytes / int4 time) GB/s=%.1f  MBU=%.1f%% (target>2x fp8 time)\n",
         gbps(f8T,aAi+aBi), 100*gbps(f8T,aAi+aBi)/PEAK);
  printf("\n  SPEEDUP int4 vs fp8:  A=%.2fx  B=%.2fx  fused=%.2fx   %s\n",
         aA8/aAi, aB8/aBi, (aA8+aB8)/(aAi+aBi), ((aAi+aBi)<(aA8+aB8))?"INT4 WINS":"fp8 still faster");
  printf("  per-layer fused: fp8=%.2f us  int4=%.2f us  (engine K5 gate+up GEMM+down GEMV today = ~23.1 us)\n",
         aA8+aB8, aAi+aBi);

  // ---- SPEED-CEILING probe (M=1 only): leanest int4 dequant (half2 __hfma2) for kernel A ----
  if(M==1){
    std::vector<__half> y2h(HIDDEN); for(int k=0;k<HIDDEN;k++) y2h[k]=__float2half(y_h[k]);
    __half* y2d=(__half*)up(y2h,HIDDEN*2);
    const size_t smemAh=(size_t)(HIDDEN/2)*sizeof(__half2);
    CK(cudaFuncSetAttribute(i4_kA_half<RA>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smemAh));
    auto runAh=[&](){ i4_kA_half<RA><<<cA,BLK,smemAh>>>((const __half2*)y2d,sel_d,Wgu4dd,Sgu4dd,a_d,E,INTER); };
    runAh(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float aAh=bench(runAh)*1e3;
    printf("\n  [speed-ceiling] int4 gate+up HALF-accum(A): %.2f us  %.1f GB/s  %.1f%%MBU   vs fp8 A=%.2f us -> %.2fx\n",
           aAh, gbps(i4A,aAh), 100*gbps(i4A,aAh)/PEAK, aA8, aA8/aAh);
    printf("  (lean half2 __hfma2 = cheapest possible dequant+contract; if THIS can't beat fp8, the int4 unpack ALU is the wall)\n");
  }
  return 0;
}
