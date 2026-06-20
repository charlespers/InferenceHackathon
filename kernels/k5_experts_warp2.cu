// K5 down-proj v2 — front-loaded candidate fix for the measured weak link (k5b e=0.405 vs gate/up 0.490).
//
// Diagnosis (from the A/B split + roofline): the winner's k5b_down_warp stages ALL 8 experts' `a`
// (8*1536*4 = 48 KB smem) per CTA, which caps occupancy (~4 CTAs/SM by smem alone). This v2 instead puts
// ONE slot per CTA block (grid.y = slot) and stages only that slot's `a` (1536*4 = 6 KB) -> ~6-8x more
// CTAs/SM allowed by smem, so the down GEMV is no longer smem-occupancy-bound.
//
// Everything else identical to the winner: coalesced warp-per-row split-K + fp8x2->half2 dequant.
// Pairs with k5a_gateup_warp from kernels/k5_experts_warp.cu (gate/up unchanged — it already hits 0.490).
//
// HYPOTHESIS to confirm with Nsight (E4): k5b is occupancy/reduce-overhead bound, not DRAM-bound. If
// Nsight shows k5b already DRAM-saturated (>90% throughput), this won't help and the real fix is the
// sub-warp split-K (fewer lanes/row, more rows/warp) to cut the 5-step shuffle on the short 1536 reduce.
//
// NOT COMPILED on this box (darwin). Build + validate on-box; see kernels/k5_downproj_bench.cu.
//   Launch: grid = dim3(NT_D_tiles, nslot); block; smem = MOE_INTER*sizeof(float) (6 KB).
//     blockIdx.y = slot ; blockIdx.x tiles the warps over HIDDEN outputs for that slot.
#include "common.cuh"
using namespace q3;

static __device__ __forceinline__ float warp_dot_v2(const fp8* __restrict__ w,
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

// One slot per CTA-y; stage only a[slot] (6 KB). Warps grid-stride over HIDDEN outputs for that slot.
extern "C" __global__ void k5b_down_warp_v2(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot){
  const int slot = blockIdx.y;
  if (slot >= nslot) return;
  extern __shared__ float as[];                       // MOE_INTER (6 KB) — only this slot's activation
  for(int i=threadIdx.x;i<MOE_INTER;i+=blockDim.x) as[i]=a_glb[(size_t)slot*MOE_INTER+i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int wib=threadIdx.x>>5, wpb=blockDim.x>>5;
  const int gwarp=blockIdx.x*wpb+wib, nwarp=gridDim.x*wpb;
  const int e=sel_idx[slot]; const float gw=sel_w[slot];
  const fp8* W=Wd[e]; const float* S=Wd_scale[e];
  for(int o=gwarp; o<HIDDEN; o+=nwarp){
    const float d=warp_dot_v2(W+(size_t)o*MOE_INTER, as, MOE_INTER, lane);
    if(lane==0) atomicAdd(&h_io[o], gw*d*S[o]);
  }
}
