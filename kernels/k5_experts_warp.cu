// K5 experts — MEASURED-BEST B=1 decode kernel for the MoE bottleneck (~14.2B params/token).
// Validated + benchmarked on a real H100 80GB (sm_90a). Numerically equivalent to k5_experts_fused
// (max relative error 3.2e-5 vs the scalar reference over the residual output).
//
// Optimization journey (measured, 8 active experts, fp8, 151 MB/call, peak 3.35 TB/s):
//   scalar reference (k5_experts.cu)                       9.82 ms   15 GB/s   e=0.005   1x
//   +128-bit loads + smem-y + hoisted scale (fused)        1.15 ms  131 GB/s   e=0.039   8.5x
//   +tile each expert across CTAs (fill the 132 SMs)       0.175ms  862 GB/s   e=0.257   56x
//   +warp-per-row, split-K across lanes (coalesced HBM)    0.105ms 1444 GB/s   e=0.431   94x
//   +fp8x2->half2 hardware dequant, full occupancy        0.098ms 1538 GB/s   e=0.459  107x   <-- this file
//
// Two ideas carry the win at B=1:
//   1. WARP-PER-ROW + split-K: a warp owns one output channel; its 32 lanes split the contraction so
//      consecutive lanes read consecutive 16-byte chunks of the SAME row -> fully coalesced HBM, then a
//      shuffle reduce. (Thread-per-row is memory-divergent: 32 rows HIDDEN apart per warp instruction.)
//   2. FILL THE MACHINE: one CTA per expert leaves 124/132 SMs idle. Grid-stride warps over (slot,channel)
//      with ~4k-8k warps. Best measured launch: 264 CTAs x 1024 threads (8448 warps).
//
// Split into two kernels around a global `a` buffer so gate/up and down can each tile to fill the SMs.
// STILL OPEN (diminishing returns past e~0.46; measure before chasing): down-proj (kernel B) has a
// shorter contraction (1536) so its warp-reduce overhead is proportionally larger and its 48KB all-`a`
// smem caps occupancy — a block-level (not warp) reduce or per-expert partial buffers may lift B;
// atomicAdd over 8 experts -> tree reduce; int4 expert weights = the next ~2x byte win.
//
// Build + bench:  see kernels/k5_microbench.cu  (nvcc -arch=sm_90a -O3 --use_fast_math)
#include "common.cuh"
using namespace q3;

// Dot of a contiguous fp8 row w[0..n) with ys[0..n), split coalesced across a warp's 32 lanes, reduced.
// Dequant via the hardware fp8x2->half2 path (8 vector conversions / 128-bit load vs 16 scalar casts).
static __device__ __forceinline__ float warp_dot(const fp8* __restrict__ w,
                                                 const float* __restrict__ ys, int n, int lane){
  float a0=0.f, a1=0.f;
  const uint4* __restrict__ wv=reinterpret_cast<const uint4*>(w);
  const int nv=n>>4;                                  // 16 fp8 per uint4
  for(int v=lane; v<nv; v+=32){                       // lanes 0..31 -> consecutive uint4 (coalesced)
    uint4 p=wv[v];
    const unsigned* wu=reinterpret_cast<const unsigned*>(&p);
    const float* yy=ys+(v<<4);
    #pragma unroll
    for(int q=0;q<4;q++){
      unsigned wq=wu[q];
      __nv_fp8x2_e4m3 lo,hi; lo.__x=(unsigned short)(wq&0xffffu); hi.__x=(unsigned short)(wq>>16);
      float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi);
      const float* yq=yy+(q<<2);
      a0+=yq[0]*fl.x; a1+=yq[1]*fl.y; a0+=yq[2]*fh.x; a1+=yq[3]*fh.y;   // 2 accumulators -> ILP
    }
  }
  float acc=a0+a1;
  #pragma unroll
  for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,o);
  return acc;                                         // valid on lane 0
}

// A: a[slot][j] = silu(s_g*<y,gate_j>) * (s_u*<y,up_j>).  grid-stride warps over (slot,j).
// launch: <<<CTAs, block, HIDDEN*sizeof(float)>>>  (smem stages y once per CTA)
extern "C" __global__ void k5a_gateup_warp(
    const float* __restrict__ y, const int* __restrict__ sel_idx,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    float* __restrict__ a_glb, int nslot){
  extern __shared__ float ys[];                       // HIDDEN
  for(int k=threadIdx.x;k<HIDDEN;k+=blockDim.x) ys[k]=y[k];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int total=nslot*MOE_INTER;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/MOE_INTER, j=item-slot*MOE_INTER; const int e=sel_idx[slot];
    const fp8* W=Wgu[e]; const float* S=Wgu_scale[e];
    const float g=warp_dot(W+(size_t)j*HIDDEN,             ys, HIDDEN, lane);
    const float u=warp_dot(W+(size_t)(MOE_INTER+j)*HIDDEN, ys, HIDDEN, lane);
    if(lane==0) a_glb[(size_t)slot*MOE_INTER+j]=silu(g*S[j])*(u*S[MOE_INTER+j]);
  }
}

// B: h_io[o] += gw * s_d * <a[slot], down_o>  (atomic over experts).  grid-stride warps over (slot,o).
// launch: <<<CTAs, block, nslot*MOE_INTER*sizeof(float)>>>  (smem stages all-`a` once per CTA; opt-in >48KB)
extern "C" __global__ void k5b_down_warp(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const fp8* const* __restrict__ Wd, const float* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot){
  extern __shared__ float as[];                       // nslot*MOE_INTER
  const int na=nslot*MOE_INTER;
  for(int i=threadIdx.x;i<na;i+=blockDim.x) as[i]=a_glb[i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int total=nslot*HIDDEN;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/HIDDEN, o=item-slot*HIDDEN;
    const int e=sel_idx[slot]; const float gw=sel_w[slot];
    const fp8* W=Wd[e]; const float* S=Wd_scale[e];
    const float d=warp_dot(W+(size_t)o*MOE_INTER, as+(size_t)slot*MOE_INTER, MOE_INTER, lane);
    if(lane==0) atomicAdd(&h_io[o], gw*d*S[o]);
  }
}
