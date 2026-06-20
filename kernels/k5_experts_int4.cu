// K5 INT4 expert GEMV — the biggest remaining BYTE lever: halve the dominant routed-expert term
// (66% of per-token bytes) by storing expert weights as 4-bit instead of fp8. At B=1 decode is
// bandwidth-bound, so the ceiling is ~2x the fp8 kernel — IF the in-register int4 unpack doesn't become
// issue-bound. This kernel + k5_int4_bench.cu MEASURE that on the H100 (fp8 winner is e=0.46 / 1538 GB/s).
//
// Format: symmetric per-output-channel int4 (W4A16). Each weight nibble n in [0,15] -> value (n-8)*scale,
// scale per output row (hoisted out of the contraction, same as the fp8 kernel). 8 int4 / uint32,
// 32 int4 / 128-bit load. (A real deploy would use group-wise AWQ/GPTQ scales; per-channel is the
// throughput-equivalent skeleton — the unpack cost, the thing we're measuring, is identical.)
//
// Mirrors k5_experts_warp.cu (warp-per-row, split-K across lanes, coalesced). NOT COMPILED on this box;
// build + bench on-box (k5_int4_bench.cu). Validate against an int4 reference before trusting numbers.
#include "common.cuh"
using namespace q3;

// Raw dot of a contiguous int4 row (n weights, packed 8/uint32) with ys[0..n), split coalesced across a
// warp's 32 lanes, reduced. Returns sum(y_i * (nibble_i - 8)); caller multiplies the per-row scale once.
static __device__ __forceinline__ float warp_dot_int4(const unsigned* __restrict__ wq,
                                                      const float* __restrict__ ys, int n, int lane){
  float a0=0.f, a1=0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(wq);   // 4 words = 32 int4 / load
  const int nv = n >> 5;                                               // n/32 128-bit loads
  for(int v=lane; v<nv; v+=32){                                        // lanes -> consecutive uint4 (coalesced)
    uint4 p = wv[v];
    const unsigned* w4 = reinterpret_cast<const unsigned*>(&p);
    const float* yy = ys + (v << 5);
    #pragma unroll
    for(int q=0;q<4;q++){
      unsigned w = w4[q];
      const float* yq = yy + (q << 3);
      #pragma unroll
      for(int t=0;t<8;t++){                                            // 8 nibbles / word
        int nib = (int)((w >> (4*t)) & 0xFu) - 8;                      // [-8,7]
        (t & 1 ? a1 : a0) += yq[t] * (float)nib;                       // 2 accumulators -> ILP
      }
    }
  }
  float acc = a0 + a1;
  #pragma unroll
  for(int o=16;o>0;o>>=1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                                          // lane 0
}

// Packed int4 layout: gate|up = [2*MOE_INTER, HIDDEN] -> HIDDEN/8 uint32 per row.
//                     down    = [HIDDEN, MOE_INTER]   -> MOE_INTER/8 uint32 per row.
// A: a[slot][j] = silu(s_g*<y,gate_j>) * (s_u*<y,up_j>).  grid-stride warps over (slot,j).
extern "C" __global__ void k5a_gateup_int4(
    const float* __restrict__ y, const int* __restrict__ sel_idx,
    const unsigned* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    float* __restrict__ a_glb, int nslot){
  extern __shared__ float ys[];                                        // HIDDEN
  for(int k=threadIdx.x;k<HIDDEN;k+=blockDim.x) ys[k]=y[k];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int words_per_row = HIDDEN >> 3;                               // HIDDEN/8 uint32
  const int total=nslot*MOE_INTER;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/MOE_INTER, j=item-slot*MOE_INTER; const int e=sel_idx[slot];
    const unsigned* W=Wgu[e]; const float* S=Wgu_scale[e];
    const float g=warp_dot_int4(W + (size_t)j*words_per_row,                ys, HIDDEN, lane);
    const float u=warp_dot_int4(W + (size_t)(MOE_INTER+j)*words_per_row,    ys, HIDDEN, lane);
    if(lane==0) a_glb[(size_t)slot*MOE_INTER+j]=silu(g*S[j])*(u*S[MOE_INTER+j]);
  }
}

// B: h_io[o] += gw * s_d * <a[slot], down_o>, atomic over slots.  grid-stride warps over (slot,o).
extern "C" __global__ void k5b_down_int4(
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const unsigned* const* __restrict__ Wd, const float* const* __restrict__ Wd_scale,
    const float* __restrict__ a_glb, float* __restrict__ h_io, int nslot){
  extern __shared__ float as[];                                        // nslot*MOE_INTER
  const int na=nslot*MOE_INTER;
  for(int i=threadIdx.x;i<na;i+=blockDim.x) as[i]=a_glb[i];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  const int words_per_row = MOE_INTER >> 3;                            // MOE_INTER/8 uint32
  const int total=nslot*HIDDEN;
  for(int item=gwarp; item<total; item+=nwarp){
    const int slot=item/HIDDEN, o=item-slot*HIDDEN;
    const int e=sel_idx[slot]; const float gw=sel_w[slot];
    const unsigned* W=Wd[e]; const float* S=Wd_scale[e];
    const float d=warp_dot_int4(W + (size_t)o*words_per_row, as+(size_t)slot*MOE_INTER, MOE_INTER, lane);
    if(lane==0) atomicAdd(&h_io[o], gw*d*S[o]);
  }
}
