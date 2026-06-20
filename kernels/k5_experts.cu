// K5 — fused MoE experts (the bandwidth bottleneck, ~14.2B/token).
// Per active expert: fuse gate+up GEMV ([W 4096x3072]) -> silu(gate)*up (1536),
// then down GEMV (1536->4096) with the routing weight folded into the epilogue and
// accumulated into the residual. Grouped/persistent over the TOP_K active experts.
// EP hooks marked for the multi-GPU dispatch/combine (see docs/kernel-design/ep-parallel-schedule.md).
#include "common.cuh"
using namespace q3;

// y:        [HIDDEN] router-normed activation (from K4)
// sel_idx:  [TOP_K] active expert ids; sel_w: [TOP_K] gate weights
// Wgu[e]:   fp8 [2*MOE_INTER, HIDDEN] stacked gate|up; Wd[e]: fp8 [HIDDEN, MOE_INTER]
// h_io:     [HIDDEN] residual stream, accumulated in place (+= sum_j w_j * down_j)
extern "C" __global__ void k5_experts_fused(
    const float* __restrict__ y,
    const int* __restrict__ sel_idx, const float* __restrict__ sel_w,
    const fp8* const* __restrict__ Wgu, const float* const* __restrict__ Wgu_scale,
    const fp8* const* __restrict__ Wd,  const float* const* __restrict__ Wd_scale,
    float* __restrict__ h_io) {
  // One CTA per active expert slot (grid.x = TOP_K_local after EP dispatch). Persistent
  // variant: keep y resident in smem, loop the local experts to amortize launch.
  const int slot = blockIdx.x;                 // 0..TOP_K_local-1
  const int e = sel_idx[slot];
  const float gw = sel_w[slot];

  __shared__ float a[MOE_INTER];               // silu(gate)*up
  // gate+up fused: a[j] = silu( sum_k y[k]*Wgu[e][j,k] ) * ( sum_k y[k]*Wgu[e][MOE_INTER+j,k] )
  for (int j = threadIdx.x; j < MOE_INTER; j += blockDim.x) {
    const fp8* grow = Wgu[e] + (size_t)j * HIDDEN;
    const fp8* urow = Wgu[e] + (size_t)(MOE_INTER + j) * HIDDEN;
    float g = 0.f, u = 0.f;
    for (int k = 0; k < HIDDEN; ++k) { float yk = y[k];
      g += yk * deq(grow[k], Wgu_scale[e][j]);
      u += yk * deq(urow[k], Wgu_scale[e][MOE_INTER + j]); }
    a[j] = silu(g) * u;
  }
  __syncthreads();
  // down GEMV with routing weight folded in + atomic accumulate into residual.
  for (int o = threadIdx.x; o < HIDDEN; o += blockDim.x) {
    const fp8* drow = Wd[e] + (size_t)o * MOE_INTER;
    float acc = 0.f;
    for (int j = 0; j < MOE_INTER; ++j) acc += a[j] * deq(drow[j], Wd_scale[e][o]);
    atomicAdd(&h_io[o], gw * acc);             // <-- routing weight + accumulate fused
  }
  // EP NOTE: when experts are sharded across GPUs, `y` arrives via dispatch all-to-all and
  // the gw*acc partials return via combine all-to-all (summed into h_io on the owner rank).
  // TODO(on-box): grouped/persistent kernel over local experts; 128-bit fp8 loads; split the
  //   MOE_INTER reduce across warps; replace atomicAdd with a per-CTA partial + tree reduce;
  //   try int4 on expert weights (biggest byte win); capture inside the whole-step CUDA graph.
}
