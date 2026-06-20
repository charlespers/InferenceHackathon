// K3 — attention epilogue: O-proj GEMV (8192->4096) with FUSED residual add.
// h_out = h_in + Wo @ attn_out.  One fewer dispatch than proj-then-add.
#include "common.cuh"
using namespace q3;

extern "C" __global__ void k3_attn_epilogue(
    const float* __restrict__ attn_out,     // [Q_DIM=8192]
    const fp8*  __restrict__ Wo, const float* __restrict__ Wo_scale,  // [HIDDEN, Q_DIM]
    const float* __restrict__ h_in,         // [HIDDEN] residual
    float* __restrict__ h_out) {            // [HIDDEN]
  const int o = blockIdx.x * blockDim.y + threadIdx.y;   // output row 0..HIDDEN-1
  if (o >= HIDDEN) return;
  float acc = 0.f;
  const fp8* wrow = Wo + (size_t)o * Q_DIM;
  for (int k = threadIdx.x; k < Q_DIM; k += warpSize) acc += attn_out[k] * deq(wrow[k], Wo_scale[o]);
  for (int s = warpSize/2; s; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);
  if (threadIdx.x == 0) h_out[o] = h_in[o] + acc;        // <-- fused residual
  // TODO(on-box): 128-bit vectorized fp8 loads, warp-per-row tiling, split-K for Q_DIM=8192.
}
