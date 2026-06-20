// k6_device_functions.cu — real __device__ implementations of k6_overlap_decode.cu's three extern
// declarations (multimem_allreduce_8kb, stream_weight_tile, expert_gemv), adapted from the team's
// already-existing kernels rather than written from scratch. NOT YET RUN ON GPU — this is the
// non-GPU-bound part of "wire up k6": real code, compiled against the box's headers where checkable,
// but not launched/profiled/correctness-checked end-to-end inside k6_overlap_decode.cu itself yet.
//
// #include this AFTER the extern declarations in k6_overlap_decode.cu (or merge it in directly) to
// replace the stubs with real bodies.

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda/pipeline>
#include <cstdint>
namespace cg = cooperative_groups;

// =================================================================================================
// multimem_allreduce_8kb — adapted from nvls_allreduce.cu's nvls_allreduce_half KERNEL into a plain
// __device__ FUNCTION (k6 calls this from inside its own grid, not as a separate launch). Same
// multimem.ld_reduce+st PTX, same v4.f16x2 coalescing. Caller (k6) provides mc_ptr already mapped to
// the multicast VA -- this function doesn't do any setup, only the reduce+broadcast.
//
// NOTE ON THE BARRIER: per the round-3 measurement (docs/config-sweep.md), an NVSHMEM HOST barrier is
// ~200x slower (1107us) than an in-kernel device-side flag-spin barrier (5.34us) -- this function
// deliberately does NOT include a barrier itself (matches variant (A), 3.52us, the fastest measured).
// k6's own grid.sync() after the call site (see k6_overlap_decode.cu's `grid.sync()` right after the
// `is_reduce` branch) IS the barrier here -- correct, since grid.sync() is exactly the in-kernel,
// device-side mechanism the round-3 finding says to use. Do not ALSO add an NVSHMEM barrier inside
// this function; that would silently reintroduce the 200x regression underneath the existing grid.sync().
// =================================================================================================
__device__ void multimem_allreduce_8kb(half* __restrict__ mc_ptr, int n) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= n) return;
  uint32_t a, b, c, d;
  asm volatile("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(mc_ptr + i));
  asm volatile("multimem.st.global.v4.f16x2 [%0], {%1,%2,%3,%4};"
               :: "l"(mc_ptr + i), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

// =================================================================================================
// stream_weight_tile — cp.async prefetch of one weight tile from HBM into shared memory, using the
// same cuda::pipeline mechanism as k5_experts_pipelined.cu (not raw PTX) so it composes with that
// file's STAGES/TILE_K double-buffering if/when expert_gemv below is swapped for the full pipelined
// version. This issues ONE stage's copy; the caller (k6's per-layer loop) is responsible for calling
// it once per stage and committing/waiting the pipeline around the matching expert_gemv call.
// =================================================================================================
__device__ void stream_weight_tile(const void* __restrict__ hbm_src, void* __restrict__ smem_dst,
                                    int bytes, cuda::pipeline<cuda::thread_scope_block>& pipe) {
  cg::thread_block block = cg::this_thread_block();
  // cuda::memcpy_async cooperatively splits `bytes` across the block's threads; matches the
  // call shape k5_experts_pipelined.cu uses internally for its weight-tile loads. Needs a pipeline
  // (or barrier) object -- ANOTHER widening of k6's declared signature (it only passes hbm_src/
  // smem_dst/bytes, no pipeline handle); the caller must own one cuda::pipeline per stream/compute
  // block and pass it through, matching k5_experts_pipelined.cu's per-block `cuda::make_pipeline()`.
  cuda::memcpy_async(block, smem_dst, hbm_src, bytes, pipe);
}

// =================================================================================================
// expert_gemv — INTEGRATION GAP, FLAGGED RATHER THAN PAPERED OVER.
//
// k6_overlap_decode.cu declares: expert_gemv(const half* x, const void* w_smem, half* y, int rows, int k)
// k5_experts_pipelined.cu's real kernel is: expert_gemv_pipelined(const uint8_t* W [fp8], const half* x_g,
//   const half* scales [per-row dequant], half* y, int ROWS, int K) -- it ALSO needs a per-row scale
// array that k6's declared signature has NO PARAMETER for. `w_smem` being `const void*` can legitimately
// hold the fp8 bytes (void* doesn't commit to a type), but the missing `scales` pointer is a real gap,
// not just a casting issue -- the dequant math needs it (deq2() in k5_experts_pipelined.cu).
//
// This adapter:
//  1. reuses k5_experts_pipelined.cu's deq2() dequant + warp-per-row dot-product structure exactly,
//  2. adds the missing scale pointer as an extra parameter (widening k6's call site is required --
//     this is NOT optional, the math is wrong without it: fp8 weights with no scale = wrong magnitude).
//  3. is a __device__ function (one warp = one output row), not k5_experts_pipelined.cu's __global__
//     kernel -- callable from k6's "compute" blocks directly, matching k6's SM-specialization design.
//
// STILL NEEDS, before this can be trusted: (a) k6_overlap_decode.cu's own extern declaration updated
// to add `const half* scales`, (b) an actual on-box correctness check against k5_experts_pipelined.cu's
// own validated reference (max_rel < 1e-3 per that file's own validation note), (c) the STAGES/TILE_K
// cp.async double-buffering this single-warp version doesn't yet use (it does ONE direct dot product
// per row -- the real perf win in k5_experts_pipelined.cu comes from overlapping STAGES loads with
// compute, not captured here yet).
// =================================================================================================
__device__ __forceinline__ half2 k6_deq2(uint16_t packed, half scale) {
  __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3);
  half2 h = *reinterpret_cast<half2*>(&r);
  return __hmul2(h, __halves2half2(scale, scale));
}

__device__ void expert_gemv(const half* __restrict__ x, const void* __restrict__ w_smem,
                             const half* __restrict__ scales,   // ADDED -- missing from k6's declared signature
                             half* __restrict__ y, int rows, int k) {
  const uint8_t* W = reinterpret_cast<const uint8_t*>(w_smem);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int nwarps = blockDim.x >> 5;
  for (int row = warp; row < rows; row += nwarps) {
    const uint8_t* w_row = W + (size_t)row * k;
    half scale = scales[row];
    float acc = 0.f;
    for (int j = lane * 2; j < k; j += 64) {
      uint16_t packed = *reinterpret_cast<const uint16_t*>(w_row + j);
      half2 wv = k6_deq2(packed, scale);
      half2 xv = *reinterpret_cast<const half2*>(x + j);
      float2 prod = __half22float2(__hmul2(wv, xv));
      acc += prod.x + prod.y;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[row] = __float2half(acc);
  }
}
