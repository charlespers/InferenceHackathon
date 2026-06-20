// common.cuh — shared shapes + fp8 helpers for Qwen3-235B-A22B B=1 decode kernels.
// Public model facts + standard CUDA only. Build target: sm_90a (H100).
//   nvcc -arch=sm_90a -O3 --use_fast_math -c k*.cu
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace q3 {

// ---- Qwen3-235B-A22B verified shapes (config.json) ----
constexpr int HIDDEN      = 4096;
constexpr int N_LAYERS    = 94;     // all MoE
constexpr int N_Q_HEADS   = 64;
constexpr int N_KV_HEADS   = 4;     // GQA 16:1
constexpr int HEAD_DIM    = 128;    // explicit; N_Q_HEADS*HEAD_DIM = 8192 != HIDDEN
constexpr int GQA_GROUP   = N_Q_HEADS / N_KV_HEADS;   // 16 Q heads per KV head
constexpr int Q_DIM       = N_Q_HEADS  * HEAD_DIM;    // 8192
constexpr int KV_DIM      = N_KV_HEADS * HEAD_DIM;    // 512
constexpr int QKV_OUT     = Q_DIM + 2 * KV_DIM;       // 9216 (fused QKV)
constexpr int N_EXPERTS   = 128;
constexpr int TOP_K       = 8;      // no shared expert
constexpr int MOE_INTER   = 1536;   // expert gate/up out, down in
constexpr int VOCAB       = 151936;
constexpr float RMS_EPS   = 1e-6f;
constexpr float ROPE_THETA = 1000000.0f;

using fp8  = __nv_fp8_e4m3;          // weight storage (e5m2 also an option for KV)
using bf16 = __nv_bfloat16;

// Row-major fp8 weight with per-output-channel scale (K-major load for coalescing).
struct Fp8Weight {            // logical shape [out, in], stored K-major (in contiguous)
  const fp8*  __restrict__ w;     // out*in
  const float* __restrict__ scale; // per-out-channel dequant scale [out]
  int out, in;
};

// Dequant one fp8 value. TODO(on-box): vectorize to 128-bit loads (16 fp8/thread) + ILP.
__device__ __forceinline__ float deq(fp8 v, float s) { return float(v) * s; }

// RMSNorm scale for a vector x[n] (returns 1/rms); apply as x_i * rms_inv * weight_i.
__device__ __forceinline__ float rms_inv(const float* x, int n) {
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
  // TODO(on-box): warp/block reduce (cub::BlockReduce) instead of this sketch.
  return rsqrtf(ss / n + RMS_EPS);
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

} // namespace q3
