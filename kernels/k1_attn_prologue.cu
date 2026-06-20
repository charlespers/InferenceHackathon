// K1 — attention prologue (one fused kernel per layer, B=1 decode).
// Fuses: input-RMSNorm -> fused QKV GEMV (W 4096x9216) -> per-head QK-norm (RMSNorm over
// the 128-dim, fp32) -> RoPE(theta=1e6) -> write K,V to cache. QK-norm + RoPE are free
// epilogue ops; never round-trip them to HBM.
#include "common.cuh"
using namespace q3;

// h:        [HIDDEN] residual-stream input (bf16/fp32)
// w_in_norm:[HIDDEN] input RMSNorm weights
// Wqkv:     fp8 [QKV_OUT, HIDDEN] (K-major), scale [QKV_OUT]
// q_norm,k_norm: [HEAD_DIM] per-head QK-norm weights (shared across heads)
// rope_cos,rope_sin: [HEAD_DIM/2] for this position
// out_q:    [Q_DIM] normed+roped query (kept resident for K2)
// kv_cache_k/v: [.., pos, KV_DIM] write slot for this token (fp8)
extern "C" __global__ void k1_attn_prologue(
    const float* __restrict__ h, const float* __restrict__ w_in_norm,
    const fp8* __restrict__ Wqkv, const float* __restrict__ Wqkv_scale,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ out_q, fp8* __restrict__ kv_k, fp8* __restrict__ kv_v) {
  // 1) input RMSNorm -> x (broadcast rms_inv across the block)
  __shared__ float x[HIDDEN];
  float ri = rms_inv(h, HIDDEN);              // TODO(on-box): block-reduce
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) x[i] = h[i] * ri * w_in_norm[i];
  __syncthreads();

  // 2) fused QKV GEMV: each output row o = sum_k x[k]*deq(Wqkv[o,k]).
  //    grid-stride over QKV_OUT rows; one warp per row recommended.
  // TODO(on-box): tile N=9216, 128-bit vectorized fp8 loads, K-major, split-K if needed.
  const int o = blockIdx.x * blockDim.y + threadIdx.y;   // output row index (sketch)
  if (o >= QKV_OUT) return;
  float acc = 0.f;
  const fp8* wrow = Wqkv + (size_t)o * HIDDEN;
  for (int k = threadIdx.x; k < HIDDEN; k += warpSize) acc += x[k] * deq(wrow[k], Wqkv_scale[o]);
  // warp-reduce acc -> lane0 holds row result (sketch; use __shfl_down_sync)
  for (int s = warpSize/2; s; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);
  if (threadIdx.x != 0) return;

  // 3) route the row into Q / K / V regions and apply epilogue.
  if (o < Q_DIM) {                            // query lane: QK-norm + RoPE, keep resident
    int head = o / HEAD_DIM, d = o % HEAD_DIM;
    // per-head RMSNorm(q) over HEAD_DIM then RoPE. TODO(on-box): do per-head reduce in a
    // 2nd small pass or cooperative groups; here we mark the math.
    // q' = q * rms_inv_head * q_norm[d]; then rotate (d, d+HEAD_DIM/2) by rope.
    out_q[o] = acc; // placeholder: QK-norm+RoPE applied in the head-grouped epilogue (TODO)
    (void)head; (void)d; (void)q_norm; (void)rope_cos; (void)rope_sin;
  } else if (o < Q_DIM + KV_DIM) {            // key lane: QK-norm + RoPE -> write cache (fp8)
    int kd = o - Q_DIM;                       // TODO(on-box): k_norm + rope then quantize
    kv_k[kd] = fp8(acc);
    (void)k_norm;
  } else {                                    // value lane: no norm/rope -> write cache (fp8)
    int vd = o - Q_DIM - KV_DIM;
    kv_v[vd] = fp8(acc);
  }
  // NOTE: the per-head QK-norm needs a head-local reduce over HEAD_DIM=128. Cleanest impl:
  // assign one warp per (head) so the 128-dim reduce is warp-local; fuse rope in the same warp.
  // TODO(on-box): restructure to warp-per-head; current layout is the math sketch.
}

// Launch: one block per ~32 output rows (warp-per-row), or warp-per-head variant for Q/K.
// TODO(on-box): tune block/grid; benchmark vs separate norm+gemv+rope to confirm the fusion win.
