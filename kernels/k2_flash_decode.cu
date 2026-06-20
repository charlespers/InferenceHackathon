// K2 — single-query flash-decode attention, B=1, GQA 16:1, head_dim 128.
// Split-KV across CTAs (a lone query underfills 132 SMs) + 2-pass online-softmax reduce.
// In-register KV dequant (fp8/int8). GQA broadcast: KV head g = q_head / 16.
#include "common.cuh"
using namespace q3;

// Pass 1: each CTA handles (one q_head, one KV chunk). Online softmax over its chunk ->
// partial (m=row-max, l=denom, acc[HEAD_DIM]). Striped partials for pass 2.
extern "C" __global__ void k2_flash_decode_partial(
    const float* __restrict__ q,            // [Q_DIM] normed+roped query (from K1)
    const fp8*  __restrict__ kv_k,          // [ctx_len, KV_DIM]
    const fp8*  __restrict__ kv_v,          // [ctx_len, KV_DIM]
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale,
    int ctx_len, int n_splits,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc) {
  const int qh    = blockIdx.y;                 // query head 0..63
  const int split = blockIdx.x;                 // KV-chunk index 0..n_splits-1
  const int kvh   = qh / GQA_GROUP;             // GQA broadcast -> KV head 0..3
  const int chunk = (ctx_len + n_splits - 1) / n_splits;
  const int t0 = split * chunk, t1 = min(t0 + chunk, ctx_len);
  const float scale = rsqrtf((float)HEAD_DIM);
  // online softmax over [t0,t1) for this (qh): m,l,acc over HEAD_DIM.
  // TODO(on-box): load q-head into regs/smem; dequant K,V rows; FMA dot; rescale acc on new max.
  // TODO(on-box): tune n_splits to fill SMs (~ceil(132 / N_Q_HEADS) chunks min); smem budget.
  (void)q;(void)kv_k;(void)kv_v;(void)kv_k_scale;(void)kv_v_scale;(void)kvh;(void)scale;
  (void)t0;(void)t1;(void)part_m;(void)part_l;(void)part_acc;
}

// Pass 2: merge the n_splits partials per head -> attn_out[Q_DIM] (standard flash reduce).
extern "C" __global__ void k2_flash_decode_reduce(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, int n_splits, float* __restrict__ attn_out) {
  // TODO(on-box): combine partials with the log-sum-exp trick; write attn_out per head.
  (void)part_m;(void)part_l;(void)part_acc;(void)n_splits;(void)attn_out;
}
// TODO(on-box): pick #splits by ctx_len; fp8 vs int8 KV; consider fusing K1's rope into q load.
