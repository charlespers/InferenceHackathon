/* sdpa_tree_ref.h — LOOP-A: CPU fp32 NUMERICS REFERENCE for the M=k TREE attention (the gate the .cu
 * masked M=k attention kernel must match). Conventions matched to Charles's engine (common.cuh / k1 /
 * k2): GQA (kv_head = q_head / (n_q/n_kv)), scale = 1/sqrt(head_dim), RoPE theta=1e6 GPT-NeoX
 * "rotate-half", per-head RMSNorm(head_dim, fp32) * qnorm_w on Q then RoPE. K/V come from the cache
 * ALREADY normed+roped+written (k1 does that at KV-write time) — so this reference ropes only Q and
 * reads K/V straight from the cache. Tree mask (which positions each query attends) from tree_attn.h.
 *
 * The .cu kernel (extend k2_flash_decode: k queries, each = flash-decode over context [0..context_len)
 * + a masked dot over its ancestor draft slots) must reproduce this within fp8 tolerance. Dims are
 * args so tiny cases unit-test the math and the real (128/64/4) config drops in. Pure C (gcc/nvcc).
 */
#ifndef SDPA_TREE_REF_H
#define SDPA_TREE_REF_H

#include <math.h>
#include <string.h>

/* per-head RMSNorm over head_dim (fp32) * weight, in place. */
static inline void str_rmsnorm(float* v, const float* w, int hd) {
    float ss = 0.0f;
    for (int i = 0; i < hd; ++i) ss += v[i] * v[i];
    float inv = 1.0f / sqrtf(ss / hd + 1e-6f);
    for (int i = 0; i < hd; ++i) v[i] = v[i] * inv * w[i];
}

/* GPT-NeoX rotate-half RoPE in place: pairs (i, i+hd/2), angle = pos * theta^(-2i/hd). */
static inline void str_rope(float* v, int hd, int pos, float theta) {
    int half = hd / 2;
    for (int i = 0; i < half; ++i) {
        float freq = powf(theta, -2.0f * (float)i / (float)hd);
        float a = (float)pos * freq;
        float c = cosf(a), s = sinf(a);
        float x = v[i], y = v[i + half];
        v[i]        = x * c - y * s;
        v[i + half] = y * c + x * s;
    }
}

/* M=k tree masked SDPA reference.
 *   n_query, n_q_heads, n_kv_heads, head_dim
 *   q_proj [n_query * n_q_heads * head_dim]   : raw Q projections (pre-norm/rope)
 *   qnorm_w[head_dim]                         : per-head-dim RMSNorm weight (shared across heads)
 *   q_pos_id[n_query]                         : RoPE absolute position per query (from tree_attn.h)
 *   k_cache/v_cache [n_total_pos * n_kv_heads * head_dim] : already normed+roped+written
 *   context_len                               : every query attends [0..context_len)
 *   anc_off[n_query+1], anc_slots[]           : extra (ancestor draft) KV positions per query (tree mask)
 *   theta                                     : RoPE theta (1e6)
 *   out [n_query * n_q_heads * head_dim]      : attention output
 */
static inline void sdpa_tree(int n_query, int n_q_heads, int n_kv_heads, int head_dim,
                             const float* q_proj, const float* qnorm_w, const int* q_pos_id,
                             const float* k_cache, const float* v_cache, int n_total_pos,
                             int context_len, const int* anc_off, const int* anc_slots,
                             float theta, float* out) {
    (void)n_total_pos;
    int group = n_q_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    /* scratch for one query-head (max head_dim 256 covers 128) */
    float q[256];
    /* attended-position buffer: context_len + ancestors */
    for (int j = 0; j < n_query; ++j) {
        int n_anc = anc_off[j + 1] - anc_off[j];
        int n_att = context_len + n_anc;
        for (int h = 0; h < n_q_heads; ++h) {
            int kv = h / group;
            /* prepare Q: copy, rmsnorm, rope */
            memcpy(q, q_proj + ((size_t)j * n_q_heads + h) * head_dim, head_dim * sizeof(float));
            str_rmsnorm(q, qnorm_w, head_dim);
            str_rope(q, head_dim, q_pos_id[j], theta);
            /* scores over the attended set (online max for stability) */
            float maxs = -INFINITY;
            /* first pass: max */
            for (int a = 0; a < n_att; ++a) {
                int t = (a < context_len) ? a : anc_slots[anc_off[j] + (a - context_len)];
                const float* k = k_cache + ((size_t)t * n_kv_heads + kv) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q[d] * k[d];
                dot *= scale;
                if (dot > maxs) maxs = dot;
            }
            /* second pass: softmax-weighted V */
            float denom = 0.0f;
            float acc[256]; for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;
            for (int a = 0; a < n_att; ++a) {
                int t = (a < context_len) ? a : anc_slots[anc_off[j] + (a - context_len)];
                const float* k = k_cache + ((size_t)t * n_kv_heads + kv) * head_dim;
                const float* vv = v_cache + ((size_t)t * n_kv_heads + kv) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q[d] * k[d];
                float w = expf(dot * scale - maxs);
                denom += w;
                for (int d = 0; d < head_dim; ++d) acc[d] += w * vv[d];
            }
            float* o = out + ((size_t)j * n_q_heads + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) o[d] = (denom > 0.0f) ? acc[d] / denom : 0.0f;
        }
    }
}

#endif /* SDPA_TREE_REF_H */
