/* spec_accept_tree.h — LOOP-A: host-side LOSSLESS TREE acceptance for the native spec loop.
 *
 * Generalizes spec_accept.h (chain) to a branching draft TREE — the lever that pushes tau on Charles's
 * FLAT M=k GEMM verify (a wider tree verifies ~free up to the 16-tile but accepts more often). C port of
 * the proven Rust engine/src/spec/tree.rs::accept_tree (SpecInfer-style residual multi-child rule).
 *
 * Charles's loop: build the draft tree (EAGLE3 head, top-b per node) -> run the M=n_nodes GEMM verify with
 * a TREE-ATTENTION mask (each node attends only to its ancestors) -> memcpy the n_nodes lm_head rows ->
 * call spec_accept_tree() -> accepted path + real bonus -> append to context/KV, continue.
 *
 * CONTRACT: target_rows[i] (row i, length vocab) = P(. | context + path-to-node-i) = the distribution
 * PREDICTING node i's children (the token AFTER node i). Node 0 is the root = last confirmed token
 * (sentinel, never emitted); row[0] predicts the first-level draft tokens. Adjacency is CSR:
 * child_list[child_off[i] .. child_off[i+1]) are the children of node i.
 *
 * Rule per node: try children in draft-prob desc order; accept the first with u < min(1, p_resid(x)/q);
 * on reject subtract that child's draft mass from the residual and renormalize; if all reject, bonus is a
 * sample from the final residual; at a leaf (whole path accepted), bonus is a pure sample of its row.
 * LOSSLESS for ANY tree shape (validated in tree.rs: emitted = target greedy ramp regardless of topology).
 *
 * Pure C (gcc/nvcc, verified nvcc cuda-12.6 sm_90a). Single tree (the EAGLE3 chain-of-trees per round).
 */
#ifndef SPEC_ACCEPT_TREE_H
#define SPEC_ACCEPT_TREE_H

#include <math.h>
#include <stddef.h>

/* softmax(logits) -> probs in `out` (caller buffer, length vocab). */
static inline void sat_softmax(const float* logits, int vocab, float* out) {
    float mx = logits[0];
    for (int i = 1; i < vocab; ++i) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) { out[i] = expf(logits[i] - mx); sum += out[i]; }
    if (sum <= 0.0f) { for (int i = 0; i < vocab; ++i) out[i] = 1.0f / vocab; return; }
    for (int i = 0; i < vocab; ++i) out[i] /= sum;
}

/* sample an index from a probability vector `p` (length vocab) using u in [0,1). */
static inline int sat_sample(const float* p, int vocab, float u) {
    float total = 0.0f; for (int i = 0; i < vocab; ++i) total += p[i];
    float t = u * (total > 0.0f ? total : 1.0f), cum = 0.0f;
    for (int i = 0; i < vocab; ++i) { cum += p[i]; if (t <= cum) return i; }
    return vocab - 1;
}

/* Lossless tree acceptance.
 *   n_nodes, tokens[n_nodes], draft_logprob[n_nodes]  : the tree (node 0 = root sentinel).
 *   child_off[n_nodes+1], child_list[...]             : CSR adjacency (children of i).
 *   target_rows[n_nodes*vocab]                        : row[i] predicts node i's children (CONTRACT).
 *   rng_u[], n_u                                      : U(0,1) stream (consumed per accept test + bonus).
 *   prob_scratch[vocab]                               : caller scratch buffer.
 *   out_tokens[>= max tree depth + 1]                 : accepted path tokens + bonus.
 *   out_n_accepted                                    : #accepted draft tokens.
 * Returns #emitted (= *out_n_accepted + 1).
 */
static inline int spec_accept_tree(int n_nodes, const int* tokens, const float* draft_logprob,
                                   const int* child_off, const int* child_list,
                                   const float* target_rows, int vocab,
                                   const float* rng_u, int n_u, float* prob_scratch,
                                   int* out_tokens, int* out_n_accepted) {
    int accepted = 0, node = 0, ui = 0;
    for (;;) {
        int beg = child_off[node], end = child_off[node + 1];
        const float* row = target_rows + (size_t)node * vocab;
        if (beg == end) { /* leaf: pure target sample */
            sat_softmax(row, vocab, prob_scratch);
            out_tokens[accepted] = sat_sample(prob_scratch, vocab, ui < n_u ? rng_u[ui++] : 0.5f);
            *out_n_accepted = accepted; return accepted + 1;
        }
        /* children sorted by draft prob desc (insertion sort; small fan-out) */
        int order[64]; int nb = 0;
        for (int e = beg; e < end && nb < 64; ++e) {
            int c = child_list[e], j = nb++;
            while (j > 0 && draft_logprob[order[j - 1]] < draft_logprob[c]) { order[j] = order[j - 1]; --j; }
            order[j] = c;
        }
        sat_softmax(row, vocab, prob_scratch); /* residual */
        int descended = -1;
        for (int s = 0; s < nb; ++s) {
            int c = order[s], x = tokens[c];
            float q = expf(draft_logprob[c]);
            float px = (x >= 0 && x < vocab) ? prob_scratch[x] : 0.0f;
            float a = (q > 0.0f) ? (px / q) : 1.0f; if (a > 1.0f) a = 1.0f;
            float u = (ui < n_u) ? rng_u[ui++] : 0.5f;
            if (u < a) { descended = c; break; }
            if (x >= 0 && x < vocab) prob_scratch[x] = (prob_scratch[x] - q > 0.0f) ? prob_scratch[x] - q : 0.0f;
            float ssum = 0.0f; for (int i = 0; i < vocab; ++i) ssum += prob_scratch[i];
            if (ssum > 1e-12f) for (int i = 0; i < vocab; ++i) prob_scratch[i] /= ssum;
        }
        if (descended >= 0) { out_tokens[accepted++] = tokens[descended]; node = descended; }
        else { /* all rejected: bonus from final residual */
            out_tokens[accepted] = sat_sample(prob_scratch, vocab, ui < n_u ? rng_u[ui++] : 0.5f);
            *out_n_accepted = accepted; return accepted + 1;
        }
    }
}

#endif /* SPEC_ACCEPT_TREE_H */
