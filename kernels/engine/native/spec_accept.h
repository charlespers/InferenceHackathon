/* spec_accept.h — LOOP-A (djamoils): host-side LOSSLESS speculative acceptance for the M=k GEMM
 * verify loop (assembly-plan milestone M3; see engine/docs/native-spec-loop-wiring.md).
 *
 * This is the device-ready port of the proven-lossless Rust accept (engine/src/spec/accept.rs +
 * the full-accept bonus fix in eagle3_engine.rs). It runs HOST-SIDE on the (k+1) target logit rows
 * copied from device — at B=1 that copy is ~a few MB and the logic is trivially cheap, so there is
 * no need to put it on the GPU. Charles's .cu loop: run the M=(k+1) GEMM verify, cudaMemcpy the
 * (k+1) lm_head rows to host, call spec_accept(), append the emitted tokens to context + KV, advance
 * the EAGLE3 head, repeat.
 *
 * THE TWO CONTRACTS the caller's verify forward MUST satisfy (engine/docs/spec-accept-correctness-notes.md):
 *   (1) LAYOUT: target_logits[pos] (row pos, length `vocab`) must be the target distribution
 *       P(. | context + draft[0..pos]) — i.e. the row that PREDICTS draft_tokens[pos]. Concretely,
 *       run the verify over the input sequence [ctx_last, d0, d1, ..., d_{k-1}] (that is M = k+1
 *       columns); the "predict-next" output at input slot i is the row that predicts draft token i.
 *       => row[0] predicts d0, ..., row[k-1] predicts d_{k-1}, row[k] predicts the bonus. Do NOT pass
 *       the row sitting AT draft[pos] (that predicts draft[pos+1] — the classic off-by-one).
 *   (2) BONUS: row[k] is a REAL distribution P(. | context + draft[0..k]); the bonus is sampled from
 *       it. Never reuse the last accepted row as a greedy stand-in (that duplicates the last token).
 *
 * Losslessness: speculative sampling is exact for ANY draft distribution, so the emitted tokens are
 * a sample from the target regardless of the EAGLE3 head's choices (validated end-to-end in Rust:
 * decode_is_lossless_invariant_to_lambda_and_verify_depth).
 *
 * Pure C (no CUDA types) so it compiles with gcc or nvcc and is unit-testable on CPU. Single-drafter
 * (the EAGLE3 chain); the multi-drafter tournament is a straightforward extension if ever needed.
 */
#ifndef SPEC_ACCEPT_H
#define SPEC_ACCEPT_H

#include <math.h>
#include <stddef.h>

/* log p(token) from a raw logit row, numerically stable (log-sum-exp). */
static inline float sa_log_softmax_at(const float* logits, int vocab, int token) {
    float mx = logits[0];
    for (int i = 1; i < vocab; ++i) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) sum += expf(logits[i] - mx);
    return logits[token] - mx - logf(sum);
}

/* Categorical argmax (greedy) — used only as a degenerate fallback. */
static inline int sa_argmax(const float* logits, int vocab) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

/* Sample from softmax(logits) using u ~ U(0,1). For a peaked target this returns the argmax. */
static inline int sa_sample_categorical(const float* logits, int vocab, float u) {
    float mx = logits[0];
    for (int i = 1; i < vocab; ++i) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) sum += expf(logits[i] - mx);
    float cum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        cum += expf(logits[i] - mx) / sum;
        if (u <= cum) return i;
    }
    return vocab - 1;
}

/* Adjusted distribution on rejection: p_adj(x) ∝ max(0, p_target(x) - p_draft(x)), sampled with u.
 * Preserves the exact target distribution in expectation (Leviathan et al. 2023). */
static inline int sa_sample_adjusted(const float* logits, int vocab,
                                     int draft_token, float draft_logprob, float u) {
    float mx = logits[0];
    for (int i = 1; i < vocab; ++i) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; ++i) sum += expf(logits[i] - mx);
    float draft_prob = expf(draft_logprob);
    float total = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        float pt = expf(logits[i] - mx) / sum;
        float pd = (i == draft_token) ? draft_prob : 0.0f;
        float w = pt - pd; if (w < 0.0f) w = 0.0f;
        total += w;
    }
    if (total < 1e-9f) return sa_argmax(logits, vocab);  /* degenerate */
    float cum = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        float pt = expf(logits[i] - mx) / sum;
        float pd = (i == draft_token) ? draft_prob : 0.0f;
        float w = pt - pd; if (w < 0.0f) w = 0.0f;
        cum += w / total;
        if (u <= cum) return i;
    }
    return vocab - 1;
}

/* Accept one speculative round (single drafter / EAGLE3 chain).
 *
 *   target_logits : (k+1) rows x vocab, row[pos] = P(. | context + draft[0..pos]) (CONTRACT 1).
 *   k             : number of draft tokens.
 *   draft_tokens  : [k] proposed token ids.
 *   draft_logprobs: [k] log p_draft(draft_tokens[i] | ...).
 *   rng_u         : >= (k+1) draws of U(0,1): rng_u[pos] for the accept test / adjusted sample at
 *                   position pos; rng_u[k] for the full-accept bonus.
 *   out_tokens    : caller buffer >= (k+1); receives accepted prefix + the 1 bonus token.
 *   out_n_accepted: receives the number of accepted DRAFT tokens (0..k).
 *
 * Returns the number of EMITTED tokens (= *out_n_accepted + 1, the bonus is always emitted).
 */
static inline int spec_accept(const float* target_logits, int vocab, int k,
                              const int* draft_tokens, const float* draft_logprobs,
                              const float* rng_u, int* out_tokens, int* out_n_accepted) {
    int accepted = 0;
    for (int pos = 0; pos < k; ++pos) {
        const float* row = target_logits + (size_t)pos * vocab;  /* predicts draft[pos] (CONTRACT 1) */
        float lp_t = sa_log_softmax_at(row, vocab, draft_tokens[pos]);
        float ratio = expf(lp_t - draft_logprobs[pos]);
        if (ratio > 1.0f) ratio = 1.0f;                          /* standard spec criterion */
        if (rng_u[pos] < ratio) {
            out_tokens[accepted++] = draft_tokens[pos];          /* accept this draft token */
            continue;
        }
        /* reject at pos -> bonus from the adjusted distribution at THIS row, then stop */
        out_tokens[accepted] = sa_sample_adjusted(row, vocab, draft_tokens[pos],
                                                  draft_logprobs[pos], rng_u[k]);
        *out_n_accepted = accepted;
        return accepted + 1;
    }
    /* all k accepted -> bonus is a REAL sample from row[k] (CONTRACT 2), never a stand-in */
    const float* bonus_row = target_logits + (size_t)k * vocab;
    out_tokens[accepted] = sa_sample_categorical(bonus_row, vocab, rng_u[k]);
    *out_n_accepted = accepted;
    return accepted + 1;
}

#endif /* SPEC_ACCEPT_H */
