/* draft_vocab.h — LOOP-A (djamoils): EAGLE3 draft-vocab d2t map for the native spec loop
 * (assembly-plan milestone M2; see engine/docs/native-spec-loop-wiring.md).
 *
 * Port of the proven Rust DraftVocabMap (engine/src/spec/draft_vocab.rs). The EAGLE3 head emits logits
 * over a SMALL draft vocabulary (RedHat head = 64000; broken nm-testing = 32000) — NOT the target's
 * 151936. This is THE thing the broken nm-testing conversion got wrong (measured accept-length 1.4 vs
 * the correct ~2.7). Two rules, or acceptance silently collapses:
 *   (1) the candidate TOKEN is d2t[i]  (map draft index -> target token id), and
 *   (2) draft_logprob is the DRAFT-SPACE log-softmax at index i (logit[i] - logsumexp over the draft
 *       vocab) — NOT a full-vocab softmax, NOT the raw logit. spec_accept divides p_target/p_draft, so a
 *       mis-scaled p_draft poisons every acceptance test (-> the 1.4 failure).
 *
 * Charles's M2: after the head forward, call draft_top_candidates() on the head's raw draft-vocab logits
 * to get (target_token, draft_logprob) pairs; feed them as the draft chain into the M=(k+1) verify + the
 * spec_accept() in spec_accept.h. Pure C (gcc/nvcc), CPU-unit-tested.
 */
#ifndef DRAFT_VOCAB_H
#define DRAFT_VOCAB_H

#include <math.h>

/* Top-`m` draft candidates from the head's draft-vocab logits.
 *   d2t          : [draft_vocab] draft index -> target token id (the head's published map).
 *   draft_logits : [draft_vocab] raw head logits over the DRAFT vocab.
 *   m            : number of candidates wanted (capped at draft_vocab and at 64).
 *   out_tokens   : [>=m] receives the TARGET token ids (d2t[i]), sorted by draft prob desc.
 *   out_logprobs : [>=m] receives the DRAFT-SPACE log-softmax log p_draft(token) (<= 0).
 * Returns the number of candidates written.
 */
static inline int draft_top_candidates(const int* d2t, const float* draft_logits, int draft_vocab,
                                       int m, int* out_tokens, float* out_logprobs) {
    if (draft_vocab <= 0 || m <= 0) return 0;
    int n = m < draft_vocab ? m : draft_vocab;
    if (n > 64) n = 64;  /* reference cap; native path uses a proper top-k */

    /* (2) draft-space log-sum-exp normalizer (stable) — the bug fix */
    float mx = draft_logits[0];
    for (int i = 1; i < draft_vocab; ++i) if (draft_logits[i] > mx) mx = draft_logits[i];
    float sum = 0.0f;
    for (int i = 0; i < draft_vocab; ++i) sum += expf(draft_logits[i] - mx);
    float lse = mx + logf(sum);

    /* select top-n by logit (n-pass; n small). track chosen indices to skip. */
    int sel[64];
    for (int s = 0; s < n; ++s) {
        int best = -1; float bestv = -INFINITY;
        for (int i = 0; i < draft_vocab; ++i) {
            int taken = 0;
            for (int t = 0; t < s; ++t) if (sel[t] == i) { taken = 1; break; }
            if (taken) continue;
            if (best < 0 || draft_logits[i] > bestv) { bestv = draft_logits[i]; best = i; }
        }
        sel[s] = best;
        out_tokens[s]   = d2t[best];               /* (1) draft index -> TARGET token id */
        out_logprobs[s] = draft_logits[best] - lse; /* (2) draft-space log p_draft */
    }
    return n;
}

#endif /* DRAFT_VOCAB_H */
