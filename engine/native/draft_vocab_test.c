/* draft_vocab_test.c — LOOP-A: CPU unit tests for draft_vocab.h (assembly-plan M2).
 * Build & run (no GPU): gcc -O2 draft_vocab_test.c -lm -o /tmp/dvt && /tmp/dvt
 * Mirrors the Rust tests (engine/src/spec/draft_vocab.rs).
 */
#include <stdio.h>
#include <math.h>
#include "draft_vocab.h"

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    /* draft vocab of 4 -> target tokens [500,501,502,503] (the Rust test's map) */
    int d2t[4] = {500, 501, 502, 503};

    /* T1: top candidate maps to TARGET token space (d2t[i]) */
    {
        float logits[4] = {0.0f, 1.0f, 5.0f, 0.5f}; /* favors draft idx 2 -> token 502 */
        int tok[4]; float lp[4];
        int n = draft_top_candidates(d2t, logits, 4, 2, tok, lp);
        CHECK(n == 2, "T1: returns 2 candidates");
        CHECK(tok[0] == 502, "T1: top candidate token = d2t[2] = 502");
        CHECK(tok[1] == 501, "T1: second = d2t[1] = 501");
    }

    /* T2: draft_logprob is the DRAFT-SPACE log-softmax (uniform -> ln(1/4)) */
    {
        float logits[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        int tok[4]; float lp[4];
        int n = draft_top_candidates(d2t, logits, 4, 4, tok, lp);
        CHECK(n == 4, "T2: 4 candidates");
        float ref = logf(0.25f);
        int ok = 1; float psum = 0.0f;
        for (int i = 0; i < n; ++i) { if (fabsf(lp[i] - ref) > 1e-5f) ok = 0; if (lp[i] > 1e-6f) ok = 0; psum += expf(lp[i]); }
        CHECK(ok, "T2: each logprob = ln(1/4) and <= 0");
        CHECK(fabsf(psum - 1.0f) < 1e-5f, "T2: draft probs sum to 1 (draft-space normalized)");
    }

    /* T3: capped at vocab; m=0 -> empty */
    {
        float logits[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        int tok[16]; float lp[16];
        CHECK(draft_top_candidates(d2t, logits, 4, 10, tok, lp) == 4, "T3: capped at draft_vocab (4)");
        CHECK(draft_top_candidates(d2t, logits, 4, 0, tok, lp) == 0, "T3: m=0 -> empty");
        CHECK(tok[0] == 503, "T3: highest-logit idx 3 -> token 503 first");
    }

    if (failures == 0) printf("draft_vocab: ALL TESTS PASSED\n");
    else printf("draft_vocab: %d FAILURE(S)\n", failures);
    return failures ? 1 : 0;
}
