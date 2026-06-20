/* spec_accept_test.c — LOOP-A: CPU unit tests for spec_accept.h (assembly-plan M3).
 * Build & run (no GPU needed):  gcc -O2 spec_accept_test.c -lm -o /tmp/sat && /tmp/sat
 * Mirrors the Rust accept tests (engine/src/spec/accept.rs) + the losslessness invariant
 * (eagle3_engine.rs::decode_is_lossless_invariant_to_lambda_and_verify_depth), at the accept level.
 */
#include <stdio.h>
#include <string.h>
#include "spec_accept.h"

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } } while (0)

/* Build (k+1) logit rows where row[pos] is PEAKED at peak_tokens[pos] (a deterministic target). */
static void peaked_rows(float* logits, int vocab, int kp1, const int* peak_tokens) {
    memset(logits, 0, (size_t)kp1 * vocab * sizeof(float));
    for (int r = 0; r < kp1; ++r) logits[(size_t)r * vocab + peak_tokens[r]] = 30.0f;
}

int main(void) {
    const int vocab = 64;

    /* ---- Test 1: all draft tokens match the target peaks -> all accepted + real bonus from row[k] */
    {
        int k = 3;
        int peaks[4] = {1, 2, 3, 7};                 /* rows predict d0=1,d1=2,d2=3, bonus=7 */
        float logits[4 * 64];
        peaked_rows(logits, vocab, 4, peaks);
        int draft[3] = {1, 2, 3};
        float dlp[3] = {-0.1f, -0.1f, -0.1f};
        float u[4] = {0.01f, 0.01f, 0.01f, 0.01f};   /* always accept; bonus sample picks the peak */
        int out[4]; int nacc = -1;
        int emit = spec_accept(logits, vocab, k, draft, dlp, u, out, &nacc);
        CHECK(nacc == 3, "T1: all 3 draft tokens accepted");
        CHECK(emit == 4, "T1: emitted = accepted + bonus");
        CHECK(out[0]==1 && out[1]==2 && out[2]==3, "T1: accepted prefix = draft");
        CHECK(out[3]==7, "T1: full-accept bonus = real sample from row[k] (token 7), not a duplicate");
    }

    /* ---- Test 2: draft tokens the target hates -> 0 accepted, bonus from adjusted dist ~ the peak */
    {
        int k = 2;
        int peaks[3] = {5, 5, 5};                    /* target wants token 5 everywhere */
        float logits[3 * 64];
        peaked_rows(logits, vocab, 3, peaks);
        int draft[2] = {40, 40};                     /* draft proposes 40 (target hates it) */
        float dlp[2] = {0.0f, 0.0f};                 /* p_draft = 1 -> ratio = p_t(40) ~ 0 -> reject */
        float u[3] = {0.99f, 0.99f, 0.99f};
        int out[3]; int nacc = -1;
        int emit = spec_accept(logits, vocab, k, draft, dlp, u, out, &nacc);
        CHECK(nacc == 0, "T2: zero accepted (all rejected)");
        CHECK(emit == 1, "T2: still emits the bonus");
        CHECK(out[0] == 5, "T2: rejection bonus = adjusted sample ~ target peak (5)");
    }

    /* ---- Test 3: ACCEPT-LEVEL LOSSLESSNESS — for a deterministic target, the EMITTED TOKENS are the
     * target ramp regardless of the draft (the draft only changes HOW MANY are emitted). Mirrors the
     * Rust decode-loop invariance: ctx_last=5 -> ramp 6,7,8,(bonus 9). */
    {
        int k = 3;
        int peaks[4] = {6, 7, 8, 9};                 /* row[pos] predicts ramp token 6+pos; bonus 9 */
        float logits[4 * 64];
        peaked_rows(logits, vocab, 4, peaks);
        float dlp[3] = {-0.2f, -0.2f, -0.2f};
        float u[4] = {0.01f, 0.01f, 0.01f, 0.01f};
        /* Several different draft chains (simulating different λ / head choices): */
        int drafts[4][3] = {
            {6, 7, 8},        /* perfect draft -> all accept */
            {6, 7, 40},       /* wrong at pos2 -> accept 6,7 then bonus */
            {6, 40, 40},      /* wrong at pos1 -> accept 6 then bonus */
            {40, 40, 40},     /* all wrong -> bonus only */
        };
        for (int d = 0; d < 4; ++d) {
            int out[4]; int nacc = -1;
            int emit = spec_accept(logits, vocab, k, drafts[d], dlp, u, out, &nacc);
            int ok = 1;
            for (int i = 0; i < emit; ++i) if (out[i] != 6 + i) ok = 0;  /* every emitted == ramp */
            CHECK(ok, "T3: emitted tokens are the target ramp regardless of draft (lossless)");
        }
    }

    if (failures == 0) printf("spec_accept: ALL TESTS PASSED\n");
    else printf("spec_accept: %d FAILURE(S)\n", failures);
    return failures ? 1 : 0;
}
