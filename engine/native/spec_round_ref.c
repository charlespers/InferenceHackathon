/* spec_round_ref.c — LOOP-A: host-side REFERENCE of one native spec-decode round, composing the two
 * LOOP-A correctness pieces (M2 draft_vocab.h + M3 spec_accept.h) into the exact call sequence Charles's
 * .cu loop wires around his GPU kernels. This is the host-side TEMPLATE + parity reference for M2+M3.
 * Build & run (no GPU): gcc -O2 spec_round_ref.c -lm -o /tmp/srr && /tmp/srr
 *
 * The native loop, per round (GPU parts in <angle brackets>):
 *   <EAGLE3 head forward>  -> raw draft-vocab logits per chain position
 *   draft_top_candidates() -> (target_token, draft_logprob) per position           [M2, this file]
 *   <M=(k+1) GEMM verify over [ctx_last, d0..d_{k-1}]> -> (k+1) target lm_head rows  [Charles M1]
 *   memcpy rows to host; spec_accept() -> accepted prefix + real bonus              [M3, this file]
 *   append emitted tokens to context + KV; advance head; repeat
 *
 * Here the head logits and the verify rows are MOCKED (deterministic) so the host-side glue is exercised
 * and shown lossless on CPU; the native path swaps in the two <GPU> steps unchanged around these calls.
 */
#include <stdio.h>
#include <string.h>
#include "draft_vocab.h"   /* M2: draft_top_candidates */
#include "spec_accept.h"   /* M3: spec_accept */

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); failures++; } } while (0)

/* Mock target: deterministic greedy continuation = prev_token + 1. Builds the (k+1) verify rows with
 * the CORRECT layout (row[p] predicts draft position p, from ctx_last): peaked at (ctx_last+1+p). */
static void mock_verify_rows(float* rows, int vocab, int kp1, int ctx_last) {
    memset(rows, 0, (size_t)kp1 * vocab * sizeof(float));
    for (int p = 0; p < kp1; ++p) rows[(size_t)p * vocab + ((ctx_last + 1 + p) % vocab)] = 30.0f;
}

/* Run one host-side spec round (the template). Returns #emitted; fills out_tokens. */
static int spec_round(const int* d2t, int draft_vocab,
                      const float* head_logits, /* [k * draft_vocab] one row per chain position */
                      int k, int vocab, int ctx_last, const float* rng_u, int* out_tokens) {
    /* M2: turn each position's head logits into the top-1 draft (target_token, draft_logprob). */
    int draft_tokens[16]; float draft_logprobs[16];
    for (int p = 0; p < k; ++p) {
        int tk_[1]; float lp_[1];
        draft_top_candidates(d2t, head_logits + (size_t)p * draft_vocab, draft_vocab, 1, tk_, lp_);
        draft_tokens[p] = tk_[0];
        draft_logprobs[p] = lp_[0];
    }
    /* <GPU verify> mocked: the (k+1) target rows with the correct layout. */
    static float rows[ (16 + 1) * 4096 ];
    mock_verify_rows(rows, vocab, k + 1, ctx_last);
    /* M3: lossless accept -> accepted prefix + real bonus. */
    int nacc = 0;
    return spec_accept(rows, vocab, k, draft_tokens, draft_logprobs, rng_u, out_tokens, &nacc);
}

int main(void) {
    const int vocab = 4096;       /* target vocab (small for the demo) */
    const int draft_vocab = 8;    /* head's draft vocab */
    int d2t[8] = {100, 101, 102, 103, 104, 105, 106, 107}; /* draft idx -> target token */
    int ctx_last = 100;           /* last confirmed token; ramp continuation = 101,102,... */
    float rng_u[17]; for (int i = 0; i < 17; ++i) rng_u[i] = 0.01f;

    /* Two DIFFERENT head outputs (simulating different drafts) -> SAME lossless ramp output. */
    /* head A: peaks at draft idx {1,2,3} -> tokens {101,102,103} (matches the ramp -> all accept). */
    float headA[3 * 8]; memset(headA, 0, sizeof(headA));
    headA[0*8 + 1] = 9; headA[1*8 + 2] = 9; headA[2*8 + 3] = 9;
    /* head B: peaks at draft idx {7,7,7} -> token 107 (wrong -> reject early, resample to ramp). */
    float headB[3 * 8]; memset(headB, 0, sizeof(headB));
    headB[0*8 + 7] = 9; headB[1*8 + 7] = 9; headB[2*8 + 7] = 9;

    int outA[4], outB[4];
    int eA = spec_round(d2t, draft_vocab, headA, 3, vocab, ctx_last, rng_u, outA);
    int eB = spec_round(d2t, draft_vocab, headB, 3, vocab, ctx_last, rng_u, outB);

    printf("round(headA): emitted %d ->", eA); for (int i=0;i<eA;++i) printf(" %d", outA[i]); printf("\n");
    printf("round(headB): emitted %d ->", eB); for (int i=0;i<eB;++i) printf(" %d", outB[i]); printf("\n");

    /* Lossless: every emitted token is the target ramp (ctx_last+1+i), regardless of the draft. */
    int okA = 1; for (int i=0;i<eA;++i) if (outA[i] != ctx_last+1+i) okA = 0;
    int okB = 1; for (int i=0;i<eB;++i) if (outB[i] != ctx_last+1+i) okB = 0;
    CHECK(okA, "headA round emits the target ramp");
    CHECK(okB, "headB round emits the target ramp (different draft, SAME output -> lossless)");
    CHECK(eA >= eB, "matching draft accepts more (throughput), but output is identical (correctness)");

    if (failures == 0) printf("spec_round_ref: ALL TESTS PASSED (M2+M3 compose, host-side glue lossless)\n");
    else printf("spec_round_ref: %d FAILURE(S)\n", failures);
    return failures ? 1 : 0;
}
