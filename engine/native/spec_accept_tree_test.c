/* spec_accept_tree_test.c — LOOP-A: CPU tests for spec_accept_tree.h (host-side tree acceptance).
 * gcc -O2 spec_accept_tree_test.c -lm -o /tmp/satt && /tmp/satt
 * Mirrors engine/src/spec/tree.rs tests: chain, correct-branch-among-siblings, lossless-invariant.
 */
#include <stdio.h>
#include <string.h>
#include "spec_accept_tree.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); failures++; } } while (0)

#define V 64
static float ROWS[8 * V];
/* ramp target: row[i] peaks at tokens[i]+1 (deterministic greedy next), independent of topology. */
static void ramp_rows(const int* tokens, int n) {
    memset(ROWS, 0, sizeof(ROWS));
    for (int i = 0; i < n; ++i) ROWS[(size_t)i * V + ((tokens[i] + 1) % V)] = 30.0f;
}

int main(void) {
    float u[8]; for (int i = 0; i < 8; ++i) u[i] = 0.01f;
    float scratch[V]; int out[8]; int na;

    /* T1: chain 5->6->7->8 -> accept [6,7,8], bonus 9 */
    {
        int tokens[4] = {5,6,7,8}; float dlp[4] = {0,-0.1f,-0.1f,-0.1f};
        int coff[5] = {0,1,2,3,3}; int clist[3] = {1,2,3};
        ramp_rows(tokens, 4);
        int e = spec_accept_tree(4, tokens, dlp, coff, clist, ROWS, V, u, 8, scratch, out, &na);
        CHECK(na == 3 && e == 4, "T1: chain accepts 3 + bonus");
        CHECK(out[0]==6 && out[1]==7 && out[2]==8 && out[3]==9, "T1: emitted = ramp 6,7,8,9");
    }

    /* T2: root 5; siblings 40 (wrong, higher prob) & 6 (ramp, lower); 6->7. Accept [6,7], bonus 8. */
    {
        int tokens[4] = {5,40,6,7}; float dlp[4] = {0,-0.1f,-0.5f,-0.2f};
        int coff[5] = {0,2,2,3,3}; int clist[3] = {1,2,3}; /* node0:[1,2], node2:[3] */
        ramp_rows(tokens, 4);
        int e = spec_accept_tree(4, tokens, dlp, coff, clist, ROWS, V, u, 8, scratch, out, &na);
        CHECK(na == 2 && e == 3, "T2: tree picks correct branch (2 accepted)");
        CHECK(out[0]==6 && out[1]==7 && out[2]==8, "T2: emitted = ramp 6,7,8 (rejects wrong sibling 40)");
    }

    /* T3: lossless invariance — different topologies, SAME ramp output. all-wrong chain -> bonus only. */
    {
        int tokens[4] = {5,90,91,92}; float dlp[4] = {0,-0.1f,-0.1f,-0.1f};
        int coff[5] = {0,1,2,3,3}; int clist[3] = {1,2,3};
        ramp_rows(tokens, 4);
        int e = spec_accept_tree(4, tokens, dlp, coff, clist, ROWS, V, u, 8, scratch, out, &na);
        CHECK(na == 0 && e == 1, "T3: all-wrong tree accepts nothing");
        CHECK(out[0] == 6, "T3: bonus = ramp 6 (lossless regardless of topology)");
    }

    if (failures == 0) printf("spec_accept_tree: ALL TESTS PASSED\n");
    else printf("spec_accept_tree: %d FAILURE(S)\n", failures);
    return failures ? 1 : 0;
}
