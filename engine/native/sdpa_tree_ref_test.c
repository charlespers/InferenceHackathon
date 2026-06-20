/* sdpa_tree_ref_test.c — LOOP-A: CPU test for sdpa_tree_ref.h (M=k tree-attention numerics gate).
 * gcc -O2 sdpa_tree_ref_test.c -lm -o /tmp/sdr && /tmp/sdr   (also nvcc-verified)
 */
#include <stdio.h>
#include <math.h>
#include "sdpa_tree_ref.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a,b) (fabsf((a)-(b)) < 1e-3f)

int main(void) {
    const int HQ = 2, HKV = 1, HD = 4;     /* tiny GQA: 2 Q heads share 1 KV head */
    const float theta = 1000000.0f;
    float qn[4] = {1,1,1,1};               /* identity RMSNorm weight */

    /* ---- T1: uniform K over attended positions -> uniform attention -> out = mean of attended V ---- */
    {
        float qp[2*4] = {0.5f,0.1f,0.2f,0.3f, 0.4f,0.6f,0.1f,0.2f}; /* 1 query, 2 heads */
        float kc[3*4] = {1,1,1,1, 1,1,1,1, 1,1,1,1};               /* pos 0,1,2 all identical K */
        float vc[3*4] = {1,1,1,1, 3,3,3,3, 100,100,100,100};
        int pos[1] = {0};
        int anc_off[2] = {0,0}; int anc_slots[1] = {0};            /* no ancestors */
        float out[2*4];
        sdpa_tree(1,HQ,HKV,HD, qp,qn,pos, kc,vc,3, /*context_len*/2, anc_off,anc_slots, theta, out);
        /* attends pos 0,1 only (context_len=2): uniform K -> mean(V0,V1) = 2 everywhere; pos2 IGNORED */
        for (int h=0; h<HQ; ++h) for (int d=0; d<HD; ++d)
            CHECK(NEAR(out[h*HD+d], 2.0f), "T1: uniform K -> mean of attended V (=2); non-attended pos2 ignored");
    }

    /* ---- T2: adding pos2 as an ANCESTOR draft slot pulls it into the attended set ---- */
    {
        float qp[1*4] = {0.5f,0.1f,0.2f,0.3f};
        float kc[3*4] = {1,1,1,1, 1,1,1,1, 1,1,1,1};
        float vc[3*4] = {1,1,1,1, 3,3,3,3, 8,8,8,8};
        int pos[1] = {0};
        int anc_off[2] = {0,1}; int anc_slots[1] = {2};            /* query attends ancestor slot 2 */
        float out[1*4];
        sdpa_tree(1,1,1,HD, qp,qn,pos, kc,vc,3, 2, anc_off,anc_slots, theta, out);
        /* now attends 0,1,2 -> mean(1,3,8)=4 */
        for (int d=0; d<HD; ++d) CHECK(NEAR(out[d], 4.0f), "T2: ancestor slot pulled in -> mean(V0,V1,V2)=4");
    }

    /* ---- T3: RoPE is applied — non-uniform K, output depends on the query position ---- */
    {
        float qp[1*4] = {1,0,0,0};
        float kc[2*4] = {1,0,0,0, 0,0,1,0};                        /* distinct K rows */
        float vc[2*4] = {10,10,10,10, 20,20,20,20};
        int anc_off[2] = {0,0}; int anc_slots[1] = {0};
        float out0[4], out5[4];
        int p0[1]={0}, p5[1]={5};
        sdpa_tree(1,1,1,HD, qp,qn,p0, kc,vc,2, 2, anc_off,anc_slots, theta, out0);
        sdpa_tree(1,1,1,HD, qp,qn,p5, kc,vc,2, 2, anc_off,anc_slots, theta, out5);
        int differ = 0; for (int d=0; d<HD; ++d) if (!NEAR(out0[d], out5[d])) differ = 1;
        CHECK(differ, "T3: RoPE applied -> output differs between pos 0 and pos 5");
        /* sanity: both are convex combos of V0=10 and V1=20 -> in [10,20] */
        for (int d=0; d<HD; ++d) CHECK(out0[d] >= 9.9f && out0[d] <= 20.1f, "T3: output in [V0,V1] range");
    }

    if (failures == 0) printf("sdpa_tree_ref: ALL TESTS PASSED\n");
    else printf("sdpa_tree_ref: %d FAILURE(S)\n", failures);
    return failures ? 1 : 0;
}
