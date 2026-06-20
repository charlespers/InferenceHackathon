/* tree_attn_test.c — LOOP-A: CPU test for tree_attn.h (M=k tree-attention structure).
 * gcc -O2 tree_attn_test.c -o /tmp/tat && /tmp/tat   (also nvcc-verified)
 */
#include <stdio.h>
#include "tree_attn.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); failures++; } } while (0)

int main(void) {
    /* Tree: root0 -> {1,2}; 1 -> {3}.  context_len = 10.  draft KV slot(i) = 10 + (i-1). */
    int parent[4] = {0, 0, 0, 1};
    int ctx = 10;
    int pos_id[4], anc_off[5], anc_slots[64];
    int total = tree_attn_build(4, parent, ctx, pos_id, anc_off, anc_slots);

    /* RoPE positions: depth(0,1,1,2) -> ctx+depth-1 = 9,10,10,11 */
    CHECK(pos_id[0]==9 && pos_id[1]==10 && pos_id[2]==10 && pos_id[3]==11, "pos_ids = ctx+depth-1");

    /* ancestor draft-slots (slot(i)=10+i-1): node1->[10], node2->[11], node3->[10,12] (anc node1 + self) */
    CHECK(anc_off[1]-anc_off[0]==0, "root attends no draft slots");
    CHECK(anc_off[2]-anc_off[1]==1 && anc_slots[anc_off[1]]==10, "node1 attends its own slot 10");
    CHECK(anc_off[3]-anc_off[2]==1 && anc_slots[anc_off[2]]==11, "node2 attends its own slot 11");
    CHECK(anc_off[4]-anc_off[3]==2 &&
          anc_slots[anc_off[3]]==10 && anc_slots[anc_off[3]+1]==12,
          "node3 attends ancestor path [10 (node1), 12 (self)] in root->self order");
    CHECK(total==4, "total ancestor slots = sum of depths");

    /* Key correctness property: a node NEVER attends to a non-ancestor branch (node3 must NOT see slot 11 = node2). */
    int sees_node2 = 0;
    for (int s = anc_off[3]; s < anc_off[4]; ++s) if (anc_slots[s]==11) sees_node2 = 1;
    CHECK(!sees_node2, "node3 does NOT attend the sibling branch (node2/slot 11) — tree isolation");

    if (failures == 0) printf("tree_attn: ALL TESTS PASSED\n");
    else printf("tree_attn: %d FAILURE(S)\n", failures);
    return failures ? 1 : 0;
}
