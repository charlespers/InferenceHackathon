/* tree_attn.h — LOOP-A: the TREE-ATTENTION structure for the native M=k spec verify (the spec-specific
 * ingredient of the new M=k attention kernel; the GQA/RoPE/softmax math is already in Charles's M=1
 * k1/k2/k3 kernels — this is the part that makes it TREE-aware).
 *
 * In a spec round the target verifies a draft TREE in one M=(n_drafts) attention. Each draft node must
 * attend to: ALL committed context [0..context_len)  +  its OWN ancestors' draft tokens (the path from
 * the first draft down to itself) — and to NO other tree branch (siblings/cousins are different
 * continuations). RoPE position of a node = context_len + depth-1 (its absolute sequence position).
 *
 * This header builds, from the tree's parent[] array, exactly what the M=k attention kernel needs:
 *   - pos_id[i]      : RoPE absolute position for draft node i.
 *   - anc CSR        : for each draft node, the list of draft KV-slots it attends to (its ancestor path
 *                      INCLUDING itself). Draft node i writes K/V at slot context_len + (i-1); so the
 *                      kernel's mask = {all context} ∪ {context_len + (a-1) : a in anc(i)}.
 * Node indexing: 0 = root (the last committed token, NOT a draft — never has a KV draft-slot); draft
 * nodes are 1..n_nodes-1 (matching SpecTree / spec_accept_tree.h).
 *
 * Pure C (gcc/nvcc-verified). Build the tree (EAGLE3 head) -> tree_attn_build() -> feed pos_id + the
 * ancestor mask into the M=k attention -> the rest of the verify (GEMM proj + MoE) is the flat path.
 */
#ifndef TREE_ATTN_H
#define TREE_ATTN_H

/* Depth of node i (root = 0). */
static inline int tree_depth(const int* parent, int i) {
    int d = 0;
    while (i != 0) { i = parent[i]; ++d; }
    return d;
}

/* Build the tree-attention structure.
 *   n_nodes, parent[n_nodes]   : the draft tree (parent[0]=0 root).
 *   context_len                : committed sequence length (KV positions [0..context_len)).
 *   out_pos_id[n_nodes]        : RoPE absolute position per node (root included).
 *   out_anc_off[n_nodes+1]     : CSR offsets into out_anc_slots, per draft node (root entry empty).
 *   out_anc_slots[cap]         : flattened ancestor draft-KV-slots (context_len + a-1) per node,
 *                                root→self order; cap must be >= sum of depths (<= n_nodes^2).
 * Returns the total number of ancestor slots written (the used length of out_anc_slots).
 */
static inline int tree_attn_build(int n_nodes, const int* parent, int context_len,
                                  int* out_pos_id, int* out_anc_off, int* out_anc_slots) {
    int w = 0;
    out_anc_off[0] = 0;
    for (int i = 0; i < n_nodes; ++i) {
        int d = tree_depth(parent, i);
        out_pos_id[i] = context_len + d - 1; /* root (d=0) -> last context pos; draft d -> ctx+d-1 */
        if (i != 0) {
            /* walk ancestors root..self (draft nodes only), record their KV slots, root→self order */
            int start = w;
            int node = i;
            while (node != 0) {            /* collect self + ancestors (draft nodes) */
                out_anc_slots[w++] = context_len + (node - 1);
                node = parent[node];
            }
            /* reverse [start..w) so the order is root-side → self (ascending position) */
            for (int a = start, b = w - 1; a < b; ++a, --b) {
                int t = out_anc_slots[a]; out_anc_slots[a] = out_anc_slots[b]; out_anc_slots[b] = t;
            }
        }
        out_anc_off[i + 1] = w;
    }
    return w;
}

#endif /* TREE_ATTN_H */
