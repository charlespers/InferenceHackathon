# mk_tree_attn clean latency (2026-06-20 19:02 UTC, box free 81GB, H100 sm_90a)

| ctx | w1 | w4 | w8 | w16 | w32 | flatness (w1->w32) |
|---|--:|--:|--:|--:|--:|--:|
| 1024 | 765.7 | 766.9 | 775.2 | 795.4 | 844.8 us | +10.3% |
| 4096 | 3045 | 3052 | 3075 | 3134 | 3293 us | +8.2% |

## Conclusions
1. **Flat in tree width** (the spec-relevant property): a 32-node tree ~= 1.08-1.10x a 1-node verify ->
   wide trees ~free -> push tau via width. The attention analog of Charles's flat M=k GEMM verify.
2. **Absolutes slow = correctness-first serial KV scan** (linear in ctx: 765@1024 -> 3045@4096). NOT a
   limit: the OPTIMIZATION is to reuse k2_flash_decode (split-KV + 2-pass online softmax) for the context
   part (~100x). My kernel's value-add = tree mask + draft-slot attention + k-query batching (done,
   GPU-validated err 2e-7). Correctness banked; perf = reuse k2.
