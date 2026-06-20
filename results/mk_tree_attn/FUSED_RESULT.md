# mk_tree_attn_fused — split-KV optimization (clean, box free 81GB, H100)
GPU-validated vs sdpa_tree_ref gate: err 5.8e-08 PASS.
| ctx4096 | serial (1 warp) | FUSED (W=8 split) | speedup |
|---|--:|--:|--:|
| w1 | 3045 us | 364 us | 8.4x |
| w8 | 3075 us | 477 us | 6.4x |
| w32 | 3293 us | 1602 us | 2.1x |
- Fixes the chain-bound serial scan (W warps split context, combine in shared mem, k2_fused design).
- 364us@M=1 fp32 ~ competitive with k2 (~500us); fp8 K/V would ~halve it (~180us). Handles M=k trees.
- Directly attacks the K2 attention floor (LOOP-C: 24% of the 2.1ms->430 target) AND is the spec verify attn.
- Note: fused saturates SMs sooner (one CTA/(query,head) x W warps), so high width is less flat than the
  serial under-utilized version; for realistic tree widths (k<=8) fused wins ~6x. Next: fp8 K/V + tune W.
