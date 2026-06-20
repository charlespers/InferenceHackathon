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

## fp8 K/V (mk_tree_attn_fp8.cu) — the K2-floor win
GPU-validated vs fp32 gate: max abs err 3.9e-3 = 0.78% of max|ref| PASS (fp8 tol ~5%).
| ctx4096 w1 | serial | fp32 fused | FP8 fused |
|---|--:|--:|--:|
| us/round | 3045 | 364 | **173.7** |
| speedup | 1x | 8.4x | **17.5x** |
- 173.7us @ M=1 BEATS k2 (~500us) by ~2.9x; fp8 matches Charles's decode_step_tp8/k2 cache (per-channel
  scale, k2_load4 dequant) -> drop-in. Per-node @w8 = 30.8us. Candidate for BOTH the K2 forward floor and
  the M=k spec verify attention. Next: W tune + fold into the spec verify / A-B vs k2 at decode ctx.

## W (warps/CTA) tune — the K2-floor number
W sweep @ctx4096, M=1 (fp8): W4=331, W8=174, W16=94.7, **W32=62.5us** (more warps = shorter chains).
W=32 full width: w1=62.5 w4=123 w8=246 w16=499 w32=1035 us (per-node ~31us, flat).
**FINAL arc (M=1, ctx4096, the K2 forward-floor case): serial 3045 -> fp32-fused 364 -> fp8 174 -> fp8 W=32 62.5us = 49x.**
At ~62.5us, the K2 floor (was ~500us placeholder / 24% of 2.1ms) drops ~8x -> ~3% of the forward. Candidate
K2 replacement (M=1) AND the M=k spec verify attn (~31us/node). PENDING: real k2 number (A/B) + Charles's
exact KV slot/scale layout to drop in. W should be tuned per regime (M=1->W32; M=k->lower W as k adds CTAs).
