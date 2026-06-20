# K2 multi-query flatness A/B — the spec free-ride test (FALSIFIED)

**Date:** 2026-06-20 ~19:40 UTC. Device: NVIDIA H100 80GB HBM3, SMs=132, sm_90a, idle box
(model-free microbench, ~2MB KV — no GPU slot needed). Correctness gated vs CPU fp32 (err ~1e-8 PASS).

## Question
Charles (commit 2905218, `kernels/k2_batched_decode.cu`, written off-GPU) hypothesized that a
multi-query K2 that **parallelizes M over warps + shares KV in L2 + shrinks n_splits as M grows**
would go FLAT in the draft width M (`us(M=8)/us(M=1) -> ~1.0`), restoring the "spec rides free on a
flat forward" projection (840-1000 tok/s). His current `tp8_k2_partial_mq` (serial per-warp loop over
M) measured 1.5x at M=8 and capped honest spec e2e at 107 (n-gram) / 294 (EAGLE3).

I validated his kernel (he had not run it) and swept the launch geometry. **Answer: NO — K2 scales
~linearly in M even with the best geometry. The flat-K2 hypothesis is falsified.**

## Result 1 — Charles's kernel as written (k2b_pick_splits, target ~4096 warps), ctx=4096
| M | n_splits | warps | us/K2-fwd | ratio vs M=1 |
|--:|--:|--:|--:|--:|
| 1 | 64 | 4096 | 45.3 | 1.00 |
| 4 | 16 | 4096 | 101.8 | 2.25 |
| 8 | 8 | 4096 | 191.4 | **4.23** |
| 16 | 4 | 4096 | 368.3 | 8.14 |

The FLAT test (`<1.25x`) fails at every M>1. Same shape at other ctx (M=8/M=1): **2.96x @ctx2048,
4.23x @ctx4096, 5.57x @ctx8192** — scaling *worsens* with context (longer per-warp serial chains).

## Result 2 — best-tuned (n_splits swept per M), ctx=4096
Charles's `k2b_pick_splits` caps warps at ~4096; the real H100 fill point is **~12-16K warps**.
Re-tuned optimum:
| M | best n_splits | warps | us/K2-fwd | ratio vs M=1 | us/query |
|--:|--:|--:|--:|--:|--:|
| 1 | 48 | 3072 | 41.3 | 1.00 | 41.3 |
| 2 | 32 | 4096 | 58.7 | 1.42 | 29.4 |
| 4 | 48 | 12288 | 97.5 | 2.36 | 24.4 |
| 8 | 32 | 16384 | 164.8 | **3.99** | 20.6 |
| 16 | 32 | 32768 | 300.9 | 7.29 | 18.8 |

Best-tuned M=8 = ~4.0x (vs his 4.23x). **Tuning shaves ~10-14% at M=4-8 but does NOT change the
conclusion.** Per-query cost *does* amortize (41 -> 19 us/query, ~2.2x via L2-shared KV) — but total
work = M x ctx and the GPU is ~saturated, so wall-clock scales ~linearly. Sublinearity (M=8 is 4.0x
not 8x) comes only from M=1 being under-saturated (idle SM headroom the first few queries fill).

## Cross-check — my own kernel (mk_tree_attn_fp8, W=32, independent geometry)
`FUSED_RESULT.md`: w1=62.5 -> w8=246 -> w32=1035 us (w8/w1 = 3.9x). **Two structurally different
kernels (his adaptive-split L2-reuse; my one-CTA-per-(query,head) W-warp split) agree: ~4x at M=8.**
The K2 M-scaling is a property of the B=1 flash-decode workload, not a single kernel's geometry.

## Implications (honest)
1. **The flat-K2 spec free-ride is dead.** A better K2 kernel does NOT flatten the verify. This
   confirms Charles's measured honest e2e (294 EAGLE3) from the kernel side; the 840-1000 flat
   projection should be retired.
2. **Spec still wins, realistically ~2.6-3.4x** (294 EAGLE3 vs 112 his forward / 85.7 vLLM bf16).
   The K2 M-tax is a real cap but GEMM panels (T16/T1=1.001) + comms are M-flat and dominate the
   forward, so the net spec multiplier survives — just bounded, not free.
3. **M=1 K2 is still a real floor win.** Plain-decode attention at M=1 = ~41us (k2b) / 62.5us
   (my fp8 W=32) vs the ~500us placeholder = ~8-12x. Unaffected by the spec-scaling finding.
4. **Free fix for Charles:** retune `k2b_pick_splits` `target_warps` 4096 -> ~12000-16000 (fill point
   is higher than assumed) for ~10-14% at M=4-8. M=1 optimum is splits~48 (not 64).

## Reconciliation with Charles's two flatness numbers (they're BOTH right)
- His OLD `kernels/spec_loop_e2e.txt` PART 2 "flat 1.003" = the **weight-GEMM panels only** (experts
  gate/up/down, attn QKV, attn O, router, lm_head). Those are weight-read-bound (2.74 GB/fwd) and
  genuinely M-flat on tensor cores (cuBLAS wgmma). TRUE.
- His NEW 2905218 "T8/T1=1.5, T16/T1=2.3" = the **full forward incl. the flash-decode attention**
  (QK^T·softmax·V over the KV cache) — which is NOT in the GEMM panel list and is exactly my K2 kernel.
- **Back-solve the K2 fraction** f of the forward from `(1-f) + f·(K2(M)/K2(1)) = T(M)/T(1)`:
  M=8: `(1-f)+4.0f = 1.5 -> f≈0.17`; M=16: `(1-f)+7.29f = 2.3 -> f≈0.21`. So **K2 attention ≈ 17-21%
  of the decode forward**, scaling ~4x@M=8 -> blended forward 1.5x. My isolated K2 (4x) *explains* his
  blended 1.5x and confirms his batched kernel does NOT remove it.

## What it means for the headline (optimal-k — CAVEATED projection)
Net spec tok/s ∝ `τ(k) / (T_blended(k) + draft)`, with `T_blended(M) = 0.80 + 0.20·K2(M)/K2(1)` (f≈0.20)
and τ from Charles's expected anchor (k4→2.8, k8→3.76, k16→~3.9), draft small:
| k (=γ) | M=k+1 | K2(M)/K2(1) | T_blended | τ | τ/T_blended |
|--:|--:|--:|--:|--:|--:|
| 3 | 4 | 2.36 | 1.07 | 2.80 | **2.62** |
| 7 | 8 | 3.99 | 1.40 | 3.76 | **2.69** |
| 15 | 16 | 7.29 | 2.06 | 3.90 | 1.89 |
- **Optimal k ≈ 4-8**, net spec ≈ **2.6-2.7x** over plain forward (matches Charles's measured ~294/112).
  k=16 is over-drafting — the K2 M-tax overtakes the τ gain. The 840-1000 flat-projection is dead.
- The **splits-fix** (warps 4096->~14K) trims T_blended(8) from ~1.45 to ~1.40 (~3-4% e2e at k=8). Small.
- CAVEAT: f≈0.20 is back-solved from Charles's commit-msg ratios, not his per-component breakdown; τ is
  the EAGLE3 "expected" anchor, not measured on the real model at these k. Confirm f + τ(k) with Charles
  before banking. The DIRECTION (optimal k≈4-8, ~2.6x, not flat) is robust to f in [0.15,0.25].

## Repro
Box `/tmp/mkta`: `k2b` (Charles, flatness table), `k2bs` (my splits-sweep harness, `k2b_splits.cu`).
`nvcc -arch=sm_90a -O3 --use_fast_math -I. k2_batched_decode.cu -o k2b` ; `./k2b <ctx> <iters>`.
`./k2bs <ctx> <iters> <M>` sweeps n_splits at fixed M.
