# Small-MoE strategy validation (real numerics, on-box H100) — 2026-06-20

The custom decode step is a latency *proxy* (dummy weights). To validate the **strategies** with real
numerics cheaply, run them on a small MoE. OLMoE-1B-7B-0924 (64 experts, top-8) is the fast sanity
target (E8/E9); **Qwen3-30B-A3B** (same architecture as the 235B — 128 experts, top-8) is the
arch-faithful target whose numbers transfer directly to the 235B (download in progress).

Run on the box: `python tools/verify_route_prediction.py --model <m> --device cuda --dtype bfloat16`.

## E8 — Route prediction (DirectProxy) — VALIDATED ✅  (OLMoE-1B-7B, cuda/bf16, 8 prompts, 24 tok)
Validates `engine/routing/predictor.rs` + the prefetch in `scheduler.rs`. Random baseline = top_k/E = 0.125.

| signal | mean | vs random | reading |
|--------|-----:|----------:|---------|
| **(A) DirectProxy accuracy** — top-k(W_gate[L+1]·h_L) vs actual top-k(L+1) | **0.808** | **6.5×** | predictor.rs premise holds; prefetch is sound; ~19% misprediction (wasted-prefetch budget) |
| (B) layer-to-layer expert overlap | 0.115 | ≈1.0× (≈random) | consecutive layers do NOT reuse experts — naive "reuse last layer" fails |
| (C) token-to-token expert overlap | 0.470 | 3.8× | consecutive tokens reuse experts → markov/n-gram route cache is viable |

**Key insight:** (A) is high *despite* (B) ≈ random — DirectProxy works not because layers share
experts, but because `h_L` is a good enough proxy for `h_{L+1}` to drive L+1's router. That's a
specific validation of the DirectProxy approach over naive layer-reuse. Re-run on Qwen3-30B-A3B /
235B for the deploy number; (A)=0.81 here says the routing-prefetch lever is real.

## E9 — Self-speculation (shallow-pass agreement) — NOT VIABLE on OLMoE ❌  (cuda/bf16)
Does a shallow pass (first L_d layers + logit-lens) predict the next token well enough to draft without
a trained head? τ ≈ (top1 agree)·k; break-even ≈ 1.86 (L_d=12,k=2, `self-speculation-design.md`).

| depth L_d/L | top1 agree | top1∈full4 | τ (k=2) |
|------------:|-----------:|-----------:|--------:|
| 4/16  | 0.032 | 0.040 | 0.06 |
| 8/16  | 0.048 | 0.104 | 0.10 |
| 12/16 | 0.264 | 0.432 | 0.53 |
| 14/16 | 0.544 | 0.712 | 1.09 |
| 15/16 | 0.680 | 0.872 | **1.36** |

**Even a near-full-depth shallow pass (15/16 layers) only agrees 68% → τ=1.36 < 1.86 break-even.** So
self-speculation does not beat the draft cost on OLMoE → **use n-gram / a trained MTP head, not
self-spec** (decides the E6 fallback). Re-run on Qwen3-30B-A3B (48 layers, arch-faithful) to confirm
for the deploy family; the trend (shallow passes agree poorly until near-full depth) argues against
self-spec generally.

## Caveats + next
- OLMoE (16 layers, 64 experts) is the *fast* sanity model. **Qwen3-30B-A3B** (48 layers, 128 experts,
  top-8 — the 235B's small sibling) is downloading; rerunning E8/E9 there gives the **235B-transferable**
  numbers. Script note: `verify_self_speculation.py` needed single-GPU (a cross-device bug in `lens_top`
  bites when the model shards — must fix before the 235B run, which always shards 8-way).
- Conifer has GEMM/kernel-lab tuning methodology, but it targets edge/Metal/llama.cpp (not H100 sm_90a)
  and is clean-room-private — reference only, not copied here.
- Found bug to fix: `tools/verify_self_speculation.py` `lens_top` must move `hidden` to the lm_head's
  device (not just dtype) so the 235B (sharded) run works. **(FIXED + pushed.)**

## Qwen3-30B-A3B — arch-faithful confirmation (same arch as the 235B: 128 experts, top-8) ✅
The 235B's small sibling — identical router/RoPE/RMSNorm/top-8 structure → these numbers transfer
directly to the 235B. Run on the box (GPU 0, bf16, 8 prompts, 24 tok). Random baseline = 8/128 = 0.062.

| signal | OLMoE-1B-7B | **Qwen3-30B-A3B** | reading |
|--------|------------:|------------------:|---------|
| (A) DirectProxy accuracy | 0.808 (6.5×) | **0.718 (11.6× random)** | predictor.rs premise holds on the 235B arch; ~28% mispredict |
| (B) layer-to-layer overlap | 0.115 (≈rand) | 0.066 (≈rand 0.062) | layers do NOT reuse experts — naive layer-reuse fails (both models) |
| (C) token-to-token overlap | 0.470 | 0.510 (8.2×) | consecutive tokens reuse experts → markov/n-gram route cache viable |

**Bottom line:** route-prefetch (`engine/routing/predictor.rs` + `scheduler.rs`) is validated on the
235B's exact architecture — DirectProxy predicts the next layer's top-8 at **72% accuracy, 11.6× random**,
*not* by layer-reuse (B≈random) but because `h_L` proxies `h_{L+1}` for L+1's router. Re-run on the full
235B (8-way sharded; needs the `lens_top` device fix shipped here) for the final deploy number.
E9 self-spec on Qwen3-30B-A3B: in progress (GPU 1).
