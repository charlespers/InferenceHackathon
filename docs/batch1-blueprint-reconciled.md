# Batch-1 single-stream blueprint ↔ our measured evidence (Qwen3-235B-A22B / 8×H100)

An external expert blueprint (user-provided, 2026-06-20) lays out the canonical batch-1 plan. It agrees
with where the team landed; this doc **maps each lever to our own measured/validated evidence**, and
flags the **one correction our data forces**. Source-of-truth numbers are ours unless noted.

## The blueprint's thesis (verbatim gist) — and our evidence
| Blueprint claim | Our evidence | Status |
|---|---|---|
| Batch-1 is memory+latency bound, ~1–2 FLOP/byte, pinned to the HBM roofline | `efficiency.py` roofline; live engine measured at **9–19% of the HBM ceiling** (kernel_gap dominant) | ✅ confirmed |
| Roofline ceilings: BF16 ~610 / FP8 ~1220 / INT4 ~2440 tok/s (weight-read only) | our `sweep`/roofline ceilings match (fp8 ~1231, bf16 ~616 at ctx2k) | ✅ matches |
| **Pure TP=8; EP is actively bad at B=1** (8 experts fire → idle GPUs + all-to-all) | team measured TP8 ≫ EP; `latency.py` model + `levers` agree; EP→TP inversion | ✅ confirmed |
| **Comms trap: ~188 collectives/token, ~0.5–0.9 ms pure latency** — co-equal with the INT4 read | the **floor finding**: TPOT ≈ overhead 60% / comms 26% / weight 14% (F≈0.86), measured floor-bound | ✅ confirmed (this is "the floor is the game") |
| CUDA graphs non-negotiable (hundreds of launches × ~5–10 µs) | **k6 built + runs on H100**: 755-node graph; measured **per-launch 1.71 µs → 661 launches = 1.13 ms/token** launch overhead the graph collapses (~16% of the 7 ms floor) | ✅ built + quantified (`docs/k6-decode-step-result.md`) |
| Fused grouped-GEMV MoE (TMA + wgmma + in-kernel INT4→FP8 dequant) | `kernels/k5_experts*` (measured e=0.46 fp8; int4 variants); the K5 bottleneck | 🔧 in progress |
| Flash-Decoding / split-KV + FP8 KV | `kernels/k2_flash_decode.cu` (split-KV); KV first-order only past ~64–128k (our depth-sweep) | ✅ design matches |
| Route-prefetch / early-dispatch to hide the collective | **DirectProxy route-prediction validated: accuracy 0.808 (6.5× random) on a real MoE** (E8, `engine/routing/predictor.rs`) | ✅ validated (`small-moe-strategy-validation.md`) |
| Speculative decoding 2–3× (EAGLE-2/3), amortizes the weight read | floor-aware spec model (`bench/spec_model.py`): at the measured F≈0.86 **big trees win, ~3.4–4×**; EAGLE3 wired | ✅ confirmed + sized |
| (blueprint omits) self-speculation as a draft source | **ruled out** (E9): shallow-pass agreement τ≈1.36 < 1.86 break-even even at 15/16 depth → use n-gram/MTP | ✅ our addition |

## The one correction our data forces: sequence the levers by the *floor*, not by the roofline
The blueprint ranks **quantization as the #1 dial** (BF16→FP8→INT4, ~2× each). That is the right *ceiling*
analysis, but our measured engine sits at **9–19% of that ceiling** — it is **floor-bound** (overhead/
comms/launch), not bandwidth-bound *yet*. We proved the consequence empirically: `ab_adaptive` showed a
byte-saving lever (adaptive-top-k) was **invisible / net-negative while floor-bound**, and Alyssa's data
showed **fp8 ~19% *slower* than bf16 at B=1** for the same reason. So the **correct order for THIS engine**:

1. **Drop the floor first** — CUDA graphs (k6, ~1.1 ms/token recoverable) + fused all-reduce+RMSNorm /
   NVLS comms + kernel efficiency (K5 e→1). This is "the floor is the game."
2. **Then** the bandwidth levers (INT4/FP8 weights, FP8 KV) become visible and pay the ~2× the blueprint
   predicts — gated on the §9 accuracy validation.
3. **Speculative decode** stacks on top throughout (amortizes whatever the per-pass cost is); at the
   current high floor, **bigger trees win** (floor amortized once), shrinking as the floor falls.

Same destination as the blueprint (TP8 + quant + comms-fusion + graphs + spec → ~700–1700 tok/s pre-spec,
×2–3 with spec); the **ordering** is the hard-won, data-backed difference.

## Validated-on-real-hardware so far (this session, on-box)
- k6 / full decode step: compiles (nvcc 12.6 sm_90a) + runs; CUDA-graph capture works (755 nodes). Proxy.
- Route-prediction (DirectProxy): 0.808 accuracy → prefetch lever real.
- Self-speculation: ruled out → n-gram/MTP for drafting.
- Next on the arch-faithful **Qwen3-30B-A3B** (downloading): re-run E8/E9 for 235B-transferable numbers.
