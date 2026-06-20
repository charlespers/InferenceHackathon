# djamoils — adaptive top-k expert reduction (B=1 latency optimization)

## CLAIMED AVENUE: confidence-adaptive top-k (an actual perf win, unclaimed)
Qwen3-235B routes every token to a fixed **top-8 of 128** experts. The router softmax
is often concentrated in the first few — the 7th/8th expert can carry near-zero weight.
**Routing to fewer experts when the router is confident cuts the #1 B=1 term (expert
weight bytes, ~66% of decode) and STACKS with FP8/int4 (Charles) and placement (Jaymin)
instead of overlapping them.** Documented as avenue #10 in `b1-latency-architecture.md`
(Tier-5) but **not in the experiment queue (E0–E8) and unbuilt** — I'm implementing it.

Policy (configurable): `k=4 if top-4 router mass > 0.9 ; elif k=6 if top-6 mass > 0.95 ;
else 8`. Quality gate: ≤0.5% MMLU / ≤3% HumanEval delta vs k=8 (per the arch doc).

Work in progress:
- `experiments/adaptive_topk/` — vLLM Qwen3-MoE patch (the optimization) + SOTA brief.
- `tools/router_mass.py` — measures router concentration on the real model → sets the
  threshold + projects the expert-byte savings (run in my :45 slot).

Estimated gain: ~1.1× (adaptive top-6, low risk) to ~1.3× (top-4 + light depth skip),
orthogonal to the rest of the stack.

---

## (earlier) measurement tooling — feeds the above + the team baseline
Everyone else builds (Jaymin: placement/predictor/spec/vLLM serving; Charles:
CUDA kernels k1–k6 + quant + comms). The measurement tools below remain useful for the
baseline + route-prediction go/no-go, but the **active deliverable is the optimization above**.

## Hardware reality (confirmed on-box)
- **8×H100 80GB HBM3** (~3.35 TB/s), full NVSwitch mesh (`NV18` any-to-any). *Not* H200.
- Roofline floor ~1.85 ms/token bf16, ~0.93 ms FP8; realistic ceiling ~500–540 tok/s.
- Shared box, 15-min testing slots/hour; djamoils owns **:45–:00**. A full bf16 load
  monopolizes all 8 GPUs, so model-loading tests only run in-slot when GPUs are free.

## Key adversarial finding (saves wasted effort)
**Expert *prefetch-to-hide-transfer* is moot at B=1.** All expert weights are HBM-resident,
so there is no PCIe fetch to overlap (Charles's L4; confirmed against the roofline). So:
- jminding's **hot-expert replication** still helps — it cuts the *per-token imbalance*
  (busiest GPU among the 8 chosen experts), not transfer latency.
- **Route prediction** only has value as a **spec-decode draft primer** (pre-stage which
  experts drafted tokens will hit), not as a standalone prefetch lever.

## Measured so far
- **Persistence (token t→t+1 top-8 overlap): 44.6%** — far above the 8/128≈6% chance line.
  ⇒ routes are temporally predictable enough to prime a spec-decode draft. (From an early
  run; needs the corrected script for the cross-layer + per-token numbers.)
- **Aggregate expert imbalance ~6.5× (max/mean), worst layer ~10×** (from `routing_stats`).
  This is *throughput* imbalance; the **per-token (latency-relevant)** number is being
  measured now — that's the one that bounds B=1 EP latency.
- Hot-expert coverage: top-16/128 experts ≈ 40% of activations, top-32 ≈ 61% ⇒ the HBM
  budget knob for replication.

## Tools in this branch
- `tools/routing_predict.py` — clean per-token route tracer (fixes the layer×decode-step
  interleaving in `routing_analysis.py`). Emits: **DirectProxy** cross-layer predictor
  accuracy (jminding's Tier-1, L→L+1, zero-training, banded early/mid/late), persistence
  hit-rate, per-token GPU imbalance (round-robin vs placement vs +replication), hot-expert
  coverage, and raw per-token routes for offline placement eval.
- `tools/measure_baseline.py` — stdlib OpenAI-SSE latency probe for the live vLLM engine:
  **TTFT / TPOT / decode tok/s**, % of roofline, dominant-term hint. This is the **E1
  baseline** the whole team is gated on ("every gain is × this number").
- `src/inferutil/latency.py` — added `measured_max_experts` override so a real trace's
  per-token imbalance can drive the roofline projection.

## Next (in-slot)
1. Rerun `routing_predict.py` on free GPUs → real DirectProxy + per-token imbalance.
2. When the FP8 vLLM server is up in-slot → `measure_baseline.py` → first measured tok/s.
3. Feed both into the roofline; record measured deltas per lever.
