# djamoils — measurement & route-prediction verdict (B=1, 8×H100)

My lane: **measurement/validation** — turn the team's components into measured deltas.
Everyone else is building (Jaymin: placement/predictor/spec/vLLM serving; Charles:
CUDA kernels k1–k6 + quant + comms). The gap I own: the **measured numbers** that say
which lever actually pays, and the **route-prediction go/no-go**.

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
