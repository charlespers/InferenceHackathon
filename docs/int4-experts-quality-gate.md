# int4-experts — the 1000-tok/s cushion + its quality gate

> **❌ SUPERSEDED / DEAD at B=1 (`results-reaction-04.md`).** The squeeze round MEASURED int4 (v3, LOP3 half2
> unpack) at **0.58× fp8 — SLOWER** — the int4→half unpack is ALU-bound at B=1, so the half-the-bytes win never
> materializes. **int4 is ruled out as a B=1 latency lever.** The weight floor is fp8 (0.78 ms). The comms-side
> cushions (EP collective-count reduction, stale-TP) replace this. Kept for the record / for the *throughput*
> regime where int4 still pays. Do **not** put int4 on the B=1 path.

The lossless path to 1000 (fp8 + NVLS ≤ ~3 µs + small-spec → ~1060–1170, `ladder_to_1000.py`) is tight on the
comms. **int4 experts is the cushion**: it cuts the weight 0.78 → 0.51 ms, which **relaxes the NVLS requirement
from ≤3 µs to ≤4 µs** (int4-experts + NVLS@4µs + small-spec → ~953) and gives margin if e-tuning or spec
under-deliver. It's a *quality-gated* lever, so it must be validated — here's the plan, scoped so it's the
smallest quality risk for the biggest weight win.

## What to quantize (and what NOT to)
- **int4 the EXPERTS only** (gate/up/down — the 14.2 B, 68% of active weight). **Keep fp8 on attention + router**
  (6.7 B): attention is more quality-sensitive and is the smaller win. This "int4-experts + fp8-rest" is the
  sweet spot — most of the weight reduction, least of the quality risk.
- **Per-GROUP int4** (group_size 64–128), not per-tensor — per-group tracks the expert weight distribution and
  is what AWQ/GPTQ MoE recipes use. Weight-only (W4A16-style: int4 weights, fp16/fp8 activations) — at B=1 the
  activation is one vector, so activation precision barely matters; keep it high.
- **Calibration-based (AWQ or GPTQ)**, not naive RTN — AWQ's activation-aware scaling matters for MoE experts
  (the salient channels). Calibrate on a small in-distribution set (chat + code, a few hundred sequences).

## The gate (must pass before int4 is allowed on the path)
Run the team's `quality_compare.py` (and `quality_probe.py`) int4-experts vs the fp8 reference:
1. **Greedy token-parity** on a held-out chat+code set: target **≥ 98–99% next-token match** (lossless-ish at
   temp 0). A big drop here = the quantization is broken (bad scales/group size) — fix before trusting tok/s.
2. **Downstream eval delta**: a small MMLU/GSM8K/needle slice — **within ~1%** of fp8. This catches quality loss
   that token-parity misses (reasoning, long-context recall).
3. **Per-expert sanity**: no single expert's output MSE blows up (a mis-scaled hot expert can dominate). Log the
   worst-expert error; if one expert is an outlier, keep *it* in fp8 (mixed-precision experts — cheap insurance).

## Where it sits in the decision tree (`path-to-1000.md`)
- **If NVLS C ≤ 2–3 µs:** lossless path wins (fp8 + small-spec → ~1100). **int4 not needed** — skip the quality
  risk. Keep it on the shelf.
- **If NVLS C ~4 µs (or e-tuning stalls / spec under-delivers):** int4-experts is the cushion → restores ~1000
  at C=4 µs. **Then the gate above is mandatory and on the critical path.**
- **If stale-TP (LOOP-C) passes its own gate:** comms is hidden → ~1588, int4 unneeded (different cushion).
So: **validate int4-experts in parallel (it's cheap to prep), but only ship it if the C measurement says the
lossless path is short.** The order is: measure C first → decide → gate int4 only if needed.

## Effort note
int4-experts is mostly a *quantization recipe* (llmcompressor/AWQ on the experts) + the gate — not new kernels
(the K5 int4 path exists, `k5_experts_int4.cu`). So it's a low-effort cushion to *prepare*; the cost is the
quality validation, which is why it's gated, not assumed. Don't quantize-and-hope — measure C, then decide.
