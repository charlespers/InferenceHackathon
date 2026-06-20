# Reacting to data round 2 — adaptive-top-k & route-prediction results

Two more committed results (`results/ab_*.json`, `results/routing_predict_early.json`) land on two of the
levers I analyzed. Both validate the analysis; one delivers a sharp strategic confirmation.

## `ab_baseline` vs `ab_adaptive` — adaptive top-k is INVISIBLE while floor-bound (the key confirmation)
| run | tok/s | TPOT | % of roofline | hint |
|---|---|---|---|---|
| baseline (top-8) | 13.23 | 75.6 ms | 2.5% | "DOMINATED by floor (launch/host/comms)" |
| adaptive top-k | **9.67** | 103 ms | 1.8% | same — and *slower* |

Adaptive top-k (a weight-read lever) made it **worse**, not better, because this engine is at **2.5% of
roofline** — massively floor/overhead-bound. The expert-byte savings are invisible; the adaptive routing's
extra per-token work shows up as pure overhead. **This is direct on-box evidence for `overhead-attribution.md`:
while the floor dominates, weight levers (adaptive-top-k, int4, fp8) don't move the needle — fix the floor
first.** (Note: this engine at 13 tok/s is even more overhead-bound than vLLM's 85.7; it's a different/slower
server — but the lesson transfers: measure where you are on the roofline before reaching for a weight lever.)

## `routing_predict_early` — validates the route-prediction + EP-placement analysis
Measured on the real Qwen3-235B (200 decode tokens):
- **Busiest-rank imbalance: round-robin 2.53** (≈ my balls-in-bins 2.6) → with **replication 16× → 1.73**.
  Confirms `ep-placement-for-b1.md`: replication lowers the per-step busiest factor.
- **Co-activation (affinity) placement works:** cross-GPU expert traffic 87.7% → **68.3%**, local fraction
  0.123 → **0.317** (2.6×). Exactly the co-activation-aware placement I argued the optimizer needs (vs its
  marginal-count greedy). The data says affinity placement is real.
- **DirectProxy route prediction is viable — and depth-dependent:** persistence **0.446** overall, rising by
  layer (0.10 early → 0.34+ and climbing). Matches `predictor.rs`'s premise that the residual stream makes
  *later* layers predictable. **Design refinement: prefetch the deep layers** (high persistence); early
  layers are near-random — skip prefetch there or use a learned predictor.

## Updated priorities (now confirmed by two data rounds)
1. **Fix the floor/overhead first** — confirmed twice: vLLM at 16% of roofline (overhead-attribution), the
   adaptive engine at 2.5%. Until the floor is down, **every weight lever (adaptive-top-k, int4, fp8) is
   invisible.** Run `E-attr` (Nsight Systems) to split the floor into comms vs kernels vs host.
2. **Layout = TP8** (structural; EP busiest-rank 2.53 measured).
3. **Comms tuning (E0b)** — the floor's comms component (16µs all-reduce).
4. **Route-prefetch (`scheduler.rs`), deep layers only** — now justified by data (persistence 0.45 rising):
   if `E-attr` says comms-bound, prefetching the deep-layer expert union hides the all-to-all there.
5. **EP-path mitigation = affinity placement** (0.123→0.317 locality) + replication (2.53→1.73), *if* EP is
   ever forced. TP8 still avoids it entirely.
6. **Weight levers (fp8/int4/adaptive-top-k) — LAST.** The ab_adaptive result proves they're invisible while
   floor-bound; they pay only *after* the floor is fixed (then their ~14% share grows).

## One-line takeaway
The data says the game is **the floor, not the bytes.** Two independent results show weight-read levers do
nothing while the engine runs at 2–16% of roofline. **Attribute the floor (`E-attr`), kill comms + kernel
overhead, serve TP8 — then the quant/adaptive levers finally matter.**
