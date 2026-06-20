# Comms is resolved & measured end-to-end — the lever is NVLS latency on TP8, not EP count-reduction

**LOOP-C, 2026-06-20.** Closing the comms thread now that all three numbers are measured. Two payloads:
**(1)** my in-engine C prediction is independently CONFIRMED by Charles's working engine; **(2)** one stale
steering signal in `path-to-1000.md` should be retired (it points at an EP detour the NVLS measurement made
unnecessary). Supports Charles/Alyssa's comms work — flagging their doc, not editing it.

## The comms picture, fully pinned (three numbers, three provenances)
| all-reduce path | C (per collective) | how measured | role |
|---|---|---|---|
| stock NCCL ring | ~33–35 µs | E0 `nccl-tests`, standalone-launched | upper bound; the engine bypasses it |
| **in-engine AR (current)** | **~17–18 µs** | **Charles's 70.1 tok/s engine: comms = 23.8% of step = 3.2–3.4ms / 189 AR** | what runs today |
| NVLS multimem in-switch | **3.84 µs** | `nvls_ar.cu`, bit-exact (Alyssa 3.52–5.34) | the replacement |

**My E0 reconcile is confirmed.** I argued (`comms_floor_reconcile_e0.md`, from TPOT×e consistency) that the
in-engine C is **~10–18 µs, not the 35 µs stock ring**. Charles's custom engine — a completely independent
measurement (real TP8 decode, not a consistency argument) — lands the in-engine AR at **~17–18 µs**. Two
unrelated methods agree; 35 µs as "the engine's comms" is dead. Use C≈17 µs for the *current* engine, 3.84 µs
post-NVLS.

## Retire the stale EP-count steering in `path-to-1000.md` (lines 10–12)
The reaction-04 banner still reads:
> *"the comms lever is the **COUNT** (188 TP → ~94 via EP 1-barrier/layer) + batched spec, **NOT per-collective
> latency** — unless the multimem in-switch reduce beats the barrier, **which is still the make-or-break to
> measure**."*

**That "unless" condition has fired.** The multimem reduce was measured at **3.84 µs < the 16 µs barrier** —
it *beat* the barrier. So:
- The comms lever **IS per-collective latency** (NVLS multimem), exactly what the banner said it wasn't-unless.
- The **EP count-reduction detour (188→94) is now unnecessary** — and harmful at B=1: the doc's own analysis
  says EP "never wins at B=1" (busiest-rank E[max]=2.6 experts; all-to-all = 125 µs = 3.5× the TP all-reduce).
  You don't need EP's count cut when NVLS+overlap drive the *latency* of all 189 TP collectives to ~0.
- **Keep TP8.** The body of `path-to-1000.md` (line 26: `0.78 fp8 + 0.19 NVLS + ~0 = 0.97ms → 1033`) already
  assumes NVLS-on-TP8 — so the banner now contradicts the body. Recommend deleting the EP-count clause and the
  "NOT per-collective latency / still to measure" caveat (superseded by the 3.84 µs measurement).

## Net comms plan (measured, not projected)
**TP8 + NVLS multimem (3.84 µs) + exact deferred-overlap (hides it under the ~4.3 µs fp8 weight cover → comms
→ ~0, lossless) + spec on top.** No EP layout change, no count reduction, no retraining. Of the original
"three non-negotiables" (fp8, comms-handled, overhead→0), **comms is now the most de-risked**: the kernel
exists and is bit-exact; the only remaining comms work is the in-graph/k6 overlap wiring (Charles/Alyssa).
The binding terms are now the **~7 ms kernel floor** (router K4 + experts K5 → e→1, Charles) and **spec**
(EAGLE3+graphs, LOOP-A).
