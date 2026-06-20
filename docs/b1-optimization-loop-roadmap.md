# B=1 decode optimization loop — ranked roadmap & ceiling proof

**Source of truth is runnable, not this prose:** `PYTHONPATH=src python3 -m inferutil.optimize`
(deterministic, CPU-only, 12 tests in `tests/test_optimize.py`). This doc is the decision
summary; the driver is the calibrated model that produces every number below.

## What this is

An automated optimization loop for the B=1 decode engine (Qwen3-235B-A22B, 8×H100). It
cannot run CUDA from a GPU-less box, so it drives an **additive TPOT model calibrated to the
team's measured on-box anchors** and runs the loop the goal asks for: profile the hottest
term → apply ONE lever → re-derive tok/s → keep only on strict improvement. Each lever is
tagged `MEASURED` / `PREDICTED` / `MISSING` with a doc citation and a quality gate.

Why additive (not the multiplicative efficiency knob): at B=1 a step's wall-time is a **sum**
of terms on different hardware paths — `overhead + comms + weight_read + kv_read`. Each lever
attacks ONE term; "profile the hottest path and attack it" is literally that.

## Calibration (reproduces the measured baseline)

```
baseline  TPOT 11.66 ms = overhead 7.00 (60%) + comms 3.00 (26%) + weight 1.56 (13%) + kv 0.10
          -> 85.8 tok/s   (vLLM bf16-TP8 = 85.7, reproduced to <1%)
```
Weight term cross-checked against the analytical roofline (`latency.decode_latency`, tp-plan:
bf16 1.61 ms / fp8 0.80 ms). The team's own engine is at **74.5 tok/s** today (commit
`21c5f10`, climbing toward the vLLM target); re-anchor by editing `MEASURED_BASELINE`.

## The loop (greedy, profile-driven)

| # | attack | lever | status | tok/s | next bottleneck |
|---|--------|-------|--------|-------|-----------------|
| 1 | overhead | Megakernel / CUDA-graph (overhead→~0.5ms) | PREDICTED | 85.8 → 193.8 | comms |
| 2 | comms | Deferred-overlap (hide comms under weight stream) | **MISSING** | 193.8 → 463.0 | weight |
| 3 | weight | FP8 weights (fused dequant) | PREDICTED | 463.0 → 724.6 | weight |

**Plain-decode best (all lossless levers):** TPOT 1.38 ms → **724.6 tok/s**.

## Ceiling proof — is 1000 tok/s reachable?

- **Hard single-stream floor** = fp8 active-weight read alone = 0.78 ms = **1282 tok/s**.
  1000 sits *below* this wall, so 1000 is **physically possible** for plain decode — the
  blocker is overhead+comms, not the memory wall.
- **Conservative** (overhead→0, NVLS 2 µs, comms NOT hidden): **865 tok/s** → misses 1000.
- **Optimistic** (deferred-overlap hides comms→0): **1205 tok/s** → clears 1000.

The plain-decode path to 1000 hinges **entirely on the deferred-overlap kernel** — which is
**unmeasured e2e**. This surfaces a real tension between two team docs: `path-to-1000.md`'s
865 ceiling assumes comms is *not* overlapped; `comms-breakthrough-nvls.md` says it *is*.

### Sensitivity — exactly when plain decode reaches 1000

```
 overhead | 0% hidden   50%    76%   100% | min-overlap→1000
   0.00ms |      644    840    997  1205* | 76% hidden
   0.25ms |      555    694    798   926  | unreachable
   0.50ms |      487    591    665   752  | unreachable
   0.75ms |      434    515    570   633  | unreachable
```

Plain decode clears 1000 **only** at overhead≈0 **and** ≥76% of comms hidden — the top-right
corner, two unmeasured kernel bets stacked. At any realistic overhead residual (≥0.25 ms) it
is unreachable **even with perfectly hidden comms**.

## The missing capability: speculative decode

Spec is the only lever that clears 1000 **robustly** — in both the conservative and optimistic
comms scenarios, **losslessly** (exact verification). It does not bet 1000 on one unmeasured
kernel gate.

- **Buildable headline:** EAGLE3 **W1×D2** (single draft head), accept 0.72, τ≈2.24 emitted
  per verify → **1.82×** → **724.6 → 1318 tok/s**.
- The floor-aware sweep's optimum is a *bigger* tree (W4×D8, 2.09×) — but that assumes N
  *independent* drafters; real EAGLE3 is one correlated head, so W1×D2 is the engineering pick.
  (The sweep correctly prefers bigger trees here because F≈0.63 is still floor-bound: a 0.5 ms
  overhead residual + structural non-expert weight keep F high.)

## Ranked roadmap of remaining gains

| rank | lever | status | gain | gate |
|------|-------|--------|------|------|
| 1 | NVLS in-kernel all-reduce | **MEASURED** (3.84 µs) | comms 3.0→0.72 ms | lossless ✓ |
| 2 | Megakernel / CUDA-graph (overhead→0) | PREDICTED | +108 tok/s | lossless |
| 3 | FP8 weights (fused dequant) | PREDICTED | +262 tok/s | perplexity parity (gated) |
| 4 | **Deferred-overlap (comms→0)** | **MISSING — the open gate** | +269 tok/s | C < weight-cover; **measure this** |
| 5 | int4 experts | PREDICTED | safety net | **LOSSY** — only if fp8 slips |
| 6 | **Speculative decode (EAGLE3 W1×D2)** | **MISSING — the make-or-break** | ×1.82 → 1318 | lossless ✓ |

## The one measurement that unblocks everything

**Wire the measured NVLS reduce (3.84 µs) into the megakernel/k6 step with deferred-overlap and
measure e2e TPOT.** That single number decides whether plain decode can approach 1000 on its own
or whether spec is mandatory. Either way **spec (W1×D2) is the robust path to 1000** and should
be built in parallel — it clears the target without depending on the overlap gate landing.

## Honest scope

All numbers are analytical/calibrated, not fresh GPU measurements (this box has no GPU). The
calibration *reproduces* the measured baseline and every lever carries its measured-or-predicted
status; the go/no-go gate is always a realized on-box tok/s. Run `inferutil.optimize` to
regenerate; edit `LEVERS` / `MEASURED_BASELINE` to fold in new on-box data.
