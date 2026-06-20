# experiments/stale_tp — LOOP-C: Speculative / Stale Tensor Parallelism

**Avenue.** Break the ~188 serial TP all-reduces/token (the dominant B=1 floor term)
by computing on **stale/predicted** reduced activations so the collective no longer
blocks the critical path. Full design + literature: [`research/n4_speculative_stale_tp.md`](../../research/n4_speculative_stale_tp.md).
Ceiling model: [`tools/stale_tp_ceiling.py`](../../tools/stale_tp_ceiling.py).

**This dir is the QUALITY PROBE.** It does **not** make decode faster — it measures the
single go/no-go number: *how much all-reduce staleness does Qwen3-235B tolerate before
greedy output diverges from exact?* If quality holds at K≥2 **without retraining**, a real
stale-TP kernel is worth building (the ceiling model says it then reaches ~roofline,
stacking with Charles's multimem one-shot). If it collapses → honest KILL (the literature's
prior: Ladder-Residual/Kog both needed retraining).

## Files
- `stale_tp.py` — env/ctl-configured monkeypatch of vLLM's `tensor_model_parallel_all_reduce`.
  Decision logic is the pure `StaleScheduler` (no torch) → unit-testable with no GPU.
- `test_stale_tp.py` — 8 pure unit tests of the scheduler (run anywhere, no GPU).
- `run_stale_probe.sh` — box slot runner: ONE bf16-TP8 launch, sweeps K/mode/policy via
  the control file, captures greedy per point, runs the parity gate. Full lock/mem/time guards.

## De-risk status (done, no GPU)
```
python experiments/stale_tp/test_stale_tp.py     # ALL 8 TESTS PASSED
```
The scheduler logic (refresh-vs-substitute, per-slot caching, period wrap, decode-only
gating, layer & temporal modes, ctl reload) is verified. Only the vLLM symbol-rebind in
`install()` is untested until a slot.

## Knobs (env defaults; control file overrides live)
`STALE_TP_ENABLE` `STALE_TP_K` `STALE_TP_MODE`(layer|temporal) `STALE_TP_POLICY`(proxy|local)
`STALE_TP_DECODE_ONLY`(keep prefill exact) `STALE_TP_PERIOD`(all-reduces/pass, 2×layers=188)
`STALE_TP_CTL`(JSON `{enable,K,mode,policy,decode_only}` re-read per pass → sweep without relaunch).

## What the probe sweeps & how to read it
| point | meaning | expectation |
|---|---|---|
| `exact` | reference (enable=false) | the greedy baseline |
| `lyr_proxy_k2/k4/k8` | reuse last real reduced value every K layers | **the N4 hypothesis** — does parity hold? |
| `lyr_local_k2` | return un-reduced local partial | **sanity floor** — should degrade (matches Kog "naive removal heavily degrades") |
| `tmp_proxy_k2` | reuse previous token's reduce | across-token staleness variant |

Parity via `tools/quality_compare.py` (greedy prefix agreement + exact-match). **GO** if
`lyr_proxy_k2` stays ≳99% vs `exact`; **conditional GO** if only K=2 or attention-exact holds;
**NO-GO** if it collapses even at K=2 → defer to Charles's multimem lever; Ladder-Residual
(retrain) is out of hackathon scope.

## Run on the box (in a locked, in-window, mem-checked slot only)
```bash
# stage standalone copy (avoid disturbing jminding's checkout)
mkdir -p /alloc/data/stale_tp_tools
cp experiments/stale_tp/stale_tp.py tools/quality_probe.py tools/quality_compare.py \
   tools/measure_baseline.py /alloc/data/stale_tp_tools/
# arm (survives the ssh session; fires in-window under the atomic lock)
nohup bash experiments/stale_tp/run_stale_probe.sh >/dev/null 2>&1 &
# watch
tail -f /alloc/data/stale_tp/slot_stale.log    # results -> origin/loopc-results
```
First slot: confirm `observed_calls_per_pass` in `vllm_stale.log` == 188 (calibrates PERIOD),
verify `exact` reproduces the baseline, then read the K=2 parity gate.
</content>
