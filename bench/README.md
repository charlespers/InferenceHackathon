# bench/ — autoresearch + benchmark harness (B=1, 8×H100)

Measures the real wall-clock latency that the kernel/parallelism tuning is chasing, and tells
you which term to attack next. It speaks the **same OpenAI `/v1/chat/completions` SSE contract**
the UI streams, so one endpoint serves the demo and the benchmark. Stdlib-only (no installs).

## Files
- `roofline.py` — byte budget + the bandwidth ceiling; given a measured TPOT, prints **MFU, MBU,
  arithmetic intensity vs the roofline ridge, the regime, and the dominant term** via principled
  attribution (`argmax(weight_ms, kv_ms, kernel_gap_ms)` — not a hand-tuned threshold cascade).
- `measure.py` — streams one request, times **TTFT / TPOT / decode tok/s** from the wall clock
  (not from a self-reported number); parses optional `x_summary`.
- `sweep.py` — the **autoresearch driver**: the DoF search space + a **decision tree** mapping the
  dominant term → the next lever to try; appends each run to `results.jsonl`.

## Metric definitions
- **TTFT** — time to first streamed token (prefill).
- **TPOT** — mean inter-token wall-clock latency during decode (the number to minimize).
- **decode tok/s** — `(tokens-1)/(t_last-t_first)`.
- **MFU** — `2·N_active·tok/s / peak_FLOPS` (compute utilization; meaningful for prefill).
- **MBU** — `bytes/token·tok/s / peak_BW` (bandwidth utilization; the decode number that matters).
- **AI / ridge** — arithmetic intensity `FLOPs/byte` vs `peak_FLOPS/peak_BW`; below the ridge ⇒
  memory-bound.
- **% of roofline / floor** — achieved tok/s ÷ ceiling at that ctx/precision.

## Analytical suite (`src/inferutil/bench`, no GPU required)
The rigorous, statistically-honest version of this loop — runs on a laptop against the analytical
roofline model (MockEngine) and, when the engine lands, against real streamed runs:
```bash
PYTHONPATH=src python -m inferutil.bench run --name fp8ep --dtype 1 --kv-dtype 2 --tp 2 --ep 8 --repeats 5
PYTHONPATH=src python -m inferutil.bench diagnose --name fp8ep     # bottleneck → ranked next levers
PYTHONPATH=src python -m inferutil.bench export --name fp8ep --format csv --out runs.csv
```
- Reports MFU/MBU, a full latency panel (TTFT / E2E / throughput / TPOT) with **Student-t 95% CIs**,
  roofline regime, and a **bottleneck → ranked-lever** recommendation (predicted speedup + effort).
- `run` also writes a reproducibility `*.manifest.json` (host, git commit, model hash, hardware).
- Design rationale: `docs/superpowers/specs/2026-06-19-rigorous-benching-design.md`.

## Loop (use it the moment the engine is up)
```bash
# 1) baseline number
python bench/measure.py --base http://localhost:8000 --ctx 32768 --decode 128

# 2) where's the bottleneck?
python bench/roofline.py --ctx 32768 --weight-bytes 1 --kv-bytes 1 --tpot-ms <measured>

# 3) record + get the next lever, then change ONE engine setting and repeat
python bench/sweep.py --base http://localhost:8000 --ctx 32768 --label tp4ep2-fp8
```
Change one DoF (precision / layout / graph / spec / draft length), relaunch the engine, re-run.
Stop when % of roofline plateaus — then you're physics-limited (see `trajectory.md` §3).

## Extra signals to capture on the box (alongside this harness)
- `nvidia-smi dmon -s u` (per-GPU utilization / balance — is one GPU a routing hotspot?)
- Nsight Systems trace (gaps between kernels = launch/sync overhead → CUDA graph)
- Nsight Compute on K5 (the expert kernel = the bottleneck; check % of HBM peak)
- speculative **accept rate** (from the engine / `x_summary.spec_accept_rate`)

## Notes
- `--engine conifer|vllm` only matters for the mock's head-to-head profiles; a real OpenAI
  server ignores it.
- `relaunch_hook` in `sweep.py` is intentionally not wired — engine launch flags differ per
  config; the operator relaunches between runs (or wire it to your launcher).
