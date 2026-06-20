# Inference Benchmark Harness — Design Spec

**Date:** 2026-06-19
**Status:** Draft, pending user review
**Branch:** `benchmarking-system`
**Context:** Hackathon. Build an offline benchmark harness that measures real B=1 decode
performance for **Qwen3-235B-A22B on 8×H100** (running on Prime Intellect), and reports it
**against the analytical roofline** the repo already models. This repo *is* the engine — it
is seeded from the Conifer engine, so the harness instruments our own in-process decode loop.

**Repo state (verified 2026-06-19):** `main` now contains both the `inferutil` analytical
model (`src/inferutil/`) and the merged inference console (`server/` + `ui/`, the former
`build/inference-console` branch — already merged at commit `5946a90`, so there is no longer
an unrelated-history problem). The Conifer engine itself is **not yet in the repo** on any
branch; the only engine present is `server/mock_engine.py` (a UI mock). The `Engine` seam in
§3 exists precisely so we build now and adapt when real Conifer code lands.

## 1. Goal & Non-Goals

### Goal
A self-contained `inferutil.bench` package that, for a named configuration, runs a fixed-window
B=1 benchmark and produces a structured result capturing:
- **Latency:** TTFT, prefill tok/s, decode tok/s, TPOT (inter-token p50/p95), total.
- **Bandwidth / efficiency:** bytes-moved/token, achieved HBM bandwidth, % of peak bandwidth,
  and % of the analytical floor (tok/s) — the roofline metrics the whole repo is built around.
- **Device telemetry:** per-GPU temperature, SM/memory utilization %, power, energy/token,
  clocks, HBM memory used; aggregated as min/mean/max plus per-GPU spread (imbalance).

Results are saved to a structured store and surfaced via a CLI report that diffs **measured vs
analytical floor** and **measured vs prior runs**. This realizes the "change one thing,
re-measure, keep what wins" measurement loop from `docs/h100-tuning-playbook.md` §G as code,
matching the chosen **experiment-compare-and-report** workflow.

### Non-Goals (YAGNI / deliberately excluded)
- No live dashboard or web server (the `server/` + `ui/` console, now merged to `main`, is the
  live view; the harness feeds it via the §7 adapter rather than duplicating it).
- No DCGM/Nsight *required* path — they are optional, runtime-detected plug-ins only.
- No throughput / multi-request / continuous-batching modes — **B=1 only**.
- No changes to the `server/` or `ui/` console code — we consume its metric vocabulary, not edit it.
- The harness does not implement inference; it drives an `Engine` behind an interface.

## 2. Architecture

A new package alongside the existing analytical model, same stdlib-first, dependency-light
style as `inferutil`:

```
src/inferutil/bench/
  engine.py      # Engine protocol (the seam) + MockEngine; ConiferEngine lands later
  runner.py      # drives the fixed-window benchmark for one config; the decode loop wrapper
  telemetry.py   # background NVML sampler thread (temps/util/power/clocks/mem); pluggable
  metrics.py     # raw timings + samples + model facts -> BenchResult (derived BW, efficiency)
  store.py       # write/read JSON results keyed by config; load prior runs
  report.py      # CLI report: metrics + diff vs analytical floor + diff vs prior run
  cli.py         # `python -m inferutil.bench run/report/compare ...`
tests/test_bench_*.py
results/         # JSON results store (git-ignored; results/.gitkeep tracked)
```

The harness reuses the existing analytical modules as the **roofline oracle**:
`model.py` (bytes-per-token, param accounting), `hardware.py` (peak HBM BW, GPU specs),
`latency.py` (`decode_latency` -> the floor tok/s for a given plan/dtype/seq).

## 3. The engine seam

Everything depends on a small `Engine` protocol, never on Conifer directly. This is the seam
that lets us build and test now and plug the real engine in later; **the data ingestion path
is expected to change once the engine is set up, and only `ConiferEngine` changes when it does.**

```python
class Engine(Protocol):
    def reset(self, *, plan, dtype_bytes, kv_dtype_bytes, tp, ep, seq_len) -> None: ...
    def prefill(self, token_ids: list[int]) -> PrefillResult: ...   # first token + timing
    def decode_step(self) -> DecodeStep: ...                        # one token; optional routing
```

- **`MockEngine`** — synthesizes per-step timings seeded from `inferutil.latency.decode_latency`
  for the active config, plus small deterministic (seeded) jitter. Makes the entire pipeline
  runnable on a laptop with **no GPU**. A perfectly-efficient MockEngine reports ~100% of the
  analytical floor, which doubles as the wiring sanity check.
- **`ConiferEngine`** — thin adapter over the in-repo decode loop once Conifer code lands.
  Implements only the three protocol methods.

`PrefillResult` / `DecodeStep` are small frozen dataclasses: a step carries its wall-time and
optional per-token expert routing (`[{layer, expert_id, gpu}]`) when the engine exposes it
(absent for MockEngine and non-augmented engines).

## 4. Metrics

Three families flow into one `BenchResult` dataclass.

### 4.1 Latency (timed in `runner.py`)
TTFT (prefill end → first token), prefill tok/s, decode tok/s, TPOT inter-token p50/p95,
total wall-time. `warmup_steps` are discarded before timing (CUDA-graph capture, cold caches).

### 4.2 Bandwidth / efficiency (derived in `metrics.py`)
Per the telemetry research, achieved bandwidth is **derived, not hardware-counter-based**, so it
works on any box:
- `bytes_per_token` — from `model.py` active-param + KV accounting for the config's dtype/plan.
- `achieved_hbm_bw = bytes_per_token / TPOT_seconds`.
- `pct_of_peak_bw = achieved_hbm_bw / cluster.aggregate_hbm_bw`.
- `pct_of_floor = measured_decode_tok_s / analytical_floor_tok_s`
  (floor from `decode_latency(...).tokens_per_s`).

These four are the headline roofline numbers: how close to physics are we, and where is the leak
(compare measured breakdown against the analytical weight/kv/comms terms).

### 4.3 Device telemetry (`telemetry.py`)
A background thread samples NVML at ~50–100 ms for all GPUs, bracketing the decode window so it
does not perturb the timing path. Per GPU: temperature, SM util %, memory util %, power (W),
SM/memory clocks, HBM bytes used. Aggregated to min/mean/max across the window and across GPUs,
plus **per-GPU spread** (max−mean util = imbalance, which exposes the expert-routing skew the
analytical model predicts). **Energy/token** = integral of power over the decode window ÷ tokens.

Telemetry is behind a `TelemetrySource` interface:
- `NvmlTelemetry` (default, via `pynvml`) — the guaranteed baseline.
- `NullTelemetry` — when NVML is unavailable (laptop); metrics report telemetry as unavailable.
- `DcgmTelemetry` (optional) — auto-detected at startup; if profiling counters are permitted on
  the box it adds measured DRAM-active % to cross-check the derived bandwidth. Absence is silent.
- `FakeTelemetrySource` — canned samples for tests.

## 5. Experiment / config model

A run is one named `BenchConfig` — the knobs from the playbook measurement loop:

```python
@dataclass(frozen=True)
class BenchConfig:
    name: str                      # e.g. "fp8-hybrid-tp2ep8"
    plan: str                      # tp | ep | hybrid
    dtype_bytes: int               # 2=bf16, 1=fp8
    kv_dtype_bytes: int
    tp: int
    ep: int
    prompt_tokens: int = 512       # fixed window (playbook §G)
    decode_tokens: int = 128
    seed: int = 0
    warmup_steps: int = 8
```

The config serializes to a stable id (hash of its fields) so re-running the same knobs extends
the same result lineage. Encoding the loop this way — one variable changed per config — is the
point: comparisons become a command, not a manual spreadsheet.

## 6. Results store (`store.py`)

- One JSON file per run: `results/<config-name>/<runid>.json`. `runid` is supplied by the caller
  (the CLI stamps a timestamp) — the harness never calls a wall-clock internally so runs are
  reproducible and the store is deterministic.
- Each file is self-contained and diffable:
  `{config, env (gpu, n_gpus, driver, host), metrics: BenchResult, analytical: floor}`.
- `results/` is git-ignored (a run can be large/noisy); `results/.gitkeep` keeps the directory.
- `store.load_latest(name)` and `store.load_all(name)` back the report's prior-run diff.

## 7. Report (`report.py` + `cli.py`)

Mirrors the existing `inferutil` report style (aligned text tables + takeaways; text-first so it
reads over SSH). Three subcommands:

- **`run <config>`** — execute a benchmark, print the result, save it to the store.
- **`report <runid|latest>`** — full breakdown for one run **next to the analytical floor**:
  measured tok/s vs floor tok/s, achieved BW vs peak, telemetry summary, and where the leak is
  (compare against the analytical weight/kv/comms breakdown).
- **`compare <a> <b>`** — side-by-side diff of two runs: Δ tok/s, Δ TTFT, Δ efficiency,
  Δ temp/power. Answers "did FP8 actually help?" in one command.

Each subcommand takes `--json` for piping. A single adapter function serializes a `BenchResult`
into the console's `x_summary` shape (`ttft_ms`, `decode_tok_per_s`, `prefill_tokens`,
`completion_tokens`, …) — the contract defined in `server/schemas.py` / `server/mock_engine.py`,
now on `main` — so the console can consume harness output later. A thin add, not built now.

## 8. Testing (stdlib `unittest`, matching `tests/test_model.py`)

- `MockEngine` makes the full pipeline runnable with **no GPU**: assert the runner produces a
  well-formed `BenchResult`, derived-bandwidth math is correct, the store round-trips JSON, the
  report renders, and `compare` computes diffs.
- `FakeTelemetrySource` (canned samples) tests sampler aggregation (min/mean/max, spread,
  energy/token) without NVML.
- Sanity test: a perfectly-efficient `MockEngine` reports ~100% of the analytical floor, proving
  the measured-vs-roofline wiring.

## 9. Integration with the rest of the repo

- **Reuses** `inferutil.model`, `inferutil.hardware`, `inferutil.latency` as the roofline oracle —
  no duplication of model facts or peak specs.
- **Extends, does not reinvent,** the console metric vocabulary (`server/schemas.py`,
  `server/mock_engine.py`, now on `main`): the `BenchResult -> x_summary` adapter keeps field
  names aligned so harness results can feed the console and its stubbed `RealEngineBackend`
  (`server/backend.py`) can later pull from this instrumentation layer.
- **Branch hygiene:** the console (formerly `build/inference-console`) is already merged into
  `main` at `5946a90`, so the earlier unrelated-history concern is resolved. The
  `benchmarking-system` branch should be synced onto current `main` before implementation so it
  builds on top of `server/` rather than the pre-merge tree it was cut from.

## 10. Build order (for the plan)

1. `engine.py` protocol + `MockEngine` (seeded from `latency.py`).
2. `metrics.py` `BenchResult` + derived bandwidth/efficiency math.
3. `runner.py` fixed-window loop (prefill, warmup discard, decode timing).
4. `telemetry.py` sampler + `NvmlTelemetry`/`NullTelemetry`/`FakeTelemetrySource`.
5. `store.py` JSON read/write.
6. `report.py` + `cli.py` (`run`/`report`/`compare`, `--json`).
7. Tests across the pipeline (no GPU required).
8. `BenchResult -> x_summary` adapter + `results/.gitkeep`, `.gitignore` entry.
