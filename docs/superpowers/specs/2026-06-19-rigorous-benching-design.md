# Rigorous benching suite — design spec

**Date:** 2026-06-19
**Author:** Charles (+ Claude)
**Status:** Approved — implementing

## Goal

Upgrade our B=1 decode benchmarking suite (`src/inferutil/bench/`) into a
**research-grade** tool that reports MFU, MBU, a full latency panel (E2E /
throughput / per-token), and rich statistics, and that **closes the loop from
bottleneck diagnosis to a ranked "what to optimize next" recommendation.**

Informed by — but **not copied from** — Conifer's `coniferbench` methodology.
The metric definitions are public (MFU: PaLM paper; MBU: Databricks; roofline:
Williams et al.). We reimplement them clean-room. We do **not** import Conifer
code or its Metal/CPU machinery (wrong hardware: we target 8×H100 / vLLM /
the Rust engine).

## Scoping decisions (from the user)

1. **"TTS" = all latency views**: report E2E latency, throughput (tok/s), and
   per-token latency (TPOT/ITL) together as one latency panel.
2. **Measure target = both**: metrics must work on the analytical model
   (MockEngine) now AND on real measured runs (vLLM adapter) later, with an
   explicit measured-vs-predicted gap.
3. **Decision driver = bottleneck → lever**: diagnose the dominant bottleneck
   per run, then map it to the specific next optimization that would move it.

## Hard constraints (from the existing codebase)

- **Pure stdlib** (`pyproject.toml` declares no deps). Student-t CIs come from a
  small lookup table; no numpy.
- **`BenchResult` is built positionally** in tests. New fields are appended with
  defaults; nested dataclasses get explicit handling in `store.py`'s round-trip.
- Keep the full test suite green (`python3 -m pytest tests/`, currently 45
  passing) before and after. Rust suite (`cargo test --package engine`) is
  unaffected but verified.

## The math (clean-room, single source of truth)

New module `efficiency.py` holds pure functions:

```
flops_per_token(active_params)         = 2 * active_params          # 2N (PaLM); attention O(seq) optional, off by default
mfu(tok_s, active_params, peak_flops)  = flops_per_token * tok_s / peak_flops
mbu(tok_s, bytes_per_token, peak_bw)   = bytes_per_token * tok_s / peak_bw
ai_decode(active_params, bpt)          = flops_per_token / bytes_per_token         # FLOPs/byte
ai_prefill(active_params, wbytes, P)   = flops_per_token * P / weight_bytes
roofline_ridge(peak_flops, peak_bw)    = peak_flops / peak_bw
classify_regime(ai, ridge)             = "memory-bound" if ai < ridge else "compute-bound"
peak_flops(gpu, dtype_bytes)           = gpu.fp8_flops if dtype_bytes < 2 else gpu.bf16_flops
```

- **MFU** reported separately for prefill (compute-bound, meaningful) and decode
  (memory-bound, structurally low — reported for completeness), plus
  `achieved_tflops_{prefill,decode}`.
- **MBU** uses `bytes_per_token` from our existing accounting
  (`active_params*dtype + seq*kv_bytes_per_token`). **Deliberate improvement over
  Conifer:** Conifer drops KV from MBU as "second-order"; at our depths
  (640 → sweeping 32k) KV is first-order (~9.6 GB vs ~21.6 GB weights), so we
  KEEP KV and additionally report `kv_byte_share`.
- Peaks use SI (decimal) units consistent with `hardware.py`. Aggregate
  bandwidth = `gpu.hbm_bw * n_gpus`; aggregate FLOPs = `peak_flops * n_gpus`.

## Statistics

New module `stats.py`:

```
Stat = { n, mean, std (ddof=1), cv, p50, p90, p95, p99, min, max, ci95_lo, ci95_hi, samples }
summarize(samples) -> Stat        # Student-t CI for small n; None (never 0) when n==0
```

- `runner.py` default `repeats` raised 1 → 5; per-repeat decode tok/s, TPOT,
  TTFT, E2E collected into `Stat`s.
- `report.is_significant` upgraded to a CI-overlap / Welch-style check (current
  2σ heuristic kept as fallback for n<2).

## Decision support

New module `attribution.py`:

```
diagnose(result) -> Bottleneck {
    dominant_term: "weight_bw" | "kv_bw" | "comms" | "compute" | "kernel_gap",
    share: float,                # fraction of per-token time
    headroom_to_floor: float,    # 1 - pct_of_floor  (measured) or 0 (analytical)
    regime: str, ai: float, ridge: float,
    confidence: float, note: str,
}
```

Replaces `bench/roofline.py`'s hand-tuned 0.7/0.5 thresholds with
roofline-principled logic over the existing weight/kv/comms/compute breakdown,
plus the measured-vs-floor gap (`kernel_gap` term: time lost to real kernels /
launch / dequant beyond the analytical floor).

New module `levers.py`:

```
recommend(cfg, cluster, config) -> [Lever]   # ranked by predicted speedup
Lever { name, predicted_tok_s, speedup, effort: "S"|"M"|"L", rationale }
```

Each candidate lever is scored by **re-running the analytical model
(`decode_latency`) with the lever applied** and comparing tok/s to baseline.
Candidate levers: int4 weights (`dtype_bytes 1→0.5`-equiv via a half-byte
weight model), fp8/int8 KV (`kv_dtype_bytes 2→1`), spec-decode tree (apply an
acceptance-rate speedup model), EP/TP layout changes (sweep tp/ep), ideal
routing (placement optimizer payoff = imbalance elimination). Tagged with rough
effort. Ranked by predicted speedup; ties broken by lower effort.

## Reproducibility & export

New module `manifest.py`:

```
build_manifest(cfg, cluster, config, *, peak_bw=None, cli=None) -> dict
# host (hostname, platform), git commit (best-effort), model-config hash,
# bench config, seed, measured-or-spec peak BW, exact CLI, schema version.
```

`store.py` gains tidy long-format **CSV** export (one row per run, em-dash for
unmeasured — never fake 0) and lossless **JSONL** (full samples). New CLI:
`inferutil.bench export --format {csv,jsonl}` and `inferutil.bench diagnose`
(prints bottleneck + ranked levers). `run` writes a `manifest.json` alongside
results.

## Module map

| File | Status | Purpose |
|------|--------|---------|
| `src/inferutil/bench/efficiency.py` | NEW | MFU/MBU/AI/ridge pure math |
| `src/inferutil/bench/stats.py` | NEW | `Stat` + Student-t summarize |
| `src/inferutil/bench/attribution.py` | NEW | bottleneck diagnosis |
| `src/inferutil/bench/levers.py` | NEW | ranked next-lever recommender |
| `src/inferutil/bench/manifest.py` | NEW | reproducibility capture |
| `src/inferutil/bench/metrics.py` | EDIT | `Efficiency` field on `BenchResult` |
| `src/inferutil/bench/runner.py` | EDIT | repeats→5, collect latency Stats |
| `src/inferutil/bench/report.py` | EDIT | MFU/MBU/regime + latency panel + diagnosis |
| `src/inferutil/bench/store.py` | EDIT | round-trip new field; CSV/JSONL export |
| `src/inferutil/bench/cli.py` | EDIT | `export`, `diagnose` subcommands |
| `bench/roofline.py` | EDIT | use principled attributor |

## Testing

Extend `tests/test_bench_*.py`:
- `test_bench_efficiency.py`: known inputs → known MFU/MBU/AI/ridge/regime.
- `test_bench_stats.py`: percentile interpolation, Student-t CI width, CV, n==0→None.
- `test_bench_attribution.py` (extend): synthetic breakdown → expected dominant term; kernel-gap when below floor.
- `test_bench_levers.py`: predicted-speedup monotonicity (int4 < fp8 bytes ⇒ higher tok/s); ranking order.
- `test_bench_manifest.py`: required keys present; deterministic model hash.
- Round-trip test for `store.py` with the new `Efficiency` field.

## Out of scope (YAGNI)

- HTML report / plots (separate UI exists; not requested).
- Metal/CPU STREAM-triad bandwidth probe (wrong hardware; reuse `kernels/k5_microbench` HBM number when available).
- Continuous-batching / B>1 sweeps (B=1 is our regime).
