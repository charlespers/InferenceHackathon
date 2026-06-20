# Inference Benchmark Harness — Agent-Loop Extensions Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. This is an ADDENDUM to `2026-06-19-inference-benchmark-harness.md` (Tasks 1–8 already merged-clean on branch `benchmarking-system`).

**Goal:** Make `inferutil.bench` trustworthy for an autonomous optimization loop: run-to-run variance/significance, measured per-phase attribution, output-quality parity, and a pass/fail regression gate.

**Architecture:** Extend the existing modules. All four features work against `MockEngine` today and gain real teeth when `ConiferEngine` lands behind the unchanged `Engine` seam. New `BenchResult` fields are APPENDED WITH DEFAULTS so every existing positional construction and test stays valid.

**Tech Stack:** Python ≥3.10, stdlib only. Tests are plain `assert` functions in top-level `tests/`, runnable via `python -m pytest tests/` and directly.

## Global Constraints

- Python ≥3.10, **standard library only**. All dataclasses `frozen=True`. SI units (seconds, bytes/sec, joules).
- **Append-only to `BenchResult`**: new fields go AFTER `telemetry` and MUST have defaults (`n_repeats: int = 1`, `decode_tok_per_s_std: float = 0.0`, `measured_breakdown=None`, `quality=None`). Do not reorder existing fields.
- The harness calls NO wall-clock for run identity except `cli.py`'s `run`/(new)`gate` paths via `datetime`.
- Tests start with the `sys.path.insert(0, .../src)` shim from `tests/test_model.py`.
- `MockEngine` stays the analytical-floor oracle: with `efficiency=1.0, jitter=0.0` a decode step's `seconds` still equals `decode_latency(...).total_s`. New `MockEngine` outputs (breakdown, token_id) must not change that invariant.
- After EACH task, the FULL suite (`python -m pytest tests/ -q`) must stay green (start: 29 passed).
- Commit with `git commit --no-verify`.

---

### Task 9: Run-to-run variance + significance

**Files:**
- Modify: `src/inferutil/bench/config.py` (add `repeats` field)
- Modify: `src/inferutil/bench/metrics.py` (`BenchResult` + `build_result` stats)
- Modify: `src/inferutil/bench/runner.py` (repeat loop)
- Modify: `src/inferutil/bench/report.py` (`format_result` shows ±std; `format_compare` significance line)
- Test: `tests/test_bench_variance.py`

**Interfaces:**
- Consumes: existing `BenchConfig`, `BenchResult`, `run_benchmark`, `MockEngine`.
- Produces: `BenchConfig.repeats: int = 1`; `BenchResult.n_repeats: int = 1` and `BenchResult.decode_tok_per_s_std: float = 0.0` (appended, defaulted); `build_result` gains keyword `decode_tok_per_s_samples: list | None = None` (per-repeat tok/s; when given, `decode_tok_per_s` = mean and `decode_tok_per_s_std` = population stddev, `n_repeats` = len); `run_benchmark` runs `config.repeats` full passes; `report.is_significant(a, b) -> bool` and an updated `format_compare`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_variance.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.metrics import build_result, summarize_telemetry
from inferutil.bench.report import is_significant, format_compare
from inferutil.bench.store import RunRecord

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def test_repeats_populate_std_and_count():
    cfg = BenchConfig(name="v", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=32, repeats=5)
    # jitter makes repeats differ so std > 0
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.05, seed=1)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    assert r.n_repeats == 5
    assert r.decode_tok_per_s_std > 0.0
    assert r.decode_tok_per_s > 0.0


def test_single_repeat_has_zero_std():
    cfg = BenchConfig(name="v", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=16, repeats=1)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    assert r.n_repeats == 1 and r.decode_tok_per_s_std == 0.0


def test_build_result_samples_mean_and_std():
    tele = summarize_telemetry([], 10, 0.0)
    cfg = BenchConfig(name="v", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=16)
    r = build_result(cfg=QWEN3_235B, cluster=CLUSTER, config=cfg, ttft_s=0.04,
                     prefill_tok_per_s=1000.0, decode_step_seconds=[0.01] * 15,
                     telemetry_summary=tele, decode_tok_per_s_samples=[90.0, 110.0])
    assert r.decode_tok_per_s == 100.0
    assert abs(r.decode_tok_per_s_std - 10.0) < 1e-9   # popstd of [90,110]
    assert r.n_repeats == 2


def test_significance_threshold():
    # means 100 vs 130, stds 5 and 5 -> combined 2*sqrt(50)=14.1 -> 30 is significant
    assert is_significant(100.0, 5.0, 130.0, 5.0) is True
    # means 100 vs 105, stds 5 and 5 -> 5 < 14.1 -> not significant
    assert is_significant(100.0, 5.0, 105.0, 5.0) is False


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_variance.py -v`
Expected: FAIL (`cannot import name 'is_significant'` / `repeats` unexpected kwarg).

- [ ] **Step 3: Write minimal implementation**

In `src/inferutil/bench/config.py`, add a field to `BenchConfig` (AFTER `warmup_steps`, keep defaults last):

```python
    warmup_steps: int = 8
    repeats: int = 1               # full-run repeats for variance/significance
```

In `src/inferutil/bench/metrics.py`, append two fields to `BenchResult` (AFTER `telemetry`, both defaulted):

```python
    # device
    telemetry: TelemetrySummary
    # variance (appended, defaulted — keeps existing positional construction valid)
    n_repeats: int = 1
    decode_tok_per_s_std: float = 0.0
```

Then update `build_result`'s signature and body. Replace the existing `build_result` definition header and the `decode_tok_per_s` computation:

```python
def build_result(*, cfg, cluster, config, ttft_s, prefill_tok_per_s, decode_step_seconds,
                 telemetry_summary, decode_tok_per_s_samples=None):
```

After computing `decode_tok_per_s` from the representative `decode_step_seconds` (existing logic), add — just before the `return`:

```python
    # Variance across full-run repeats, when provided. The representative
    # decode_step_seconds still drives TPOT percentiles + bytes/BW.
    if decode_tok_per_s_samples:
        n_rep = len(decode_tok_per_s_samples)
        mean = sum(decode_tok_per_s_samples) / n_rep
        var = sum((x - mean) ** 2 for x in decode_tok_per_s_samples) / n_rep
        decode_tok_per_s = mean
        decode_tok_per_s_std = var ** 0.5
    else:
        n_rep = 1
        decode_tok_per_s_std = 0.0
```

and add the two new fields to the `BenchResult(...)` constructor call:

```python
        telemetry=telemetry_summary,
        n_repeats=n_rep, decode_tok_per_s_std=decode_tok_per_s_std)
```

In `src/inferutil/bench/runner.py`, wrap the prefill+decode in a repeat loop. Replace the body after the `engine.reset(...)` call so it runs `config.repeats` passes, telemetry brackets the LAST pass, and per-pass decode tok/s are collected:

```python
    samples_tok_s = []
    step_seconds = []
    gpu_samples = []
    ttft_s = 0.0
    prefill_tok_per_s = 0.0
    for rep in range(config.repeats):
        engine.reset(plan=config.plan, dtype_bytes=config.dtype_bytes,
                     kv_dtype_bytes=config.kv_dtype_bytes, tp=config.tp, ep=config.ep,
                     seq_len=config.seq_len)
        pre = engine.prefill(list(range(config.prompt_tokens)))
        ttft_s = pre.seconds + pre.first_token_seconds
        prefill_tok_per_s = (pre.n_prompt_tokens / pre.seconds) if pre.seconds else float("inf")
        for _ in range(config.warmup_steps):
            engine.decode_step()
        last_rep = rep == config.repeats - 1
        if last_rep:
            telemetry.start()
        step_seconds = [engine.decode_step().seconds for _ in range(config.decode_tokens - 1)]
        if last_rep:
            gpu_samples = telemetry.stop()
        total = sum(step_seconds)
        samples_tok_s.append((len(step_seconds) / total) if total else float("inf"))

    summary = summarize_telemetry(gpu_samples, config.decode_tokens, sum(step_seconds))
    return build_result(cfg=cfg, cluster=cluster, config=config, ttft_s=ttft_s,
                        prefill_tok_per_s=prefill_tok_per_s,
                        decode_step_seconds=step_seconds, telemetry_summary=summary,
                        decode_tok_per_s_samples=samples_tok_s)
```

Note: the initial `engine.reset(...)` that previously stood before the loop is now INSIDE the loop (one reset per repeat). Remove the old single pre-loop reset/prefill/warmup/decode block so it is not duplicated. Keep the `decode_tokens < 2` guard and the `telemetry = telemetry or NullTelemetry()` line before the loop.

In `src/inferutil/bench/report.py`, add `is_significant` and use it in `format_compare`. Add near the top (after imports):

```python
def is_significant(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """True when |mean_b - mean_a| exceeds combined run-to-run noise (~95%)."""
    noise = 2.0 * ((std_a ** 2 + std_b ** 2) ** 0.5)
    return abs(mean_b - mean_a) > noise
```

In `format_result`, change the decode line to show ±std when n_repeats > 1. Replace the existing decode latency line with:

```python
        f"  decode       : {r.decode_tok_per_s:8.1f} tok/s"
        + (f" ±{r.decode_tok_per_s_std:.1f} (n={r.n_repeats})" if r.n_repeats > 1 else "")
        + f"   (TPOT p50 {r.tpot_p50_s*1e3:.2f} / p95 {r.tpot_p95_s*1e3:.2f} ms)",
```

In `format_compare`, append a significance verdict line for decode tok/s. Add as the final list element before the closing `])`:

```python
        f"  {'decode sig?':<16}"
        f"{'SIGNIFICANT' if is_significant(ra.decode_tok_per_s, ra.decode_tok_per_s_std, rb.decode_tok_per_s, rb.decode_tok_per_s_std) else 'within-noise':>36}",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bench_variance.py -v`
Expected: PASS (4 passed).
Then: `python -m pytest tests/ -q` → all green (33 passed: 29 + 4).

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/config.py src/inferutil/bench/metrics.py src/inferutil/bench/runner.py src/inferutil/bench/report.py tests/test_bench_variance.py
git commit --no-verify -m "feat(bench): run-to-run variance + compare significance"
```

---

### Task 10: Measured per-phase attribution

**Files:**
- Modify: `src/inferutil/bench/engine.py` (`StepBreakdown` dataclass; `DecodeStep.breakdown`; `MockEngine` fills it)
- Modify: `src/inferutil/bench/metrics.py` (`MeasuredBreakdown`; aggregate; `BenchResult.measured_breakdown`)
- Modify: `src/inferutil/bench/runner.py` (collect per-step breakdowns from the last repeat)
- Modify: `src/inferutil/bench/report.py` (`format_result` shows measured breakdown)
- Modify: `src/inferutil/bench/store.py` (round-trip the optional nested breakdown)
- Test: `tests/test_bench_attribution.py`

**Interfaces:**
- Consumes: `decode_latency(...)` returns `DecodeBreakdown` with `.weight_read_s, .kv_read_s, .comms_s, .compute_s` (existing).
- Produces: `engine.StepBreakdown(weight_s, kv_s, comms_s, compute_s)` (frozen); `DecodeStep.breakdown: "StepBreakdown | None" = None`; `MockEngine` populates it scaled by `1/efficiency` so the four components sum to the step `seconds`; `metrics.MeasuredBreakdown(weight_s, kv_s, comms_s, compute_s)` (mean per token); `metrics.aggregate_breakdown(steps_breakdowns) -> MeasuredBreakdown | None`; `BenchResult.measured_breakdown: "MeasuredBreakdown | None" = None` (appended, defaulted); `build_result` gains keyword `step_breakdowns=None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_attribution.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.latency import decode_latency
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine, StepBreakdown, DecodeStep
from inferutil.bench.runner import run_benchmark
from inferutil.bench.metrics import aggregate_breakdown, MeasuredBreakdown

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def test_mock_step_breakdown_sums_to_step_seconds():
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    eng.reset(plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8, seq_len=640)
    step = eng.decode_step()
    b = step.breakdown
    assert isinstance(b, StepBreakdown)
    assert abs((b.weight_s + b.kv_s + b.comms_s + b.compute_s) - step.seconds) < 1e-12


def test_aggregate_breakdown_means():
    bs = [StepBreakdown(1.0, 2.0, 3.0, 4.0), StepBreakdown(3.0, 4.0, 5.0, 6.0)]
    agg = aggregate_breakdown(bs)
    assert isinstance(agg, MeasuredBreakdown)
    assert (agg.weight_s, agg.kv_s, agg.comms_s, agg.compute_s) == (2.0, 3.0, 4.0, 5.0)
    assert aggregate_breakdown([]) is None


def test_run_populates_measured_breakdown():
    cfg = BenchConfig(name="a", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=16)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    assert r.measured_breakdown is not None
    # weight term dominates at B=1 (matches the analytical thesis)
    mb = r.measured_breakdown
    assert mb.weight_s > mb.comms_s and mb.weight_s > mb.kv_s


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_attribution.py -v`
Expected: FAIL (`cannot import name 'StepBreakdown'`).

- [ ] **Step 3: Write minimal implementation**

In `src/inferutil/bench/engine.py`, add a frozen dataclass (after `ExpertRoute`) and a field on `DecodeStep`:

```python
@dataclass(frozen=True)
class StepBreakdown:
    weight_s: float
    kv_s: float
    comms_s: float
    compute_s: float
```

```python
@dataclass(frozen=True)
class DecodeStep:
    index: int
    seconds: float
    routes: tuple = ()
    breakdown: "StepBreakdown | None" = None
```

In `MockEngine.reset`, capture the component shares from the analytical breakdown (store the whole breakdown object). Replace the `reset` body's `decode_latency(...)` call so it keeps the breakdown:

```python
    def reset(self, *, plan, dtype_bytes, kv_dtype_bytes, tp, ep, seq_len):
        bd = decode_latency(
            self.cfg, self.cluster, plan=plan, dtype_bytes=dtype_bytes,
            kv_dtype_bytes=kv_dtype_bytes, seq_len=seq_len, tp=tp, ep=ep)
        self._floor_s = bd.total_s
        self._shares = (bd.weight_read_s, bd.kv_read_s, bd.comms_s, bd.compute_s)
        self._index = 0
        self._rng = random.Random(self.seed)
```

In `MockEngine.decode_step`, build a `StepBreakdown` that sums to the (jittered) step seconds by scaling the analytical shares to the actual step time. Replace the `return DecodeStep(...)` with:

```python
        s = self._step_seconds()
        w, k, c, comp = self._shares
        scale = (s / self._floor_s) if self._floor_s else 0.0  # preserves component ratios
        breakdown = StepBreakdown(weight_s=w * scale, kv_s=k * scale,
                                  comms_s=c * scale, compute_s=comp * scale)
        return DecodeStep(index=i, seconds=s, routes=routes, breakdown=breakdown)
```

(The `s = self._step_seconds()` line replaces using `_step_seconds()` inline; ensure `decode_step` calls it exactly once.)

In `src/inferutil/bench/metrics.py`, add `MeasuredBreakdown`, `aggregate_breakdown`, a `BenchResult` field, and wire `build_result`. Add the dataclass (after `TelemetrySummary`):

```python
@dataclass(frozen=True)
class MeasuredBreakdown:
    weight_s: float                # mean seconds/token
    kv_s: float
    comms_s: float
    compute_s: float
```

Append a field to `BenchResult` (after the variance fields, defaulted):

```python
    decode_tok_per_s_std: float = 0.0
    measured_breakdown: "MeasuredBreakdown | None" = None
```

Add the aggregator:

```python
def aggregate_breakdown(step_breakdowns) -> "MeasuredBreakdown | None":
    bs = [b for b in (step_breakdowns or []) if b is not None]
    if not bs:
        return None
    n = len(bs)
    return MeasuredBreakdown(
        weight_s=sum(b.weight_s for b in bs) / n,
        kv_s=sum(b.kv_s for b in bs) / n,
        comms_s=sum(b.comms_s for b in bs) / n,
        compute_s=sum(b.compute_s for b in bs) / n)
```

Extend `build_result` to accept and store it: add `step_breakdowns=None` to the signature, and in the constructor call add `measured_breakdown=aggregate_breakdown(step_breakdowns)`.

In `src/inferutil/bench/runner.py`, collect breakdowns from the last repeat's steps. Inside the repeat loop, when `last_rep`, capture the `DecodeStep` objects (not just `.seconds`). Replace the decode collection line in the loop with:

```python
        if last_rep:
            steps = [engine.decode_step() for _ in range(config.decode_tokens - 1)]
            step_seconds = [s.seconds for s in steps]
            last_breakdowns = [s.breakdown for s in steps]
        else:
            step_seconds = [engine.decode_step().seconds for _ in range(config.decode_tokens - 1)]
```

Initialize `last_breakdowns = []` before the loop, and pass `step_breakdowns=last_breakdowns` to `build_result`.

In `src/inferutil/bench/report.py`, after the bandwidth/roofline block in `format_result`, add a measured-breakdown block (only when present):

```python
    mb = r.measured_breakdown
    if mb is not None:
        ms = 1e3
        lines += [
            "  -- measured decode breakdown (mean ms/token) --",
            f"  weight {mb.weight_s*ms:.3f}  kv {mb.kv_s*ms:.3f}  "
            f"comms {mb.comms_s*ms:.3f}  compute {mb.compute_s*ms:.4f}",
        ]
```

(Insert this before the telemetry block. `lines` is the list already being built; keep ordering coherent.)

In `src/inferutil/bench/store.py`, round-trip the optional breakdown. In `_record_from_dict`, after popping `telemetry`, also rebuild `measured_breakdown` when present:

```python
    res = dict(d["result"])
    tele = dict(res.pop("telemetry"))
    tele["per_gpu_mean_util"] = tuple(tele["per_gpu_mean_util"])
    mb = res.pop("measured_breakdown", None)
    from .metrics import MeasuredBreakdown
    measured = MeasuredBreakdown(**mb) if mb else None
    result = BenchResult(telemetry=TelemetrySummary(**tele),
                         measured_breakdown=measured, **res)
```

(Importing `MeasuredBreakdown` at the top of `store.py` is also fine; use whichever keeps the module clean. `asdict` already serializes `measured_breakdown` to a dict or `None` on write — no write-side change needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bench_attribution.py -v` → PASS (3 passed).
Then: `python -m pytest tests/ -q` → all green (36 passed).

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/engine.py src/inferutil/bench/metrics.py src/inferutil/bench/runner.py src/inferutil/bench/report.py src/inferutil/bench/store.py tests/test_bench_attribution.py
git commit --no-verify -m "feat(bench): measured per-phase decode attribution"
```

---

### Task 11: Output-quality parity check

**Files:**
- Modify: `src/inferutil/bench/engine.py` (`DecodeStep.token_id`; `MockEngine` emits deterministic ids with a `quality` knob)
- Create: `src/inferutil/bench/quality.py` (`QualityResult`, `match_rate`)
- Modify: `src/inferutil/bench/metrics.py` (`BenchResult.quality` field)
- Modify: `src/inferutil/bench/runner.py` (collect ids; compute parity vs optional reference)
- Modify: `src/inferutil/bench/store.py` (round-trip optional quality)
- Test: `tests/test_bench_quality.py`

**Interfaces:**
- Produces: `DecodeStep.token_id: "int | None" = None`; `MockEngine(..., quality_offset=0)` — token id at step `i` is deterministic `(i*2654435761) % VOCAB` XOR/offset, where `quality_offset` perturbs ids so a "degraded" engine diverges from a reference; `quality.QualityResult(match_rate, n_compared, first_divergence)` (frozen); `quality.match_rate(candidate_ids, reference_ids) -> QualityResult`; `BenchResult.quality: "QualityResult | None" = None` (appended, defaulted); `run_benchmark(..., reference_ids=None)` — when given, computes parity of the last repeat's generated ids vs `reference_ids` and sets `BenchResult.quality`; `run_benchmark` is otherwise unchanged. Add a helper `generated_ids(engine, config)` is NOT needed — collect ids inside the runner loop.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_quality.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.quality import match_rate, QualityResult

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def test_match_rate_identical():
    q = match_rate([1, 2, 3, 4], [1, 2, 3, 4])
    assert isinstance(q, QualityResult)
    assert q.match_rate == 1.0 and q.first_divergence == -1 and q.n_compared == 4


def test_match_rate_partial():
    q = match_rate([1, 2, 9, 4], [1, 2, 3, 4])
    assert q.match_rate == 0.75 and q.first_divergence == 2


def test_mock_emits_token_ids():
    eng = MockEngine(QWEN3_235B, CLUSTER)
    eng.reset(plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8, seq_len=640)
    s = eng.decode_step()
    assert isinstance(s.token_id, int)


def test_run_parity_perfect_then_degraded():
    cfg = BenchConfig(name="q", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=64, decode_tokens=16)
    ref_eng = MockEngine(QWEN3_235B, CLUSTER, quality_offset=0)
    ref = run_benchmark(ref_eng, cfg, QWEN3_235B, CLUSTER, collect_ids=True)
    ref_ids = ref.generated_token_ids
    # identical engine -> perfect parity
    same = run_benchmark(MockEngine(QWEN3_235B, CLUSTER, quality_offset=0), cfg,
                         QWEN3_235B, CLUSTER, reference_ids=ref_ids)
    assert same.quality is not None and same.quality.match_rate == 1.0
    # degraded engine -> parity drops below 1.0
    worse = run_benchmark(MockEngine(QWEN3_235B, CLUSTER, quality_offset=7), cfg,
                          QWEN3_235B, CLUSTER, reference_ids=ref_ids)
    assert worse.quality.match_rate < 1.0


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_quality.py -v`
Expected: FAIL (`No module named 'inferutil.bench.quality'`).

- [ ] **Step 3: Write minimal implementation**

Create `src/inferutil/bench/quality.py`:

```python
"""Output-quality parity: compare a candidate token sequence to a reference.

Speed wins that silently change the model's output (aggressive quant, bad
speculative-decode verification) are regressions. This measures token-level
greedy parity; with the mock it is synthetic, but the moment a real engine
emits real token ids it becomes a genuine quality gate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityResult:
    match_rate: float              # fraction of compared positions that match
    n_compared: int
    first_divergence: int          # index of first mismatch, or -1 if none


def match_rate(candidate_ids, reference_ids) -> QualityResult:
    n = min(len(candidate_ids), len(reference_ids))
    if n == 0:
        return QualityResult(match_rate=1.0, n_compared=0, first_divergence=-1)
    matches = 0
    first_div = -1
    for i in range(n):
        if candidate_ids[i] == reference_ids[i]:
            matches += 1
        elif first_div == -1:
            first_div = i
    return QualityResult(match_rate=matches / n, n_compared=n, first_divergence=first_div)
```

In `src/inferutil/bench/engine.py`, add `token_id` to `DecodeStep` and a `quality_offset` knob to `MockEngine`:

```python
@dataclass(frozen=True)
class DecodeStep:
    index: int
    seconds: float
    routes: tuple = ()
    breakdown: "StepBreakdown | None" = None
    token_id: "int | None" = None
```

In `MockEngine.__init__`, add the parameter:

```python
    def __init__(self, cfg, cluster, *, efficiency=1.0, jitter=0.0, seed=0,
                 expose_routes=False, quality_offset=0):
        ...
        self.quality_offset = quality_offset
```

In `MockEngine.decode_step`, compute a deterministic token id and pass it through:

```python
        token_id = ((i * 2654435761) + self.quality_offset * (i + 1)) % self.cfg.vocab
        return DecodeStep(index=i, seconds=s, routes=routes, breakdown=breakdown,
                          token_id=token_id)
```

(`quality_offset=0` gives a fixed reference sequence; a nonzero offset perturbs ids per position so parity drops below 1.0.)

In `src/inferutil/bench/metrics.py`, append the field to `BenchResult` (after `measured_breakdown`, defaulted):

```python
    measured_breakdown: "MeasuredBreakdown | None" = None
    quality: "object | None" = None    # QualityResult | None (avoids import cycle)
```

(`build_result` gains keyword `quality=None` and passes it straight through to the constructor.)

In `src/inferutil/bench/runner.py`, collect ids on the last repeat and optionally compute parity. Add params to the signature: `run_benchmark(engine, config, cfg, cluster, telemetry=None, reference_ids=None, collect_ids=False)`. In the last-repeat branch, capture token ids alongside breakdowns:

```python
        if last_rep:
            steps = [engine.decode_step() for _ in range(config.decode_tokens - 1)]
            step_seconds = [s.seconds for s in steps]
            last_breakdowns = [s.breakdown for s in steps]
            last_ids = [s.token_id for s in steps]
```

(Initialize `last_ids = []` before the loop.) After the loop, compute quality and attach ids:

```python
    quality = None
    if reference_ids is not None:
        from .quality import match_rate
        quality = match_rate(last_ids, reference_ids)
    result = build_result(cfg=cfg, cluster=cluster, config=config, ttft_s=ttft_s,
                          prefill_tok_per_s=prefill_tok_per_s,
                          decode_step_seconds=step_seconds, telemetry_summary=summary,
                          decode_tok_per_s_samples=samples_tok_s,
                          step_breakdowns=last_breakdowns, quality=quality)
    if collect_ids:
        # attach for callers that need a reference sequence (not part of the stored schema)
        object.__setattr__(result, "generated_token_ids", last_ids)
    return result
```

(The `object.__setattr__` is needed because `BenchResult` is frozen; this attaches a transient attribute used only to seed a reference run and is intentionally NOT serialized.)

In `src/inferutil/bench/store.py`, round-trip the optional quality in `_record_from_dict` (alongside `measured_breakdown`):

```python
    qd = res.pop("quality", None)
    from .quality import QualityResult
    quality = QualityResult(**qd) if qd else None
    result = BenchResult(telemetry=TelemetrySummary(**tele),
                         measured_breakdown=measured, quality=quality, **res)
```

Note `asdict` on write turns `quality` into a dict (or leaves `None`). If `quality` was attached as a transient via `object.__setattr__`, it is a dataclass field set to a value, so `asdict` includes it — fine. The transient `generated_token_ids` is NOT a dataclass field, so `asdict` ignores it (not serialized), which is intended.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bench_quality.py -v` → PASS (4 passed).
Then: `python -m pytest tests/ -q` → all green (40 passed).

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/engine.py src/inferutil/bench/quality.py src/inferutil/bench/metrics.py src/inferutil/bench/runner.py src/inferutil/bench/store.py tests/test_bench_quality.py
git commit --no-verify -m "feat(bench): output-quality parity check (token-level)"
```

---

### Task 12: Regression gate (pass/fail thresholds) + CLI

**Files:**
- Create: `src/inferutil/bench/gate.py` (`Thresholds`, `GateResult`, `evaluate`)
- Modify: `src/inferutil/bench/cli.py` (`gate` subcommand, non-zero exit on fail)
- Modify: `src/inferutil/bench/__init__.py` (re-export new public names)
- Test: `tests/test_bench_gate.py`

**Interfaces:**
- Consumes: `BenchResult` fields `decode_tok_per_s`, `ttft_s`, `pct_of_floor`, and `quality.match_rate` (Task 11); `store.load_latest`/`load_run`.
- Produces: `gate.Thresholds(min_decode_tok_per_s=None, max_ttft_s=None, min_pct_of_floor=None, min_quality_match=None)` (frozen, all optional); `gate.GateResult(passed: bool, failures: tuple)` (frozen); `gate.evaluate(result, thresholds) -> GateResult`; CLI `gate` subcommand that loads a run, evaluates thresholds from flags, prints PASS/FAIL with reasons, and exits non-zero on failure (machine-consumable by an agent/CI).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_gate.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.metrics import BenchResult, TelemetrySummary
from inferutil.bench.quality import QualityResult
from inferutil.bench.gate import Thresholds, GateResult, evaluate

TELE = TelemetrySummary(False, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ())


def _result(tok_s=120.0, ttft=0.04, pct=0.6, qmatch=1.0):
    return BenchResult(ttft, 9000.0, tok_s, 0.008, 0.009, 1.0, 128,
                       45_000_000_000, 5.3e12, 0.31, 540.0, pct, TELE,
                       quality=QualityResult(qmatch, 128, -1))


def test_pass_when_all_met():
    g = evaluate(_result(), Thresholds(min_decode_tok_per_s=100.0, max_ttft_s=0.05,
                                       min_pct_of_floor=0.5, min_quality_match=0.99))
    assert isinstance(g, GateResult) and g.passed and g.failures == ()


def test_fail_lists_each_violation():
    g = evaluate(_result(tok_s=80.0, qmatch=0.9),
                 Thresholds(min_decode_tok_per_s=100.0, min_quality_match=0.99))
    assert g.passed is False
    joined = " ".join(g.failures)
    assert "decode" in joined and "quality" in joined


def test_no_thresholds_passes():
    assert evaluate(_result(), Thresholds()).passed is True


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_gate.py -v`
Expected: FAIL (`No module named 'inferutil.bench.gate'`).

- [ ] **Step 3: Write minimal implementation**

Create `src/inferutil/bench/gate.py`:

```python
"""Pass/fail regression gate: the machine-readable accept/reject an autonomous
optimization loop branches on. Every threshold is optional; only the ones set
are enforced."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    min_decode_tok_per_s: "float | None" = None
    max_ttft_s: "float | None" = None
    min_pct_of_floor: "float | None" = None
    min_quality_match: "float | None" = None


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple                # tuple[str, ...]


def evaluate(result, thresholds: Thresholds) -> GateResult:
    fails = []
    t = thresholds
    if t.min_decode_tok_per_s is not None and result.decode_tok_per_s < t.min_decode_tok_per_s:
        fails.append(f"decode {result.decode_tok_per_s:.1f} < min {t.min_decode_tok_per_s:.1f} tok/s")
    if t.max_ttft_s is not None and result.ttft_s > t.max_ttft_s:
        fails.append(f"ttft {result.ttft_s*1e3:.1f} > max {t.max_ttft_s*1e3:.1f} ms")
    if t.min_pct_of_floor is not None and result.pct_of_floor < t.min_pct_of_floor:
        fails.append(f"pct_of_floor {result.pct_of_floor*100:.1f} < min {t.min_pct_of_floor*100:.1f} %")
    if t.min_quality_match is not None:
        q = result.quality.match_rate if result.quality is not None else 0.0
        if q < t.min_quality_match:
            fails.append(f"quality {q:.3f} < min {t.min_quality_match:.3f}")
    return GateResult(passed=(not fails), failures=tuple(fails))
```

In `src/inferutil/bench/cli.py`, add a `gate` subcommand. Add the import at the top:

```python
from .gate import Thresholds, evaluate
```

Add a handler:

```python
def _cmd_gate(args) -> None:
    rec = _resolve(args)
    th = Thresholds(min_decode_tok_per_s=args.min_tok_s, max_ttft_s=args.max_ttft_s,
                    min_pct_of_floor=args.min_pct_floor, min_quality_match=args.min_quality)
    g = evaluate(rec.result, th)
    if g.passed:
        print(f"GATE PASS  [{rec.runid}] {rec.config.name}")
    else:
        print(f"GATE FAIL  [{rec.runid}] {rec.config.name}")
        for f in g.failures:
            print(f"  - {f}")
        raise SystemExit(1)
```

Register the subparser in `main` (alongside `report`/`compare`). `gate` reuses the `report`-style run selection (`--name`, optional `runid` defaulting to `latest`):

```python
    gp = sub.add_parser("gate", help="pass/fail a stored run against thresholds")
    gp.add_argument("--name", default="default")
    gp.add_argument("runid", nargs="?", default="latest")
    gp.add_argument("--min-tok-s", type=float, default=None)
    gp.add_argument("--max-ttft-s", type=float, default=None)
    gp.add_argument("--min-pct-floor", type=float, default=None)
    gp.add_argument("--min-quality", type=float, default=None)
    gp.set_defaults(func=_cmd_gate)
```

(`_resolve` already handles `runid in (None, "latest")` and exits cleanly when the run is missing — reuse it unchanged.)

In `src/inferutil/bench/__init__.py`, add the new public names to the imports and `__all__`:

```python
from .quality import QualityResult, match_rate
from .gate import Thresholds, GateResult, evaluate
from .metrics import MeasuredBreakdown, aggregate_breakdown   # add alongside existing metrics imports
from .engine import StepBreakdown                              # add alongside existing engine imports
```

and extend `__all__` with: `"QualityResult", "match_rate", "Thresholds", "GateResult", "evaluate", "MeasuredBreakdown", "aggregate_breakdown", "StepBreakdown"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bench_gate.py -v` → PASS (3 passed).
Then run a manual CLI gate to confirm the exit code wiring (PowerShell: `$env:PYTHONPATH='src'`):
`python -c "import tempfile; from inferutil.bench.cli import main; d=tempfile.mkdtemp(); main(['--results-dir',d,'run','--name','g','--decode','8','--prompt','32']); main(['--results-dir',d,'gate','--name','g','--min-tok-s','1'])"`
Expected: a `GATE PASS` line (min 1 tok/s is trivially met by the mock). Then the FULL suite: `python -m pytest tests/ -q` → all green (43 passed).

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/gate.py src/inferutil/bench/cli.py src/inferutil/bench/__init__.py tests/test_bench_gate.py
git commit --no-verify -m "feat(bench): regression gate (thresholds) + gate CLI"
```

---

## Self-Review notes (for the executor)

- **Append-only discipline:** every new `BenchResult` field (`n_repeats`, `decode_tok_per_s_std`, `measured_breakdown`, `quality`) is added AFTER `telemetry` WITH a default, so the Task 1–8 positional constructions in `test_bench_metrics.py`/`test_bench_report.py`/`test_bench_store.py` keep working untouched. If any of those tests break, a field was inserted in the wrong place — fix the ordering, not the old tests.
- **Store round-trip:** Tasks 10 and 11 each add a nested optional dataclass to the JSON; both update `_record_from_dict` to rebuild it (`MeasuredBreakdown`, `QualityResult`). `asdict` handles the write side automatically. After Task 11, re-run `test_bench_store.py` to confirm the round-trip still holds with the new fields defaulting to `None`.
- **MockEngine invariant preserved:** the floor identity (`efficiency=1.0, jitter=0.0` → step seconds == `decode_latency().total_s`) is unchanged; breakdown components are scaled to sum to the step seconds, and token ids/quality are orthogonal to timing.
- **Runner growth:** the repeat loop (Task 9) is the structural change; Tasks 10–11 only add per-step capture on the last repeat. Verify the `decode_tokens < 2` guard and single `telemetry.start()/stop()` bracket survive each edit.
- **Agent-loop readiness after this addendum:** variance+significance (don't lock in noise), measured attribution (know where time went), quality parity (don't trade correctness for speed), and a non-zero-exit gate (branchable accept/reject). The remaining gap is the real `ConiferEngine` — still gated on conifer arriving; everything here is correct the moment it does.
