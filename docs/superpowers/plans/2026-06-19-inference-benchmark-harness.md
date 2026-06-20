# Inference Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `inferutil.bench` — an offline B=1 decode benchmark harness that drives an in-process engine behind a seam, captures latency + derived bandwidth/efficiency + NVML telemetry, stores JSON results, and reports measured-vs-roofline and run-to-run diffs.

**Architecture:** A new subpackage `src/inferutil/bench/`. Everything depends on a tiny `Engine` protocol (`MockEngine` now, `ConiferEngine` later). The existing `inferutil.{model,hardware,latency}` modules are reused as the roofline oracle. A `MockEngine` seeded from `latency.decode_latency` makes the whole pipeline runnable and testable with no GPU.

**Tech Stack:** Python ≥3.10, **standard library only** (no torch/numpy). `pynvml` is an *optional* runtime import inside `NvmlTelemetry` (guarded; absence is non-fatal). Tests use plain `assert` functions, pytest-compatible and standalone-runnable, mirroring `tests/test_model.py`.

## Global Constraints

- **Python ≥3.10**, stdlib only in the import path; `pynvml` imported lazily inside `NvmlTelemetry.__init__` under `try/except`, never at module top level.
- **SI units internally**: time in **seconds**, bandwidth in **bytes/sec**, energy in **joules**. Only `report.py` converts to ms for display (matching `cli.py`'s existing style).
- **All dataclasses `frozen=True`.**
- **No internal wall-clock**: the harness never calls `datetime`/`time.time` for run identity; `runid` is supplied by the caller (`cli.py` stamps it). `NvmlTelemetry` may use `time.monotonic` for *sample offsets only*.
- **Test layout**: tests live in top-level `tests/`, named `test_bench_<area>.py`, each starting with the `sys.path.insert(0, .../src)` shim from `tests/test_model.py`, runnable via `python -m pytest tests/` or directly.
- **Model/hardware facts** come only from `inferutil.model` / `inferutil.hardware` — never hard-coded.
- Commit after every task with `git commit --no-verify` (repo has LF/CRLF warnings; not errors).

---

### Task 1: Package scaffold + `BenchConfig`

**Files:**
- Create: `src/inferutil/bench/__init__.py`
- Create: `src/inferutil/bench/config.py`
- Test: `tests/test_bench_config.py`

**Interfaces:**
- Produces: `BenchConfig` (frozen dataclass, fields per Appendix A + `seq_len` property + `kv_dtype_bytes`), and `config_id(config: BenchConfig) -> str` (stable 12-hex-char hash of all fields).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_config.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.config import BenchConfig, config_id


def test_seq_len_is_prompt_plus_decode():
    c = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                    tp=2, ep=8, prompt_tokens=512, decode_tokens=128)
    assert c.seq_len == 640


def test_config_id_is_stable_and_field_sensitive():
    a = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8)
    b = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8)
    c = BenchConfig(name="x", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2, tp=2, ep=8)
    assert config_id(a) == config_id(b)
    assert config_id(a) != config_id(c)
    assert len(config_id(a)) == 12


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inferutil.bench'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inferutil/bench/__init__.py
"""inferutil.bench — offline B=1 decode benchmark harness."""
```

```python
# src/inferutil/bench/config.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass, astuple


@dataclass(frozen=True)
class BenchConfig:
    name: str
    plan: str                      # "tp" | "ep" | "hybrid"
    dtype_bytes: int               # 2=bf16, 1=fp8
    kv_dtype_bytes: int
    tp: int
    ep: int
    prompt_tokens: int = 512       # fixed window (playbook §G)
    decode_tokens: int = 128
    seed: int = 0
    warmup_steps: int = 8

    @property
    def seq_len(self) -> int:
        """Representative decode context: prompt + generated tokens."""
        return self.prompt_tokens + self.decode_tokens


def config_id(config: BenchConfig) -> str:
    """Stable 12-hex-char id from all config fields → results lineage key."""
    raw = "|".join(str(x) for x in astuple(config))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/__init__.py src/inferutil/bench/config.py tests/test_bench_config.py
git commit --no-verify -m "feat(bench): BenchConfig + stable config_id"
```

---

### Task 2: Engine seam + `MockEngine`

**Files:**
- Create: `src/inferutil/bench/engine.py`
- Test: `tests/test_bench_engine.py`

**Interfaces:**
- Consumes: `BenchConfig` (Task 1); `inferutil.model.MoEConfig`, `inferutil.hardware.Cluster`, `inferutil.latency.decode_latency`.
- Produces: `ExpertRoute`, `PrefillResult`, `DecodeStep` (frozen dataclasses per Appendix A); `Engine` Protocol with `reset(*, plan, dtype_bytes, kv_dtype_bytes, tp, ep, seq_len)`, `prefill(token_ids) -> PrefillResult`, `decode_step() -> DecodeStep`; `MockEngine(cfg, cluster, *, efficiency=1.0, jitter=0.0, seed=0, expose_routes=False)` whose per-decode-step `seconds` equals `decode_latency(...).total_s / efficiency` (so `efficiency=1.0, jitter=0.0` reproduces the analytical floor exactly).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_engine.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.latency import decode_latency
from inferutil.bench.engine import MockEngine, DecodeStep, PrefillResult

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def _eng(**kw):
    e = MockEngine(QWEN3_235B, CLUSTER, **kw)
    e.reset(plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8, seq_len=640)
    return e


def test_perfect_mock_step_equals_floor():
    floor = decode_latency(QWEN3_235B, CLUSTER, plan="hybrid", dtype_bytes=2,
                           kv_dtype_bytes=2, seq_len=640, tp=2, ep=8).total_s
    step = _eng(efficiency=1.0, jitter=0.0).decode_step()
    assert isinstance(step, DecodeStep)
    assert abs(step.seconds - floor) < 1e-12


def test_efficiency_scales_step_time():
    floor = decode_latency(QWEN3_235B, CLUSTER, plan="hybrid", dtype_bytes=2,
                           kv_dtype_bytes=2, seq_len=640, tp=2, ep=8).total_s
    step = _eng(efficiency=0.5, jitter=0.0).decode_step()
    assert abs(step.seconds - floor / 0.5) < 1e-12


def test_prefill_shape_and_indices():
    e = _eng()
    pre = e.prefill(list(range(512)))
    assert isinstance(pre, PrefillResult) and pre.n_prompt_tokens == 512
    assert e.decode_step().index == 0 and e.decode_step().index == 1


def test_routes_optional():
    assert _eng(expose_routes=False).decode_step().routes == ()
    assert len(_eng(expose_routes=True).decode_step().routes) == QWEN3_235B.top_k


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inferutil.bench.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inferutil/bench/engine.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..model import MoEConfig
from ..hardware import Cluster
from ..latency import decode_latency


@dataclass(frozen=True)
class ExpertRoute:
    layer: int
    expert_id: int
    gpu: int


@dataclass(frozen=True)
class PrefillResult:
    n_prompt_tokens: int
    seconds: float                 # prompt-processing wall-time
    first_token_seconds: float     # time to emit the first decoded token


@dataclass(frozen=True)
class DecodeStep:
    index: int
    seconds: float
    routes: tuple = ()             # tuple[ExpertRoute, ...]; empty if not exposed


@runtime_checkable
class Engine(Protocol):
    def reset(self, *, plan: str, dtype_bytes: int, kv_dtype_bytes: int,
              tp: int, ep: int, seq_len: int) -> None: ...
    def prefill(self, token_ids: list) -> PrefillResult: ...
    def decode_step(self) -> DecodeStep: ...


class MockEngine:
    """Engine whose timings are seeded from the analytical roofline.

    With efficiency=1.0 and jitter=0.0 each decode step takes exactly
    decode_latency(...).total_s, so an end-to-end run reports 100% of floor —
    the wiring sanity check. efficiency<1 models real kernels leaving BW on the
    table; jitter adds seeded per-step noise for realistic p50/p95.
    """

    def __init__(self, cfg: MoEConfig, cluster: Cluster, *, efficiency: float = 1.0,
                 jitter: float = 0.0, seed: int = 0, expose_routes: bool = False):
        self.cfg = cfg
        self.cluster = cluster
        self.efficiency = efficiency
        self.jitter = jitter
        self.seed = seed
        self.expose_routes = expose_routes
        self._floor_s = 0.0
        self._index = 0
        self._rng = random.Random(seed)

    def reset(self, *, plan, dtype_bytes, kv_dtype_bytes, tp, ep, seq_len):
        self._floor_s = decode_latency(
            self.cfg, self.cluster, plan=plan, dtype_bytes=dtype_bytes,
            kv_dtype_bytes=kv_dtype_bytes, seq_len=seq_len, tp=tp, ep=ep).total_s
        self._index = 0
        self._rng = random.Random(self.seed)

    def _step_seconds(self) -> float:
        s = self._floor_s / self.efficiency
        if self.jitter:
            s *= 1.0 + self._rng.uniform(-self.jitter, self.jitter)
        return s

    def prefill(self, token_ids) -> PrefillResult:
        n = len(token_ids)
        # Crude prefill model (compute-bound; refined when ConiferEngine lands):
        # ~one floor-step of weight reads per prompt token.
        return PrefillResult(n_prompt_tokens=n, seconds=self._floor_s * n,
                             first_token_seconds=self._step_seconds())

    def decode_step(self) -> DecodeStep:
        i = self._index
        self._index += 1
        routes = ()
        if self.expose_routes:
            routes = tuple(
                ExpertRoute(
                    layer=i % self.cfg.n_layers,
                    expert_id=(i * 7 + k * 13) % self.cfg.n_experts,
                    gpu=((i * 7 + k * 13) % self.cfg.n_experts) % self.cluster.n_gpus,
                )
                for k in range(self.cfg.top_k)
            )
        return DecodeStep(index=i, seconds=self._step_seconds(), routes=routes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_engine.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/engine.py tests/test_bench_engine.py
git commit --no-verify -m "feat(bench): Engine seam + roofline-seeded MockEngine"
```

---

### Task 3: Telemetry sources + `GpuSample`

**Files:**
- Create: `src/inferutil/bench/telemetry.py`
- Test: `tests/test_bench_telemetry.py`

**Interfaces:**
- Produces: `GpuSample` (frozen dataclass per Appendix A); `TelemetrySource` Protocol (`start() -> None`, `stop() -> list[GpuSample]`, `available: bool`); `NullTelemetry` (`available=False`, `stop()->[]`); `FakeTelemetrySource(samples)` (`available=True`, returns canned samples); `NvmlTelemetry(interval_s=0.05)` (guarded `pynvml` import; background sampler thread).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_telemetry.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.telemetry import (
    GpuSample, NullTelemetry, FakeTelemetrySource, NvmlTelemetry,
)


def _s(gpu, t, sm, power, temp=60.0):
    return GpuSample(gpu_index=gpu, t_seconds=t, temp_c=temp, sm_util_pct=sm,
                     mem_util_pct=sm, power_w=power, sm_clock_mhz=1500.0,
                     mem_clock_mhz=2600.0, mem_used_bytes=40_000_000_000)


def test_null_source_unavailable_and_empty():
    n = NullTelemetry()
    n.start()
    assert n.available is False and n.stop() == []


def test_fake_source_returns_canned():
    samples = [_s(0, 0.0, 80.0, 600.0), _s(1, 0.0, 40.0, 500.0)]
    f = FakeTelemetrySource(samples)
    f.start()
    assert f.available is True and f.stop() == samples


def test_nvml_available_is_bool_without_gpu():
    # No GPU / no pynvml on the dev box: must not raise; available is False-ish bool.
    t = NvmlTelemetry()
    assert isinstance(t.available, bool)
    t.start()
    assert isinstance(t.stop(), list)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inferutil.bench.telemetry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inferutil/bench/telemetry.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GpuSample:
    gpu_index: int
    t_seconds: float
    temp_c: float
    sm_util_pct: float
    mem_util_pct: float
    power_w: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    mem_used_bytes: int


class TelemetrySource(Protocol):
    available: bool
    def start(self) -> None: ...
    def stop(self) -> list: ...     # list[GpuSample]


class NullTelemetry:
    """Used when no NVML is present (e.g. laptop). Reports nothing."""
    available = False
    def start(self) -> None: pass
    def stop(self) -> list: return []


class FakeTelemetrySource:
    """Canned samples for tests."""
    def __init__(self, samples):
        self._samples = list(samples)
        self.available = True
    def start(self) -> None: pass
    def stop(self) -> list: return self._samples


class NvmlTelemetry:
    """Background NVML sampler. pynvml is imported lazily; if it (or the driver)
    is unavailable, `available` is False and start/stop are no-ops."""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            self._p = pynvml
            self._n = pynvml.nvmlDeviceGetCount()
            self.available = True
        except Exception:
            self._p = None
            self._n = 0
            self.available = False

    def start(self) -> None:
        if not self.available:
            return
        self._stop.clear()
        self._samples = []
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        p = self._p
        handles = [p.nvmlDeviceGetHandleByIndex(i) for i in range(self._n)]
        while not self._stop.is_set():
            t = time.monotonic() - self._t0
            for i, h in enumerate(handles):
                u = p.nvmlDeviceGetUtilizationRates(h)
                self._samples.append(GpuSample(
                    gpu_index=i, t_seconds=t,
                    temp_c=float(p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU)),
                    sm_util_pct=float(u.gpu), mem_util_pct=float(u.memory),
                    power_w=p.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    sm_clock_mhz=float(p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_SM)),
                    mem_clock_mhz=float(p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_MEM)),
                    mem_used_bytes=int(p.nvmlDeviceGetMemoryInfo(h).used)))
            self._stop.wait(self.interval_s)

    def stop(self) -> list:
        if not self.available:
            return []
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return list(self._samples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_telemetry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/telemetry.py tests/test_bench_telemetry.py
git commit --no-verify -m "feat(bench): pluggable telemetry sources (Null/Fake/NVML)"
```

---

### Task 4: Metrics — derivation + `BenchResult`

**Files:**
- Create: `src/inferutil/bench/metrics.py`
- Test: `tests/test_bench_metrics.py`

**Interfaces:**
- Consumes: `BenchConfig` (Task 1); `GpuSample` (Task 3); `inferutil.model.MoEConfig`, `inferutil.hardware.Cluster`, `inferutil.latency.decode_latency`.
- Produces:
  - `bytes_per_token(cfg, seq_len, dtype_bytes, kv_dtype_bytes) -> int`
  - `percentile(sorted_vals: list[float], p: float) -> float`
  - `summarize_telemetry(samples, n_decode_tokens, decode_window_s) -> TelemetrySummary`
  - `build_result(*, cfg, cluster, config, ttft_s, prefill_tok_per_s, decode_step_seconds, telemetry_summary) -> BenchResult`
  - `TelemetrySummary`, `BenchResult` (frozen dataclasses per Appendix A).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_metrics.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.latency import decode_latency
from inferutil.bench.config import BenchConfig
from inferutil.bench.telemetry import GpuSample
from inferutil.bench.metrics import (
    bytes_per_token, percentile, summarize_telemetry, build_result,
)

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)
CFG = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                  tp=2, ep=8, prompt_tokens=512, decode_tokens=128)


def test_bytes_per_token_matches_model_accounting():
    expected = QWEN3_235B.active_params * 2 + 640 * QWEN3_235B.kv_bytes_per_token(2)
    assert bytes_per_token(QWEN3_235B, 640, 2, 2) == expected


def test_percentile_interpolates():
    assert percentile([0.0, 1.0, 2.0, 3.0], 0.5) == 1.5


def test_perfect_run_is_100pct_of_floor():
    floor_s = decode_latency(QWEN3_235B, CLUSTER, plan="hybrid", dtype_bytes=2,
                             kv_dtype_bytes=2, seq_len=640, tp=2, ep=8).total_s
    steps = [floor_s] * (CFG.decode_tokens - 1)
    r = build_result(cfg=QWEN3_235B, cluster=CLUSTER, config=CFG, ttft_s=0.05,
                     prefill_tok_per_s=10000.0, decode_step_seconds=steps,
                     telemetry_summary=summarize_telemetry([], CFG.decode_tokens, 0.0))
    assert abs(r.pct_of_floor - 1.0) < 1e-9
    assert r.bytes_per_token == bytes_per_token(QWEN3_235B, 640, 2, 2)
    assert 0.0 < r.pct_of_peak_bw < 1.0


def test_summarize_telemetry_aggregates_and_imbalance():
    # gpu0 busy (80%), gpu1 idle (40%); two timestamps each.
    def s(g, t, sm, pw):
        return GpuSample(g, t, 65.0, sm, sm, pw, 1500.0, 2600.0, 40_000_000_000)
    samples = [s(0, 0.0, 80.0, 600.0), s(1, 0.0, 40.0, 500.0),
               s(0, 0.1, 80.0, 600.0), s(1, 0.1, 40.0, 500.0)]
    summ = summarize_telemetry(samples, n_decode_tokens=10, decode_window_s=0.1)
    assert summ.available and summ.n_gpus == 2
    assert summ.per_gpu_mean_util == (80.0, 40.0)
    assert abs(summ.util_imbalance - (80.0 / 60.0)) < 1e-9   # max / mean
    # total instantaneous power = 1100 W; energy/token = 1100*0.1/10 = 11 J
    assert abs(summ.energy_j_per_token - 11.0) < 1e-9


def test_summarize_empty_is_unavailable():
    summ = summarize_telemetry([], 10, 0.0)
    assert summ.available is False and summ.per_gpu_mean_util == ()


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inferutil.bench.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inferutil/bench/metrics.py
from __future__ import annotations

from dataclasses import dataclass

from ..model import MoEConfig
from ..hardware import Cluster
from ..latency import decode_latency
from .config import BenchConfig


@dataclass(frozen=True)
class TelemetrySummary:
    available: bool
    n_gpus: int
    temp_c_max: float
    sm_util_pct_mean: float
    mem_util_pct_mean: float
    power_w_mean: float            # mean TOTAL power across all GPUs (W)
    energy_j_per_token: float
    util_imbalance: float          # busiest-GPU mean util / mean-across-GPUs
    per_gpu_mean_util: tuple       # tuple[float, ...]


@dataclass(frozen=True)
class BenchResult:
    # latency
    ttft_s: float
    prefill_tok_per_s: float
    decode_tok_per_s: float
    tpot_p50_s: float
    tpot_p95_s: float
    total_s: float
    n_decode_tokens: int
    # bandwidth / efficiency (derived)
    bytes_per_token: int
    achieved_hbm_bw: float
    pct_of_peak_bw: float
    analytical_floor_tok_per_s: float
    pct_of_floor: float
    # device
    telemetry: TelemetrySummary


def bytes_per_token(cfg: MoEConfig, seq_len: int, dtype_bytes: int,
                    kv_dtype_bytes: int) -> int:
    """Active weight bytes + whole-KV read for this context (per latency.py terms)."""
    return cfg.active_params * dtype_bytes + seq_len * cfg.kv_bytes_per_token(kv_dtype_bytes)


def percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize_telemetry(samples, n_decode_tokens: int,
                        decode_window_s: float) -> TelemetrySummary:
    if not samples:
        return TelemetrySummary(False, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ())
    gpu_ids = sorted({s.gpu_index for s in samples})
    by_gpu = {g: [s for s in samples if s.gpu_index == g] for g in gpu_ids}
    per_gpu_mean_util = tuple(
        sum(s.sm_util_pct for s in by_gpu[g]) / len(by_gpu[g]) for g in gpu_ids)
    mean_util = sum(per_gpu_mean_util) / len(per_gpu_mean_util)
    imbalance = (max(per_gpu_mean_util) / mean_util) if mean_util else 0.0
    samples_per_gpu = len(samples) / len(gpu_ids)   # assumes equal sampling cadence
    power_total_mean = sum(s.power_w for s in samples) / samples_per_gpu
    energy = (power_total_mean * decode_window_s / n_decode_tokens) if n_decode_tokens else 0.0
    return TelemetrySummary(
        available=True, n_gpus=len(gpu_ids),
        temp_c_max=max(s.temp_c for s in samples),
        sm_util_pct_mean=sum(s.sm_util_pct for s in samples) / len(samples),
        mem_util_pct_mean=sum(s.mem_util_pct for s in samples) / len(samples),
        power_w_mean=power_total_mean, energy_j_per_token=energy,
        util_imbalance=imbalance, per_gpu_mean_util=per_gpu_mean_util)


def build_result(*, cfg: MoEConfig, cluster: Cluster, config: BenchConfig,
                 ttft_s: float, prefill_tok_per_s: float, decode_step_seconds: list,
                 telemetry_summary: TelemetrySummary) -> BenchResult:
    steps_sorted = sorted(decode_step_seconds)
    n = len(decode_step_seconds)
    total_decode = sum(decode_step_seconds)
    tpot_mean = (total_decode / n) if n else float("inf")
    decode_tok_per_s = (1.0 / tpot_mean) if tpot_mean else float("inf")
    bpt = bytes_per_token(cfg, config.seq_len, config.dtype_bytes, config.kv_dtype_bytes)
    achieved = (bpt / tpot_mean) if tpot_mean else 0.0
    floor_tok_s = decode_latency(
        cfg, cluster, plan=config.plan, dtype_bytes=config.dtype_bytes,
        kv_dtype_bytes=config.kv_dtype_bytes, seq_len=config.seq_len,
        tp=config.tp, ep=config.ep).tokens_per_s
    return BenchResult(
        ttft_s=ttft_s, prefill_tok_per_s=prefill_tok_per_s,
        decode_tok_per_s=decode_tok_per_s,
        tpot_p50_s=percentile(steps_sorted, 0.5),
        tpot_p95_s=percentile(steps_sorted, 0.95),
        total_s=ttft_s + total_decode, n_decode_tokens=config.decode_tokens,
        bytes_per_token=bpt, achieved_hbm_bw=achieved,
        pct_of_peak_bw=achieved / cluster.aggregate_hbm_bw,
        analytical_floor_tok_per_s=floor_tok_s,
        pct_of_floor=(decode_tok_per_s / floor_tok_s) if floor_tok_s else 0.0,
        telemetry=telemetry_summary)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_metrics.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/metrics.py tests/test_bench_metrics.py
git commit --no-verify -m "feat(bench): metrics — derived bandwidth/efficiency + telemetry summary"
```

---

### Task 5: Runner — fixed-window benchmark loop

**Files:**
- Create: `src/inferutil/bench/runner.py`
- Test: `tests/test_bench_runner.py`

**Interfaces:**
- Consumes: `Engine` (Task 2), `BenchConfig` (Task 1), `metrics.build_result`/`summarize_telemetry` (Task 4), `telemetry.NullTelemetry`/`TelemetrySource` (Task 3).
- Produces: `run_benchmark(engine, config, cfg, cluster, telemetry=None) -> BenchResult`. Sequence: `engine.reset(...)` → `prefill(range(prompt_tokens))` → discard `warmup_steps` decode steps → `telemetry.start()` → collect `decode_tokens - 1` decode-step seconds → `telemetry.stop()` → `build_result(...)`. Requires `decode_tokens >= 2`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_runner.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.telemetry import FakeTelemetrySource, GpuSample

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)
CFG = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                  tp=2, ep=8, prompt_tokens=512, decode_tokens=128, warmup_steps=8)


def test_perfect_mock_run_hits_floor():
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, CFG, QWEN3_235B, CLUSTER)
    assert abs(r.pct_of_floor - 1.0) < 1e-9
    assert r.n_decode_tokens == 128
    assert r.ttft_s > 0.0


def test_runner_uses_telemetry_source():
    eng = MockEngine(QWEN3_235B, CLUSTER)
    fake = FakeTelemetrySource([
        GpuSample(0, 0.0, 70.0, 90.0, 90.0, 600.0, 1500.0, 2600.0, 4 * 10**10),
        GpuSample(1, 0.0, 60.0, 30.0, 30.0, 400.0, 1500.0, 2600.0, 4 * 10**10),
    ])
    r = run_benchmark(eng, CFG, QWEN3_235B, CLUSTER, telemetry=fake)
    assert r.telemetry.available and r.telemetry.n_gpus == 2


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inferutil.bench.runner'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inferutil/bench/runner.py
from __future__ import annotations

from ..model import MoEConfig
from ..hardware import Cluster
from .config import BenchConfig
from .metrics import build_result, summarize_telemetry, BenchResult
from .telemetry import NullTelemetry


def run_benchmark(engine, config: BenchConfig, cfg: MoEConfig, cluster: Cluster,
                  telemetry=None) -> BenchResult:
    """Drive one fixed-window B=1 benchmark and return a BenchResult.

    Telemetry brackets ONLY the timed decode window (warmup discarded first) so
    sampling never perturbs the latency path.
    """
    if config.decode_tokens < 2:
        raise ValueError("decode_tokens must be >= 2 (need inter-token samples)")
    telemetry = telemetry or NullTelemetry()

    engine.reset(plan=config.plan, dtype_bytes=config.dtype_bytes,
                 kv_dtype_bytes=config.kv_dtype_bytes, tp=config.tp, ep=config.ep,
                 seq_len=config.seq_len)

    pre = engine.prefill(list(range(config.prompt_tokens)))
    ttft_s = pre.seconds + pre.first_token_seconds
    prefill_tok_per_s = (pre.n_prompt_tokens / pre.seconds) if pre.seconds else float("inf")

    for _ in range(config.warmup_steps):
        engine.decode_step()

    telemetry.start()
    step_seconds = [engine.decode_step().seconds for _ in range(config.decode_tokens - 1)]
    gpu_samples = telemetry.stop()

    summary = summarize_telemetry(gpu_samples, config.decode_tokens, sum(step_seconds))
    return build_result(cfg=cfg, cluster=cluster, config=config, ttft_s=ttft_s,
                        prefill_tok_per_s=prefill_tok_per_s,
                        decode_step_seconds=step_seconds, telemetry_summary=summary)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_runner.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/runner.py tests/test_bench_runner.py
git commit --no-verify -m "feat(bench): fixed-window benchmark runner"
```

---

### Task 6: Store — `RunRecord` JSON round-trip + `x_summary` adapter

**Files:**
- Create: `src/inferutil/bench/store.py`
- Test: `tests/test_bench_store.py`

**Interfaces:**
- Consumes: `BenchConfig` (Task 1); `BenchResult`, `TelemetrySummary` (Task 4).
- Produces: `RunRecord` (frozen: `runid: str`, `config: BenchConfig`, `env: dict`, `result: BenchResult`); `write_run(record, results_dir) -> str` (path `results_dir/<config.name>/<runid>.json`); `load_run(path) -> RunRecord`; `load_all(name, results_dir) -> list[RunRecord]` (sorted by `runid`); `load_latest(name, results_dir) -> RunRecord | None`; `result_to_x_summary(record) -> dict` (console-compatible keys: `ttft_ms`, `decode_tok_per_s`, `prefill_tokens`, `completion_tokens`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_store.py
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.config import BenchConfig
from inferutil.bench.metrics import BenchResult, TelemetrySummary
from inferutil.bench.store import (
    RunRecord, write_run, load_run, load_all, load_latest, result_to_x_summary,
)

CFG = BenchConfig(name="demo", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                  tp=2, ep=8, prompt_tokens=512, decode_tokens=128)
TELE = TelemetrySummary(True, 8, 71.0, 55.0, 60.0, 4800.0, 12.3, 1.4, (55.0, 50.0))
RESULT = BenchResult(0.041, 9000.0, 118.3, 0.0085, 0.0091, 1.13, 128,
                     45_000_000_000, 5.3e12, 0.31, 540.0, 0.219, TELE)


def _record():
    return RunRecord(runid="20260619-120000", config=CFG,
                     env={"gpu": "H100-SXM-80GB", "n_gpus": 8}, result=RESULT)


def test_round_trip_through_json():
    with tempfile.TemporaryDirectory() as d:
        path = write_run(_record(), d)
        back = load_run(path)
        assert back.runid == "20260619-120000"
        assert back.config == CFG
        assert back.result.decode_tok_per_s == 118.3
        assert back.result.telemetry.per_gpu_mean_util == (55.0, 50.0)


def test_load_all_and_latest_ordered():
    with tempfile.TemporaryDirectory() as d:
        write_run(RunRecord("20260619-100000", CFG, {}, RESULT), d)
        write_run(RunRecord("20260619-130000", CFG, {}, RESULT), d)
        allr = load_all("demo", d)
        assert [r.runid for r in allr] == ["20260619-100000", "20260619-130000"]
        assert load_latest("demo", d).runid == "20260619-130000"
        assert load_latest("missing", d) is None


def test_x_summary_has_console_keys():
    xs = result_to_x_summary(_record())
    assert abs(xs["ttft_ms"] - 41.0) < 1e-6
    assert xs["decode_tok_per_s"] == 118.3
    assert xs["prefill_tokens"] == 512 and xs["completion_tokens"] == 128


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inferutil.bench.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inferutil/bench/store.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

from .config import BenchConfig
from .metrics import BenchResult, TelemetrySummary


@dataclass(frozen=True)
class RunRecord:
    runid: str
    config: BenchConfig
    env: dict
    result: BenchResult


def _record_to_dict(r: RunRecord) -> dict:
    return {"runid": r.runid, "config": asdict(r.config), "env": r.env,
            "result": asdict(r.result)}


def _record_from_dict(d: dict) -> RunRecord:
    res = dict(d["result"])
    tele = dict(res.pop("telemetry"))
    tele["per_gpu_mean_util"] = tuple(tele["per_gpu_mean_util"])
    result = BenchResult(telemetry=TelemetrySummary(**tele), **res)
    return RunRecord(runid=d["runid"], config=BenchConfig(**d["config"]),
                     env=d["env"], result=result)


def write_run(record: RunRecord, results_dir: str) -> str:
    out_dir = os.path.join(results_dir, record.config.name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, record.runid + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_record_to_dict(record), f, indent=2)
    return path


def load_run(path: str) -> RunRecord:
    with open(path, encoding="utf-8") as f:
        return _record_from_dict(json.load(f))


def load_all(name: str, results_dir: str) -> list:
    d = os.path.join(results_dir, name)
    if not os.path.isdir(d):
        return []
    files = sorted(fn for fn in os.listdir(d) if fn.endswith(".json"))
    return [load_run(os.path.join(d, fn)) for fn in files]


def load_latest(name: str, results_dir: str):
    runs = load_all(name, results_dir)
    return runs[-1] if runs else None


def result_to_x_summary(record: RunRecord) -> dict:
    """Serialize to the console's x_summary shape (server/schemas.py vocabulary)."""
    r = record.result
    return {
        "ttft_ms": round(r.ttft_s * 1e3, 3),
        "decode_tok_per_s": r.decode_tok_per_s,
        "prefill_tokens": record.config.prompt_tokens,
        "completion_tokens": r.n_decode_tokens,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/store.py tests/test_bench_store.py
git commit --no-verify -m "feat(bench): JSON results store + x_summary adapter"
```

---

### Task 7: Report + CLI (`run` / `report` / `compare`)

**Files:**
- Create: `src/inferutil/bench/report.py`
- Create: `src/inferutil/bench/cli.py`
- Create: `src/inferutil/bench/__main__.py`
- Test: `tests/test_bench_report.py`

**Interfaces:**
- Consumes: `RunRecord`/store fns (Task 6); `BenchResult` (Task 4); `inferutil.{QWEN3_235B, Cluster, GPUS}`; `MockEngine` (Task 2); `run_benchmark` (Task 5); `NvmlTelemetry`/`NullTelemetry` (Task 3); `BenchConfig` (Task 1).
- Produces: `format_result(record) -> str` (single-run breakdown incl. measured-vs-floor); `format_compare(a, b) -> str` (two-run diff); `main(argv=None)` argparse CLI with subcommands `run`, `report`, `compare`, each accepting `--json`; `run` also `--gpu/--n-gpus/--name/--plan/--dtype/--kv-dtype/--tp/--ep/--prompt/--decode/--results-dir`, stamps `runid` from `datetime` (the one allowed clock, CLI-only), uses `NvmlTelemetry` if available else `NullTelemetry`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_report.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.config import BenchConfig
from inferutil.bench.metrics import BenchResult, TelemetrySummary
from inferutil.bench.store import RunRecord
from inferutil.bench.report import format_result, format_compare

CFG = BenchConfig(name="demo", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                  tp=2, ep=8)
TELE = TelemetrySummary(True, 8, 71.0, 55.0, 60.0, 4800.0, 12.3, 1.4, (55.0, 50.0))


def _rec(runid, tok_s, pct):
    res = BenchResult(0.041, 9000.0, tok_s, 0.0085, 0.0091, 1.13, 128,
                      45_000_000_000, 5.3e12, 0.31, 540.0, pct, TELE)
    return RunRecord(runid, CFG, {"gpu": "H100-SXM-80GB", "n_gpus": 8}, res)


def test_format_result_mentions_floor_and_tokps():
    out = format_result(_rec("r1", 118.3, 0.219))
    assert "118.3" in out and "floor" in out.lower() and "%" in out


def test_format_compare_shows_delta():
    out = format_compare(_rec("r1", 100.0, 0.18), _rec("r2", 120.0, 0.22))
    assert "r1" in out and "r2" in out and ("+20" in out or "20.0" in out)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inferutil.bench.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inferutil/bench/report.py
from __future__ import annotations

from .store import RunRecord


def format_result(record: RunRecord) -> str:
    r = record.result
    c = record.config
    ms = 1e3
    t = r.telemetry
    lines = [
        f"== BENCH: {c.name}  [{record.runid}]  plan={c.plan} "
        f"dtype={c.dtype_bytes*8}b tp={c.tp} ep={c.ep} "
        f"ctx={c.seq_len} ({c.prompt_tokens}+{c.decode_tokens}) ==",
        f"  env          : {record.env.get('gpu','?')} x{record.env.get('n_gpus','?')}",
        "  -- latency --",
        f"  TTFT         : {r.ttft_s*ms:8.2f} ms",
        f"  prefill      : {r.prefill_tok_per_s:8.1f} tok/s",
        f"  decode       : {r.decode_tok_per_s:8.1f} tok/s   "
        f"(TPOT p50 {r.tpot_p50_s*ms:.2f} / p95 {r.tpot_p95_s*ms:.2f} ms)",
        "  -- bandwidth / roofline --",
        f"  bytes/token  : {r.bytes_per_token/1e9:8.2f} GB",
        f"  achieved BW  : {r.achieved_hbm_bw/1e12:8.2f} TB/s "
        f"({r.pct_of_peak_bw*100:.1f}% of peak)",
        f"  vs floor     : {r.decode_tok_per_s:8.1f} / "
        f"{r.analytical_floor_tok_per_s:.1f} tok/s = "
        f"{r.pct_of_floor*100:.1f}% of floor",
    ]
    if t.available:
        lines += [
            "  -- device telemetry --",
            f"  temp (max)   : {t.temp_c_max:8.1f} C",
            f"  SM util mean : {t.sm_util_pct_mean:8.1f} %   "
            f"mem util {t.mem_util_pct_mean:.1f} %",
            f"  power (total): {t.power_w_mean:8.0f} W   "
            f"energy/token {t.energy_j_per_token:.2f} J",
            f"  GPU imbalance: {t.util_imbalance:8.2f}x  "
            f"(busiest vs mean util across {t.n_gpus} GPUs)",
        ]
    else:
        lines.append("  -- device telemetry unavailable --")
    return "\n".join(lines)


def _delta(a: float, b: float) -> str:
    d = b - a
    return f"{d:+.1f}"


def format_compare(a: RunRecord, b: RunRecord) -> str:
    ra, rb = a.result, b.result
    ms = 1e3
    return "\n".join([
        f"== COMPARE: {a.runid} (A) vs {b.runid} (B) ==",
        f"  {'metric':<16}{'A':>12}{'B':>12}{'delta':>12}",
        f"  {'decode tok/s':<16}{ra.decode_tok_per_s:>12.1f}"
        f"{rb.decode_tok_per_s:>12.1f}{_delta(ra.decode_tok_per_s, rb.decode_tok_per_s):>12}",
        f"  {'TTFT ms':<16}{ra.ttft_s*ms:>12.2f}{rb.ttft_s*ms:>12.2f}"
        f"{_delta(ra.ttft_s*ms, rb.ttft_s*ms):>12}",
        f"  {'% of floor':<16}{ra.pct_of_floor*100:>12.1f}{rb.pct_of_floor*100:>12.1f}"
        f"{_delta(ra.pct_of_floor*100, rb.pct_of_floor*100):>12}",
        f"  {'% of peak BW':<16}{ra.pct_of_peak_bw*100:>12.1f}"
        f"{rb.pct_of_peak_bw*100:>12.1f}"
        f"{_delta(ra.pct_of_peak_bw*100, rb.pct_of_peak_bw*100):>12}",
        f"  {'temp max C':<16}{ra.telemetry.temp_c_max:>12.1f}"
        f"{rb.telemetry.temp_c_max:>12.1f}"
        f"{_delta(ra.telemetry.temp_c_max, rb.telemetry.temp_c_max):>12}",
    ])
```

```python
# src/inferutil/bench/cli.py
from __future__ import annotations

import argparse
import json
from datetime import datetime

from ..model import QWEN3_235B
from ..hardware import GPUS, Cluster
from .config import BenchConfig
from .engine import MockEngine
from .runner import run_benchmark
from .telemetry import NvmlTelemetry, NullTelemetry
from .store import (write_run, load_run, load_latest, result_to_x_summary, RunRecord)
from .report import format_result, format_compare

DEFAULT_RESULTS_DIR = "results"


def _build_config(args) -> BenchConfig:
    return BenchConfig(
        name=args.name, plan=args.plan, dtype_bytes=args.dtype,
        kv_dtype_bytes=args.kv_dtype, tp=args.tp, ep=args.ep,
        prompt_tokens=args.prompt, decode_tokens=args.decode)


def _cmd_run(args) -> None:
    cluster = Cluster(gpu=GPUS[args.gpu], n_gpus=args.n_gpus)
    config = _build_config(args)
    engine = MockEngine(QWEN3_235B, cluster, efficiency=args.efficiency,
                        jitter=args.jitter, seed=config.seed)
    tele = NvmlTelemetry()
    if not tele.available:
        tele = NullTelemetry()
    result = run_benchmark(engine, config, QWEN3_235B, cluster, telemetry=tele)
    runid = datetime.now().strftime("%Y%m%d-%H%M%S")     # CLI-only clock (run identity)
    record = RunRecord(runid=runid, config=config,
                       env={"gpu": cluster.gpu.name, "n_gpus": cluster.n_gpus}, result=result)
    path = write_run(record, args.results_dir)
    if args.json:
        print(json.dumps(result_to_x_summary(record), indent=2))
    else:
        print(format_result(record))
        print(f"\nsaved -> {path}")


def _resolve(args) -> RunRecord:
    if args.runid in (None, "latest"):
        rec = load_latest(args.name, args.results_dir)
        if rec is None:
            raise SystemExit(f"no runs for '{args.name}' in {args.results_dir}")
        return rec
    import os
    return load_run(os.path.join(args.results_dir, args.name, args.runid + ".json"))


def _cmd_report(args) -> None:
    rec = _resolve(args)
    print(json.dumps(result_to_x_summary(rec), indent=2) if args.json
          else format_result(rec))


def _cmd_compare(args) -> None:
    import os
    a = load_run(os.path.join(args.results_dir, args.name, args.a + ".json"))
    b = load_run(os.path.join(args.results_dir, args.name, args.b + ".json"))
    print(format_compare(a, b))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="inferutil.bench",
                                 description="Offline B=1 decode benchmark harness")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark and store it")
    r.add_argument("--name", default="default")
    r.add_argument("--gpu", default="H100-SXM-80GB", choices=list(GPUS))
    r.add_argument("--n-gpus", type=int, default=8)
    r.add_argument("--plan", default="hybrid", choices=["tp", "ep", "hybrid"])
    r.add_argument("--dtype", type=int, default=2, choices=[1, 2])
    r.add_argument("--kv-dtype", type=int, default=2, choices=[1, 2])
    r.add_argument("--tp", type=int, default=2)
    r.add_argument("--ep", type=int, default=8)
    r.add_argument("--prompt", type=int, default=512)
    r.add_argument("--decode", type=int, default=128)
    r.add_argument("--efficiency", type=float, default=1.0,
                   help="MockEngine BW efficiency (1.0 = analytical floor)")
    r.add_argument("--jitter", type=float, default=0.0)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=_cmd_run)

    rp = sub.add_parser("report", help="print a stored run (default: latest)")
    rp.add_argument("--name", default="default")
    rp.add_argument("runid", nargs="?", default="latest")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=_cmd_report)

    cp = sub.add_parser("compare", help="diff two stored runs")
    cp.add_argument("--name", default="default")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.set_defaults(func=_cmd_compare)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

```python
# src/inferutil/bench/__main__.py
from .cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench_report.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/report.py src/inferutil/bench/cli.py src/inferutil/bench/__main__.py tests/test_bench_report.py
git commit --no-verify -m "feat(bench): report + CLI (run/report/compare)"
```

---

### Task 8: Package exports, results-dir hygiene, end-to-end smoke

**Files:**
- Modify: `src/inferutil/bench/__init__.py`
- Modify: `.gitignore:1-16`
- Create: `results/.gitkeep`
- Modify: `README.md` (append a "Benchmark harness" section)
- Test: `tests/test_bench_smoke.py`

**Interfaces:**
- Produces: `inferutil.bench` re-exports (`BenchConfig`, `config_id`, `MockEngine`, `Engine`, `run_benchmark`, `BenchResult`, `RunRecord`, `write_run`, `load_run`, `load_latest`, `result_to_x_summary`, `NvmlTelemetry`, `NullTelemetry`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_smoke.py
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench import (
    BenchConfig, MockEngine, run_benchmark, RunRecord, write_run, load_latest,
)
from inferutil.bench.cli import main

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def test_end_to_end_run_store_load():
    cfg = BenchConfig(name="smoke", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=1,
                      tp=2, ep=8, prompt_tokens=256, decode_tokens=64)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=0.7, jitter=0.0)
    result = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    assert abs(result.pct_of_floor - 0.7) < 1e-9     # efficiency flows through end-to-end
    with tempfile.TemporaryDirectory() as d:
        write_run(RunRecord("20260619-0001", cfg, {"gpu": "H100-SXM-80GB", "n_gpus": 8},
                            result), d)
        assert load_latest("smoke", d).result.pct_of_floor == result.pct_of_floor


def test_cli_run_and_report(capsys=None):
    with tempfile.TemporaryDirectory() as d:
        main(["--results-dir", d, "run", "--name", "cli", "--decode", "16",
              "--prompt", "64", "--dtype", "1"])
        main(["--results-dir", d, "report", "--name", "cli"])  # latest


if __name__ == "__main__":
    test_end_to_end_run_store_load(); print("ok  test_end_to_end_run_store_load")
    test_cli_run_and_report(); print("ok  test_cli_run_and_report")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench_smoke.py -v`
Expected: FAIL with `ImportError: cannot import name 'BenchConfig' from 'inferutil.bench'`

- [ ] **Step 3: Write minimal implementation**

Replace `src/inferutil/bench/__init__.py` with:

```python
# src/inferutil/bench/__init__.py
"""inferutil.bench — offline B=1 decode benchmark harness.

Drives an in-process Engine (MockEngine now, ConiferEngine later) over a fixed
window, captures latency + derived bandwidth/efficiency + NVML telemetry, stores
JSON results, and reports measured-vs-roofline. See
docs/superpowers/specs/2026-06-19-inference-benchmark-harness-design.md.
"""

from .config import BenchConfig, config_id
from .engine import Engine, MockEngine, ExpertRoute, PrefillResult, DecodeStep
from .telemetry import (GpuSample, TelemetrySource, NullTelemetry,
                        FakeTelemetrySource, NvmlTelemetry)
from .metrics import (BenchResult, TelemetrySummary, bytes_per_token,
                      summarize_telemetry, build_result)
from .runner import run_benchmark
from .store import (RunRecord, write_run, load_run, load_all, load_latest,
                    result_to_x_summary)

__all__ = [
    "BenchConfig", "config_id",
    "Engine", "MockEngine", "ExpertRoute", "PrefillResult", "DecodeStep",
    "GpuSample", "TelemetrySource", "NullTelemetry", "FakeTelemetrySource",
    "NvmlTelemetry",
    "BenchResult", "TelemetrySummary", "bytes_per_token", "summarize_telemetry",
    "build_result", "run_benchmark",
    "RunRecord", "write_run", "load_run", "load_all", "load_latest",
    "result_to_x_summary",
]
```

Append to `.gitignore` (after line 16):

```
# benchmark results (noisy/large; keep the dir, drop the runs)
results/*
!results/.gitkeep
```

Create `results/.gitkeep` (empty file):

```
```

Append to `README.md`:

```markdown

## Benchmark harness (`inferutil.bench`)

Offline B=1 decode benchmarks measured against the analytical roofline. Runs
today on a `MockEngine` (no GPU); a `ConiferEngine` slots in behind the same
`Engine` seam when the engine lands.

```bash
PYTHONPATH=src python -m inferutil.bench run --name fp8-hybrid --dtype 1 --plan hybrid --tp 2 --ep 8
PYTHONPATH=src python -m inferutil.bench report --name fp8-hybrid          # latest run
PYTHONPATH=src python -m inferutil.bench compare <runidA> <runidB> --name fp8-hybrid
```

Each run captures TTFT, decode/prefill tok/s, TPOT p50/p95, derived achieved
bandwidth (% of peak and % of analytical floor), and NVML device telemetry
(temps, util, power, energy/token, per-GPU imbalance). Results are JSON under
`results/<name>/` and diffable run-to-run.
```

- [ ] **Step 4: Run the FULL suite to verify everything passes**

Run: `python -m pytest tests/ -v`
Expected: PASS — all `test_bench_*` plus the existing `test_model.py` (no regressions).

- [ ] **Step 5: Commit**

```bash
git add src/inferutil/bench/__init__.py .gitignore results/.gitkeep README.md tests/test_bench_smoke.py
git commit --no-verify -m "feat(bench): package exports, results-dir hygiene, end-to-end smoke + README"
```

---

## Self-Review notes (for the executor)

- **Spec coverage:** §2 layout → Tasks 1–8 (note: `BenchConfig` lives in `config.py`, a focused split not drawn in the §2 tree). §3 Engine seam → Task 2. §4.1 latency → Tasks 4–5. §4.2 derived bandwidth/efficiency → Task 4 (`build_result`). §4.3 telemetry → Tasks 3–4. §5 config model → Task 1. §6 store → Task 6. §7 report/CLI + `--json` + `x_summary` adapter → Tasks 6–7. §8 testing (no GPU) → every task's tests + Task 8 smoke. §9 console vocabulary alignment → Task 6 `result_to_x_summary`. Appendix A data shapes → Tasks 1–6 verbatim.
- **Roofline wiring check** is enforced three times: `MockEngine` step == floor (Task 2), `build_result` pct_of_floor == 1.0 (Task 4), end-to-end pct_of_floor == efficiency (Tasks 5 & 8).
- **`ConiferEngine`** is intentionally NOT built here — it implements the same three `Engine` methods over the real decode loop once conifer code lands; nothing downstream changes (the whole point of the seam).
