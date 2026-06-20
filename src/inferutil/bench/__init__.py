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
