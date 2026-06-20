# src/inferutil/bench/__init__.py
"""inferutil.bench — offline B=1 decode benchmark harness.

Drives an in-process Engine (MockEngine now, ConiferEngine later) over a fixed
window, captures latency + derived bandwidth/efficiency + NVML telemetry, stores
JSON results, and reports measured-vs-roofline. See
docs/superpowers/specs/2026-06-19-inference-benchmark-harness-design.md.
"""

from .config import BenchConfig, config_id
from .engine import Engine, MockEngine, ExpertRoute, PrefillResult, DecodeStep, StepBreakdown
from .telemetry import (GpuSample, TelemetrySource, NullTelemetry,
                        FakeTelemetrySource, NvmlTelemetry)
from .metrics import (BenchResult, TelemetrySummary, bytes_per_token,
                      summarize_telemetry, build_result, MeasuredBreakdown,
                      aggregate_breakdown)
from .runner import run_benchmark
from .store import (RunRecord, write_run, load_run, load_all, load_latest,
                    result_to_x_summary, record_row, export_csv, export_jsonl,
                    export_markdown)
from .quality import QualityResult, match_rate
from .gate import Thresholds, GateResult, evaluate, regression_gate
from .efficiency import Efficiency, compute_efficiency
from .stats import Stat, summarize, means_differ
from .attribution import Bottleneck, diagnose
from .levers import Lever, recommend
from .prefill import prefill_latency
from .sweep import (SweepPoint, depth_sweep, config_sweep,
                    quant_grid, layout_grid, full_grid)
from .cost import energy_metrics, rental_usd_per_mtok
from .manifest import build_manifest, model_hash

__all__ = [
    "BenchConfig", "config_id",
    "Engine", "MockEngine", "ExpertRoute", "PrefillResult", "DecodeStep",
    "StepBreakdown",
    "GpuSample", "TelemetrySource", "NullTelemetry", "FakeTelemetrySource",
    "NvmlTelemetry",
    "BenchResult", "TelemetrySummary", "bytes_per_token", "summarize_telemetry",
    "build_result", "run_benchmark",
    "MeasuredBreakdown", "aggregate_breakdown",
    "RunRecord", "write_run", "load_run", "load_all", "load_latest",
    "result_to_x_summary", "record_row", "export_csv", "export_jsonl",
    "export_markdown",
    "QualityResult", "match_rate",
    "Thresholds", "GateResult", "evaluate", "regression_gate",
    # rigorous-suite additions
    "Efficiency", "compute_efficiency",
    "Stat", "summarize", "means_differ",
    "Bottleneck", "diagnose",
    "Lever", "recommend",
    "prefill_latency",
    "SweepPoint", "depth_sweep", "config_sweep",
    "quant_grid", "layout_grid", "full_grid",
    "energy_metrics", "rental_usd_per_mtok",
    "build_manifest", "model_hash",
]
