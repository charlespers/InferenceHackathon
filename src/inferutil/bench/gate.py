"""Pass/fail regression gate: the machine-readable accept/reject an autonomous
optimization loop branches on. Every threshold is optional; only the ones set
are enforced."""
from __future__ import annotations

from dataclasses import dataclass

from .stats import stat_from_dict, means_differ


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
        if result.quality is None:
            fails.append(f"quality not measured (run had no reference) — required min {t.min_quality_match:.3f}")
        elif result.quality.match_rate < t.min_quality_match:
            fails.append(f"quality {result.quality.match_rate:.3f} < min {t.min_quality_match:.3f}")
    return GateResult(passed=(not fails), failures=tuple(fails))


def _throughput_stat(result):
    lat = getattr(result, "latency", None) or {}
    return stat_from_dict(lat.get("throughput_tok_s"))


def regression_gate(baseline_result, candidate_result) -> GateResult:
    """Fail iff the candidate's decode throughput is *significantly* below the
    baseline — i.e. the mean dropped AND the 95% CIs are disjoint. A significant
    improvement, or a within-noise change, passes. Falls back to a point
    comparison when either run lacks repeats (no CI)."""
    b = _throughput_stat(baseline_result)
    c = _throughput_stat(candidate_result)
    fails = []
    if b is None or c is None or b.mean is None or c.mean is None \
            or b.ci95_lo is None or c.ci95_lo is None:
        # No CIs (n<2 on a side): conservative point comparison.
        if candidate_result.decode_tok_per_s < baseline_result.decode_tok_per_s:
            fails.append(
                f"throughput {candidate_result.decode_tok_per_s:.1f} < baseline "
                f"{baseline_result.decode_tok_per_s:.1f} tok/s (no CI; n<2)")
    elif c.mean < b.mean and means_differ(b, c):
        fails.append(
            f"throughput regressed {c.mean:.1f} vs baseline {b.mean:.1f} tok/s "
            f"(95% CIs disjoint)")
    return GateResult(passed=(not fails), failures=tuple(fails))
