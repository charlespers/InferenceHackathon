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
        if result.quality is None:
            fails.append(f"quality not measured (run had no reference) — required min {t.min_quality_match:.3f}")
        elif result.quality.match_rate < t.min_quality_match:
            fails.append(f"quality {result.quality.match_rate:.3f} < min {t.min_quality_match:.3f}")
    return GateResult(passed=(not fails), failures=tuple(fails))
