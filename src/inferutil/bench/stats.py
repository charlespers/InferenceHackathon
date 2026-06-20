"""Honest summary statistics over repeated measurements — pure stdlib.

A `Stat` summarizes one quantity measured `n` times: central tendency, spread,
percentiles, and a Student-t 95% confidence interval on the mean (honest for the
small `n` a benchmark produces). `None` means "not measured" — we never
substitute a fabricated 0, so downstream reports can render an em-dash instead of
implying a real zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List, Optional


# Two-sided 95% Student-t critical values by degrees of freedom (df = n-1).
# For df beyond the table we fall back to the normal z (1.96); rounding to the
# nearest tabulated df <= request keeps the interval conservative (slightly wide).
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}
_Z95 = 1.96


def t95(df: int) -> float:
    """Two-sided 95% critical value for `df` degrees of freedom."""
    if df <= 0:
        return float("inf")
    if df in _T95:
        return _T95[df]
    if df > 30:
        return _Z95
    return _T95[max(k for k in _T95 if k <= df)]


def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    """Linear-interpolation percentile; p in [0, 1]. None for empty input."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@dataclass(frozen=True)
class Stat:
    n: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None          # sample std (ddof=1); None for n<2
    cv: Optional[float] = None           # coefficient of variation = std/mean
    p50: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    ci95_lo: Optional[float] = None      # 95% CI on the MEAN (Student-t)
    ci95_hi: Optional[float] = None
    samples: tuple = ()

    @property
    def ci95_half_width(self) -> Optional[float]:
        if self.ci95_lo is None or self.ci95_hi is None:
            return None
        return (self.ci95_hi - self.ci95_lo) / 2.0


def summarize(values) -> Stat:
    """Summarize a list of repeated measurements into a `Stat`."""
    xs = [float(v) for v in (values or []) if v is not None]
    n = len(xs)
    if n == 0:
        return Stat(n=0)
    mean = sum(xs) / n
    sv = sorted(xs)
    if n == 1:
        return Stat(n=1, mean=mean, p50=mean, p90=mean, p95=mean, p99=mean,
                    min=sv[0], max=sv[0], samples=tuple(xs))
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = sqrt(var)
    cv = (std / mean) if mean else None
    se = std / sqrt(n)
    half = t95(n - 1) * se
    return Stat(
        n=n, mean=mean, std=std, cv=cv,
        p50=percentile(sv, 0.5), p90=percentile(sv, 0.9),
        p95=percentile(sv, 0.95), p99=percentile(sv, 0.99),
        min=sv[0], max=sv[-1],
        ci95_lo=mean - half, ci95_hi=mean + half, samples=tuple(xs))


def stat_to_dict(s: Stat) -> dict:
    """JSON-friendly dict (drops the raw samples tuple's type)."""
    return {
        "n": s.n, "mean": s.mean, "std": s.std, "cv": s.cv,
        "p50": s.p50, "p90": s.p90, "p95": s.p95, "p99": s.p99,
        "min": s.min, "max": s.max,
        "ci95_lo": s.ci95_lo, "ci95_hi": s.ci95_hi,
        "samples": list(s.samples),
    }


def stat_from_dict(d: Optional[dict]) -> Optional[Stat]:
    if not d:
        return None
    d = dict(d)
    d["samples"] = tuple(d.get("samples", ()) or ())
    return Stat(**d)


def means_differ(a: Stat, b: Stat) -> bool:
    """True when two means are statistically distinguishable (95% CIs disjoint).

    Conservative: requires the confidence intervals not to overlap. Falls back to
    False when either side lacks a CI (n<2)."""
    if a.ci95_lo is None or b.ci95_lo is None:
        return False
    return a.ci95_hi < b.ci95_lo or b.ci95_hi < a.ci95_lo
