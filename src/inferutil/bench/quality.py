"""Output-quality parity: compare a candidate token sequence to a reference.

Speed wins that silently change the model's output (aggressive quant, bad
speculative-decode verification) are regressions. This measures token-level
greedy parity; with the mock it is synthetic, but the moment a real engine
emits real token ids it becomes a genuine quality gate.

NOTE: On MockEngine the token ids are synthetic and independent of the timing
knobs, so the parity check is a wiring SCAFFOLD that only becomes a real
correctness safeguard once ConiferEngine emits real greedy token ids; do not
trust min_quality as a real quality gate until then.
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
