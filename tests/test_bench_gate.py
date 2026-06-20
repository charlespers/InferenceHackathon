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


def test_quality_none_fails_with_not_measured_message():
    result_no_quality = BenchResult(0.04, 9000.0, 120.0, 0.008, 0.009, 1.0, 128,
                                    45_000_000_000, 5.3e12, 0.31, 540.0, 0.6, TELE,
                                    quality=None)
    g = evaluate(result_no_quality, Thresholds(min_quality_match=0.99))
    assert g.passed is False
    joined = " ".join(g.failures)
    assert "not measured" in joined


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
