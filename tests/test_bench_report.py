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
