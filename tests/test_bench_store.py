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
