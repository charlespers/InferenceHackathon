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
