# tests/test_bench_variance.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.metrics import build_result, summarize_telemetry
from inferutil.bench.report import is_significant, format_compare
from inferutil.bench.store import RunRecord

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def test_repeats_populate_std_and_count():
    cfg = BenchConfig(name="v", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=32, repeats=5)
    # jitter makes repeats differ so std > 0
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.05, seed=1)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    assert r.n_repeats == 5
    assert r.decode_tok_per_s_std > 0.0
    assert r.decode_tok_per_s > 0.0


def test_single_repeat_has_zero_std():
    cfg = BenchConfig(name="v", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=16, repeats=1)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    assert r.n_repeats == 1 and r.decode_tok_per_s_std == 0.0


def test_build_result_samples_mean_and_std():
    tele = summarize_telemetry([], 10, 0.0)
    cfg = BenchConfig(name="v", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=16)
    r = build_result(cfg=QWEN3_235B, cluster=CLUSTER, config=cfg, ttft_s=0.04,
                     prefill_tok_per_s=1000.0, decode_step_seconds=[0.01] * 15,
                     telemetry_summary=tele, decode_tok_per_s_samples=[90.0, 110.0])
    assert r.decode_tok_per_s == 100.0
    assert abs(r.decode_tok_per_s_std - 10.0) < 1e-9   # popstd of [90,110]
    assert r.n_repeats == 2


def test_significance_threshold():
    # means 100 vs 130, stds 5 and 5 -> combined 2*sqrt(50)=14.1 -> 30 is significant
    assert is_significant(100.0, 5.0, 130.0, 5.0) is True
    # means 100 vs 105, stds 5 and 5 -> 5 < 14.1 -> not significant
    assert is_significant(100.0, 5.0, 105.0, 5.0) is False


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
