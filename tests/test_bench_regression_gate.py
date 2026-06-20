import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.gate import regression_gate

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def _run(efficiency, repeats=5, jitter=0.01, seed=1):
    cfg = BenchConfig(name="g", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=32,
                      repeats=repeats)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=efficiency, jitter=jitter, seed=seed)
    return run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)


def test_significant_drop_fails():
    base = _run(1.0)          # ~260 tok/s
    cand = _run(0.75)         # ~195 tok/s, disjoint CIs -> regression
    g = regression_gate(base, cand)
    assert not g.passed and any("regressed" in f for f in g.failures)


def test_improvement_passes():
    base = _run(0.75)         # slower baseline
    cand = _run(1.0)          # faster candidate -> not a regression
    assert regression_gate(base, cand).passed


def test_within_noise_passes():
    base = _run(1.0, seed=1)
    cand = _run(1.0, seed=2)  # same efficiency, only jitter differs -> within noise
    assert regression_gate(base, cand).passed


def test_no_ci_falls_back_to_point_comparison():
    base = _run(1.0, repeats=1, jitter=0.0)
    cand = _run(0.8, repeats=1, jitter=0.0)   # slower, no CI -> point-compare fail
    g = regression_gate(base, cand)
    assert not g.passed and any("no CI" in f for f in g.failures)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
