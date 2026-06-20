import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.telemetry import FakeTelemetrySource, GpuSample

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)
CFG = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                  tp=2, ep=8, prompt_tokens=512, decode_tokens=128, warmup_steps=8)


def test_perfect_mock_run_hits_floor():
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, CFG, QWEN3_235B, CLUSTER)
    assert abs(r.pct_of_floor - 1.0) < 1e-9
    assert r.n_decode_tokens == 128
    assert r.ttft_s > 0.0


def test_decode_tokens_below_two_raises():
    # the runner's only guard of the "need inter-token samples" invariant
    cfg = BenchConfig(name="e", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=64, decode_tokens=1)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    with pytest.raises(ValueError):
        run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)


def test_runner_uses_telemetry_source():
    eng = MockEngine(QWEN3_235B, CLUSTER)
    fake = FakeTelemetrySource([
        GpuSample(0, 0.0, 70.0, 90.0, 90.0, 600.0, 1500.0, 2600.0, 4 * 10**10),
        GpuSample(1, 0.0, 60.0, 30.0, 30.0, 400.0, 1500.0, 2600.0, 4 * 10**10),
    ])
    r = run_benchmark(eng, CFG, QWEN3_235B, CLUSTER, telemetry=fake)
    assert r.telemetry.available and r.telemetry.n_gpus == 2


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
