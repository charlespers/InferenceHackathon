import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.latency import decode_latency
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine, StepBreakdown, DecodeStep
from inferutil.bench.runner import run_benchmark
from inferutil.bench.metrics import aggregate_breakdown, MeasuredBreakdown

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def test_mock_step_breakdown_sums_to_step_seconds():
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    eng.reset(plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8, seq_len=640)
    step = eng.decode_step()
    b = step.breakdown
    assert isinstance(b, StepBreakdown)
    assert abs((b.weight_s + b.kv_s + b.comms_s + b.compute_s) - step.seconds) < 1e-12


def test_aggregate_breakdown_means():
    bs = [StepBreakdown(1.0, 2.0, 3.0, 4.0), StepBreakdown(3.0, 4.0, 5.0, 6.0)]
    agg = aggregate_breakdown(bs)
    assert isinstance(agg, MeasuredBreakdown)
    assert (agg.weight_s, agg.kv_s, agg.comms_s, agg.compute_s) == (2.0, 3.0, 4.0, 5.0)
    assert aggregate_breakdown([]) is None


def test_run_populates_measured_breakdown():
    cfg = BenchConfig(name="a", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=16)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    assert r.measured_breakdown is not None
    # weight term dominates at B=1 (matches the analytical thesis)
    mb = r.measured_breakdown
    assert mb.weight_s > mb.comms_s and mb.weight_s > mb.kv_s


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
