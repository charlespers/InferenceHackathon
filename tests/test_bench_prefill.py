import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.prefill import (
    prefill_latency, prefill_weight_bytes, expected_distinct_experts,
)

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def test_distinct_experts_saturates_with_prompt_length():
    few = expected_distinct_experts(128, 8, 8)
    many = expected_distinct_experts(128, 8, 2048)
    assert few < many <= 128
    assert many > 127.0                       # long prompt touches ~all experts
    assert expected_distinct_experts(128, 8, 0) == 0.0


def test_prefill_weight_bytes_grows_with_prompt():
    short = prefill_weight_bytes(QWEN3_235B, 1, 8)
    long = prefill_weight_bytes(QWEN3_235B, 1, 2048)
    assert short < long                       # more experts activated by longer prompt


def test_prefill_far_faster_than_decode():
    # prefill amortizes the weight read over the whole prompt -> way higher tok/s
    pf_s = prefill_latency(QWEN3_235B, CLUSTER, dtype_bytes=1, kv_dtype_bytes=2,
                           prompt_tokens=512)
    prefill_tok_s = 512 / pf_s
    # a single decode step rereads active weights -> ~hundreds tok/s
    cfg = BenchConfig(name="p", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=512, decode_tokens=8, repeats=1)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    assert prefill_tok_s > 5 * r.decode_tok_per_s


def test_prefill_makes_mfu_meaningful():
    cfg = BenchConfig(name="p", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=512, decode_tokens=16, repeats=1)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    eff = r.efficiency
    # prefill MFU is now a real, non-trivial compute-utilization number,
    # and far above the structurally-tiny decode MFU
    assert eff.mfu_prefill > eff.mfu_decode * 10
    assert eff.mfu_prefill > 0.01


def test_prompt_tokens_zero_is_finite_and_json_safe():
    import math, json, tempfile
    from inferutil.bench.store import RunRecord, write_run
    cfg = BenchConfig(name="z", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=0, decode_tokens=8, repeats=1)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    # no inf leaks into prefill metrics (would serialize as invalid JSON `Infinity`)
    assert math.isfinite(r.prefill_tok_per_s)
    assert math.isfinite(r.efficiency.mfu_prefill)
    assert math.isfinite(r.efficiency.achieved_tflops_prefill)
    with tempfile.TemporaryDirectory() as d:
        p = write_run(RunRecord("zr", cfg, {"gpu": "H100-SXM-80GB", "n_gpus": 8}, r), d)
        with open(p) as f:
            txt = f.read()
        assert "Infinity" not in txt and "NaN" not in txt
        json.loads(txt)   # strict parse must succeed


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
