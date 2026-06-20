# tests/test_bench_quality.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.quality import match_rate, QualityResult

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def test_match_rate_identical():
    q = match_rate([1, 2, 3, 4], [1, 2, 3, 4])
    assert isinstance(q, QualityResult)
    assert q.match_rate == 1.0 and q.first_divergence == -1 and q.n_compared == 4


def test_match_rate_partial():
    q = match_rate([1, 2, 9, 4], [1, 2, 3, 4])
    assert q.match_rate == 0.75 and q.first_divergence == 2


def test_mock_emits_token_ids():
    eng = MockEngine(QWEN3_235B, CLUSTER)
    eng.reset(plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8, seq_len=640)
    s = eng.decode_step()
    assert isinstance(s.token_id, int)


def test_run_parity_perfect_then_degraded():
    cfg = BenchConfig(name="q", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=64, decode_tokens=16)
    ref_eng = MockEngine(QWEN3_235B, CLUSTER, quality_offset=0)
    ref = run_benchmark(ref_eng, cfg, QWEN3_235B, CLUSTER, collect_ids=True)
    ref_ids = ref.generated_token_ids
    # identical engine -> perfect parity
    same = run_benchmark(MockEngine(QWEN3_235B, CLUSTER, quality_offset=0), cfg,
                         QWEN3_235B, CLUSTER, reference_ids=ref_ids)
    assert same.quality is not None and same.quality.match_rate == 1.0
    # degraded engine -> parity drops below 1.0
    worse = run_benchmark(MockEngine(QWEN3_235B, CLUSTER, quality_offset=7), cfg,
                          QWEN3_235B, CLUSTER, reference_ids=ref_ids)
    assert worse.quality.match_rate < 1.0


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
