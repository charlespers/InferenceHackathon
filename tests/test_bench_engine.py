# tests/test_bench_engine.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.latency import decode_latency
from inferutil.bench.engine import MockEngine, DecodeStep, PrefillResult

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def _eng(**kw):
    e = MockEngine(QWEN3_235B, CLUSTER, **kw)
    e.reset(plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8, seq_len=640)
    return e


def test_perfect_mock_step_equals_floor():
    floor = decode_latency(QWEN3_235B, CLUSTER, plan="hybrid", dtype_bytes=2,
                           kv_dtype_bytes=2, seq_len=640, tp=2, ep=8).total_s
    step = _eng(efficiency=1.0, jitter=0.0).decode_step()
    assert isinstance(step, DecodeStep)
    assert abs(step.seconds - floor) < 1e-12


def test_efficiency_scales_step_time():
    floor = decode_latency(QWEN3_235B, CLUSTER, plan="hybrid", dtype_bytes=2,
                           kv_dtype_bytes=2, seq_len=640, tp=2, ep=8).total_s
    step = _eng(efficiency=0.5, jitter=0.0).decode_step()
    assert abs(step.seconds - floor / 0.5) < 1e-12


def test_prefill_shape_and_indices():
    e = _eng()
    pre = e.prefill(list(range(512)))
    assert isinstance(pre, PrefillResult) and pre.n_prompt_tokens == 512
    assert e.decode_step().index == 0 and e.decode_step().index == 1


def test_routes_optional():
    assert _eng(expose_routes=False).decode_step().routes == ()
    assert len(_eng(expose_routes=True).decode_step().routes) == QWEN3_235B.top_k


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
