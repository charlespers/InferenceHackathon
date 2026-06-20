import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.attribution import diagnose
from inferutil.bench.levers import recommend, Lever, _spec_speedup

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def _cfg(**kw):
    base = dict(name="L", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                tp=2, ep=8, prompt_tokens=512, decode_tokens=128)
    base.update(kw)
    return BenchConfig(**base)


def test_returns_ranked_levers_all_positive_gain():
    levers = recommend(QWEN3_235B, CLUSTER, _cfg())
    assert levers and all(isinstance(lv, Lever) for lv in levers)
    sps = [lv.speedup for lv in levers]
    assert sps == sorted(sps, reverse=True)        # descending by speedup
    assert all(lv.speedup > 1.0 for lv in levers)
    assert "int4 experts" in {lv.name for lv in levers}


def test_int4_beats_kv_quant_when_weight_bound():
    by = {lv.name: lv for lv in recommend(QWEN3_235B, CLUSTER,
                                          _cfg(dtype_bytes=1, kv_dtype_bytes=2))}
    assert "int4 experts" in by
    # halving 21.6GB weights helps far more than halving tiny KV at this ctx
    if "fp8/int8 KV" in by:
        assert by["int4 experts"].speedup > by["fp8/int8 KV"].speedup


def test_min_speedup_filter():
    levers = recommend(QWEN3_235B, CLUSTER, _cfg(), min_speedup=1.5)
    assert all(lv.speedup >= 1.5 for lv in levers)


def test_bottleneck_annotation_marks_targeting_lever():
    cfg = _cfg(dtype_bytes=1, decode_tokens=16, prompt_tokens=128, repeats=1)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    b = diagnose(r)                                # weight_bw dominant
    levers = recommend(QWEN3_235B, CLUSTER, cfg, bottleneck=b)
    tagged = [lv for lv in levers if "targets diagnosed bottleneck" in lv.rationale]
    assert tagged, "lever targeting the diagnosed bottleneck should be annotated"


def test_spec_speedup_monotonic_in_accept_rate():
    # floor-aware model (bench.spec_model); monotone in acceptance at fixed floor
    assert _spec_speedup(0.8, 4, 1, 0.5) > _spec_speedup(0.3, 4, 1, 0.5)


def test_spec_lever_regime_aware_floor():
    # floor-bound (high F) -> big trees win; weight-bound (F=0) -> small trees win
    small_wb = _spec_speedup(0.7, 2, 1, 0.0)
    big_wb = _spec_speedup(0.7, 8, 4, 0.0)
    small_fb = _spec_speedup(0.7, 2, 1, 0.9)
    big_fb = _spec_speedup(0.7, 8, 4, 0.9)
    assert small_wb > big_wb     # weight-bound: big tree's expert-union tax loses
    assert big_fb > small_fb     # floor-bound: big tree wins


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
