import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.attribution import diagnose
from inferutil.bench.levers import recommend
from inferutil.bench.sweep import config_sweep, full_grid
from inferutil.bench.store import RunRecord
from inferutil.bench.report import format_plan

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def _plan_text(dtype=1.0):
    cfg = BenchConfig(name="plan", plan="hybrid", dtype_bytes=dtype, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=512, decode_tokens=64, repeats=1)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    rec = RunRecord(runid="analytical", config=cfg,
                    env={"gpu": "H100-SXM-80GB", "n_gpus": 8}, result=r)
    b = diagnose(r)
    levers = recommend(QWEN3_235B, CLUSTER, cfg, bottleneck=b)
    best = config_sweep(QWEN3_235B, CLUSTER, full_grid(cfg, 8))[0]
    return format_plan(rec, b, levers, best), best


def test_plan_has_all_sections():
    text, best = _plan_text()
    assert "PLAN:" in text
    assert "bottleneck" in text
    assert "biggest wins" in text
    assert "suggested order" in text
    assert "best reachable config" in text


def test_plan_best_is_lowest_byte_config():
    _, best = _plan_text()
    # the analytically fastest config is the most-quantized one
    assert best.dtype_bytes == 0.5 and best.kv_dtype_bytes == 1


def test_plan_suggested_order_starts_cheap():
    text, _ = _plan_text()
    order = text.split("suggested order")[1]
    # the first numbered step should be a low-effort [S] lever
    first = [ln for ln in order.splitlines() if ln.strip().startswith("1.")][0]
    assert "[S]" in first


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
