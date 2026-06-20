import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.metrics import build_result, summarize_telemetry
from inferutil.bench.attribution import diagnose, Bottleneck

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def _cfg(**kw):
    base = dict(name="d", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2,
                tp=2, ep=8, prompt_tokens=128, decode_tokens=16, repeats=1)
    base.update(kw)
    return BenchConfig(**base)


def test_at_floor_is_weight_bound_no_gap():
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.0, jitter=0.0)
    r = run_benchmark(eng, _cfg(), QWEN3_235B, CLUSTER)
    b = diagnose(r)
    assert isinstance(b, Bottleneck)
    assert b.dominant_term == "weight_bw"     # B=1 decode is weight-read bound
    assert b.headroom_to_floor < 1e-9          # at the floor -> no kernel gap
    assert b.regime == "memory-bound"
    assert 0.5 <= b.confidence <= 1.0


def test_below_floor_flags_kernel_gap():
    # efficiency=0.5 -> pct_of_floor=0.5 -> gap == floor time -> kernel_gap dominates
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=0.5, jitter=0.0)
    r = run_benchmark(eng, _cfg(), QWEN3_235B, CLUSTER)
    b = diagnose(r)
    assert b.dominant_term == "kernel_gap"
    assert abs(b.headroom_to_floor - 0.5) < 1e-6
    assert "below the analytical floor" in b.note


def test_above_floor_shares_stay_sane():
    # efficiency>1 -> run BEATS the analytical floor (pct_of_floor>1); a reachable
    # regime (better-than-modeled routing) that was previously untested.
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=1.5, jitter=0.0)
    r = run_benchmark(eng, _cfg(), QWEN3_235B, CLUSTER)
    assert r.pct_of_floor > 1.0
    b = diagnose(r)
    assert b.dominant_term == "weight_bw"          # gap=0 can't win
    assert b.headroom_to_floor == 0.0
    # shares are fractions of ACTUAL per-token time: in [0,1], not inflated past 1
    assert 0.0 < b.share <= 1.0 and 0.0 <= b.second_share <= 1.0
    assert b.share + b.second_share <= 1.0 + 1e-9


def test_fallback_without_measured_breakdown():
    # build a result with no per-term breakdown (step_breakdowns=None)
    cfg = _cfg()
    from inferutil.latency import decode_latency
    floor_s = decode_latency(QWEN3_235B, CLUSTER, plan="hybrid", dtype_bytes=1,
                             kv_dtype_bytes=2, seq_len=cfg.seq_len, tp=2, ep=8).total_s
    steps = [floor_s] * (cfg.decode_tokens - 1)
    r = build_result(cfg=QWEN3_235B, cluster=CLUSTER, config=cfg, ttft_s=0.04,
                     prefill_tok_per_s=9000.0, decode_step_seconds=steps,
                     telemetry_summary=summarize_telemetry([], cfg.decode_tokens, 0.0))
    assert r.measured_breakdown is None
    b = diagnose(r)
    assert b.dominant_term == "weight_bw"      # KV share tiny at short ctx
    assert b.regime == "memory-bound"


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
