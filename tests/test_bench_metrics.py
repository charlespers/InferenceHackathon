import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.latency import decode_latency
from inferutil.bench.config import BenchConfig
from inferutil.bench.telemetry import GpuSample
from inferutil.bench.metrics import (
    bytes_per_token, percentile, summarize_telemetry, build_result,
)

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)
CFG = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                  tp=2, ep=8, prompt_tokens=512, decode_tokens=128)


def test_bytes_per_token_matches_model_accounting():
    expected = QWEN3_235B.active_params * 2 + 640 * QWEN3_235B.kv_bytes_per_token(2)
    assert bytes_per_token(QWEN3_235B, 640, 2, 2) == expected


def test_percentile_interpolates():
    assert percentile([0.0, 1.0, 2.0, 3.0], 0.5) == 1.5


def test_perfect_run_is_100pct_of_floor():
    floor_s = decode_latency(QWEN3_235B, CLUSTER, plan="hybrid", dtype_bytes=2,
                             kv_dtype_bytes=2, seq_len=640, tp=2, ep=8).total_s
    steps = [floor_s] * (CFG.decode_tokens - 1)
    r = build_result(cfg=QWEN3_235B, cluster=CLUSTER, config=CFG, ttft_s=0.05,
                     prefill_tok_per_s=10000.0, decode_step_seconds=steps,
                     telemetry_summary=summarize_telemetry([], CFG.decode_tokens, 0.0))
    assert abs(r.pct_of_floor - 1.0) < 1e-9
    assert r.bytes_per_token == bytes_per_token(QWEN3_235B, 640, 2, 2)
    assert 0.0 < r.pct_of_peak_bw < 1.0


def test_summarize_telemetry_aggregates_and_imbalance():
    # gpu0 busy (80%), gpu1 idle (40%); two timestamps each.
    def s(g, t, sm, pw):
        return GpuSample(g, t, 65.0, sm, sm, pw, 1500.0, 2600.0, 40_000_000_000)
    samples = [s(0, 0.0, 80.0, 600.0), s(1, 0.0, 40.0, 500.0),
               s(0, 0.1, 80.0, 600.0), s(1, 0.1, 40.0, 500.0)]
    summ = summarize_telemetry(samples, n_decode_tokens=10, decode_window_s=0.1)
    assert summ.available and summ.n_gpus == 2
    assert summ.per_gpu_mean_util == (80.0, 40.0)
    assert abs(summ.util_imbalance - (80.0 / 60.0)) < 1e-9   # max / mean
    # total instantaneous power = 1100 W; energy/token = 1100*0.1/10 = 11 J
    assert abs(summ.energy_j_per_token - 11.0) < 1e-9


def test_summarize_empty_is_unavailable():
    summ = summarize_telemetry([], 10, 0.0)
    assert summ.available is False and summ.per_gpu_mean_util == ()


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
