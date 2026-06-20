import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.efficiency import (
    flops_per_token, mfu, mbu, achieved_tflops, roofline_ridge, classify_regime,
    arithmetic_intensity_decode, arithmetic_intensity_prefill,
    peak_flops_for_dtype, compute_efficiency,
)
from inferutil.bench.metrics import bytes_per_token

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def test_flops_per_token_is_2n():
    assert flops_per_token(10_000_000_000) == 2e10
    # attention term is opt-in
    base = flops_per_token(1000)
    with_attn = flops_per_token(1000, seq_len=128, n_layers=2, d_model=4096,
                                include_attention=True)
    assert with_attn > base


def test_mfu_known_value():
    # 2 * 20e9 * 1000 / 1e15 = 0.04
    assert abs(mfu(1000.0, 20_000_000_000, 1e15) - 0.04) < 1e-12
    assert mfu(None, 20e9, 1e15) is None
    assert mfu(1000.0, 20e9, 0) is None      # no peak -> None, never a fake 0


def test_mbu_known_value():
    # 30e9 bytes/token * 100 tok/s / 26.8e12 = 0.111940...
    val = mbu(100.0, 30_000_000_000, 26.8e12)
    assert abs(val - (3e12 / 26.8e12)) < 1e-12
    assert mbu(100.0, None, 26.8e12) is None


def test_achieved_tflops():
    assert abs(achieved_tflops(1000.0, 20e9) - 40.0) < 1e-9   # 2*20e9*1000/1e12


def test_ridge_and_regime():
    peak_flops = H100_SXM.fp8_flops * 8
    peak_bw = H100_SXM.hbm_bw * 8
    ridge = roofline_ridge(peak_flops, peak_bw)
    assert abs(ridge - (1978.9 / 3.35)) < 1e-6        # ~590.7 FLOPs/byte
    assert classify_regime(1.0, ridge) == "memory-bound"
    assert classify_regime(2000.0, ridge) == "compute-bound"
    assert classify_regime(None, ridge) == "unknown"


def test_decode_is_memory_bound_for_our_model():
    bpt = bytes_per_token(QWEN3_235B, 640, 1, 1)
    ai = arithmetic_intensity_decode(QWEN3_235B.active_params, bpt)
    ridge = roofline_ridge(H100_SXM.fp8_flops * 8, H100_SXM.hbm_bw * 8)
    assert ai < ridge                                  # B=1 decode is memory-bound
    # prefill intensity is prompt_tokens-times larger -> far higher
    weight_bytes = QWEN3_235B.active_params * 1
    ai_p = arithmetic_intensity_prefill(QWEN3_235B.active_params, weight_bytes, 512)
    assert ai_p > ai * 100


def test_peak_flops_for_dtype():
    assert peak_flops_for_dtype(989.4e12, 1978.9e12, 2) == 989.4e12
    assert peak_flops_for_dtype(989.4e12, 1978.9e12, 1) == 1978.9e12


def test_compute_efficiency_bundle():
    bpt = bytes_per_token(QWEN3_235B, 640, 1, 1)
    kv = 640 * QWEN3_235B.kv_bytes_per_token(1)
    eff = compute_efficiency(
        active_params=QWEN3_235B.active_params,
        weight_bytes=QWEN3_235B.active_params * 1,
        bytes_per_token=bpt, kv_bytes=kv, prompt_tokens=512,
        prefill_tok_s=5000.0, decode_tok_s=800.0,
        peak_flops=H100_SXM.fp8_flops * 8, peak_bw=H100_SXM.hbm_bw * 8)
    assert eff.regime_decode == "memory-bound"
    assert 0.0 < eff.mbu_decode < 1.5          # plausible utilization
    assert eff.mfu_prefill > eff.mfu_decode    # prefill throughput >> => higher MFU
    assert 0.0 < eff.kv_byte_share < 1.0
    assert eff.mfu_decode is not None and eff.ai_decode is not None


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
