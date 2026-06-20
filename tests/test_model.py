"""Sanity checks: reproduce Qwen3-235B-A22B's headline numbers from the config.

Run: python -m pytest tests/  (or just `python tests/test_model.py`)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, expected_max_experts_per_gpu


def test_total_params_is_235b():
    p = QWEN3_235B.total_params
    assert 230e9 < p < 240e9, f"{p/1e9:.1f}B not ~235B"


def test_active_params_is_22b():
    a = QWEN3_235B.active_params
    assert 20e9 < a < 24e9, f"{a/1e9:.1f}B not ~22B"


def test_active_is_small_fraction():
    assert QWEN3_235B.active_params / QWEN3_235B.total_params < 0.12


def test_kv_cache_per_token():
    # 2 (k,v) * 94 layers * 512 kv_dim * 2 bytes = 192.5 KB/token
    assert QWEN3_235B.kv_bytes_per_token(2) == 2 * 94 * 512 * 2


def test_expert_imbalance_8_into_8():
    # 8 distinct experts into 8 GPUs: busiest holds ~2.0-2.8 in expectation
    e = expected_max_experts_per_gpu(8, 8)
    assert 2.0 < e < 3.0, e
    # ideal (perfect placement) would be 1.0
    assert expected_max_experts_per_gpu(8, 8) > 1.0


def test_no_imbalance_single_gpu():
    assert expected_max_experts_per_gpu(8, 1) == 8.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
