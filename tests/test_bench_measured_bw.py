import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.latency import decode_latency
from inferutil.bench.cli import _measured_cluster

SPEC = Cluster(gpu=H100_SXM, n_gpus=8)


def test_none_is_passthrough():
    assert _measured_cluster(SPEC, None) is SPEC


def test_measured_bw_replaces_spec_and_lowers_floor():
    # measured 2500 GB/s < spec 3350 GB/s -> lower aggregate BW -> slower floor
    meas = _measured_cluster(SPEC, 2500.0)
    assert meas.gpu.hbm_bw == 2500.0 * 1e9
    assert meas.aggregate_hbm_bw == 2500.0 * 1e9 * 8
    spec_tok = decode_latency(QWEN3_235B, SPEC, plan="hybrid", dtype_bytes=1,
                              kv_dtype_bytes=2, seq_len=640, tp=2, ep=8).tokens_per_s
    meas_tok = decode_latency(QWEN3_235B, meas, plan="hybrid", dtype_bytes=1,
                              kv_dtype_bytes=2, seq_len=640, tp=2, ep=8).tokens_per_s
    assert meas_tok < spec_tok                      # honest floor against real BW


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
