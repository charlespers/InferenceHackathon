import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.sweep import depth_sweep, config_sweep, quant_grid, SweepPoint

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)
CFG = BenchConfig(name="s", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2,
                  tp=2, ep=8, prompt_tokens=512, decode_tokens=128)


def test_depth_sweep_kv_share_rises_tokps_falls():
    pts = depth_sweep(QWEN3_235B, CLUSTER, CFG, [512, 4096, 32768])
    assert all(isinstance(p, SweepPoint) for p in pts)
    assert [p.seq_len for p in pts] == [512, 4096, 32768]
    shares = [p.kv_byte_share for p in pts]
    assert shares[0] < shares[1] < shares[2]            # KV grows with context
    toks = [p.decode_tok_s for p in pts]
    assert toks[0] > toks[1] > toks[2]                  # decode slows with context
    assert all(p.regime == "memory-bound" for p in pts)


def test_depth_sweep_crosses_over_to_kv_bound():
    pts = depth_sweep(QWEN3_235B, CLUSTER, CFG, [512, 131072])
    assert pts[0].dominant_term == "weight_bw"          # short ctx: weights
    assert pts[-1].dominant_term == "kv_bw"             # long ctx: KV overtakes


def test_quant_grid_and_config_ranking():
    grid = quant_grid(CFG)
    assert len(grid) == 6                                # 3 dtypes x 2 kv dtypes
    pts = config_sweep(QWEN3_235B, CLUSTER, grid)
    toks = [p.decode_tok_s for p in pts]
    assert toks == sorted(toks, reverse=True)            # ranked fastest-first
    top = pts[0]
    assert top.dtype_bytes == 0.5 and top.kv_dtype_bytes == 1  # lowest bytes wins


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
