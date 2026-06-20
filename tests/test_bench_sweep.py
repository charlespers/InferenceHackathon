import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.sweep import (
    depth_sweep, config_sweep, quant_grid, layout_grid, full_grid, SweepPoint,
    realized_efficiency,
)

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


def test_layout_grid_keeps_quant_fixed_and_sizes():
    lg = layout_grid(CFG, 8)
    assert len(lg) == 16                                 # 4 divisors x 4 divisors
    assert all(c.dtype_bytes == CFG.dtype_bytes
               and c.kv_dtype_bytes == CFG.kv_dtype_bytes for c in lg)
    assert len(full_grid(CFG, 8)) == 3 * 2 * 16          # dtypes x kv x layouts


def test_layout_sweep_is_pure_speed_search():
    pts = config_sweep(QWEN3_235B, CLUSTER, layout_grid(CFG, 8))
    # one quant point -> all differences are layout (no quality change)
    assert len({(p.dtype_bytes, p.kv_dtype_bytes) for p in pts}) == 1
    assert pts[0].decode_tok_s == max(p.decode_tok_s for p in pts)   # top is fastest


def test_efficiency_scales_predicted_tok_s():
    floor = depth_sweep(QWEN3_235B, CLUSTER, CFG, [512])[0]
    half = depth_sweep(QWEN3_235B, CLUSTER, CFG, [512], efficiency=0.5)[0]
    assert abs(half.decode_tok_s - 0.5 * floor.decode_tok_s) < 1e-6   # linear derate
    assert half.tpot_ms > floor.tpot_ms
    # MBU also scales with the realized throughput
    assert abs(half.mbu_decode - 0.5 * floor.mbu_decode) < 1e-9


def test_realized_efficiency_roundtrips():
    floor = depth_sweep(QWEN3_235B, CLUSTER, CFG, [CFG.seq_len])[0].decode_tok_s
    # if measured == 0.2 * floor, realized e must be 0.2
    e, f = realized_efficiency(QWEN3_235B, CLUSTER, CFG, 0.2 * floor)
    assert abs(e - 0.2) < 1e-9 and abs(f - floor) < 1e-6


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
