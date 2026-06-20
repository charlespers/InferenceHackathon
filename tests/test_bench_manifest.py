import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.model import MoEConfig
from inferutil.bench.config import BenchConfig
from inferutil.bench.manifest import build_manifest, model_hash, SCHEMA

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)
CFG = BenchConfig(name="m", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=1,
                  tp=2, ep=8, prompt_tokens=512, decode_tokens=128)


def test_manifest_has_required_sections():
    m = build_manifest(QWEN3_235B, CLUSTER, CFG, cli="inferutil.bench run ...")
    for k in ("schema", "host", "code", "model", "hardware", "bench"):
        assert k in m
    assert m["schema"] == SCHEMA
    assert m["hardware"]["gpu"] == "H100-SXM-80GB" and m["hardware"]["n_gpus"] == 8
    assert m["hardware"]["aggregate_hbm_bw"] == H100_SXM.hbm_bw * 8
    assert m["bench"]["seed"] == CFG.seed
    assert m["bench"]["config_id"]
    assert m["cli"].startswith("inferutil.bench run")


def test_model_hash_deterministic_and_config_sensitive():
    assert model_hash(QWEN3_235B) == model_hash(QWEN3_235B)
    assert m_hash_differs()


def m_hash_differs():
    return model_hash(QWEN3_235B) != model_hash(MoEConfig(n_layers=80))


def test_measured_peak_bw_passthrough():
    m = build_manifest(QWEN3_235B, CLUSTER, CFG, peak_bw_measured=3.1e12)
    assert m["hardware"]["peak_bw_measured"] == 3.1e12


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
