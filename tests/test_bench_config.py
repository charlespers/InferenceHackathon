import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.config import BenchConfig, config_id


def test_seq_len_is_prompt_plus_decode():
    c = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2,
                    tp=2, ep=8, prompt_tokens=512, decode_tokens=128)
    assert c.seq_len == 640


def test_config_id_is_stable_and_field_sensitive():
    a = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8)
    b = BenchConfig(name="x", plan="hybrid", dtype_bytes=2, kv_dtype_bytes=2, tp=2, ep=8)
    c = BenchConfig(name="x", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2, tp=2, ep=8)
    assert config_id(a) == config_id(b)
    assert config_id(a) != config_id(c)
    assert len(config_id(a)) == 12


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
