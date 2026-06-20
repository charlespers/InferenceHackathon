import sys, os, csv, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil import QWEN3_235B, Cluster, H100_SXM
from inferutil.bench.config import BenchConfig
from inferutil.bench.engine import MockEngine
from inferutil.bench.runner import run_benchmark
from inferutil.bench.store import (
    RunRecord, export_csv, export_jsonl, export_markdown, record_row, load_run,
    write_run, CSV_COLUMNS, EM_DASH,
)

CLUSTER = Cluster(gpu=H100_SXM, n_gpus=8)


def _rec():
    cfg = BenchConfig(name="e", plan="hybrid", dtype_bytes=1, kv_dtype_bytes=2,
                      tp=2, ep=8, prompt_tokens=128, decode_tokens=16, repeats=3)
    eng = MockEngine(QWEN3_235B, CLUSTER, efficiency=0.8, jitter=0.02, seed=1)
    r = run_benchmark(eng, cfg, QWEN3_235B, CLUSTER)
    return RunRecord(runid="20260619-000001", config=cfg,
                     env={"gpu": "H100-SXM-80GB", "n_gpus": 8}, result=r)


def test_record_row_carries_efficiency():
    row = record_row(_rec())
    assert row["mbu_decode"] is not None
    assert row["mfu_prefill"] is not None
    assert row["regime_decode"] == "memory-bound"


def test_csv_header_and_emdash_for_unmeasured():
    with tempfile.TemporaryDirectory() as d:
        p = export_csv([_rec()], os.path.join(d, "r.csv"))
        with open(p) as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == CSV_COLUMNS
        # MockEngine has no telemetry -> em-dash, never a fabricated 0
        assert rows[0]["energy_j_per_token"] == EM_DASH
        assert rows[0]["util_imbalance"] == EM_DASH


def test_jsonl_is_lossless_with_samples():
    with tempfile.TemporaryDirectory() as d:
        p = export_jsonl([_rec()], os.path.join(d, "r.jsonl"))
        with open(p) as f:
            obj = json.loads(f.readline())
        assert obj["result"]["latency"]["throughput_tok_s"]["samples"]
        assert obj["result"]["efficiency"]["mbu_decode"] is not None


def test_markdown_table_shape():
    with tempfile.TemporaryDirectory() as d:
        p = export_markdown([_rec()], os.path.join(d, "r.md"))
        with open(p) as f:
            lines = f.read().splitlines()
        assert lines[0].startswith("| runid |") and lines[0].count("|") == len(CSV_COLUMNS) + 1
        assert set(lines[1].replace("|", "").split()) == {"---"}   # separator row
        assert lines[2].startswith("| 20260619-000001 |")
        assert EM_DASH in lines[2]                                 # unmeasured -> em-dash


def test_efficiency_and_latency_survive_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = write_run(_rec(), d)
        back = load_run(path)
        assert back.result.efficiency is not None
        assert back.result.efficiency.regime_decode == "memory-bound"
        assert back.result.latency["tpot_ms"]["p50"] is not None


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
