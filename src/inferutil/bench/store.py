from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

from .config import BenchConfig
from .metrics import BenchResult, TelemetrySummary


@dataclass(frozen=True)
class RunRecord:
    runid: str
    config: BenchConfig
    env: dict
    result: BenchResult


def _record_to_dict(r: RunRecord) -> dict:
    return {"runid": r.runid, "config": asdict(r.config), "env": r.env,
            "result": asdict(r.result)}


def _record_from_dict(d: dict) -> RunRecord:
    res = dict(d["result"])
    tele = dict(res.pop("telemetry"))
    tele["per_gpu_mean_util"] = tuple(tele["per_gpu_mean_util"])
    result = BenchResult(telemetry=TelemetrySummary(**tele), **res)
    return RunRecord(runid=d["runid"], config=BenchConfig(**d["config"]),
                     env=d["env"], result=result)


def write_run(record: RunRecord, results_dir: str) -> str:
    out_dir = os.path.join(results_dir, record.config.name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, record.runid + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_record_to_dict(record), f, indent=2)
    return path


def load_run(path: str) -> RunRecord:
    with open(path, encoding="utf-8") as f:
        return _record_from_dict(json.load(f))


def load_all(name: str, results_dir: str) -> list:
    d = os.path.join(results_dir, name)
    if not os.path.isdir(d):
        return []
    files = sorted(fn for fn in os.listdir(d) if fn.endswith(".json"))
    return [load_run(os.path.join(d, fn)) for fn in files]


def load_latest(name: str, results_dir: str):
    runs = load_all(name, results_dir)
    return runs[-1] if runs else None


def result_to_x_summary(record: RunRecord) -> dict:
    """Serialize to the console's x_summary shape (server/schemas.py vocabulary)."""
    r = record.result
    return {
        "ttft_ms": round(r.ttft_s * 1e3, 3),
        "decode_tok_per_s": r.decode_tok_per_s,
        "prefill_tokens": record.config.prompt_tokens,
        "completion_tokens": r.n_decode_tokens,
    }
