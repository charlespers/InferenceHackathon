from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict

from .config import BenchConfig
from .metrics import BenchResult, TelemetrySummary, MeasuredBreakdown


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
    mb = res.pop("measured_breakdown", None)
    measured = MeasuredBreakdown(**mb) if mb else None
    qd = res.pop("quality", None)
    from .quality import QualityResult
    quality = QualityResult(**qd) if qd else None
    eff = res.pop("efficiency", None)
    from .efficiency import Efficiency
    efficiency = Efficiency(**eff) if eff else None
    # `latency` stays a plain dict (of Stat dicts) — round-trips as-is via **res.
    result = BenchResult(telemetry=TelemetrySummary(**tele),
                         measured_breakdown=measured, quality=quality,
                         efficiency=efficiency, **res)
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
    files = sorted(fn for fn in os.listdir(d)
                   if fn.endswith(".json") and not fn.endswith(".manifest.json"))
    return [load_run(os.path.join(d, fn)) for fn in files]


def load_latest(name: str, results_dir: str):
    runs = load_all(name, results_dir)
    return runs[-1] if runs else None


EM_DASH = "—"  # rendered for unmeasured values; never a fabricated 0

# Tidy long-format columns: one row per run.
CSV_COLUMNS = [
    "runid", "name", "plan", "dtype_bytes", "kv_dtype_bytes", "tp", "ep",
    "prompt_tokens", "decode_tokens", "seq_len", "n_repeats",
    "decode_tok_per_s", "decode_tok_per_s_std", "ttft_ms_p50", "tpot_p50_ms",
    "tpot_p95_ms", "e2e_ms_p50", "mbu_decode", "mfu_prefill", "mfu_decode",
    "ai_decode", "roofline_ridge", "regime_decode", "kv_byte_share",
    "pct_of_peak_bw", "pct_of_floor", "energy_j_per_token", "util_imbalance",
]


def _stat(latency: dict | None, key: str, field: str):
    if not latency or key not in latency or latency[key] is None:
        return None
    return latency[key].get(field)


def record_row(record: RunRecord) -> dict:
    """Flatten a run into the tidy CSV vocabulary. None stays None here; the CSV
    writer renders it as an em-dash so unmeasured never reads as a real 0."""
    r, c = record.result, record.config
    eff = r.efficiency
    t = r.telemetry
    ms = 1e3
    row = {
        "runid": record.runid, "name": c.name, "plan": c.plan,
        "dtype_bytes": c.dtype_bytes, "kv_dtype_bytes": c.kv_dtype_bytes,
        "tp": c.tp, "ep": c.ep, "prompt_tokens": c.prompt_tokens,
        "decode_tokens": c.decode_tokens, "seq_len": c.seq_len,
        "n_repeats": r.n_repeats,
        "decode_tok_per_s": r.decode_tok_per_s,
        "decode_tok_per_s_std": r.decode_tok_per_s_std,
        "ttft_ms_p50": _stat(r.latency, "ttft_ms", "p50"),
        "tpot_p50_ms": r.tpot_p50_s * ms,
        "tpot_p95_ms": r.tpot_p95_s * ms,
        "e2e_ms_p50": _stat(r.latency, "e2e_ms", "p50"),
        "mbu_decode": eff.mbu_decode if eff else None,
        "mfu_prefill": eff.mfu_prefill if eff else None,
        "mfu_decode": eff.mfu_decode if eff else None,
        "ai_decode": eff.ai_decode if eff else None,
        "roofline_ridge": eff.roofline_ridge if eff else None,
        "regime_decode": eff.regime_decode if eff else None,
        "kv_byte_share": eff.kv_byte_share if eff else None,
        "pct_of_peak_bw": r.pct_of_peak_bw, "pct_of_floor": r.pct_of_floor,
        "energy_j_per_token": t.energy_j_per_token if t.available else None,
        "util_imbalance": t.util_imbalance if t.available else None,
    }
    return row


def export_csv(records: list, path: str) -> str:
    """Write runs as a tidy long-format CSV (one row per run)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for rec in records:
            row = record_row(rec)
            w.writerow({k: (EM_DASH if row.get(k) is None else row[k])
                        for k in CSV_COLUMNS})
    return path


def export_jsonl(records: list, path: str) -> str:
    """Write runs as lossless JSONL (one record per line, full samples retained)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_record_to_dict(rec)) + "\n")
    return path


def _md_cell(v) -> str:
    if v is None:
        return EM_DASH
    if isinstance(v, float):
        return f"{v:.4g}"      # compact, human-readable
    return str(v)


def export_markdown(records: list, path: str) -> str:
    """Write runs as a GitHub-flavored Markdown table (shareable in PRs/docs)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(CSV_COLUMNS) + " |\n")
        f.write("| " + " | ".join("---" for _ in CSV_COLUMNS) + " |\n")
        for rec in records:
            row = record_row(rec)
            f.write("| " + " | ".join(_md_cell(row.get(c)) for c in CSV_COLUMNS) + " |\n")
    return path


def result_to_x_summary(record: RunRecord) -> dict:
    """Serialize to the console's x_summary shape (server/schemas.py vocabulary)."""
    r = record.result
    return {
        "ttft_ms": round(r.ttft_s * 1e3, 3),
        "decode_tok_per_s": r.decode_tok_per_s,
        "prefill_tokens": record.config.prompt_tokens,
        "completion_tokens": r.n_decode_tokens,
    }
