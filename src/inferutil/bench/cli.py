# src/inferutil/bench/cli.py
from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime

from ..model import QWEN3_235B
from ..hardware import GPUS, Cluster
from .config import BenchConfig
from .engine import MockEngine
from .runner import run_benchmark
from .telemetry import NvmlTelemetry, NullTelemetry
from .store import (write_run, load_run, load_latest, result_to_x_summary, RunRecord)
from .report import format_result, format_compare
from .gate import Thresholds, evaluate

DEFAULT_RESULTS_DIR = "results"


def _build_config(args) -> BenchConfig:
    return BenchConfig(
        name=args.name, plan=args.plan, dtype_bytes=args.dtype,
        kv_dtype_bytes=args.kv_dtype, tp=args.tp, ep=args.ep,
        prompt_tokens=args.prompt, decode_tokens=args.decode)


def _cmd_run(args) -> None:
    cluster = Cluster(gpu=GPUS[args.gpu], n_gpus=args.n_gpus)
    config = _build_config(args)
    engine = MockEngine(QWEN3_235B, cluster, efficiency=args.efficiency,
                        jitter=args.jitter, seed=config.seed)
    tele = NvmlTelemetry()
    if not tele.available:
        tele = NullTelemetry()
    result = run_benchmark(engine, config, QWEN3_235B, cluster, telemetry=tele)
    runid = datetime.now().strftime("%Y%m%d-%H%M%S")     # CLI-only clock (run identity)
    driver_version = "unknown"
    try:
        if isinstance(tele, NvmlTelemetry) and tele.available:
            import pynvml
            v = pynvml.nvmlSystemGetDriverVersion()
            driver_version = v.decode() if isinstance(v, bytes) else str(v)
    except Exception:
        pass
    record = RunRecord(runid=runid, config=config,
                       env={"gpu": cluster.gpu.name, "n_gpus": cluster.n_gpus,
                            "host": socket.gethostname(),
                            "driver_version": driver_version},
                       result=result)
    path = write_run(record, args.results_dir)
    if args.json:
        print(json.dumps(result_to_x_summary(record), indent=2))
    else:
        print(format_result(record))
        print(f"\nsaved -> {path}")


def _resolve(args) -> RunRecord:
    if args.runid in (None, "latest"):
        rec = load_latest(args.name, args.results_dir)
        if rec is None:
            raise SystemExit(f"no runs for '{args.name}' in {args.results_dir}")
        return rec
    path = os.path.join(args.results_dir, args.name, args.runid + ".json")
    try:
        return load_run(path)
    except FileNotFoundError:
        raise SystemExit(f"no run '{args.runid}' for '{args.name}' in {args.results_dir}")


def _cmd_report(args) -> None:
    rec = _resolve(args)
    print(json.dumps(result_to_x_summary(rec), indent=2) if args.json
          else format_result(rec))


def _cmd_gate(args) -> None:
    rec = _resolve(args)
    th = Thresholds(min_decode_tok_per_s=args.min_tok_s, max_ttft_s=args.max_ttft_s,
                    min_pct_of_floor=args.min_pct_floor, min_quality_match=args.min_quality)
    g = evaluate(rec.result, th)
    if g.passed:
        print(f"GATE PASS  [{rec.runid}] {rec.config.name}")
    else:
        print(f"GATE FAIL  [{rec.runid}] {rec.config.name}")
        for f in g.failures:
            print(f"  - {f}")
        raise SystemExit(1)


def _cmd_compare(args) -> None:
    for runid in (args.a, args.b):
        path = os.path.join(args.results_dir, args.name, runid + ".json")
        if not os.path.exists(path):
            raise SystemExit(f"no run '{runid}' for '{args.name}' in {args.results_dir}")
    a = load_run(os.path.join(args.results_dir, args.name, args.a + ".json"))
    b = load_run(os.path.join(args.results_dir, args.name, args.b + ".json"))
    print(format_compare(a, b))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="inferutil.bench",
                                 description="Offline B=1 decode benchmark harness")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark and store it")
    r.add_argument("--name", default="default")
    r.add_argument("--gpu", default="H100-SXM-80GB", choices=list(GPUS))
    r.add_argument("--n-gpus", type=int, default=8)
    r.add_argument("--plan", default="hybrid", choices=["tp", "ep", "hybrid"])
    r.add_argument("--dtype", type=int, default=2, choices=[1, 2])
    r.add_argument("--kv-dtype", type=int, default=2, choices=[1, 2])
    r.add_argument("--tp", type=int, default=2)
    r.add_argument("--ep", type=int, default=8)
    r.add_argument("--prompt", type=int, default=512)
    r.add_argument("--decode", type=int, default=128)
    r.add_argument("--efficiency", type=float, default=1.0,
                   help="MockEngine BW efficiency (1.0 = analytical floor)")
    r.add_argument("--jitter", type=float, default=0.0)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=_cmd_run)

    rp = sub.add_parser("report", help="print a stored run (default: latest)")
    rp.add_argument("--name", default="default")
    rp.add_argument("runid", nargs="?", default="latest")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=_cmd_report)

    gp = sub.add_parser("gate", help="pass/fail a stored run against thresholds")
    gp.add_argument("--name", default="default")
    gp.add_argument("runid", nargs="?", default="latest")
    gp.add_argument("--min-tok-s", type=float, default=None)
    gp.add_argument("--max-ttft-s", type=float, default=None)
    gp.add_argument("--min-pct-floor", type=float, default=None)
    gp.add_argument("--min-quality", type=float, default=None)
    gp.set_defaults(func=_cmd_gate)

    cp = sub.add_parser("compare", help="diff two stored runs")
    cp.add_argument("--name", default="default")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.set_defaults(func=_cmd_compare)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
