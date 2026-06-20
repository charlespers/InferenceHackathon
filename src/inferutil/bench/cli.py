# src/inferutil/bench/cli.py
from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime

import sys
from dataclasses import replace

from ..model import QWEN3_235B
from ..hardware import GPUS, Cluster
from .config import BenchConfig
from .engine import MockEngine
from .runner import run_benchmark
from .telemetry import NvmlTelemetry, NullTelemetry
from .store import (write_run, load_run, load_latest, load_all,
                    result_to_x_summary, RunRecord, export_csv, export_jsonl,
                    export_markdown)
from .report import (format_result, format_compare, format_diagnosis,
                     format_sweep, format_spec_floor, format_plan)
from .spec_model import spec_sweep
from ..speculative import memory_feasibility
from .attribution import diagnose
from .levers import recommend
from .manifest import build_manifest
from .sweep import (depth_sweep, config_sweep, quant_grid, layout_grid, full_grid,
                    realized_efficiency)
from .gate import Thresholds, evaluate, regression_gate

DEFAULT_RESULTS_DIR = "results"


def _build_config(args) -> BenchConfig:
    return BenchConfig(
        name=args.name, plan=args.plan, dtype_bytes=args.dtype,
        kv_dtype_bytes=args.kv_dtype, tp=args.tp, ep=args.ep,
        prompt_tokens=args.prompt, decode_tokens=args.decode, repeats=args.repeats)


def _unique_runid(results_dir: str, name: str) -> str:
    """Second-resolution timestamp, suffixed -1/-2/... if a run from the same
    second already exists (so rapid successive runs never clobber each other)."""
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = os.path.join(results_dir, name)
    cand, i = base, 1
    while os.path.exists(os.path.join(d, cand + ".json")):
        cand, i = f"{base}-{i}", i + 1
    return cand


def _cluster_from_env(rec: RunRecord) -> Cluster:
    gpu = rec.env.get("gpu") or "H100-SXM-80GB"
    if gpu not in GPUS:
        gpu = "H100-SXM-80GB"
    return Cluster(gpu=GPUS[gpu], n_gpus=rec.env.get("n_gpus") or 8)


def _measured_cluster(cluster: Cluster, peak_bw_gbs) -> Cluster:
    """Replace the spec-sheet HBM bandwidth with a measured GB/s (e.g. from
    kernels/k5_microbench) so MBU and the floor use the real ceiling, not the
    datasheet — Conifer-style 'measure, don't trust the spec'."""
    if not peak_bw_gbs:
        return cluster
    return Cluster(gpu=replace(cluster.gpu, hbm_bw=peak_bw_gbs * 1e9),
                   n_gpus=cluster.n_gpus)


def _cmd_run(args) -> None:
    cluster = _measured_cluster(Cluster(gpu=GPUS[args.gpu], n_gpus=args.n_gpus),
                                args.peak_bw_gbs)
    config = _build_config(args)
    engine = MockEngine(QWEN3_235B, cluster, efficiency=args.efficiency,
                        jitter=args.jitter, seed=config.seed)
    tele = NvmlTelemetry()
    if not tele.available:
        tele = NullTelemetry()
    result = run_benchmark(engine, config, QWEN3_235B, cluster, telemetry=tele)
    runid = _unique_runid(args.results_dir, config.name)   # CLI-only clock (run identity)
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
    peak_bw_measured = args.peak_bw_gbs * 1e9 if args.peak_bw_gbs else None
    manifest = build_manifest(QWEN3_235B, cluster, config, cli=" ".join(sys.argv),
                              runid=runid, driver_version=driver_version,
                              peak_bw_measured=peak_bw_measured)
    mpath = os.path.join(args.results_dir, config.name, runid + ".manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if args.json:
        print(json.dumps(result_to_x_summary(record), indent=2))
    else:
        print(format_result(record))
        print(f"\nsaved -> {path}\nmanifest -> {mpath}")


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
    failures = list(evaluate(rec.result, th).failures)
    if args.baseline:
        base_path = os.path.join(args.results_dir, args.name, args.baseline + ".json")
        try:
            base = load_run(base_path)
        except FileNotFoundError:
            raise SystemExit(f"no baseline run '{args.baseline}' for '{args.name}'")
        failures += list(regression_gate(base.result, rec.result).failures)
    if not failures:
        print(f"GATE PASS  [{rec.runid}] {rec.config.name}")
    else:
        print(f"GATE FAIL  [{rec.runid}] {rec.config.name}")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)


def _cmd_diagnose(args) -> None:
    rec = _resolve(args)
    cluster = _cluster_from_env(rec)
    b = diagnose(rec.result)
    levers = recommend(QWEN3_235B, cluster, rec.config, bottleneck=b,
                       min_speedup=args.min_speedup)
    print(format_diagnosis(rec, levers))


def _cmd_plan(args) -> None:
    cluster = _measured_cluster(Cluster(gpu=GPUS[args.gpu], n_gpus=args.n_gpus),
                                args.peak_bw_gbs)
    config = BenchConfig(name="plan", plan=args.plan, dtype_bytes=args.dtype,
                         kv_dtype_bytes=args.kv_dtype, tp=args.tp, ep=args.ep,
                         prompt_tokens=args.prompt, decode_tokens=args.decode, repeats=1)
    # Analytical result (efficiency=1.0 -> floor; pass measured e for a calibrated plan).
    eng = MockEngine(QWEN3_235B, cluster, efficiency=args.efficiency, jitter=0.0)
    result = run_benchmark(eng, config, QWEN3_235B, cluster)
    rec = RunRecord(runid="analytical", config=config,
                    env={"gpu": cluster.gpu.name, "n_gpus": cluster.n_gpus},
                    result=result)
    b = diagnose(result)
    levers = recommend(QWEN3_235B, cluster, config, bottleneck=b)
    best = config_sweep(QWEN3_235B, cluster, full_grid(config, cluster.n_gpus))[0]
    print(format_plan(rec, b, levers, best))


def _cmd_calibrate(args) -> None:
    cluster = _measured_cluster(Cluster(gpu=GPUS[args.gpu], n_gpus=args.n_gpus),
                                args.peak_bw_gbs)
    config = BenchConfig(name="cal", plan=args.plan, dtype_bytes=args.dtype,
                         kv_dtype_bytes=args.kv_dtype, tp=args.tp, ep=args.ep,
                         prompt_tokens=args.prompt, decode_tokens=args.decode)
    e, floor = realized_efficiency(QWEN3_235B, cluster, config, args.measured_tok_s)
    print(f"== CALIBRATE  plan={config.plan} w{config.dtype_bytes}b "
          f"kv{config.kv_dtype_bytes}b tp={config.tp} ep={config.ep} ctx={config.seq_len} ==")
    print(f"  measured     : {args.measured_tok_s:.1f} tok/s")
    print(f"  analytical floor (e=1.0): {floor:.1f} tok/s")
    if e is not None:
        print(f"  realized whole-model efficiency e = {e:.3f}  "
              f"({e*100:.1f}% of floor)")
        print(f"  -> feed it in:  sweep/plan --efficiency {e:.3f}")


def _cmd_sweep(args) -> None:
    cluster = _measured_cluster(Cluster(gpu=GPUS[args.gpu], n_gpus=args.n_gpus),
                                args.peak_bw_gbs)
    config = BenchConfig(name="sweep", plan=args.plan, dtype_bytes=args.dtype,
                         kv_dtype_bytes=args.kv_dtype, tp=args.tp, ep=args.ep,
                         prompt_tokens=args.prompt, decode_tokens=args.decode)
    e = args.efficiency
    etag = "floor (e=1.0)" if e >= 1.0 else f"calibrated e={e:g}"
    if args.depths:
        depths = [int(x) for x in args.depths.split(",")]
        pts = depth_sweep(QWEN3_235B, cluster, config, depths, efficiency=e)
        title = (f"DEPTH SWEEP [{etag}]  plan={config.plan} w{config.dtype_bytes}b "
                 f"kv{config.kv_dtype_bytes}b tp={config.tp} ep={config.ep}")
    elif args.full:
        pts = config_sweep(QWEN3_235B, cluster, full_grid(config, cluster.n_gpus), efficiency=e)
        title = f"FULL SWEEP (quant x layout) [{etag}]  plan={config.plan} ctx={config.seq_len}"
    elif args.layout:
        pts = config_sweep(QWEN3_235B, cluster, layout_grid(config, cluster.n_gpus), efficiency=e)
        title = (f"LAYOUT SWEEP (tp x ep) [{etag}]  plan={config.plan} "
                 f"w{config.dtype_bytes}b kv{config.kv_dtype_bytes}b ctx={config.seq_len}")
    else:
        pts = config_sweep(QWEN3_235B, cluster, quant_grid(config), efficiency=e)
        title = f"CONFIG SWEEP (quant grid) [{etag}]  plan={config.plan} ctx={config.seq_len}"
    print(format_sweep(pts, n_gpus=cluster.n_gpus, usd_per_gpu_hr=args.gpu_hr,
                       title=title))


def _cmd_spec(args) -> None:
    rows = spec_sweep(args.accept, args.floor, ks=tuple(args.ks), ns=tuple(args.drafters))
    feasibility = memory_feasibility(draft_model_gb=args.draft_gb,
                                     use_fp8_target=not args.bf16_target)
    print(format_spec_floor(rows, feasibility, accept=args.accept, floor=args.floor,
                            base_tok_s=args.base_tok_s))


def _cmd_export(args) -> None:
    records = load_all(args.name, args.results_dir)
    if not records:
        raise SystemExit(f"no runs for '{args.name}' in {args.results_dir}")
    writer = {"csv": export_csv, "jsonl": export_jsonl, "md": export_markdown}[args.format]
    writer(records, args.out)
    print(f"exported {len(records)} run(s) -> {args.out}")


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
    r.add_argument("--repeats", type=int, default=5,
                   help="full-run repeats for variance/confidence intervals")
    r.add_argument("--efficiency", type=float, default=1.0,
                   help="MockEngine BW efficiency (1.0 = analytical floor)")
    r.add_argument("--jitter", type=float, default=0.0)
    r.add_argument("--peak-bw-gbs", type=float, default=None,
                   help="measured HBM GB/s per GPU (e.g. from k5_microbench); "
                        "overrides the spec sheet for MBU + floor")
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
    gp.add_argument("--baseline", default=None,
                    help="runid to compare against; fails on a significant throughput regression")
    gp.set_defaults(func=_cmd_gate)

    dp = sub.add_parser("diagnose",
                        help="diagnose the bottleneck and rank next levers")
    dp.add_argument("--name", default="default")
    dp.add_argument("runid", nargs="?", default="latest")
    dp.add_argument("--min-speedup", type=float, default=1.02,
                    help="drop levers predicting less than this speedup")
    dp.set_defaults(func=_cmd_diagnose)

    pl = sub.add_parser("plan",
                        help="one-shot decision plan: bottleneck + ranked wins + best config (no GPU)")
    pl.add_argument("--gpu", default="H100-SXM-80GB", choices=list(GPUS))
    pl.add_argument("--n-gpus", type=int, default=8)
    pl.add_argument("--plan", default="hybrid", choices=["tp", "ep", "hybrid"])
    pl.add_argument("--dtype", type=float, default=1.0, help="2=bf16, 1=fp8, 0.5=int4")
    pl.add_argument("--kv-dtype", type=int, default=2, choices=[1, 2])
    pl.add_argument("--tp", type=int, default=2)
    pl.add_argument("--ep", type=int, default=8)
    pl.add_argument("--prompt", type=int, default=512)
    pl.add_argument("--decode", type=int, default=128)
    pl.add_argument("--efficiency", type=float, default=1.0,
                    help="realized whole-model efficiency (1.0=floor; pass measured e for a calibrated plan)")
    pl.add_argument("--peak-bw-gbs", type=float, default=None,
                    help="measured HBM GB/s per GPU; overrides the spec sheet")
    pl.set_defaults(func=_cmd_plan)

    ca = sub.add_parser("calibrate",
                        help="back out realized whole-model efficiency e from a measured tok/s")
    ca.add_argument("--measured-tok-s", type=float, required=True)
    ca.add_argument("--gpu", default="H100-SXM-80GB", choices=list(GPUS))
    ca.add_argument("--n-gpus", type=int, default=8)
    ca.add_argument("--plan", default="hybrid", choices=["tp", "ep", "hybrid"])
    ca.add_argument("--dtype", type=float, default=1.0)
    ca.add_argument("--kv-dtype", type=int, default=2, choices=[1, 2])
    ca.add_argument("--tp", type=int, default=2)
    ca.add_argument("--ep", type=int, default=8)
    ca.add_argument("--prompt", type=int, default=512)
    ca.add_argument("--decode", type=int, default=128)
    ca.add_argument("--peak-bw-gbs", type=float, default=None)
    ca.set_defaults(func=_cmd_calibrate)

    sw = sub.add_parser("sweep",
                        help="analytical depth/config sweep (no GPU required)")
    sw.add_argument("--gpu", default="H100-SXM-80GB", choices=list(GPUS))
    sw.add_argument("--n-gpus", type=int, default=8)
    sw.add_argument("--plan", default="hybrid", choices=["tp", "ep", "hybrid"])
    sw.add_argument("--dtype", type=float, default=1.0, help="2=bf16, 1=fp8, 0.5=int4")
    sw.add_argument("--kv-dtype", type=int, default=2, choices=[1, 2])
    sw.add_argument("--tp", type=int, default=2)
    sw.add_argument("--ep", type=int, default=8)
    sw.add_argument("--prompt", type=int, default=512)
    sw.add_argument("--decode", type=int, default=128)
    sw.add_argument("--depths", default=None,
                    help="comma list e.g. 512,4096,32768 -> depth (KV-decay) sweep")
    sw.add_argument("--layout", action="store_true",
                    help="sweep tp x ep layouts at fixed quant (pure-speed search)")
    sw.add_argument("--full", action="store_true",
                    help="sweep the full quant x layout grid")
    sw.add_argument("--gpu-hr", type=float, default=3.0, help="$/GPU-hr for $/Mtok")
    sw.add_argument("--efficiency", type=float, default=1.0,
                    help="realized whole-model efficiency (1.0=floor; pass measured e for calibrated tok/s)")
    sw.add_argument("--peak-bw-gbs", type=float, default=None,
                    help="measured HBM GB/s per GPU; overrides the spec sheet")
    sw.set_defaults(func=_cmd_sweep)

    spc = sub.add_parser("spec",
                         help="floor-aware spec-decode sizing (drafters x k) + HBM feasibility")
    spc.add_argument("--accept", type=float, default=0.7, help="per-token acceptance rate")
    spc.add_argument("--floor", type=float, default=0.86,
                     help="floor fraction F of TPOT (overhead+comms); high=floor-bound -> big trees win")
    spc.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8],
                     help="draft lengths to sweep")
    spc.add_argument("--drafters", type=int, nargs="+", default=[1, 2, 4],
                     help="drafter counts to sweep")
    spc.add_argument("--draft-gb", type=float, default=3.4, help="one draft model size (GB)")
    spc.add_argument("--bf16-target", action="store_true",
                     help="target weights in bf16 (default fp8) -> less HBM headroom")
    spc.add_argument("--base-tok-s", type=float, default=None,
                     help="baseline decode tok/s to project absolute speedup")
    spc.set_defaults(func=_cmd_spec)

    ep = sub.add_parser("export", help="export stored runs to csv/jsonl")
    ep.add_argument("--name", default="default")
    ep.add_argument("--format", choices=["csv", "jsonl", "md"], default="csv")
    ep.add_argument("--out", required=True)
    ep.set_defaults(func=_cmd_export)

    cp = sub.add_parser("compare", help="diff two stored runs")
    cp.add_argument("--name", default="default")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.set_defaults(func=_cmd_compare)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
