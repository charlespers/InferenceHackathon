#!/usr/bin/env python3
"""Drive a REAL vLLM OpenAI server through the team's inferutil.bench harness.

The harness (src/inferutil/bench) only wires MockEngine. This adds a VllmEngine that
implements the same Engine protocol (reset/prefill/decode_step) by streaming the vLLM
/v1/chat/completions endpoint, so a real run reports the harness's headline metrics —
TPOT, achieved HBM bandwidth, and **% of the analytical floor** — in the team's store
and report format. Non-invasive: a standalone driver that imports the package, no edits
to inferutil. Stdlib-only client (matches the harness's dependency-light style).

  python3 bench/vllm_bench.py --base-url http://localhost:8001 --model qwen3-235b-a22b \
      --name fp8-tp8 --plan tp --tp 8 --ep 1 --dtype 1 --kv-dtype 2 \
      --prompt 512 --decode 128 --warmup 8 --src /alloc/data/InferenceHackathon/src
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.request


def _unique_runid(results_dir, name):
    """Collision-proof runid (suffix -1/-2 on a same-second clash) so a precious
    on-box run never silently overwrites another."""
    base = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(results_dir, name)
    c, i = base, 1
    while os.path.exists(os.path.join(d, c + ".json")):
        c, i = f"{base}-{i}", i + 1
    return c


def _import_inferutil(src_path):
    if src_path and src_path not in sys.path:
        sys.path.insert(0, src_path)
    from inferutil.model import QWEN3_235B
    from inferutil.hardware import GPUS, Cluster
    from inferutil.bench.config import BenchConfig
    from inferutil.bench.runner import run_benchmark
    from inferutil.bench.engine import PrefillResult, DecodeStep, ExpertRoute
    from inferutil.bench.store import write_run, RunRecord, result_to_x_summary
    from inferutil.bench.report import format_result
    from inferutil.bench.manifest import build_manifest
    try:
        from inferutil.bench.telemetry import NvmlTelemetry, NullTelemetry
    except Exception:
        NvmlTelemetry = NullTelemetry = None
    return dict(QWEN3_235B=QWEN3_235B, GPUS=GPUS, Cluster=Cluster, BenchConfig=BenchConfig,
                run_benchmark=run_benchmark, PrefillResult=PrefillResult, DecodeStep=DecodeStep,
                ExpertRoute=ExpertRoute, write_run=write_run, RunRecord=RunRecord,
                result_to_x_summary=result_to_x_summary, format_result=format_result,
                build_manifest=build_manifest,
                NvmlTelemetry=NvmlTelemetry, NullTelemetry=NullTelemetry)


def make_vllm_engine(M, base_url, model, prompt_tokens, max_new_tokens, temperature=0.0):
    PrefillResult, DecodeStep, ExpertRoute = M["PrefillResult"], M["DecodeStep"], M["ExpertRoute"]

    class VllmEngine:
        """Engine protocol over a streaming vLLM OpenAI server (one request per prefill).
        prefill() reads to the first token (TTFT); each decode_step() pulls the next
        streamed token and times the inter-token gap. Reads optional x_telemetry routing."""
        def __init__(self):
            self._resp = None; self._last = None; self._idx = 0; self._pending = None

        def reset(self, *, plan, dtype_bytes, kv_dtype_bytes, tp, ep, seq_len):
            self._resp = None; self._last = None; self._idx = 0; self._pending = None

        def _open(self):
            prompt = ("benchmark context " * max(1, prompt_tokens // 2)).strip()
            body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_new_tokens, "min_tokens": max_new_tokens,
                    "ignore_eos": True, "temperature": temperature, "stream": True}
            req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            self._resp = urllib.request.urlopen(req, timeout=300)

        def _next(self):
            for raw in self._resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return None
                obj = json.loads(payload)
                ch = obj.get("choices") or [{}]
                piece = (ch[0].get("delta") or {}).get("content")
                if piece:
                    routes = ()
                    tel = obj.get("x_telemetry")
                    if tel and tel.get("experts"):
                        routes = tuple(ExpertRoute(layer=e["layer"], expert_id=e["expert_id"],
                                                   gpu=e["gpu"]) for e in tel["experts"])
                    return routes
            return None

        def prefill(self, token_ids):
            t0 = time.perf_counter()
            self._open()
            self._pending = self._next()           # first content token => TTFT
            ttft = time.perf_counter() - t0
            self._last = time.perf_counter()
            return PrefillResult(n_prompt_tokens=len(token_ids), seconds=ttft,
                                 first_token_seconds=0.0)

        def decode_step(self):
            i = self._idx; self._idx += 1
            routes = self._next()
            now = time.perf_counter()
            dt = now - (self._last if self._last is not None else now)
            self._last = now
            return DecodeStep(index=i, seconds=dt, routes=routes or ())

    return VllmEngine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8001")
    ap.add_argument("--model", default="qwen3-235b-a22b")
    ap.add_argument("--name", default="vllm-fp8-tp8")
    ap.add_argument("--plan", default="tp", choices=["tp", "ep", "hybrid"])
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--dtype", type=int, default=1, help="weight bytes: 1=fp8, 2=bf16")
    ap.add_argument("--kv-dtype", type=int, default=2, help="kv bytes: 1=fp8/int8, 2=fp16")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3,
                    help="full-run repeats for CIs (each is a fresh stream; "
                         "mind the slot budget)")
    ap.add_argument("--gpu", default="H100-SXM-80GB")
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--src", default="/alloc/data/InferenceHackathon/src")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    M = _import_inferutil(a.src)
    cluster = M["Cluster"](gpu=M["GPUS"][a.gpu], n_gpus=a.n_gpus)
    config = M["BenchConfig"](name=a.name, plan=a.plan, dtype_bytes=a.dtype,
                              kv_dtype_bytes=a.kv_dtype, tp=a.tp, ep=a.ep,
                              prompt_tokens=a.prompt, decode_tokens=a.decode,
                              warmup_steps=a.warmup, repeats=a.repeats)
    needed = a.warmup + a.decode + 4               # prefill(1)+warmup+(decode-1) content tokens
    engine = make_vllm_engine(M, a.base_url, a.model, a.prompt, needed)

    tele = None
    if M["NvmlTelemetry"] is not None:
        t = M["NvmlTelemetry"]()
        tele = t if getattr(t, "available", False) else M["NullTelemetry"]()

    result = M["run_benchmark"](engine, config, M["QWEN3_235B"], cluster, telemetry=tele)
    runid = _unique_runid(a.results_dir, config.name)
    rec = M["RunRecord"](runid=runid, config=config,
                         env={"gpu": cluster.gpu.name, "n_gpus": a.n_gpus,
                              "engine": "vllm", "base_url": a.base_url}, result=result)
    path = M["write_run"](rec, a.results_dir)
    # reproducibility manifest (host, git commit, model hash, hardware) — real runs
    # are exactly where "which commit / which driver produced this" matters.
    manifest = M["build_manifest"](M["QWEN3_235B"], cluster, config,
                                   cli=" ".join(sys.argv), runid=runid)
    mpath = os.path.join(a.results_dir, config.name, runid + ".manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if a.json:
        print(json.dumps(M["result_to_x_summary"](rec), indent=2))
    else:
        print(M["format_result"](rec))
        print(f"\nsaved -> {path}\nmanifest -> {mpath}")


if __name__ == "__main__":
    main()
