#!/usr/bin/env python3
"""End-to-end B=1 latency over the OpenAI /v1/chat/completions SSE contract.

Reuses the exact contract the UI streams (incl. optional x_telemetry / x_summary), so the same
endpoint serves the demo and the benchmark. Zero third-party deps (stdlib only).

Repeats the request N times (warmup dropped) and reports each latency metric as a
distribution — mean, p50/p90/p95/p99, and a Student-t 95% CI on the mean — so a
single noisy stream never decides anything. Feed the reported TPOT into
bench/roofline.py for MFU/MBU + the dominant term.

    python bench/measure.py --base http://localhost:8000 --ctx 2048 --decode 128 --repeats 5
"""
import argparse, json, time, urllib.request
from math import sqrt

# Two-sided 95% Student-t critical values by df (=n-1); >30 falls back to z=1.96.
# Tabulated through df=30 to stay identical to inferutil.bench.stats.t95.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
        22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
        29: 2.045, 30: 2.042}


def _t95(df):
    if df <= 0:
        return float("inf")
    if df in _T95:
        return _T95[df]
    return 1.96 if df > 30 else _T95[max(k for k in _T95 if k <= df)]


def _pct(sv, p):
    if not sv:
        return None
    if len(sv) == 1:
        return sv[0]
    k = (len(sv) - 1) * p
    f = int(k); c = min(f + 1, len(sv) - 1)
    return sv[f] if f == c else sv[f] + (sv[c] - sv[f]) * (k - f)


def summarize(xs):
    """mean / std / p50,p90,p95,p99 / 95% CI for a list of samples."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {"n": 0}
    mean = sum(xs) / n
    sv = sorted(xs)
    out = {"n": n, "mean": mean, "p50": _pct(sv, .5), "p90": _pct(sv, .9),
           "p95": _pct(sv, .95), "p99": _pct(sv, .99), "min": sv[0], "max": sv[-1]}
    if n > 1:
        std = sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
        half = _t95(n - 1) * std / sqrt(n)
        out.update(std=std, ci95_lo=mean - half, ci95_hi=mean + half)
    return out


def stream_once(base, prompt, decode, engine=None, model="qwen3-235b-a22b", temperature=0.0):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": decode, "temperature": temperature, "stream": True}
    if engine:
        body["engine"] = engine  # mock-only profile selector; ignored by a real OpenAI server
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    tok_times = []
    summary = None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if "choices" in obj and obj["choices"][0].get("delta", {}).get("content"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                tok_times.append(now)
            if "x_summary" in obj:
                summary = obj["x_summary"]
    # inter-token latencies (wall clock)
    inter = [(tok_times[i] - tok_times[i-1]) * 1000 for i in range(1, len(tok_times))]
    tpot_ms = sum(inter) / len(inter) if inter else 0.0
    n = len(tok_times)
    decode_tok_s = (n - 1) / (tok_times[-1] - tok_times[0]) if n > 1 else 0.0
    return dict(ttft_ms=(ttft or 0) * 1000, tpot_ms=tpot_ms, decode_tok_s=decode_tok_s,
                tokens=n, inter_ms=inter, server_summary=summary)


def _fmt(s):
    if s.get("n", 0) < 2:
        return f"{s.get('mean', 0):.2f}"
    return (f"{s['mean']:.2f} (p50 {s['p50']:.2f} / p95 {s['p95']:.2f}, "
            f"95% CI {s['ci95_lo']:.2f}-{s['ci95_hi']:.2f}, n={s['n']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--ctx", type=int, default=2048, help="approx prompt tokens (padded)")
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--model", default="qwen3-235b-a22b", help="served model name to request")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5, help="measured repeats (warmup dropped)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature (0=greedy, default; use 0.7 for product-like spec accept-rate, spec-in-production.md)")
    a = ap.parse_args()
    prompt = ("Summarize the following. " + "context " * max(0, a.ctx)).strip()
    for _ in range(a.warmup):
        stream_once(a.base, "warm up", 8, a.engine, a.model)
    runs = [stream_once(a.base, prompt, a.decode, a.engine, a.model, a.temperature)
            for _ in range(max(1, a.repeats))]
    ttft = summarize([r["ttft_ms"] for r in runs])
    tpot = summarize([g for r in runs for g in r["inter_ms"]])   # pooled inter-token gaps
    dtoks = summarize([r["decode_tok_s"] for r in runs])
    out = {"ttft_ms": ttft, "tpot_ms": tpot, "decode_tok_s": dtoks,
           "tokens": runs[-1]["tokens"], "repeats": len(runs),
           "server_summary": runs[-1]["server_summary"]}
    print(json.dumps(out, indent=2))
    # Keep a bare "TPOT <mean>" token on the summary line (run_bench*.sh greps it).
    print(f"\nTTFT {_fmt(ttft)} ms | TPOT {tpot.get('mean', 0):.2f} ms ({_fmt(tpot)}) | "
          f"decode {_fmt(dtoks)} tok/s | {out['tokens']} tokens, {out['repeats']} repeats")
    print("Feed TPOT into bench/roofline.py --tpot-ms to get MFU/MBU + the dominant term.")
