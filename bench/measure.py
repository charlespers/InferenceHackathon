#!/usr/bin/env python3
"""End-to-end B=1 latency over the OpenAI /v1/chat/completions SSE contract.

Reuses the exact contract the UI streams (incl. optional x_telemetry / x_summary), so the same
endpoint serves the demo and the benchmark. Zero third-party deps (stdlib only).

    python bench/measure.py --base http://localhost:8000 --ctx 2048 --decode 128 [--engine conifer]
"""
import argparse, json, time, urllib.request

def stream_once(base, prompt, decode, engine=None, model="qwen3-235b-a22b"):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": decode, "temperature": 0.0, "stream": True}
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
                tokens=n, server_summary=summary)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--ctx", type=int, default=2048, help="approx prompt tokens (padded)")
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()
    prompt = ("Summarize the following. " + "context " * max(0, a.ctx)).strip()
    for _ in range(a.warmup):
        stream_once(a.base, "warm up", 8, a.engine)
    res = stream_once(a.base, prompt, a.decode, a.engine)
    print(json.dumps(res, indent=2))
    print(f"\nTTFT {res['ttft_ms']:.1f} ms | TPOT {res['tpot_ms']:.2f} ms | "
          f"decode {res['decode_tok_s']:.1f} tok/s | {res['tokens']} tokens")
    print("Feed TPOT into bench/roofline.py --tpot-ms to get the dominant term.")
