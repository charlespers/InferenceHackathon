#!/usr/bin/env python3
"""B=1 latency client for an OpenAI-compatible server (vLLM).

Streams chat completions one at a time (batch size 1), measuring TTFT and
inter-token latency, then reports decode tok/s with run-to-run variance.
Stdlib only (urllib) so it runs in any env. Writes a JSON result.

Example:
    python3 bench_b1_client.py --base http://localhost:8001 \
        --model qwen3-235b-bf16 --prompt-tokens 512 --max-tokens 128 \
        --repeats 10 --warmup 2 --out /alloc/data/vllm_b1_bench.json
"""
import argparse
import json
import statistics
import time
import urllib.request


def stream_once(base, model, prompt_tokens, max_tokens):
    """One B=1 streamed completion. Returns (ttft_s, inter_token_latencies, n_tokens)."""
    prompt = ("benchmark " * prompt_tokens).strip()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    tstamps = []
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            choices = obj.get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                tstamps.append(now)
    itl = [tstamps[i] - tstamps[i - 1] for i in range(1, len(tstamps))]
    return ttft, itl, len(tstamps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--model", default="qwen3-235b-bf16")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--plan", default="tp8")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--engine", default="vllm-0.10.1-eager")
    ap.add_argument("--out", default="/alloc/data/vllm_b1_bench.json")
    a = ap.parse_args()

    for _ in range(a.warmup):
        stream_once(a.base, a.model, a.prompt_tokens, a.max_tokens)

    ttfts, tpots, tps = [], [], []
    for i in range(a.repeats):
        ttft, itl, ntok = stream_once(a.base, a.model, a.prompt_tokens, a.max_tokens)
        if not itl or ttft is None:
            print(f"[{i+1}/{a.repeats}] no tokens streamed, skipping")
            continue
        tpot = statistics.mean(itl)
        ttfts.append(ttft)
        tpots.append(tpot)
        tps.append(1.0 / tpot)
        print(f"[{i+1}/{a.repeats}] TTFT {ttft*1e3:7.1f} ms  "
              f"TPOT {tpot*1e3:6.2f} ms  decode {1.0/tpot:6.1f} tok/s  ({ntok} tok)")

    if not tps:
        raise SystemExit("no successful runs")

    res = {
        "config": {
            "model": a.model, "prompt_tokens": a.prompt_tokens,
            "max_tokens": a.max_tokens, "repeats": a.repeats,
            "plan": a.plan, "dtype": a.dtype, "batch_size": 1, "engine": a.engine,
        },
        "ttft_ms": {"mean": statistics.mean(ttfts) * 1e3,
                    "median": statistics.median(ttfts) * 1e3,
                    "min": min(ttfts) * 1e3, "max": max(ttfts) * 1e3},
        "tpot_ms": {"mean": statistics.mean(tpots) * 1e3,
                    "p50": statistics.median(tpots) * 1e3},
        "decode_tok_per_s": {"mean": statistics.mean(tps),
                             "median": statistics.median(tps),
                             "std": statistics.pstdev(tps) if len(tps) > 1 else 0.0},
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\nWROTE", a.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
