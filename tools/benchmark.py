"""
B=1 latency benchmark against the inference server.

Measures TTFT and decode tok/s for a fixed prompt set at batch size 1.
Run this against the vLLM baseline, then again after each optimization to
track improvement.

Usage:
    python3 tools/benchmark.py                          # default: localhost:8000
    python3 tools/benchmark.py --base http://localhost:8000 --n 20 --tokens 100
"""

import argparse
import json
import statistics
import time
import urllib.request


PROMPTS = [
    "Explain the transformer attention mechanism in detail.",
    "Write a quicksort implementation in Rust.",
    "What are the main differences between TCP and UDP?",
    "Describe how gradient descent works.",
    "Explain the CAP theorem with examples.",
    "Write a Python function to find all prime numbers up to N.",
    "How does the Linux kernel handle memory allocation?",
    "What is the difference between a mutex and a semaphore?",
    "Explain how HTTPS works end to end.",
    "Describe the architecture of a large language model.",
    "Write a binary search implementation in C.",
    "How does garbage collection work in Java?",
    "Explain what a context switch is in operating systems.",
    "What is the time complexity of merge sort and why?",
    "How does a hash table handle collisions?",
    "Write a regex to parse an IP address.",
    "Explain what RLHF is and how it works.",
    "What is the difference between L1 and L2 cache?",
    "How does a neural network learn weights through backprop?",
    "Describe how Kubernetes schedules pods across nodes.",
]


def stream_request(base: str, prompt: str, max_tokens: int) -> dict:
    payload = json.dumps({
        "model": "qwen3-235b-a22b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    t_first = None
    n_tokens = 0
    buf = b""

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            buf += raw
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                line = frame.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_tokens += 1

    t_end = time.perf_counter()
    ttft_ms = (t_first - t_start) * 1000 if t_first else 0.0
    decode_s = (t_end - t_first) if t_first else (t_end - t_start)
    decode_tps = (n_tokens - 1) / decode_s if decode_s > 0 and n_tokens > 1 else 0.0

    return {
        "ttft_ms": ttft_ms,
        "decode_tok_per_s": decode_tps,
        "n_tokens": n_tokens,
        "total_s": t_end - t_start,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=10, help="number of prompts to run")
    ap.add_argument("--tokens", type=int, default=100, help="max tokens per response")
    ap.add_argument("--out", default=None, help="optional JSON output file")
    args = ap.parse_args()

    prompts = (PROMPTS * ((args.n // len(PROMPTS)) + 1))[:args.n]

    print(f"Benchmarking {args.base}  ({args.n} prompts, {args.tokens} tokens each)")
    print(f"{'#':>3}  {'TTFT ms':>10}  {'tok/s':>8}  {'tokens':>7}  {'total s':>8}")
    print("─" * 46)

    results = []
    for i, prompt in enumerate(prompts):
        try:
            r = stream_request(args.base, prompt, args.tokens)
            results.append(r)
            print(f"{i+1:>3}  {r['ttft_ms']:>10.1f}  {r['decode_tok_per_s']:>8.1f}"
                  f"  {r['n_tokens']:>7}  {r['total_s']:>8.2f}s")
        except Exception as e:
            print(f"{i+1:>3}  ERROR: {e}")

    if not results:
        print("No results.")
        return

    ttfts = [r["ttft_ms"] for r in results]
    tpss = [r["decode_tok_per_s"] for r in results if r["decode_tok_per_s"] > 0]

    print("─" * 46)
    print(f"\n{'TTFT (ms)':<20} p50={statistics.median(ttfts):.1f}  "
          f"p95={sorted(ttfts)[int(len(ttfts)*0.95)]:.1f}  "
          f"mean={statistics.mean(ttfts):.1f}")
    print(f"{'Decode tok/s':<20} p50={statistics.median(tpss):.1f}  "
          f"p95={sorted(tpss)[int(len(tpss)*0.95)]:.1f}  "
          f"mean={statistics.mean(tpss):.1f}")
    print(f"{'ms/tok (mean)':<20} {1000/statistics.mean(tpss):.1f} ms")

    summary = {
        "backend": args.base,
        "n_prompts": len(results),
        "max_tokens": args.tokens,
        "ttft_ms": {"mean": statistics.mean(ttfts), "p50": statistics.median(ttfts),
                    "p95": sorted(ttfts)[int(len(ttfts)*0.95)]},
        "decode_tok_per_s": {"mean": statistics.mean(tpss), "p50": statistics.median(tpss),
                              "p95": sorted(tpss)[int(len(tpss)*0.95)]},
        "ms_per_tok_mean": 1000 / statistics.mean(tpss),
        "results": results,
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {args.out}")

    return summary


if __name__ == "__main__":
    main()
