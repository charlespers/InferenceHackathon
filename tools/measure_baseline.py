"""Measure B=1 decode latency of a live OpenAI-compatible engine (vLLM).

Stdlib-only. Streams a chat completion, times TTFT / TPOT / decode tok/s from
the wall clock (not a self-reported number), and reports % of the H100 FP8
roofline + which term likely dominates. This is the team's E1 baseline number:
"every optimization is x this." Run it against the live FP8 vLLM server.

Usage:
    python3 tools/measure_baseline.py --base http://localhost:8001 \
        --model qwen3-235b-fp8 --decode 64 --repeats 3 --out baseline_fp8.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

# H100 FP8 B=1 roofline for Qwen3-235B-A22B (from inferutil / DESIGN.md):
# active 21.6B x 1B / (8 x 3.35 TB/s) ~= 0.93 ms floor; realistic ~500-547 tok/s.
ROOFLINE_TOK_S = 540.0  # weight-only FP8 ceiling, 8xH100


def one_run(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    t_first = None
    t_last = t0
    n = 0
    buf = b""
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            buf += raw
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                line = frame.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    continue
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n += 1
    if t_first is None or n < 2:
        return {"ok": False, "tokens": n}
    decode_s = t_last - t_first
    return {
        "ok": True,
        "ttft_ms": round((t_first - t0) * 1e3, 2),
        "tpot_ms": round(decode_s / (n - 1) * 1e3, 3),
        "decode_tok_s": round((n - 1) / decode_s, 2),
        "tokens": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--model", default="qwen3-235b-fp8")
    ap.add_argument("--prompt", default="Write a detailed explanation of how "
                    "transformer attention works, step by step.")
    ap.add_argument("--decode", type=int, default=64, help="max_tokens")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="baseline_fp8.json")
    args = ap.parse_args()

    print(f"measuring {args.base} model={args.model} decode={args.decode} "
          f"x{args.repeats} ...")
    runs = []
    for i in range(args.repeats):
        r = one_run(args.base, args.model, args.prompt, args.decode)
        runs.append(r)
        print(f"  run {i+1}: {r}")
    ok = [r for r in runs if r.get("ok")]
    if not ok:
        print("ALL RUNS FAILED — is the server serving + model name correct?")
        json.dump({"runs": runs, "ok": False}, open(args.out, "w"), indent=2)
        return

    # median by decode_tok_s
    ok.sort(key=lambda r: r["decode_tok_s"])
    med = ok[len(ok) // 2]
    pct = med["decode_tok_s"] / ROOFLINE_TOK_S * 100
    # crude dominant-term hint: roofline TPOT ~ 1000/540 = 1.85 ms; if measured
    # TPOT is far above, the gap is comms/launch/host floor, not weight bytes.
    roof_tpot = 1000.0 / ROOFLINE_TOK_S
    gap = med["tpot_ms"] / roof_tpot
    if gap < 1.5:
        term = "near weight-roofline (byte levers: int4/fp8 pay)"
    elif gap < 4:
        term = "moderate floor overhead (comms/launch — CUDA graphs/spec pay)"
    else:
        term = "DOMINATED by floor (launch/host/comms — fix fast-path first)"

    summary = {
        "ok": True,
        "base": args.base, "model": args.model, "decode": args.decode,
        "median": med,
        "ttft_ms": med["ttft_ms"], "tpot_ms": med["tpot_ms"],
        "decode_tok_s": med["decode_tok_s"],
        "pct_of_roofline": round(pct, 1),
        "roofline_tok_s": ROOFLINE_TOK_S,
        "dominant_term_hint": term,
        "runs": runs,
    }
    json.dump(summary, open(args.out, "w"), indent=2)
    print("\n=== BASELINE ===")
    print(f"  TTFT      {med['ttft_ms']} ms")
    print(f"  TPOT      {med['tpot_ms']} ms/token")
    print(f"  decode    {med['decode_tok_s']} tok/s  ({pct:.1f}% of ~{ROOFLINE_TOK_S} roofline)")
    print(f"  bottleneck hint: {term}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
