#!/usr/bin/env python3
"""B=1 decode-latency probe for the KV-cache FP8 study (stdlib only).

Streams /v1/completions (greedy, temperature 0) so we control the prompt token
count exactly and read back the ACTUAL prompt length the server saw. The KV-cache
FP8 win lands on TPOT (per-token decode) and grows with context length, because
each decode step re-reads the whole KV cache from HBM — halving KV bytes halves
that read. TTFT (prefill) is compute-bound and should barely move.

    python3 tools/kv_measure.py --base http://localhost:8088 --model qwen3 \
        --ctx 8192 --decode 128 --warmup 2 --json-out results/kv_fp8/run.json

Reports TTFT, TPOT, decode tok/s, and the server-measured prompt_tokens so the
A/B sweep compares runs at the SAME real context length, not a guessed one.
"""
import argparse, json, sys, time, urllib.request


def _filler(target_tokens: int) -> str:
    # "context " tokenizes to ~1 token each for Qwen3 BPE; first call reports the
    # real count back via usage so any drift is observed, not assumed.
    if target_tokens <= 0:
        return "Hi."
    return ("You are given a long document. " + "context " * target_tokens).strip()


def stream_once(base, prompt, decode, model):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": decode,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    tok_times = []
    prompt_tokens = None
    completion_tokens = None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            choices = obj.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                tok_times.append(now)
            usage = obj.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
    inter = [(tok_times[i] - tok_times[i - 1]) * 1000 for i in range(1, len(tok_times))]
    tpot_ms = sum(inter) / len(inter) if inter else 0.0
    n = len(tok_times)
    decode_tok_s = (n - 1) / (tok_times[-1] - tok_times[0]) if n > 1 else 0.0
    return dict(
        ttft_ms=(ttft or 0) * 1000,
        tpot_ms=tpot_ms,
        decode_tok_s=decode_tok_s,
        emitted_tokens=n,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8088")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--ctx", type=int, default=2048, help="target prompt tokens")
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=3, help="timed reps; report median")
    ap.add_argument("--label", default="", help="free-form tag e.g. kv=fp8")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    prompt = _filler(a.ctx)
    for _ in range(a.warmup):
        stream_once(a.base, "warm up please", 8, a.model)

    runs = [stream_once(a.base, prompt, a.decode, a.model) for _ in range(a.repeat)]
    runs.sort(key=lambda r: r["tpot_ms"])
    med = runs[len(runs) // 2]
    med["label"] = a.label
    med["target_ctx"] = a.ctx
    med["repeat"] = a.repeat
    med["all_tpot_ms"] = [round(r["tpot_ms"], 3) for r in runs]

    print(json.dumps(med, indent=2))
    print(
        f"\n[{a.label}] ctx≈{med['prompt_tokens']} | TTFT {med['ttft_ms']:.1f} ms | "
        f"TPOT {med['tpot_ms']:.2f} ms | decode {med['decode_tok_s']:.1f} tok/s",
        file=sys.stderr,
    )
    if a.json_out:
        import os
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(med, f, indent=2)


if __name__ == "__main__":
    main()
