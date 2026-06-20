"""Greedy completions from an OpenAI-compatible endpoint -> JSON.

Half of the adaptive-top-k quality gate: run a FIXED prompt set at temperature 0
(greedy) against an engine and save the outputs, so baseline (k=8) and adaptive
(k=4) can be compared token-for-token (tools/quality_compare.py). If adaptive
output ~matches baseline, dropping the low-confidence experts is free quality-wise.

Usage:
    python3 tools/quality_probe.py --base http://localhost:8077 --model qwen3 \
        --tokens 96 --out /alloc/data/q_baseline.json
"""
from __future__ import annotations
import argparse, json, urllib.request

PROMPTS = [
    "Write a Python function to check if a string is a palindrome.",
    "What is the capital of Australia? Answer in one sentence.",
    "Explain the difference between a stack and a queue.",
    "Compute 17 * 23 and show your steps.",
    "Write the first paragraph of a story about a lighthouse keeper.",
    "List the planets of the solar system in order from the sun.",
    "Implement binary search in Python with comments.",
    "Summarize what HTTP status code 404 means.",
    "Translate 'I would like a coffee' into French.",
    "What is recursion? Give a one-line example.",
]


def complete(base: str, model: str, prompt: str, n: int) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": n, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=payload,
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        obj = json.loads(resp.read())
    return obj["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8077")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--out", default="q_probe.json")
    a = ap.parse_args()
    comps = {}
    for i, p in enumerate(PROMPTS):
        try:
            comps[p] = complete(a.base, a.model, p, a.tokens)
            print(f"  [{i+1}/{len(PROMPTS)}] ok ({len(comps[p])} chars)")
        except Exception as e:
            comps[p] = f"<<ERROR: {e}>>"
            print(f"  [{i+1}/{len(PROMPTS)}] ERROR {e}")
    json.dump({"base": a.base, "model": a.model, "tokens": a.tokens,
               "completions": comps}, open(a.out, "w"), indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
