#!/usr/bin/env python3
"""Greedy quality gate for KV-cache FP8 (stdlib only).

KV quantization is lossy, and the loss concentrates in long-context recall: the
older a token's KV entry, the more a low-precision read can perturb attention.
So we don't just check short prompts agree — we bury a fact deep in a long filler
context (needle-in-a-haystack) and check the model still recalls it under FP8 KV.

Two phases, because baseline and fp8 servers can't be up at once on one box:

  # against the baseline server (--kv-cache-dtype auto)
  python3 tools/kv_quality.py capture --base http://localhost:8088 --model qwen3 \
      --out results/kv_fp8/q_auto.json
  # then relaunch with --kv-cache-dtype fp8 and:
  python3 tools/kv_quality.py capture --base http://localhost:8088 --model qwen3 \
      --out results/kv_fp8/q_fp8.json
  # offline:
  python3 tools/kv_quality.py compare results/kv_fp8/q_auto.json results/kv_fp8/q_fp8.json
"""
import argparse, json, sys, urllib.request


def _haystack(depth_frac: float, needle: str, total_words: int) -> str:
    filler = "The weather report noted mild conditions across the region. "
    n = max(1, total_words // 6)  # ~6 words per filler sentence
    pre = int(n * depth_frac)
    body = filler * pre + needle + " " + filler * (n - pre)
    return body


# Prompt suite: short determinism checks + long-context needle recall at varied depth.
def build_prompts():
    probes = []
    # 1. short greedy determinism (should be byte-identical regardless of KV dtype)
    probes.append(dict(id="short_math", ctx_words=0,
                       prompt="Compute 17 * 23 and explain in one sentence. Answer:"))
    probes.append(dict(id="short_def", ctx_words=0,
                       prompt="Define 'entropy' in information theory in one sentence. Answer:"))
    # 2. needle recall at increasing context length + depth (the KV-quant stressor)
    for words, depth in [(800, 0.1), (4000, 0.5), (16000, 0.5), (16000, 0.9)]:
        needle = "IMPORTANT FACT: the access code for vault 7 is ZULU-4471-OMEGA."
        ctx = _haystack(depth, needle, words)
        q = ("\n\nQuestion: What is the access code for vault 7? "
             "Answer with only the code. Answer:")
        probes.append(dict(id=f"needle_{words}w_d{int(depth*100)}",
                           ctx_words=words, prompt=ctx + q,
                           expect="ZULU-4471-OMEGA"))
    return probes


def complete(base, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False}
    req = urllib.request.Request(f"{base}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        obj = json.loads(r.read().decode())
    return obj["choices"][0]["text"], obj.get("usage", {}).get("prompt_tokens")


def cmd_capture(a):
    out = []
    for p in build_prompts():
        mt = 16 if p["id"].startswith("needle") else 64
        text, ptok = complete(a.base, a.model, p["prompt"], mt)
        rec = dict(id=p["id"], ctx_words=p["ctx_words"], prompt_tokens=ptok,
                   output=text, expect=p.get("expect"))
        if p.get("expect"):
            rec["recalled"] = p["expect"] in text
        out.append(rec)
        print(f"  {p['id']}: ptok={ptok} recalled={rec.get('recalled')} "
              f"out={text[:50]!r}", file=sys.stderr)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}", file=sys.stderr)


def cmd_compare(a):
    A = {r["id"]: r for r in json.load(open(a.baseline))}
    B = {r["id"]: r for r in json.load(open(a.fp8))}
    ids = sorted(set(A) & set(B))
    exact = 0
    rows = []
    for i in ids:
        a_out, b_out = A[i]["output"], B[i]["output"]
        match = a_out == b_out
        exact += match
        row = dict(id=i, exact=match,
                   base_recall=A[i].get("recalled"), fp8_recall=B[i].get("recalled"))
        if not match and not i.startswith("needle"):
            # first divergence char index for short determinism probes
            j = next((k for k in range(min(len(a_out), len(b_out)))
                      if a_out[k] != b_out[k]), min(len(a_out), len(b_out)))
            row["diverge_at"] = j
        rows.append(row)
    needle_ids = [i for i in ids if i.startswith("needle")]
    base_rec = sum(bool(A[i].get("recalled")) for i in needle_ids)
    fp8_rec = sum(bool(B[i].get("recalled")) for i in needle_ids)
    summary = dict(
        exact_match_rate=round(exact / len(ids), 3) if ids else 0.0,
        n=len(ids),
        needle_recall_base=f"{base_rec}/{len(needle_ids)}",
        needle_recall_fp8=f"{fp8_rec}/{len(needle_ids)}",
        rows=rows,
    )
    print(json.dumps(summary, indent=2))
    # gate verdict
    regressed = [r["id"] for r in rows if r.get("base_recall") and not r.get("fp8_recall")]
    if regressed:
        print(f"\nGATE: FAIL — fp8 lost recall on {regressed}", file=sys.stderr)
    else:
        print("\nGATE: PASS — no recall regression vs baseline", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--base", default="http://localhost:8088")
    c.add_argument("--model", default="qwen3")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_capture)
    cm = sub.add_parser("compare")
    cm.add_argument("baseline")
    cm.add_argument("fp8")
    cm.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
