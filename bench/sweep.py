#!/usr/bin/env python3
"""Autoresearch driver: sweep the tuning DoF, log results, and suggest the next lever.

It does NOT relaunch the engine itself (that's an operator step with distinct flags per config);
it measures whatever server is at --base, records the result against the roofline, and prints the
decision-tree recommendation for what to change next. Point it at each engine config in turn,
or wire `relaunch_hook` to your launcher.

    python bench/sweep.py --base http://localhost:8000 --ctx 32768 --label "tp4ep2-fp8"
"""
import argparse, json, time
import measure, roofline

# The DoF search space (operator changes the engine launch between runs; this records each).
DOF = {
    "weight_dtype": ["fp8", "int4"],          # roofline weight-bytes 1.0 / 0.5
    "kv_dtype":     ["fp8", "fp16"],          # 1.0 / 2.0
    "layout":       ["tp4ep2", "tp8", "ep8"],
    "graph":        ["off", "on"],
    "spec":         ["off", "ngram", "eagle3"],
    "draft_len":    [0, 4, 6],
}

# Decision tree: dominant term -> ordered next levers to try.
NEXT_LEVER = {
    "memory-bandwidth": ["weight_dtype: fp8->int4 on experts", "spec: ngram->eagle3", "raise draft_len"],
    "KV-bandwidth":     ["kv_dtype: fp16->fp8", "shorten ctx / prefix-KV reuse", "KV compression"],
    "comms/launch":     ["layout: ep8->tp4ep2->tp8 (fewer collectives)", "graph: off->on", "low-latency all-to-all (NVSHMEM/DeepEP)"],
    "weight-bandwidth": ["weight_dtype: bf16->fp8->int4", "check dequant not compute-bound (Nsight)"],
}

def classify(term):
    for k in NEXT_LEVER:
        if k in term:
            return NEXT_LEVER[k]
    return ["inspect Nsight; term unclear"]

def run(base, ctx, decode, engine, label, weight_bytes, kv_bytes, out):
    m = measure.stream_once(base, ("ctx " * ctx).strip() or "hi", decode, engine)
    a = roofline.analyze(ctx, m["tpot_ms"] or 1e9, weight_bytes, kv_bytes)
    rec = dict(label=label, ctx=ctx, **m, **a, next_levers=classify(a["dominant_term"]))
    with open(out, "a") as f:
        f.write(json.dumps({k: v for k, v in rec.items() if k != "server_summary"}) + "\n")
    print(f"[{label}] TPOT {m['tpot_ms']:.2f}ms  {m['decode_tok_s']:.0f} tok/s  "
          f"{a['achieved_frac']*100:.0f}% of roofline")
    print(f"  dominant: {a['dominant_term']}")
    print(f"  try next: {rec['next_levers']}")
    return rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--label", default="run")
    ap.add_argument("--weight-bytes", type=float, default=1.0)
    ap.add_argument("--kv-bytes", type=float, default=1.0)
    ap.add_argument("--out", default="bench/results.jsonl")
    a = ap.parse_args()
    print("DoF search space:", json.dumps(DOF))
    run(a.base, a.ctx, a.decode, a.engine, a.label, a.weight_bytes, a.kv_bytes, a.out)
