"""Quantify the confidence-adaptive top-k opportunity on Qwen3-235B-A22B.

The optimization (tools/../experiments/adaptive_topk): route a token to FEWER than
8 experts when the router softmax is concentrated, cutting expert-weight HBM reads
(the dominant B=1 term). This tool measures the *headroom*: across many token
positions, how much of the selected top-8 router mass sits in the top-4 / top-6?
Wherever the bottom experts carry near-zero weight, dropping them is ~free.

Outputs the numbers that set the policy + project the speedup:
  - % of (token, layer) where top-k' renorm mass > threshold (k'=4,6; thr .85/.9/.95)
  - expected average k under an adaptive policy -> expected expert-byte savings
  - per-layer concentration (early/mid/late) -- tail experts often matter more mid-stack

Uses HF `output_router_logits=True` (one forward per prompt, no generation) for speed.

Usage:
    PYTHONPATH=src python3 tools/router_mass.py --model-path /alloc/data/Qwen3-235B-A22B \
        --n-prompts 16 --gpu-mem-gib 70 --out /alloc/data/router_mass.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

PROMPTS = [
    "Explain how transformer attention works, step by step.",
    "Write a Python function that returns the nth Fibonacci number.",
    "What is the capital of France and why is it historically important?",
    "Prove that the square root of 2 is irrational.",
    "Describe the process of photosynthesis in detail.",
    "Implement quicksort in Rust with comments.",
    "Summarize the causes of the 2008 financial crisis.",
    "What are the tradeoffs between TCP and UDP?",
    "Write a haiku about gradient descent.",
    "Explain eigenvalues to a first-year student.",
    "Give three uses of the word 'set' in different senses.",
    "Translate 'good morning' into Spanish, French, and Japanese.",
    "How does a CPU branch predictor work?",
    "Derive the quadratic formula.",
    "Write SQL to find the second-highest salary per department.",
    "Explain why MoE models are memory-bandwidth bound at batch size 1.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/alloc/data/Qwen3-235B-A22B")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--gpu-mem-gib", type=float, default=70.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default="/alloc/data/router_mass.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model_path} (max_memory {args.gpu_mem_gib:.0f}GiB/GPU) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    max_mem = {i: f"{args.gpu_mem_gib:.0f}GiB" for i in range(args.n_gpus)}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", max_memory=max_mem,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    K = args.top_k
    # per-(position,layer) fraction of the selected top-8 renorm mass held by top-k'
    mass4, mass6 = [], []
    per_layer_m4: dict[int, list] = {}

    with torch.inference_mode():
        for pi, prompt in enumerate(PROMPTS[:args.n_prompts]):
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            out = model(input_ids=ids, output_router_logits=True, use_cache=False)
            rl = [r for r in (out.router_logits or []) if r is not None]
            for layer, logits in enumerate(rl):
                # logits: [n_positions, n_experts]
                probs = torch.softmax(logits.float(), dim=-1)
                top8 = torch.topk(probs, K, dim=-1).values  # [pos, 8], desc
                denom = top8.sum(dim=-1).clamp_min(1e-9)
                m4 = (top8[:, :4].sum(dim=-1) / denom)
                m6 = (top8[:, :6].sum(dim=-1) / denom)
                mass4.append(m4.cpu().numpy())
                mass6.append(m6.cpu().numpy())
                per_layer_m4.setdefault(layer, []).append(float(m4.mean()))
            print(f"  [{pi+1}/{args.n_prompts}] {ids.shape[1]} pos, {len(rl)} MoE layers")

    m4 = np.concatenate(mass4)
    m6 = np.concatenate(mass6)
    n = len(m4)

    def frac_above(arr, thr):
        return float((arr > thr).mean())

    print(f"\n=== router concentration over {n} (position,layer) samples ===")
    print(f"  mean top-4 mass (of selected 8): {m4.mean():.3f}   top-6 mass: {m6.mean():.3f}")
    table = {}
    for thr in (0.85, 0.90, 0.95):
        f4, f6 = frac_above(m4, thr), frac_above(m6, thr)
        table[str(thr)] = {"top4": f4, "top6": f6}
        print(f"  thr {thr}:  P(top4 mass>{thr})={f4*100:5.1f}%   "
              f"P(top6 mass>{thr})={f6*100:5.1f}%")

    # Adaptive policy: k=4 if top4>0.9 ; elif k=6 if top6>0.95 ; else 8.
    use4 = m4 > 0.90
    use6 = (~use4) & (m6 > 0.95)
    avg_k = (use4 * 4 + use6 * 6 + (~use4 & ~use6) * 8).mean()
    expert_byte_frac = avg_k / K  # experts ~66% of decode bytes
    e2e = 0.66 * expert_byte_frac + 0.34  # crude: only expert term shrinks
    print(f"\n  adaptive policy (k4 if m4>.9; k6 if m6>.95; else 8):")
    print(f"    tokens at k=4: {use4.mean()*100:.1f}%  k=6: {use6.mean()*100:.1f}%  "
          f"k=8: {(~use4&~use6).mean()*100:.1f}%")
    print(f"    avg experts/token: {avg_k:.2f}/8  -> expert bytes x{expert_byte_frac:.3f}")
    print(f"    crude e2e decode time x{e2e:.3f}  (~{(1/e2e-1)*100:.0f}% faster, expert-term only)")

    layers = sorted(per_layer_m4)
    band = lambda lo, hi: float(np.mean([np.mean(per_layer_m4[l]) for l in layers if lo <= l < hi]))
    nL = max(layers) + 1 if layers else 1
    bands = {"early": band(0, nL//3), "mid": band(nL//3, 2*nL//3), "late": band(2*nL//3, nL)}
    print(f"  per-layer top-4 mass: early {bands['early']:.3f} / mid {bands['mid']:.3f} / "
          f"late {bands['late']:.3f}")

    out = {
        "n_samples": n,
        "mean_top4_mass": float(m4.mean()), "mean_top6_mass": float(m6.mean()),
        "frac_above": table,
        "adaptive_policy": {
            "pct_k4": float(use4.mean()), "pct_k6": float(use6.mean()),
            "avg_k": float(avg_k), "expert_byte_frac": float(expert_byte_frac),
            "crude_e2e_time_frac": float(e2e),
        },
        "per_layer_top4_bands": bands,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
