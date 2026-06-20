"""B=1 route-prediction gate for Qwen3-235B-A22B.

Fixes the per-token/per-layer interleaving in routing_analysis.py: that script
relied on model.generate() and accumulated every (layer, decode-step) into one
flat list, so per-token routes — and the Markov matrices built from them — were
muddied. Here we drive decode **manually** (one forward = one token) and snapshot
each generated token's clean [layer -> top-8 experts] vector via a forward hook
on the LAST sequence position.

Then we compute the go/no-go numbers for the expert-prefetch (B) workstream:

  1. Per-(token,layer) GPU load imbalance  -- the real B=1 *latency* cost
     (busiest GPU among the 8 chosen experts), under:
       - round-robin placement  (expert_id % 8)
       - affinity placement      (inferutil.routing.greedy_partition)
       - round-robin + replicate top-R hot experts/layer onto all GPUs
  2. Placement local-fraction (jminding's metric): co-activation edges that
     stay on one GPU, round-robin vs affinity.        [reuses inferutil.routing]
  3. Route predictability (what a cheap prefetch predictor would hit):
       - persistence: token t's top-8 -> token t+1 (per layer)   [temporal]
       - static-hot : token's experts in to the layer's global top-K  [coverage]
       - combined   : prefetch (prev-token 8) UNION (top-K hot)
  4. Hot-expert coverage curve (HBM budget knob for replication).

Everything is framed as "how many experts must be staged to hit X% coverage"
so it feeds TokenDAG.prefetch_schedule directly.

Usage:
    PYTHONPATH=/alloc/data/InferenceHackathon/src \
      python3 tools/routing_predict.py \
        --model-path /alloc/data/Qwen3-235B-A22B \
        --n-prompts 12 --max-new-tokens 24 --out routing_predict.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch


# Clean per-forward-pass capture: layer_idx -> [top-8 expert ids] for the LAST
# sequence position of the current forward. Reset before every forward pass.
_CUR: dict[int, list[int]] = {}
# Cross-layer DirectProxy state: previous layer's gate-input hidden (h_{L-1}).
_PREV: dict = {"h": None, "layer": -1}
_MEASURE = {"on": False}  # only score DirectProxy during decode (B=1 regime)
# DirectProxy top-k overlap with the actual selection, accumulated per layer.
_DP_HITS: dict[int, list[float]] = defaultdict(list)


def reset_pass():
    _CUR.clear()
    _PREV["h"] = None
    _PREV["layer"] = -1


def make_hook(layer_idx: int, top_k: int):
    def hook(mod, inp, out):
        h = inp[0]      # gate input  = residual-stream hidden h_L  [tokens, 4096]
        logits = out    # gate output = router logits               [tokens, 128]
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        h_last = h[-1]  # the token being produced this pass
        actual = torch.topk(logits[-1], top_k).indices
        _CUR[layer_idx] = actual.tolist()
        # DirectProxy (jminding Tier-1): predict THIS layer's experts from the
        # PREVIOUS layer's hidden via this layer's router weight (residual-stream
        # proxy, zero training). top-k(W_L @ h_{L-1}) vs the actual top-k at L.
        if _MEASURE["on"] and _PREV["h"] is not None and _PREV["layer"] == layer_idx - 1:
            prev_h = _PREV["h"].to(mod.weight.device, mod.weight.dtype)
            pred = torch.topk(mod.weight @ prev_h, top_k).indices
            ov = len(set(pred.tolist()) & set(actual.tolist())) / top_k
            _DP_HITS[layer_idx].append(ov)
        _PREV["h"] = h_last
        _PREV["layer"] = layer_idx
    return hook


def attach_hooks(model, n_layers: int, top_k: int) -> list:
    handles, found = [], 0
    for name, module in model.named_modules():
        if name.endswith(".mlp.gate") or name.endswith(".block_sparse_moe.gate"):
            li = _layer_idx(name)
            if li is not None and li < n_layers:
                handles.append(module.register_forward_hook(make_hook(li, top_k)))
                found += 1
    print(f"  attached hooks to {found} MoE gates")
    if found != n_layers:
        print(f"  WARNING: expected {n_layers} gates, found {found}")
    return handles


def _layer_idx(name: str):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Manual greedy decode capturing clean per-token routes
# ---------------------------------------------------------------------------

def collect_routes(model, tok, prompts, max_new, n_layers) -> list[list[list[list[int]]]]:
    """Return routes[prompt][decode_step][layer] = [8 expert ids]."""
    routes = []
    lat = []
    for pi, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        prompt_routes = []
        with torch.inference_mode():
            reset_pass()
            _MEASURE["on"] = False                          # prefill: don't score
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            t0 = time.time()
            for _ in range(max_new):
                reset_pass()
                _MEASURE["on"] = True                       # decode: score DirectProxy
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                prompt_routes.append([list(_CUR.get(l, [])) for l in range(n_layers)])
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            dt = (time.time() - t0) / max_new * 1000
        lat.append(dt)
        routes.append(prompt_routes)
        print(f"  [{pi+1:2}/{len(prompts)}] {max_new} tok  {dt:5.1f} ms/tok")
        sys.stdout.flush()
    print(f"  decode: mean {np.mean(lat):.1f} ms/tok (HF pipeline-parallel, not engine)")
    return routes


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------

def round_robin(n_gpus: int):
    return lambda layer, e: e % n_gpus


def gpu_imbalance(routes, place, n_layers, n_gpus, replicate=None):
    """Mean over (token, layer) of max-GPU-load / mean-GPU-load among the 8
    chosen experts. mean load = top_k / n_gpus, so this == max_gpu_count scaled.

    place: fn(layer, expert) -> gpu.  replicate: optional set per layer of
    experts that may be served on ANY gpu (assigned to the least-loaded one)."""
    vals = []
    for prompt_routes in routes:
        for step in prompt_routes:
            for layer, experts in enumerate(step):
                if not experts:
                    continue
                counts = [0] * n_gpus
                free = []
                for e in experts:
                    if replicate is not None and e in replicate[layer]:
                        free.append(e)
                    else:
                        counts[place(layer, e)] += 1
                # assign replicated experts greedily to the least-loaded GPU
                for _ in free:
                    counts[counts.index(min(counts))] += 1
                mean = len(experts) / n_gpus
                vals.append(max(counts) / mean)
    return float(np.mean(vals))


def hot_experts_per_layer(routes, n_layers, n_experts, top_r):
    """Set of top-R hottest experts per layer from the collected routes."""
    counts = [np.zeros(n_experts, dtype=int) for _ in range(n_layers)]
    for prompt_routes in routes:
        for step in prompt_routes:
            for layer, experts in enumerate(step):
                for e in experts:
                    counts[layer][e] += 1
    return [set(np.argsort(c)[::-1][:top_r].tolist()) for c in counts], counts


# ---------------------------------------------------------------------------
# Predictability
# ---------------------------------------------------------------------------

def persistence_hitrate(routes, n_layers, top_k):
    """Token t's top-8 used to prefetch token t+1, per layer. Returns overlap."""
    per_layer = [[] for _ in range(n_layers)]
    for prompt_routes in routes:
        for t in range(len(prompt_routes) - 1):
            for layer in range(n_layers):
                a, b = set(prompt_routes[t][layer]), set(prompt_routes[t + 1][layer])
                if b:
                    per_layer[layer].append(len(a & b) / len(b))
    overall = np.mean([v for L in per_layer for v in L])
    by_layer = [round(float(np.mean(L)), 3) if L else None for L in per_layer]
    return float(overall), by_layer


def static_hot_coverage(routes, hot_sets):
    """Fraction of a token's actual experts that fall in the layer's top-R hot
    set (== what static replication of those R experts would serve locally)."""
    hit = tot = 0
    for prompt_routes in routes:
        for step in prompt_routes:
            for layer, experts in enumerate(step):
                for e in experts:
                    tot += 1
                    if e in hot_sets[layer]:
                        hit += 1
    return hit / tot if tot else 0.0


def combined_coverage(routes, hot_sets, n_layers):
    """Prefetch = (previous token's 8) UNION (layer top-R hot). Fraction hit."""
    hit = tot = 0
    for prompt_routes in routes:
        for t in range(1, len(prompt_routes)):
            for layer in range(n_layers):
                staged = set(prompt_routes[t - 1][layer]) | hot_sets[layer]
                for e in prompt_routes[t][layer]:
                    tot += 1
                    if e in staged:
                        hit += 1
    return hit / tot if tot else 0.0


PROMPTS = [
    "Explain the attention mechanism in transformers.",
    "Write a Python function to merge two sorted lists.",
    "What is the capital of France, and what is its history?",
    "Describe the process of photosynthesis in detail.",
    "Solve step by step: if 2x + 5 = 13, what is x?",
    "Write a short story about a robot learning to paint.",
    "What were the main causes of World War I?",
    "Implement a binary search tree in Rust with insert and lookup.",
    "Explain quantum entanglement to a curious ten-year-old.",
    "Compare supervised and unsupervised learning with examples.",
    "Write a SQL query to find the top 5 customers by total revenue.",
    "How does garbage collection work in Go, and what are its tradeoffs?",
    "Explain the CAP theorem and give a real-world example.",
    "Walk through the time complexity of quicksort in best and worst case.",
    "Describe the full architecture of a transformer model layer by layer.",
    "How would you design and implement an LRU cache from scratch?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/alloc/data/Qwen3-235B-A22B")
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--n-layers", type=int, default=94)
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--gpu-mem-gib", type=float, default=57.0,
                    help="per-GPU weight budget; keep under free HBM to avoid "
                         "the CPU offload that tanks decode speed")
    ap.add_argument("--out", default="routing_predict.json")
    args = ap.parse_args()

    print("=" * 64)
    print("Qwen3-235B-A22B  B=1 route-prediction gate")
    print("=" * 64)

    print(f"\n[1] loading {args.model_path} (device_map=auto across 8 GPUs) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    max_mem = {i: f"{args.gpu_mem_gib:.0f}GiB" for i in range(args.n_gpus)}
    print(f"    max_memory/GPU = {args.gpu_mem_gib:.0f}GiB (avoid CPU offload)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto", max_memory=max_mem,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()
    print(f"    loaded in {time.time()-t0:.1f}s")

    print("\n[2] attaching clean per-token hooks ...")
    handles = attach_hooks(model, args.n_layers, args.top_k)

    print(f"\n[3] manual decode: {args.n_prompts} prompts x {args.max_new_tokens} tok ...")
    routes = collect_routes(model, tok, PROMPTS[:args.n_prompts],
                            args.max_new_tokens, args.n_layers)
    for h in handles:
        h.remove()

    n_tokens = sum(len(p) for p in routes)
    print(f"\n[4] analysis on {n_tokens} clean decode tokens")
    print("    " + "-" * 56)

    # --- 1. per-(token,layer) GPU imbalance (the B=1 latency cost) ---
    rr = round_robin(args.n_gpus)
    imb_rr = gpu_imbalance(routes, rr, args.n_layers, args.n_gpus)
    print(f"\n  per-token GPU imbalance (max/mean over 8 GPUs, B=1 latency cost):")
    print(f"    round-robin placement              : {imb_rr:.2f}x")

    rep = {}
    for R in (4, 8, 16):
        hot, _ = hot_experts_per_layer(routes, args.n_layers, args.n_experts, R)
        v = gpu_imbalance(routes, rr, args.n_layers, args.n_gpus, replicate=hot)
        rep[R] = v
        print(f"    round-robin + replicate top-{R:<2} hot   : {v:.2f}x")

    # --- 2. placement local-fraction (jminding's metric) ---
    local = {}
    try:
        from inferutil.routing import (CoActGraph, greedy_partition,
                                       round_robin_placement, placement_stats)
        g = CoActGraph(n_experts=args.n_experts, n_layers=args.n_layers)
        for prompt_routes in routes:
            for step in prompt_routes:
                for layer, experts in enumerate(step):
                    if experts:
                        g.add_token_step(layer, experts)
                for layer in range(len(step) - 1):
                    if step[layer] and step[layer + 1]:
                        g.add_cross_layer(layer, step[layer], step[layer + 1])
        rr_place = round_robin_placement(args.n_gpus, args.n_experts, args.n_layers)
        gp_place = greedy_partition(g, args.n_gpus, args.n_experts, args.n_layers)
        s_rr = placement_stats(g, rr_place, "round_robin")
        s_gp = placement_stats(g, gp_place, "affinity")
        local = {"round_robin": s_rr, "affinity": s_gp}
        print(f"\n  placement local-fraction (co-activations staying on one GPU):")
        print(f"    round-robin : {s_rr['local_fraction']*100:5.1f}%")
        print(f"    affinity    : {s_gp['local_fraction']*100:5.1f}%  "
              f"(greedy_partition)")
    except Exception as ex:  # pragma: no cover - keep the run alive
        print(f"\n  [placement step skipped: {ex}]")

    # --- 3. predictability ---
    # 3a. DirectProxy (jminding Tier-1, cross-layer, zero-training) — the
    # predictor that enables prefetching L+1 during L's compute.
    dp_by_layer = {L: float(np.mean(v)) for L, v in _DP_HITS.items() if v}
    dp_all = float(np.mean([x for v in _DP_HITS.values() for x in v])) if _DP_HITS else 0.0
    def _band(lo, hi):
        vals = [x for L, v in _DP_HITS.items() if lo <= L < hi for x in v]
        return float(np.mean(vals)) if vals else 0.0
    nL = args.n_layers
    dp_bands = {"early(0-31)": _band(0, nL // 3),
                "mid(31-62)": _band(nL // 3, 2 * nL // 3),
                "late(62-94)": _band(2 * nL // 3, nL)}
    print(f"\n  route predictability (prefetch hit-rate):")
    print(f"    DirectProxy L->L+1 (cross-layer)   : {dp_all*100:5.1f}%   "
          f"[early {dp_bands['early(0-31)']*100:.0f}% / "
          f"mid {dp_bands['mid(31-62)']*100:.0f}% / "
          f"late {dp_bands['late(62-94)']*100:.0f}%]")

    pers, pers_by_layer = persistence_hitrate(routes, args.n_layers, args.top_k)
    print(f"    persistence (token t -> t+1)       : {pers*100:5.1f}%")
    cov = {}
    for R in (8, 16, 32):
        hot, _ = hot_experts_per_layer(routes, args.n_layers, args.n_experts, R)
        c = static_hot_coverage(routes, hot)
        comb = combined_coverage(routes, hot, args.n_layers)
        cov[R] = {"static_hot": c, "combined": comb}
        print(f"    static top-{R:<2} hot                  : {c*100:5.1f}%   "
              f"| prev-token UNION top-{R}: {comb*100:5.1f}%")

    out = {
        "config": vars(args),
        "n_decode_tokens": n_tokens,
        # raw per-decode-token routes [token][layer] = [top-8 expert ids], so any
        # placement (incl. jminding's optimized_placement.json) can be evaluated
        # offline under the per-token latency metric without re-running the GPU.
        "routes": [step for prompt_routes in routes for step in prompt_routes],
        "gpu_imbalance": {"round_robin": imb_rr,
                          "replicate": {str(k): v for k, v in rep.items()}},
        "placement_local_fraction": local,
        "predictability": {
            "direct_proxy_overall": dp_all,
            "direct_proxy_bands": dp_bands,
            "direct_proxy_by_layer": {str(k): round(v, 3) for k, v in dp_by_layer.items()},
            "persistence": pers,
            "persistence_by_layer": pers_by_layer,
            "coverage": {str(k): v for k, v in cov.items()},
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[5] written to {args.out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
