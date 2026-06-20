#!/usr/bin/env python3
"""Verify the route-prediction (DirectProxy) premise on a real MoE — small model to sanity-check,
Qwen3-235B on the H100 box for the real number. Validates `engine/routing/predictor.rs` BEFORE the
team wires the prefetch on the 235B.

The DirectProxy predictor (zero-training) predicts layer L+1's top-k experts from the residual stream
`h_{L+1} ≈ h_L` by applying L+1's router weights to the current hidden state, then prefetches /
early-dispatches before L+1 computes. This script measures, on actual routing traces, whether that
premise holds and how accurate the prediction is:

  (A) DirectProxy accuracy   — top-k(W_gate[L+1] @ h_L) vs the ACTUAL top-k of layer L+1.
                               This is exactly what predictor.rs does. >> random => prefetch is sound.
  (B) layer-to-layer overlap — top-k(router_L) ∩ top-k(router_{L+1}). The residual-stability premise.
  (C) token-to-token overlap — same layer, consecutive tokens. Basis for markov/n-gram route caching
                               (cf. routing_stats.json markov_matrices).

Random baseline for all three = top_k / num_experts.

  # small MoE sanity check (download ~14GB):
  python tools/verify_route_prediction.py --model allenai/OLMoE-1B-7B-0924 --device cpu --prompts 6 --tokens 24
  # the real target, on the box (heavy — uses a slot; or run on a smaller Qwen MoE first):
  python tools/verify_route_prediction.py --model /alloc/data/Qwen3-235B-A22B --device cuda --dtype bfloat16
"""
import argparse, statistics, sys

PROMPTS = [
    "The capital of France is Paris, and the city is famous for",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "In 1969, the Apollo 11 mission successfully landed the first humans on the",
    "The mitochondria is the powerhouse of the cell because it",
    "import numpy as np\narr = np.array([1, 2, 3])\nprint(arr",
    "Once upon a time, in a kingdom far away, there lived a wise old",
    "The derivative of x squared with respect to x is",
    "To make a good espresso, you need finely ground coffee and water at about",
]


def topk_set(vec, k):
    import torch
    return set(torch.topk(vec, k).indices.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=0, help="0 = score the prompt only (no generation)")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    kw = dict(torch_dtype=getattr(torch, a.dtype), trust_remote_code=True, output_router_logits=True)
    if a.device == "cuda":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if a.device != "cuda":
        model.to(a.device)
    cfg = model.config
    topk = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "moe_topk", 2)
    n_exp = (getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
             or getattr(cfg, "n_routed_experts", 0))
    rnd = topk / n_exp if n_exp else float("nan")
    print(f"model={a.model}  experts={n_exp}  top_k={topk}  random_overlap≈{rnd:.3f}\n", flush=True)

    # locate per-layer router/gate linear modules (Qwen2/3-MoE, Mixtral, OLMoE: layer.mlp.gate)
    gates = []
    for layer in model.model.layers:
        g = None
        mlp = getattr(layer, "mlp", None)
        for name in ("gate", "router"):
            if mlp is not None and hasattr(mlp, name):
                g = getattr(mlp, name); break
        gates.append(g)  # None for dense layers

    dp_acc, ll_ov, tt_ov = [], [], []
    for p in PROMPTS[:a.prompts]:
        ids = tok(p, return_tensors="pt").input_ids.to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(ids, output_router_logits=True, output_hidden_states=True)
        # router_logits: tuple over MoE layers, each [n_tokens, n_experts]
        rl = [r for r in (out.router_logits or []) if r is not None]
        hs = out.hidden_states  # tuple len n_layers+1; hs[i] = input to layer i
        T = ids.shape[1]
        # actual top-k per MoE layer, last token
        moe_layer_idx = [i for i, g in enumerate(gates) if g is not None]
        actual = {}
        for j, i in enumerate(moe_layer_idx):
            if j < len(rl):
                actual[i] = topk_set(rl[j][-1].float(), topk)
        # (A) DirectProxy: apply layer i's gate to hs[i] (its input residual), compare to actual[i]
        for i in moe_layer_idx:
            if i not in actual or gates[i] is None:
                continue
            h = hs[i][0, -1].float()                       # residual stream entering layer i
            W = gates[i].weight.float()                     # [n_exp, hidden]
            pred = topk_set(W @ h.to(W.device), topk)
            dp_acc.append(len(pred & actual[i]) / topk)
        # (B) layer-to-layer overlap of actual selections
        for a_i, b_i in zip(moe_layer_idx, moe_layer_idx[1:]):
            if a_i in actual and b_i in actual:
                ll_ov.append(len(actual[a_i] & actual[b_i]) / topk)
        # (C) token-to-token overlap (same layer, last two tokens)
        if T >= 2:
            for j, i in enumerate(moe_layer_idx):
                if j < len(rl) and rl[j].shape[0] >= 2:
                    s1 = topk_set(rl[j][-1].float(), topk); s2 = topk_set(rl[j][-2].float(), topk)
                    tt_ov.append(len(s1 & s2) / topk)

    def report(name, xs):
        if xs:
            print(f"  {name:38s} mean {statistics.mean(xs):.3f}  (n={len(xs)}, random≈{rnd:.3f})")
        else:
            print(f"  {name:38s} no data (model may not expose router_logits)")

    print("route-prediction signals (1.0 = perfect, random baseline shown):")
    report("(A) DirectProxy accuracy", dp_acc)
    report("(B) layer-to-layer expert overlap", ll_ov)
    report("(C) token-to-token expert overlap", tt_ov)
    print("\nInterpretation: (A) >> random => predictor.rs DirectProxy works -> prefetch/early-dispatch is")
    print("sound; the gap to 1.0 is the misprediction rate (wasted prefetch). (B)/(C) high => routing has")
    print("structure the markov/n-gram route cache can exploit. Re-run on Qwen3-235B for the deploy number.")


if __name__ == "__main__":
    sys.exit(main())
