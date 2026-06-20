#!/usr/bin/env python3
"""Measure the self-speculation viability curve on a real MoE: how well does a SHALLOW pass (first L
layers + the model's own norm + lm_head, i.e. the "logit lens") predict the FULL model's next token?

This is the deciding measurement for `docs/self-speculation-design.md`: self-spec drafts with the target's
first L_d layers, so its acceptance ≈ the top-1 agreement at depth L_d. The cost model needs τ above the
break-even (≈1.9 at L_d=12,k=2 … ≈2.8 at L_d=20,k=3), so we need the agreement curve to clear that bar at a
small-enough L_d that the draft (k·L_d/94) stays cheap.

Outputs agreement(top-1) and agreement(top-1 ∈ full top-4) at each depth, vs the full model's own logits.

  python tools/verify_self_speculation.py --model allenai/OLMoE-1B-7B-0924 --device cuda --dtype bfloat16
  python tools/verify_self_speculation.py --model /alloc/data/Qwen3-235B-A22B --device cuda --dtype bfloat16
"""
import argparse, statistics, sys

PROMPTS = [
    "The capital of France is Paris, and the city is famous for its art, fashion, and",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n",
    "In 1969, the Apollo 11 mission successfully landed the first humans on the surface of the",
    "Climate change is driven primarily by the emission of greenhouse gases such as carbon",
    "To make a good espresso you need finely ground coffee, water at about ninety-three degrees, and",
    "The theory of general relativity, published by Albert Einstein in 1915, describes gravity as the",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--depths", default="", help="comma list of layer depths; default = 25/50/75/90%% + last few")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    kw = dict(torch_dtype=getattr(torch, a.dtype), trust_remote_code=True)
    if a.device == "cuda":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if a.device != "cuda":
        model.to(a.device)

    base = model.model if hasattr(model, "model") else model
    norm = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
    lm_head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    if norm is None or lm_head is None:
        print("could not locate final norm / lm_head for this arch", file=sys.stderr); return 1
    n_layers = model.config.num_hidden_layers
    depths = ([int(x) for x in a.depths.split(",")] if a.depths
              else sorted({max(1, n_layers * p // 100) for p in (25, 50, 75, 90)}
                          | {n_layers - 2, n_layers - 1}))
    print(f"model={a.model}  layers={n_layers}  depths={depths}\n", flush=True)

    @torch.no_grad()
    def lens_top(hidden, k):  # apply the model's own norm+head to an intermediate hidden state
        logits = lm_head(norm(hidden.to(next(lm_head.parameters()).dtype)))
        return torch.topk(logits.float(), k, dim=-1).indices  # [..., k]

    agree1 = {d: [] for d in depths}     # shallow top-1 == full top-1
    agree4 = {d: [] for d in depths}     # shallow top-1 ∈ full top-4
    for p in PROMPTS[:a.prompts]:
        ids = tok(p, return_tensors="pt").input_ids.to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        full_top4 = torch.topk(out.logits.float(), 4, dim=-1).indices[0]  # [T,4]
        full_top1 = full_top4[:, 0]                                        # [T]
        hs = out.hidden_states                                            # len n_layers+1
        for d in depths:
            sh_top1 = lens_top(hs[d][0], 1)[:, 0]                          # [T]
            agree1[d].extend((sh_top1 == full_top1).float().tolist())
            in4 = [(sh_top1[t].item() in set(full_top4[t].tolist())) for t in range(full_top1.shape[0])]
            agree4[d].extend(1.0 if x else 0.0 for x in in4)

    print(f"{'depth (L_d/L)':>16}  {'top1 agree':>11}  {'top1∈full4':>11}  {'draft cost k·L_d/94':>20}")
    for d in depths:
        m1 = statistics.mean(agree1[d]) if agree1[d] else 0.0
        m4 = statistics.mean(agree4[d]) if agree4[d] else 0.0
        dc2 = 2 * d / n_layers  # draft cost for k=2 in "full decode" units
        print(f"{d:>6}/{n_layers:<3} {' ':>3} {m1:>11.3f}  {m4:>11.3f}  {dc2:>16.2f} (k=2)")
    print("\nRead: self-spec acceptance τ ≈ (top1 agree) · k. Viable if τ > break-even from")
    print("docs/self-speculation-design.md (≈1.86 at L_d=12,k=2). Pick the SMALLEST L_d that clears it —")
    print("smaller L_d = cheaper draft. If even deep L_d barely beats random, self-spec is out → use n-gram/MTP.")


if __name__ == "__main__":
    sys.exit(main())
