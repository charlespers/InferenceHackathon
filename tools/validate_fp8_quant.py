#!/usr/bin/env python3
"""validate_fp8_quant.py — DECISIVE numerical de-risk for real-text-on-our-kernels.

Question: does our per-row e4m3 fp8 quant of REAL Qwen3-235B weights, run through an fp8 GEMM
(torch._scaled_mm == the same rowwise-scaled fp8 matmul our cuBLASLt kernels do), reproduce the
bf16 output within tolerance? If yes, the engine's decode numerics are sound on real weights and
the remaining work is plumbing. If no, we caught the blocker before building the whole engine.

Samples one matrix of each kind (attn q/o, expert gate/down, lm_head) from real shards, on 1 GPU.
Run on the box:  python3 validate_fp8_quant.py /alloc/data/Qwen3-235B-A22B
"""
import sys, os, json
import torch
from safetensors import safe_open

E4M3_MAX = 448.0
def qrow(w):  # per-row e4m3, returns (fp8 tensor, row scales)
    amax = w.abs().amax(dim=1).clamp_min(1e-8)
    s = (amax / E4M3_MAX).float()
    q = (w / s[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, s
def qtensor(x):  # per-tensor e4m3 for the activation
    s = (x.abs().amax() / E4M3_MAX).clamp_min(1e-8).float()
    return (x / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), s

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/alloc/data/Qwen3-235B-A22B"
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    dev = "cuda"
    cache = {}
    def get(name):
        sh = idx[name]
        if sh not in cache: cache[sh] = safe_open(os.path.join(src, sh), framework="pt", device="cpu")
        return cache[sh].get_tensor(name).to(dev)

    samples = [
        ("L0 q_proj",   "model.layers.0.self_attn.q_proj.weight"),
        ("L0 o_proj",   "model.layers.0.self_attn.o_proj.weight"),
        ("L0 e0 gate",  "model.layers.0.mlp.experts.0.gate_proj.weight"),
        ("L0 e0 down",  "model.layers.0.mlp.experts.0.down_proj.weight"),
        ("L47 e63 up",  "model.layers.47.mlp.experts.63.up_proj.weight"),
        ("lm_head",     "lm_head.weight"),
    ]
    print(f"{'weight':<14}{'shape':>16}{'max_rel':>11}{'mean_rel':>11}{'cos_sim':>10}  verdict")
    worst = 0.0
    for label, name in samples:
        W = get(name).to(torch.bfloat16)            # [out, in]
        x = torch.randn(W.shape[1], device=dev, dtype=torch.bfloat16) * 0.1   # one activation vector (B=1)
        ref = (W.float() @ x.float())               # bf16-weight reference output
        Wq, Ws = qrow(W); xq, xs = qtensor(x)
        # fp8 GEMM with rowwise weight scale + per-tensor act scale (what our kernels compute)
        out = torch._scaled_mm(xq.view(1, -1), Wq.t(), scale_a=xs.view(1,1), scale_b=Ws.view(1,-1),
                               out_dtype=torch.bfloat16).float().view(-1)
        diff = (out - ref).abs()
        denom = ref.abs().clamp_min(1e-6)
        max_rel = (diff / denom).max().item()
        mean_rel = (diff.mean() / ref.abs().mean()).item()
        cos = torch.nn.functional.cosine_similarity(out, ref, dim=0).item()
        worst = max(worst, mean_rel)
        verd = "OK" if (mean_rel < 0.05 and cos > 0.999) else "CHECK"
        print(f"{label:<14}{str(tuple(W.shape)):>16}{max_rel:>11.4f}{mean_rel:>11.4f}{cos:>10.5f}  {verd}")
    print(f"\nVERDICT: {'fp8 quant FAITHFUL on real weights -> engine numerics sound (mean_rel '+format(worst,'.4f')+')' if worst < 0.05 else 'fp8 quant LOSSY -> needs better scheme (per-group / outliers)'}")

if __name__ == "__main__":
    main()
