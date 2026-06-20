"""Kernel-level proof for adaptive top-k: B=1 expert-chain decode time vs k.

WHY: vLLM's fused MoE pads the grid (moe_align_block_size), so dropping experts
skips the weight LOAD but not the launch/align floor -> the byte saving may not
become wall-clock. And the team's real vLLM decode is comms-bound (~85 tok/s),
not weight-bound. Both obscure the lever. This isolates the question at the
expert-GEMV level (no engine, no padding, no comms): time the B=1 expert chain
(gate_up GEMV -> SiLU*gate -> down GEMV, accumulated) for k = 2,4,6,8 distinct
Qwen3-sized experts, under CUDA-graph capture (removes Python launch overhead).

If time(k) ~ floor + k*slope with slope dominant (bandwidth-bound on expert
weights), then adaptive-k converts the byte saving directly to wall-clock — the
win Charles's K5 (grid.x = nslot, no padding floor) realizes structurally.
Combined with router-mass concentration (tools/router_mass.py: how often k drops)
this gives the honest e2e expert-term speedup.

Usage: python3 tools/moe_kernel_microbench.py --dtype bf16 --iters 300
"""
from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F

HIDDEN = 4096
INTER = 1536          # Qwen3-235B moe_intermediate_size
N_EXPERTS = 8         # allocate 8 distinct experts so each k reads distinct HBM


def build(dtype, dev):
    # gate_up: [E, 2*inter, hidden]; down: [E, hidden, inter] (Qwen3 SwiGLU)
    gu = (torch.randn(N_EXPERTS, 2 * INTER, HIDDEN, dtype=dtype, device=dev) * 0.02)
    dn = (torch.randn(N_EXPERTS, HIDDEN, INTER, dtype=dtype, device=dev) * 0.02)
    return gu, dn


def chain(x, gu, dn, k):
    out = torch.zeros_like(x)
    for e in range(k):
        g = x @ gu[e].t()              # [2*inter]
        a, b = g.chunk(2, dim=-1)
        inter = F.silu(a) * b          # [inter]
        out = out + inter @ dn[e].t()  # [hidden]
    return out


def time_k(x, gu, dn, k, iters):
    for _ in range(8):                  # warmup
        chain(x, gu, dn, k)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):       # capture k GEMVs, no launch overhead on replay
        _ = chain(x, gu, dn, k)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # ms/token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default="/alloc/data/moe_kernel_microbench.json")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    dev = "cuda:0"
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    bytes_per = 2  # bf16/fp16
    print(f"device {torch.cuda.get_device_name(0)} dtype {a.dtype}")
    gu, dn = build(dtype, dev)
    x = torch.randn(HIDDEN, dtype=dtype, device=dev)

    bytes_per_expert = (2 * INTER * HIDDEN + HIDDEN * INTER) * bytes_per  # gate_up+down
    rows = []
    for k in (2, 4, 6, 8):
        ms = time_k(x, gu, dn, k, a.iters)
        gbps = k * bytes_per_expert / (ms / 1e3) / 1e9
        rows.append((k, ms, gbps))
        print(f"  k={k}: {ms:.4f} ms/tok  |  {gbps:7.1f} GB/s  "
              f"(expert bytes {k*bytes_per_expert/1e6:.1f} MB)")

    # Decompose time(k) = floor + k*slope via the k=2..8 endpoints.
    (k0, t0, _), (k1, t1, _) = rows[0], rows[-1]
    slope = (t1 - t0) / (k1 - k0)
    floor = t0 - slope * k0
    print(f"\n  time(k) ≈ {floor:.4f} ms floor + k * {slope:.4f} ms/expert")
    frac_bw = slope * 8 / (floor + slope * 8)
    print(f"  expert weight-read share at k=8: {frac_bw*100:.1f}% "
          f"(rest = floor) -> adaptive-k can recover at most this much")
    t8 = floor + slope * 8
    for kk in (4, 6):
        tk = floor + slope * kk
        print(f"  k={kk} vs k=8: {t8/tk:.3f}x faster on the expert chain "
              f"({(1-tk/t8)*100:.1f}% less expert time)")

    import json
    json.dump({"hidden": HIDDEN, "inter": INTER, "dtype": a.dtype,
               "rows": [{"k": k, "ms": ms, "gbps": g} for k, ms, g in rows],
               "floor_ms": floor, "slope_ms_per_expert": slope,
               "expert_bw_share_k8": frac_bw},
              open(a.out, "w"), indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
