"""
Qwen3-235B-A22B routing analysis.

Loads the model across 8 GPUs, hooks the MoE router gates, runs inference on
a set of prompts, and reports:
  - GPU memory distribution after load
  - Per-layer expert activation frequency
  - Co-activation heatmaps (which experts fire together)
  - Token-to-token Markov transition patterns (feeds our predictor)

Usage:
    python3 tools/routing_analysis.py \
        --model-path /alloc/data/Qwen3-235B-A22B \
        --n-prompts 20 \
        --out routing_stats.json
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def gpu_mem_snapshot(tag: str) -> list[dict]:
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]).decode()
    rows = []
    for line in out.strip().splitlines():
        idx, used, total, util, temp = [x.strip() for x in line.split(",")]
        rows.append({
            "gpu": int(idx), "used_mb": int(used), "total_mb": int(total),
            "util_pct": int(util), "temp_c": int(temp), "tag": tag,
        })
    return rows


def print_gpu_table(rows: list[dict]) -> None:
    print(f"\n  {'GPU':<5} {'Used MB':>9} {'Total MB':>9} {'%HBM':>6} {'Util%':>6} {'Temp':>5}")
    print("  " + "-" * 46)
    for r in rows:
        pct = r["used_mb"] / r["total_mb"] * 100
        print(f"  {r['gpu']:<5} {r['used_mb']:>9,} {r['total_mb']:>9,} "
              f"{pct:>5.1f}% {r['util_pct']:>5}% {r['temp_c']:>4}°C")


# ---------------------------------------------------------------------------
# Router hook logic
# ---------------------------------------------------------------------------

class RoutingTracer:
    """Collects expert selections from every MoE layer during forward passes."""

    def __init__(self, n_layers: int, n_experts: int, top_k: int):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        # activation_counts[layer][expert] = int
        self.activation_counts: list[list[int]] = [
            [0] * n_experts for _ in range(n_layers)
        ]
        # co_act[layer][(e1,e2)] = count of co-activations
        self.co_act: list[dict] = [defaultdict(int) for _ in range(n_layers)]
        # markov[layer][(prev_experts_frozenset, next_expert)] = count
        self.markov: list[dict] = [defaultdict(int) for _ in range(n_layers)]

        self._layer_cursor = 0
        self._prev_experts: list[int] = []
        self._hooks = []
        self._token_routes: list[list[list[int]]] = []  # [token][layer] = expert_ids
        self._current_token_layers: list[list[int]] = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # output from gate linear is router logits: [batch*seq, n_experts]
            logits = output
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            # top-k selection
            topk = torch.topk(logits, self.top_k, dim=-1).indices  # [B, top_k]
            for seq_pos in range(topk.shape[0]):
                experts = topk[seq_pos].tolist()
                # update counts
                for e in experts:
                    self.activation_counts[layer_idx][e] += 1
                # co-activation pairs
                for i, e1 in enumerate(experts):
                    for e2 in experts[i + 1:]:
                        key = (min(e1, e2), max(e1, e2))
                        self.co_act[layer_idx][key] += 1
                # track per-token routes
                if seq_pos < len(self._current_token_layers):
                    self._current_token_layers[seq_pos].append(experts)
        return hook

    def attach(self, model) -> None:
        """Find all MoE gate linear layers and attach hooks."""
        moe_layers_found = 0
        for name, module in model.named_modules():
            # Qwen3MoE gate is named 'gate' inside Qwen3MoeSparseMoeBlock
            if name.endswith(".mlp.gate") or name.endswith(".block_sparse_moe.gate"):
                layer_idx = self._extract_layer_idx(name)
                if layer_idx is not None and layer_idx < self.n_layers:
                    h = module.register_forward_hook(self._make_hook(layer_idx))
                    self._hooks.append(h)
                    moe_layers_found += 1
        print(f"  Attached hooks to {moe_layers_found} MoE gate layers")

    def _extract_layer_idx(self, name: str) -> int | None:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    def begin_token(self, n_seq_positions: int = 1) -> None:
        self._current_token_layers = [[] for _ in range(n_seq_positions)]

    def end_token(self) -> None:
        if self._current_token_layers:
            self._token_routes.append(self._current_token_layers[0])

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def top_experts(self, layer: int, n: int = 16) -> list[tuple[int, int]]:
        counts = self.activation_counts[layer]
        return sorted(enumerate(counts), key=lambda x: -x[1])[:n]

    def load_imbalance(self) -> list[float]:
        """Max-to-mean expert load ratio per layer."""
        result = []
        for counts in self.activation_counts:
            total = sum(counts)
            if total == 0:
                result.append(1.0)
                continue
            mean = total / self.n_experts
            result.append(max(counts) / mean)
        return result

    def markov_matrix(self, from_layer: int, to_layer: int) -> np.ndarray:
        """128×128 transition count matrix between two adjacent layers."""
        mat = np.zeros((self.n_experts, self.n_experts), dtype=np.float32)
        # Walk token routes
        for token_layers in self._token_routes:
            if from_layer < len(token_layers) and to_layer < len(token_layers):
                for e_from in token_layers[from_layer]:
                    for e_to in token_layers[to_layer]:
                        mat[e_from, e_to] += 1
        # Row-normalize
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return mat / row_sums

    def summary(self) -> dict:
        imbalance = self.load_imbalance()
        return {
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "total_tokens_traced": len(self._token_routes),
            "mean_load_imbalance": float(np.mean(imbalance)),
            "max_load_imbalance": float(np.max(imbalance)),
            "layer_imbalance": [round(x, 3) for x in imbalance],
            "activation_counts": self.activation_counts,
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    "Explain the attention mechanism in transformers.",
    "Write a Python function to merge two sorted lists.",
    "What is the capital of France?",
    "Describe the process of photosynthesis in detail.",
    "Solve: if 2x + 5 = 13, what is x?",
    "Write a haiku about machine learning.",
    "What are the main causes of World War I?",
    "Implement a binary search tree in Rust.",
    "Explain quantum entanglement to a 10-year-old.",
    "What is the difference between supervised and unsupervised learning?",
    "Write a SQL query to find the top 5 customers by revenue.",
    "How does garbage collection work in Go?",
    "Explain the CAP theorem in distributed systems.",
    "What is the time complexity of quicksort?",
    "Write a regex to validate email addresses.",
    "Describe the architecture of a transformer model.",
    "How do you implement a LRU cache?",
    "What is backpropagation and how does it work?",
    "Explain RLHF (reinforcement learning from human feedback).",
    "Write a function to compute the Fibonacci sequence efficiently.",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/alloc/data/Qwen3-235B-A22B")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--out", default="routing_stats.json")
    parser.add_argument("--n-layers", type=int, default=94)
    parser.add_argument("--n-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-235B-A22B Routing Analysis")
    print("=" * 60)

    # Pre-load GPU snapshot
    print("\n[1] Pre-load GPU state:")
    pre_load = gpu_mem_snapshot("pre_load")
    print_gpu_table(pre_load)

    # Load model
    print(f"\n[2] Loading model from {args.model_path} ...")
    print("    (device_map='auto' — accelerate splits across all 8 GPUs)")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"    Loaded in {load_time:.1f}s")

    # Post-load GPU snapshot
    print("\n[3] Post-load GPU state (HBM after weight distribution):")
    post_load = gpu_mem_snapshot("post_load")
    print_gpu_table(post_load)
    for pre, post in zip(pre_load, post_load):
        delta = post["used_mb"] - pre["used_mb"]
        print(f"    GPU {post['gpu']}: +{delta:,} MB loaded")

    # Attach routing hooks
    print("\n[4] Attaching routing hooks ...")
    tracer = RoutingTracer(
        n_layers=args.n_layers,
        n_experts=args.n_experts,
        top_k=args.top_k,
    )
    tracer.attach(model)

    # Run inference
    prompts = PROMPTS[:args.n_prompts]
    print(f"\n[5] Running {len(prompts)} prompts ({args.max_new_tokens} tokens each) ...")
    latencies = []

    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            t_start = time.time()
            tracer.begin_token(inputs["input_ids"].shape[1])

            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            tracer.end_token()
            elapsed = (time.time() - t_start) * 1000
            n_new = outputs.shape[1] - inputs["input_ids"].shape[1]
            latencies.append(elapsed / max(n_new, 1))
            print(f"  [{i+1:2}/{len(prompts)}] {elapsed/1000:.1f}s  "
                  f"({n_new} new tokens, {elapsed/max(n_new,1):.1f} ms/tok)")
            sys.stdout.flush()

    tracer.detach()

    # Post-inference GPU snapshot
    print("\n[6] Post-inference GPU state:")
    post_inf = gpu_mem_snapshot("post_inference")
    print_gpu_table(post_inf)

    # Analysis
    summary = tracer.summary()
    print(f"\n[7] Routing analysis ({summary['total_tokens_traced']} tokens traced):")
    print(f"    Mean load imbalance (max/mean expert load): "
          f"{summary['mean_load_imbalance']:.3f}x")
    print(f"    Max  load imbalance: {summary['max_load_imbalance']:.3f}x")

    print(f"\n    Per-layer imbalance (first 10 / last 5 layers):")
    imb = summary["layer_imbalance"]
    for l in list(range(min(10, len(imb)))) + list(range(max(0, len(imb)-5), len(imb))):
        bar = "█" * int(imb[l] * 10)
        print(f"      L{l:3d}: {imb[l]:.3f}x  {bar}")

    print(f"\n    Top-16 hottest experts across all layers:")
    all_counts: list[tuple[int, int, int]] = []  # (layer, expert, count)
    for l, counts in enumerate(summary["activation_counts"]):
        for e, c in enumerate(counts):
            all_counts.append((l, e, c))
    all_counts.sort(key=lambda x: -x[2])
    for l, e, c in all_counts[:16]:
        print(f"      L{l:3d} E{e:3d}: {c:5d} activations")

    print(f"\n    Decode latency: {np.mean(latencies):.1f} ms/tok "
          f"(p50={np.percentile(latencies, 50):.1f}, "
          f"p95={np.percentile(latencies, 95):.1f})")

    # Save full stats
    out_data = {
        "routing": summary,
        "gpu_snapshots": {
            "pre_load": pre_load,
            "post_load": post_load,
            "post_inference": post_inf,
        },
        "load_time_s": load_time,
        "latency_ms_per_tok": latencies,
    }
    # Save markov matrices for the first 10 layer transitions
    out_data["markov_matrices"] = {}
    for l in range(min(10, args.n_layers - 1)):
        mat = tracer.markov_matrix(l, l + 1)
        out_data["markov_matrices"][f"{l}->{l+1}"] = mat.tolist()

    with open(args.out, "w") as f:
        json.dump(out_data, f)
    print(f"\n[8] Full stats written to {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
