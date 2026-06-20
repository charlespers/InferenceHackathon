"""
Build a load-balanced expert placement from routing_stats.json.

Algorithm:
  1. Greedy load-balanced placement: sort experts by activation count desc,
     assign each to the least-loaded GPU. This separates hot experts.
  2. Hot-expert replication: experts firing > threshold * layer-mean get a
     replica on a second GPU, splitting their load.

Output: optimized_placement.json
  {
    "placement": {"0": {"0": 2, "1": 5, ...}, ...},   # layer -> expert -> primary GPU
    "replicas":  {"17": {"78": 3}, ...},               # layer -> expert -> replica GPU
    "stats": {...}
  }

Usage:
    python3 tools/placement_optimizer.py \
        [routing_stats.json] [optimized_placement.json]
"""

import json
import sys
from pathlib import Path

N_GPUS = 8
N_EXPERTS = 128


def greedy_balanced(counts: list[int], n_gpus: int) -> list[int]:
    """Sort experts by count desc, assign each to least-loaded GPU."""
    order = sorted(range(len(counts)), key=lambda e: -counts[e])
    gpu_load = [0.0] * n_gpus
    placement = [0] * len(counts)
    for e in order:
        g = min(range(n_gpus), key=lambda g: gpu_load[g])
        placement[e] = g
        gpu_load[g] += counts[e]
    return placement


def find_replicas(counts: list[int], placement: list[int],
                  n_gpus: int, threshold_mult: float = 2.0) -> dict[int, int]:
    """Return {expert_id: replica_gpu} for experts above threshold."""
    total = sum(counts)
    if total == 0:
        return {}
    mean = total / len(counts)
    threshold = threshold_mult * mean

    gpu_load = [0.0] * n_gpus
    for e, c in enumerate(counts):
        gpu_load[placement[e]] += c

    replicas: dict[int, int] = {}
    hot = sorted(
        [e for e in range(len(counts)) if counts[e] > threshold],
        key=lambda e: -counts[e],
    )
    for e in hot:
        primary = placement[e]
        replica_g = min((g for g in range(n_gpus) if g != primary),
                        key=lambda g: gpu_load[g])
        replicas[e] = replica_g
        # Split load between primary and replica going forward
        gpu_load[replica_g] += counts[e] * 0.5
        gpu_load[primary] -= counts[e] * 0.5
    return replicas


def imbalance(counts: list[int], placement: list[int],
              replicas: dict[int, int], n_gpus: int) -> float:
    """Max/mean GPU load ratio. 1.0 = perfect balance."""
    gpu_load = [0.0] * n_gpus
    for e, c in enumerate(counts):
        p = placement[e]
        if e in replicas:
            gpu_load[p] += c * 0.5
            gpu_load[replicas[e]] += c * 0.5
        else:
            gpu_load[p] += c
    total = sum(gpu_load)
    if total == 0:
        return 1.0
    return max(gpu_load) / (total / n_gpus)


def rr_imbalance(counts: list[int], n_gpus: int) -> float:
    gpu_load = [0.0] * n_gpus
    for e, c in enumerate(counts):
        gpu_load[e % n_gpus] += c
    total = sum(gpu_load)
    if total == 0:
        return 1.0
    return max(gpu_load) / (total / n_gpus)


def main():
    stats_path = sys.argv[1] if len(sys.argv) > 1 else "/alloc/data/routing_stats.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/alloc/data/optimized_placement.json"

    print(f"Loading {stats_path} ...")
    with open(stats_path) as f:
        stats = json.load(f)

    activation_counts: list[list[int]] = stats["routing"]["activation_counts"]
    n_layers = len(activation_counts)
    print(f"  {n_layers} layers × {len(activation_counts[0])} experts")

    out_placement: dict[str, dict[str, int]] = {}
    out_replicas: dict[str, dict[str, int]] = {}

    rr_total = 0.0
    opt_total = 0.0
    total_replicated = 0

    for layer in range(n_layers):
        counts = activation_counts[layer]

        rr_imb = rr_imbalance(counts, N_GPUS)
        rr_total += rr_imb

        layer_placement = greedy_balanced(counts, N_GPUS)
        layer_replicas = find_replicas(counts, layer_placement, N_GPUS)
        opt_imb = imbalance(counts, layer_placement, layer_replicas, N_GPUS)
        opt_total += opt_imb
        total_replicated += len(layer_replicas)

        out_placement[str(layer)] = {str(e): layer_placement[e] for e in range(N_EXPERTS)}
        if layer_replicas:
            out_replicas[str(layer)] = {str(e): g for e, g in layer_replicas.items()}

    mean_rr = rr_total / n_layers
    mean_opt = opt_total / n_layers
    improvement = (1.0 - mean_opt / mean_rr) * 100

    print(f"\nResults:")
    print(f"  Round-robin mean imbalance : {mean_rr:.3f}x")
    print(f"  Optimized  mean imbalance  : {mean_opt:.3f}x")
    print(f"  Improvement                : {improvement:.1f}%")
    print(f"  Hot experts replicated     : {total_replicated} expert-layer pairs")

    # Show top 5 hottest expert-layer pairs that got replicated
    replicated_pairs = []
    for layer in range(n_layers):
        counts = activation_counts[layer]
        layer_str = str(layer)
        if layer_str in out_replicas:
            for e_str, replica_gpu in out_replicas[layer_str].items():
                e = int(e_str)
                primary = out_placement[layer_str][e_str]
                replicated_pairs.append((counts[e], layer, e, primary, replica_gpu))
    replicated_pairs.sort(reverse=True)
    print(f"\n  Top 10 replicated experts (hottest first):")
    for count, layer, expert, primary, replica in replicated_pairs[:10]:
        print(f"    L{layer:3d} E{expert:3d}: {count:5d} activations  "
              f"GPU {primary} + replica GPU {replica}")

    out = {
        "placement": out_placement,
        "replicas": out_replicas,
        "stats": {
            "n_layers": n_layers,
            "n_experts": N_EXPERTS,
            "n_gpus": N_GPUS,
            "round_robin_mean_imbalance": round(mean_rr, 4),
            "optimized_mean_imbalance": round(mean_opt, 4),
            "improvement_pct": round(improvement, 1),
            "total_replicated_expert_layers": total_replicated,
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
