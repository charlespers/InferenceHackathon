"""Graph-based expert placement and co-activation analysis.

Problem: Qwen3-235B-A22B has 128 experts per layer and 94 layers. At B=1 a
token activates top-8 experts per layer. With round-robin placement
(expert_id % 8 → GPU) each decode step touches ~6-8 of the 8 GPUs, requiring
all-to-all communication twice per layer (dispatch + combine).

Goal: find a placement of 128 experts onto 8 GPUs that minimises the expected
number of cross-GPU hops (co-activations that land on different GPUs).

Approach:
1. Collect a co-activation graph from routing telemetry.
   Nodes  = experts (identified as (layer, expert_id) pairs).
   Edges  = pairs of experts co-activated in the same token step (same layer)
            or in adjacent layers (layer L expert → layer L+1 expert).
   Weight = frequency of co-activation (higher → prefer same GPU).

2. Run a balanced k-way graph partition (k=8 GPUs, balance constraint: each GPU
   holds ≤ceil(experts/8) experts per layer).

3. Evaluate: cross_gpu_fraction = edges that cross partition / total edges.

The partition is a min-cut relaxation; we use a greedy approach (no external
deps) that is exact for the intra-layer case when routing is uniform but
exploits non-uniformity when token clusters prefer certain expert subsets.

Topological sort insight:
  Within a single token's forward pass, expert activations form a DAG:
    (layer 0, expert set) → (layer 1, expert set) → ... → (layer 93, expert set)
  A topo sort of this DAG = the layer order (it's linear, so trivial).
  The value is in *prefetching*: if we can predict the next layer's expert set
  before the current layer finishes, we can overlap the all-to-all dispatch of
  layer L+1 with the expert FFN compute of layer L (hiding comms latency).
  The routing.py module also exposes a simple DAG representation for this.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Co-activation graph
# ---------------------------------------------------------------------------

@dataclass
class CoActGraph:
    """Undirected weighted graph of expert co-activations."""
    # edge (a, b) with a < b → count
    edges: dict[tuple[int, int], int] = field(default_factory=lambda: defaultdict(int))
    # node → total activation count
    node_freq: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    n_experts: int = 128
    n_layers: int = 94

    def node_id(self, layer: int, expert_id: int) -> int:
        return layer * self.n_experts + expert_id

    def add_token_step(self, layer: int, expert_ids: list[int]) -> None:
        """Record one token's expert activations for a single layer."""
        nodes = [self.node_id(layer, e) for e in expert_ids]
        for n in nodes:
            self.node_freq[n] += 1
        # add co-activation edges for all pairs in this layer
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = min(nodes[i], nodes[j]), max(nodes[i], nodes[j])
                self.edges[(a, b)] += 1

    def add_cross_layer(self, layer: int, experts_l: list[int],
                        experts_l1: list[int]) -> None:
        """Record token path edges from layer L to L+1."""
        for e0 in experts_l:
            for e1 in experts_l1:
                a = self.node_id(layer, e0)
                b = self.node_id(layer + 1, e1)
                key = (min(a, b), max(a, b))
                self.edges[key] += 1

    def total_edge_weight(self) -> int:
        return sum(self.edges.values())

    def cross_gpu_weight(self, placement: dict[int, int]) -> int:
        """Sum of edge weights that cross GPU boundaries given a placement map
        {node_id → gpu_id}."""
        total = 0
        for (a, b), w in self.edges.items():
            if placement.get(a, -1) != placement.get(b, -2):
                total += w
        return total


# ---------------------------------------------------------------------------
# Ingest telemetry dicts (from the x_telemetry SSE stream)
# ---------------------------------------------------------------------------

def ingest_telemetry(graph: CoActGraph, telemetry_events: list[dict]) -> None:
    """Add token steps from a list of x_telemetry dicts into the graph.

    Each event has: {"token_index": int, "experts": [{"layer": int, "expert_id": int, "gpu": int}, ...]}
    """
    for ev in telemetry_events:
        experts = ev.get("experts", [])
        by_layer: dict[int, list[int]] = defaultdict(list)
        for e in experts:
            by_layer[e["layer"]].append(e["expert_id"])
        layers = sorted(by_layer)
        for layer in layers:
            graph.add_token_step(layer, by_layer[layer])
        for i in range(len(layers) - 1):
            graph.add_cross_layer(layers[i], by_layer[layers[i]],
                                  by_layer[layers[i + 1]])


# ---------------------------------------------------------------------------
# Greedy balanced partition
# ---------------------------------------------------------------------------

def greedy_partition(
    graph: CoActGraph,
    n_gpus: int = 8,
    n_experts: int = 128,
    n_layers: int = 94,
) -> dict[int, int]:
    """Greedy balanced k-way partition minimising cross-GPU edge weight.

    Approach: for each layer independently, assign experts to GPUs greedily.
    Experts with high co-activation affinity are placed together.
    Balance constraint: each GPU gets exactly n_experts // n_gpus experts/layer.

    Returns: {node_id → gpu_id}
    """
    slots = n_experts // n_gpus  # experts per GPU per layer
    placement: dict[int, int] = {}

    for layer in range(n_layers):
        nodes = [graph.node_id(layer, e) for e in range(n_experts)]
        gpu_load: dict[int, int] = {g: 0 for g in range(n_gpus)}
        assigned: dict[int, int] = {}

        # Sort nodes by total frequency descending (hot experts placed first).
        nodes_sorted = sorted(nodes, key=lambda n: graph.node_freq.get(n, 0),
                              reverse=True)

        for node in nodes_sorted:
            # Score each GPU: sum of edge weights to already-assigned neighbours.
            scores = {g: 0 for g in range(n_gpus)}
            for g in range(n_gpus):
                for other, gpu in assigned.items():
                    if gpu != g:
                        continue
                    key = (min(node, other), max(node, other))
                    scores[g] += graph.edges.get(key, 0)

            # Pick highest-scoring GPU that still has capacity.
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            for g, _ in ranked:
                if gpu_load[g] < slots:
                    assigned[node] = g
                    gpu_load[g] += 1
                    break
            else:
                # fallback: first GPU with remaining capacity
                for g in range(n_gpus):
                    if gpu_load[g] < slots:
                        assigned[node] = g
                        gpu_load[g] += 1
                        break

        placement.update(assigned)

    return placement


def round_robin_placement(n_gpus: int = 8, n_experts: int = 128,
                          n_layers: int = 94) -> dict[int, int]:
    """Baseline: expert_id % n_gpus (current default)."""
    placement = {}
    for layer in range(n_layers):
        for e in range(n_experts):
            node = layer * n_experts + e
            placement[node] = e % n_gpus
    return placement


# ---------------------------------------------------------------------------
# DAG / topological sort for pipelined prefetch scheduling
# ---------------------------------------------------------------------------

@dataclass
class TokenDAG:
    """DAG of expert activations for a single token's forward pass.

    Nodes: (layer, expert_id) pairs.
    Edges: layer L node → layer L+1 node for all pairs (sequential dependency).

    The topological sort of this DAG is just the layer order, but the real
    value is in exposing the *frontier*: at layer L, all nodes in layer L+1
    are ready to prefetch as soon as L's dispatch is issued, because the only
    dependency is the layer L output — which is computed while the prefetch
    can happen in parallel.
    """
    # layer → list of expert_ids activated
    activations: dict[int, list[int]] = field(default_factory=dict)

    def add_layer(self, layer: int, expert_ids: list[int]) -> None:
        self.activations[layer] = list(expert_ids)

    def topo_layers(self) -> list[int]:
        return sorted(self.activations)

    def prefetch_schedule(self, placement: dict[int, int],
                          n_experts: int = 128) -> list[dict]:
        """Return a per-layer prefetch plan.

        Each entry: {
          "compute_layer": L,
          "prefetch_layer": L+1,
          "prefetch_experts": [(expert_id, gpu_id), ...],
          "cross_gpu": bool  (whether prefetch experts are on multiple GPUs),
        }

        The idea: while computing layer L, issue the all-to-all dispatch for
        layer L+1. This hides the dispatch latency behind compute.
        """
        layers = self.topo_layers()
        schedule = []
        for i, layer in enumerate(layers[:-1]):
            next_layer = layers[i + 1]
            next_experts = self.activations.get(next_layer, [])
            gpus = set()
            experts_gpus = []
            for e in next_experts:
                node = next_layer * n_experts + e
                g = placement.get(node, e % 8)
                gpus.add(g)
                experts_gpus.append((e, g))
            schedule.append({
                "compute_layer": layer,
                "prefetch_layer": next_layer,
                "prefetch_experts": experts_gpus,
                "cross_gpu": len(gpus) > 1,
                "n_gpus_touched": len(gpus),
            })
        return schedule


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def placement_stats(graph: CoActGraph, placement: dict[int, int],
                    label: str = "placement") -> dict:
    total = graph.total_edge_weight()
    cross = graph.cross_gpu_weight(placement)
    return {
        "label": label,
        "total_edge_weight": total,
        "cross_gpu_weight": cross,
        "cross_gpu_fraction": round(cross / total, 4) if total else 0.0,
        "local_fraction": round(1 - cross / total, 4) if total else 1.0,
    }
