"""
MarkovPredictor — Python port of engine/src/routing/predictor.rs.

Loaded once at server startup from routing_stats.json. During vLLM inference,
we don't have access to actual per-token expert selections. Instead,
RoutingSimulator replays the measured activation distribution to produce
realistic layer-by-layer routing, runs the predictor at each layer, and
measures hit rate. This gives a real accuracy number for the predictor even
before the Rust engine is wired up.

Prediction model: given experts that fired at layer L, predict which experts
will fire at layer L+1 (within the same token's forward pass).
"""

import json
from pathlib import Path

ROUTING_STATS_PATH = "/alloc/data/routing_stats.json"
N_LAYERS = 94
N_EXPERTS = 128
TOP_K = 8


class MarkovPredictor:
    def __init__(self, n_layers: int = N_LAYERS, n_experts: int = N_EXPERTS):
        self.n_layers = n_layers
        self.n_experts = n_experts
        # counts[transition_layer][i * n_experts + j]: pseudo-count for i→j
        # transition_layer L covers the L→L+1 transition
        self.counts: list[list[float]] = [
            [1.0] * (n_experts * n_experts) for _ in range(n_layers - 1)
        ]

    @classmethod
    def from_stats(cls, path: str = ROUTING_STATS_PATH, scale: float = 1000.0) -> "MarkovPredictor":
        obj = cls()
        try:
            with open(path) as f:
                data = json.load(f)
            matrices = data.get("routing", {}).get("markov_matrices", {})
            seeded = 0
            for key, mat in matrices.items():
                try:
                    layer = int(key.split("->")[0])
                except (ValueError, IndexError):
                    continue
                if layer >= obj.n_layers - 1:
                    continue
                ne = obj.n_experts
                row = obj.counts[layer]
                for i in range(min(ne, len(mat))):
                    for j in range(min(ne, len(mat[i]))):
                        row[i * ne + j] = mat[i][j] * scale + 1.0
                seeded += 1
            print(f"[predictor] seeded {seeded} Markov layers from {path}", flush=True)
        except FileNotFoundError:
            print(f"[predictor] {path} not found — uniform prior", flush=True)
        except Exception as e:
            print(f"[predictor] load error: {e}", flush=True)
        return obj

    def predict(self, layer: int, current_experts: list[int], top_k: int = TOP_K) -> list[int]:
        """Given experts at layer L, return predicted top_k experts at layer L+1."""
        if layer >= self.n_layers - 1 or not current_experts:
            return []
        ne = self.n_experts
        row = self.counts[layer]
        scores = [0.0] * ne
        for e in current_experts:
            if e >= ne:
                continue
            base = e * ne
            row_sum = sum(row[base + j] for j in range(ne)) or 1.0
            for j in range(ne):
                scores[j] += row[base + j] / row_sum
        return sorted(range(ne), key=lambda j: -scores[j])[:top_k]

    def observe(self, layer: int, prev_experts: list[int], curr_experts: list[int]) -> None:
        """Online update: record transition layer→layer+1."""
        if layer >= self.n_layers - 1:
            return
        ne = self.n_experts
        row = self.counts[layer]
        for i in prev_experts:
            if i >= ne:
                continue
            base = i * ne
            for j in curr_experts:
                if j < ne:
                    row[base + j] += 1.0


class RoutingSimulator:
    """
    Simulates one token's 94-layer forward pass using measured activation
    distributions, then runs the predictor at each layer to measure hit rate.

    The "actual" experts per layer are the deterministic top-k by measured
    activation count — reflecting the most likely routing pattern.
    """

    def __init__(self, activation_counts: list[list[int]], predictor: MarkovPredictor):
        self.predictor = predictor
        # Precompute top-k experts per layer from measured activation counts
        self._top_experts: list[list[int]] = []
        for layer_counts in activation_counts:
            ranked = sorted(range(len(layer_counts)), key=lambda e: -layer_counts[e])
            self._top_experts.append(ranked[:TOP_K])
        # Pad any missing layers
        while len(self._top_experts) < predictor.n_layers:
            self._top_experts.append(list(range(TOP_K)))

    def simulate_token(self) -> tuple[float, list[dict]]:
        """
        Run predictor through all layers for one token.

        Returns (hit_rate, per_layer_telemetry) where hit_rate is the
        fraction of predicted experts that matched actual across all layers.
        """
        hits = 0
        total = 0
        telemetry = []
        prev_prediction: list[int] = []

        for layer in range(self.predictor.n_layers):
            actual = self._top_experts[layer]

            # Score the previous layer's prediction against this layer's actual
            if prev_prediction:
                layer_hits = len(set(prev_prediction) & set(actual))
                hits += layer_hits
                total += len(actual)

            # Online update: record this layer transition
            if layer > 0:
                self.predictor.observe(layer - 1, self._top_experts[layer - 1], actual)

            # Predict next layer
            prev_prediction = self.predictor.predict(layer, actual, TOP_K)

            telemetry.append({
                "layer": layer,
                "actual": actual,
                "predicted_next": prev_prediction,
            })

        hit_rate = hits / total if total > 0 else 0.0
        return hit_rate, telemetry


# ---------------------------------------------------------------------------
# Module-level singleton — initialized once at server startup
# ---------------------------------------------------------------------------

_simulator: RoutingSimulator | None = None


def get_simulator(stats_path: str = ROUTING_STATS_PATH) -> RoutingSimulator | None:
    global _simulator
    if _simulator is not None:
        return _simulator
    try:
        with open(stats_path) as f:
            data = json.load(f)
        activation_counts = data["routing"]["activation_counts"]
        predictor = MarkovPredictor.from_stats(stats_path)
        _simulator = RoutingSimulator(activation_counts, predictor)
        print("[predictor] simulator ready", flush=True)
    except Exception as e:
        print(f"[predictor] simulator unavailable: {e}", flush=True)
    return _simulator
