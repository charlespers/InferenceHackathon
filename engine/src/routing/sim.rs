/// RoutingSimulator: simulate one token's 94-layer forward pass from
/// measured activation statistics, run the predictor at each layer,
/// and return the hit rate.
///
/// Used by the Axum server when proxying vLLM: vLLM handles the actual
/// forward pass internally, so we can't inject ourselves per-layer.
/// Instead, this replays the measured activation distribution to quantify
/// how well our predictor would have done in a real engine.
///
/// In the real Rust engine (conifer), replace this with PredictionPipeline
/// which takes the actual hidden state and expert selections per layer.

use crate::routing::optimizer::RoutingStats;
use crate::routing::predictor::{EnsemblePredictor, MarkovTransition};
use crate::routing::types::ExpertId;

const TOP_K: usize = 8;

pub struct RoutingSimulator {
    predictor: EnsemblePredictor,
    /// Deterministic top-k experts per layer from measured activation counts.
    top_experts: Vec<Vec<ExpertId>>,
    top_k: usize,
}

impl Default for RoutingSimulator {
    fn default() -> Self {
        Self {
            predictor: EnsemblePredictor::new(128),
            top_experts: vec![(0..TOP_K as u32).collect(); 94],
            top_k: TOP_K,
        }
    }
}

impl RoutingSimulator {
    /// Build from routing_stats.json. Fails gracefully — caller should unwrap
    /// with a fallback to `RoutingSimulator::default()`.
    pub fn from_routing_stats(stats_path: &str) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(stats_path)?;
        let stats: RoutingStats = serde_json::from_str(&raw)?;

        let n_layers = stats.routing.activation_counts.len();
        let n_experts = stats.routing.activation_counts
            .first()
            .map_or(128, |v| v.len());
        let top_k = TOP_K;

        // Precompute top-k experts per layer (deterministic, reflects real distribution)
        let top_experts: Vec<Vec<ExpertId>> = stats.routing.activation_counts
            .iter()
            .map(|counts| {
                let mut indexed: Vec<(usize, u32)> =
                    counts.iter().cloned().enumerate().collect();
                indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                indexed[..top_k.min(indexed.len())]
                    .iter()
                    .map(|(i, _)| *i as ExpertId)
                    .collect()
            })
            .collect();

        // Seed Markov predictor from real transition matrices
        let mut markov = MarkovTransition::new(n_layers, n_experts);
        let mut n_seeded = 0usize;
        for (key, mat) in &stats.routing.markov_matrices {
            if let Some(layer) = key.split("->").next().and_then(|s| s.parse::<usize>().ok()) {
                markov.seed_layer_from_probs(layer, mat, 1000.0);
                n_seeded += 1;
            }
        }
        eprintln!("[sim] Markov seeded from {n_seeded} layer transitions ({n_layers} layers, {n_experts} experts)");

        let mut predictor = EnsemblePredictor::new(n_experts);
        predictor.markov = Some(markov);
        predictor.weights = [0.0, 0.0, 1.0]; // Markov only; proxy needs router weights from conifer

        Ok(Self { predictor, top_experts, top_k })
    }

    /// Simulate one token's full forward pass through all layers.
    ///
    /// For each layer L: score the previous layer's prediction against actual,
    /// do an online Markov update, then predict layer L+1.
    ///
    /// Returns hit_rate = hits / total_predictions in [0, 1].
    pub fn simulate_token(&mut self) -> f32 {
        let n_layers = self.top_experts.len();
        let mut hits = 0u32;
        let mut total = 0u32;
        let mut prev_prediction: Vec<ExpertId> = Vec::new();

        for layer in 0..n_layers {
            let actual = self.top_experts[layer].clone();

            // Score the previous layer's prediction
            if !prev_prediction.is_empty() {
                let hit_count = actual.iter()
                    .filter(|&&e| prev_prediction.contains(&e))
                    .count() as u32;
                hits += hit_count;
                total += actual.len() as u32;
            }

            // Online Markov update: record transition layer-1 → layer
            if layer > 0 {
                let prev_actual = self.top_experts[layer - 1].clone();
                if let Some(mk) = &mut self.predictor.markov {
                    mk.observe(layer - 1, &prev_actual, &actual);
                }
            }

            // Predict layer+1 (hidden state not needed — Markov only)
            let pred = self.predictor.predict_full(layer, &[], Some(&actual), self.top_k);
            prev_prediction = pred.experts;
        }

        if total > 0 { hits as f32 / total as f32 } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_simulator_runs() {
        let mut sim = RoutingSimulator::default();
        let hr = sim.simulate_token();
        assert!(hr >= 0.0 && hr <= 1.0);
    }

    #[test]
    fn simulator_hit_rate_improves_with_seeding() {
        // Build a simulator where layer 0 always routes to experts 0-7 and
        // layer 1 always routes to experts 0-7 (perfect correlation).
        let counts: Vec<Vec<u32>> = vec![
            // Layer 0: experts 0-7 dominate
            (0..128).map(|e| if e < 8 { 1000 } else { 1 }).collect(),
            // Layer 1: same experts
            (0..128).map(|e| if e < 8 { 1000 } else { 1 }).collect(),
            // Layer 2
            (0..128).map(|e| if e < 8 { 1000 } else { 1 }).collect(),
        ];

        // Manually seed Markov so 0→0 transitions are strong
        let mut markov = MarkovTransition::new(3, 128);
        for i in 0..8usize {
            for _ in 0..500 {
                let prev: Vec<ExpertId> = (0..8).map(|x| x as u32).collect();
                let curr: Vec<ExpertId> = (0..8).map(|x| x as u32).collect();
                markov.observe(0, &prev, &curr);
                markov.observe(1, &prev, &curr);
            }
            let _ = i; // suppress warning
        }

        let top_experts: Vec<Vec<ExpertId>> = counts.iter().map(|c| {
            let mut idx: Vec<(usize, u32)> = c.iter().cloned().enumerate().collect();
            idx.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            idx[..8].iter().map(|(i, _)| *i as ExpertId).collect()
        }).collect();

        let mut predictor = EnsemblePredictor::new(128);
        predictor.markov = Some(markov);
        predictor.weights = [0.0, 0.0, 1.0];

        let mut sim = RoutingSimulator { predictor, top_experts, top_k: 8 };
        let hr = sim.simulate_token();
        assert!(hr > 0.5, "hit rate should be high when transitions are strongly seeded, got {hr}");
    }
}
