/// Routing statistics: accuracy tracking and trace collection.
///
/// Two jobs:
///   1. Compare predictions against actual outcomes to compute hit rates.
///   2. Collect (hidden, routing) pairs as training data for LearnedLinear.
///
/// Designed to run on the hot path with minimal overhead. All writes are
/// to pre-allocated buffers; no allocation on the per-token path.

use crate::routing::types::{ExpertId, ExpertPrediction, LayerAccuracy, RouteOutcome};

// ---------------------------------------------------------------------------
// Accuracy tracker
// ---------------------------------------------------------------------------

/// Tracks prediction accuracy over a sliding window of recent tokens.
pub struct AccuracyTracker {
    n_layers: usize,
    top_k: usize,
    /// Per-layer rolling counters.
    hits: Vec<u64>,    // correct expert predictions
    total: Vec<u64>,   // total expert predictions attempted
}

impl AccuracyTracker {
    pub fn new(n_layers: usize, top_k: usize) -> Self {
        Self {
            n_layers,
            top_k,
            hits: vec![0; n_layers],
            total: vec![0; n_layers],
        }
    }

    /// Record one token's prediction vs actual outcome.
    pub fn record(&mut self, prediction: &ExpertPrediction, actual: &RouteOutcome) {
        let l = actual.layer;
        if l >= self.n_layers { return; }
        let actual_set: std::collections::HashSet<ExpertId> =
            actual.selected_experts.iter().copied().collect();
        let hits: u64 = prediction.experts.iter()
            .filter(|e| actual_set.contains(e))
            .count() as u64;
        self.hits[l] += hits;
        self.total[l] += self.top_k as u64;
    }

    /// Hit rate for layer l: fraction of predicted experts that were correct.
    pub fn hit_rate(&self, layer: usize) -> f32 {
        if self.total[layer] == 0 { return 0.0; }
        self.hits[layer] as f32 / self.total[layer] as f32
    }

    /// Expected correct prefetches per token at layer l.
    pub fn expected_hits(&self, layer: usize) -> f32 {
        self.hit_rate(layer) * self.top_k as f32
    }

    /// Summary across all layers (for the `x_summary` telemetry field).
    pub fn layer_summaries(&self) -> Vec<LayerAccuracy> {
        (0..self.n_layers)
            .map(|l| LayerAccuracy {
                layer: l,
                n_tokens: (self.total[l] / self.top_k as u64) as usize,
                hit_rate: self.hit_rate(l),
                expected_hits: self.expected_hits(l),
            })
            .collect()
    }

    /// Mean hit rate across all layers (single-number summary).
    pub fn mean_hit_rate(&self) -> f32 {
        let (sum, n) = (0..self.n_layers)
            .filter(|&l| self.total[l] > 0)
            .fold((0.0f32, 0usize), |(s, n), l| (s + self.hit_rate(l), n + 1));
        if n == 0 { 0.0 } else { sum / n as f32 }
    }

    pub fn reset(&mut self) {
        self.hits.iter_mut().for_each(|h| *h = 0);
        self.total.iter_mut().for_each(|t| *t = 0);
    }
}

// ---------------------------------------------------------------------------
// Trace collector
// ---------------------------------------------------------------------------

/// Collects (hidden_state, router_logits) pairs for offline training of the
/// LearnedLinear predictor.
///
/// Stored as flat f32 vecs to avoid per-token allocation. The caller
/// periodically drains the buffer and passes it to `LearnedLinear::update`.
pub struct TraceCollector {
    pub hidden_size: usize,
    pub n_experts: usize,
    pub max_traces: usize,
    /// Flat [n_traces * hidden_size].
    pub hidden_states: Vec<f32>,
    /// Flat [n_traces * n_experts].
    pub router_logits: Vec<f32>,
    /// Which layer each trace came from.
    pub layers: Vec<usize>,
}

impl TraceCollector {
    pub fn new(hidden_size: usize, n_experts: usize, max_traces: usize) -> Self {
        Self {
            hidden_size, n_experts, max_traces,
            hidden_states: Vec::with_capacity(max_traces * hidden_size),
            router_logits: Vec::with_capacity(max_traces * n_experts),
            layers: Vec::with_capacity(max_traces),
        }
    }

    /// Add one (hidden, logits) pair from the engine's forward pass.
    pub fn push(&mut self, layer: usize, hidden: &[f32], logits: &[f32]) {
        if self.layers.len() >= self.max_traces { return; }
        debug_assert_eq!(hidden.len(), self.hidden_size);
        debug_assert_eq!(logits.len(), self.n_experts);
        self.hidden_states.extend_from_slice(hidden);
        self.router_logits.extend_from_slice(logits);
        self.layers.push(layer);
    }

    pub fn n_traces(&self) -> usize { self.layers.len() }
    pub fn is_full(&self) -> bool { self.layers.len() >= self.max_traces }

    /// Drain traces for a specific layer, returning (hiddens, logits, count).
    /// Called by the training loop to update LearnedLinear.
    pub fn drain_layer(
        &mut self,
        target_layer: usize,
    ) -> (Vec<f32>, Vec<f32>, usize) {
        let hs = self.hidden_size;
        let ne = self.n_experts;
        let indices: Vec<usize> = self.layers.iter().enumerate()
            .filter(|(_, &l)| l == target_layer)
            .map(|(i, _)| i)
            .collect();
        let n = indices.len();
        let mut hiddens = Vec::with_capacity(n * hs);
        let mut logits = Vec::with_capacity(n * ne);
        for &i in &indices {
            hiddens.extend_from_slice(&self.hidden_states[i * hs..(i + 1) * hs]);
            logits.extend_from_slice(&self.router_logits[i * ne..(i + 1) * ne]);
        }
        // Remove drained entries (keep order for remaining layers).
        let keep: Vec<bool> = self.layers.iter().map(|&l| l != target_layer).collect();
        self.hidden_states = keep.iter().enumerate()
            .filter(|(_, &k)| k)
            .flat_map(|(i, _)| self.hidden_states[i * hs..(i + 1) * hs].iter().copied())
            .collect();
        self.router_logits = keep.iter().enumerate()
            .filter(|(_, &k)| k)
            .flat_map(|(i, _)| self.router_logits[i * ne..(i + 1) * ne].iter().copied())
            .collect();
        self.layers.retain(|&l| l != target_layer);
        (hiddens, logits, n)
    }

    pub fn clear(&mut self) {
        self.hidden_states.clear();
        self.router_logits.clear();
        self.layers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::types::{PredictorKind, RouteOutcome};

    fn make_prediction(layer: usize, experts: Vec<ExpertId>) -> ExpertPrediction {
        let n = experts.len();
        ExpertPrediction {
            layer,
            scores: vec![1.0; n],
            experts,
            source: PredictorKind::DirectProxy,
        }
    }

    #[test]
    fn accuracy_tracker_perfect_prediction() {
        let mut tracker = AccuracyTracker::new(4, 8);
        let pred = make_prediction(1, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let actual = RouteOutcome {
            layer: 1,
            selected_experts: vec![0, 1, 2, 3, 4, 5, 6, 7],
            logits: vec![],
        };
        tracker.record(&pred, &actual);
        assert!((tracker.hit_rate(1) - 1.0).abs() < 1e-6);
        assert!((tracker.expected_hits(1) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn accuracy_tracker_zero_prediction() {
        let mut tracker = AccuracyTracker::new(4, 8);
        let pred = make_prediction(0, vec![10, 11, 12, 13, 14, 15, 16, 17]);
        let actual = RouteOutcome {
            layer: 0,
            selected_experts: vec![0, 1, 2, 3, 4, 5, 6, 7],
            logits: vec![],
        };
        tracker.record(&pred, &actual);
        assert_eq!(tracker.hit_rate(0), 0.0);
    }

    #[test]
    fn trace_collector_drain_by_layer() {
        let mut col = TraceCollector::new(4, 8, 100);
        col.push(0, &[1.0, 0.0, 0.0, 0.0], &[0.5; 8]);
        col.push(1, &[0.0, 1.0, 0.0, 0.0], &[0.25; 8]);
        col.push(0, &[0.5, 0.5, 0.0, 0.0], &[0.1; 8]);
        assert_eq!(col.n_traces(), 3);
        let (h, _l, n) = col.drain_layer(0);
        assert_eq!(n, 2);
        assert_eq!(h.len(), 2 * 4);
        assert_eq!(col.n_traces(), 1); // only layer-1 trace remains
    }
}
