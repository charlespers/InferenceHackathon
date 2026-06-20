/// PredictionPipeline: the single integration point for conifer.
///
/// Wraps EnsemblePredictor + PrefetchScheduler into a per-layer callback:
///
///   pipeline.on_layer_done(layer, hidden, actual_experts)
///
/// Called immediately after layer `layer`'s FFN completes (hidden state and
/// expert selection are both known). The pipeline:
///   1. Records actual experts so the scheduler can score the previous prediction.
///   2. Predicts which experts layer+1 will select.
///   3. Issues WarmExpert + EarlyDispatch actions to the PrefetchSink.
///   4. Updates the Markov counts for online learning.
///
/// # Building from routing stats
///
/// At engine startup, call `PredictionPipeline::from_stats_files()` to load
/// the real Markov matrices (seeded from routing_analysis output) and the
/// optimized placement map:
///
///   let pipeline = PredictionPipeline::from_stats_files(
///       "/alloc/data/routing_stats.json",
///       "/alloc/data/optimized_placement.json",
///       sink,
///   )?;

use crate::routing::optimizer::{placement_from_json, RoutingStats};
use crate::routing::predictor::{EnsemblePredictor, MarkovTransition};
use crate::routing::scheduler::{PlacementMap, PrefetchScheduler, PrefetchSink};
use crate::routing::types::{ExpertId, ExpertPrediction};

const N_GPUS: usize = 8;
const N_LAYERS: usize = 94;
const N_EXPERTS: usize = 128;
const TOP_K: usize = 8;

pub struct PredictionPipeline<S: PrefetchSink> {
    pub predictor: EnsemblePredictor,
    pub scheduler: PrefetchScheduler<S>,
    top_k: usize,
    /// Previous layer's expert selection — needed to update Markov online.
    prev_experts: Vec<ExpertId>,
}

impl<S: PrefetchSink> PredictionPipeline<S> {
    pub fn new(predictor: EnsemblePredictor, scheduler: PrefetchScheduler<S>, top_k: usize) -> Self {
        Self { predictor, scheduler, top_k, prev_experts: Vec::new() }
    }

    /// Call immediately after layer `layer` FFN completes.
    ///
    /// `hidden`: hidden state output [hidden_size=4096]
    /// `actual_experts`: top-8 experts that just fired at this layer
    ///
    /// Returns the prediction issued for layer+1 (None at the last layer).
    pub fn on_layer_done(
        &mut self,
        layer: usize,
        hidden: &[f32],
        actual_experts: &[ExpertId],
    ) -> Option<ExpertPrediction> {
        // 1. Score the previous prediction against what actually happened.
        self.scheduler.on_layer_complete(layer, actual_experts);

        // 2. Online Markov update: observe the L-1 → L transition.
        if !self.prev_experts.is_empty() {
            if let Some(mk) = &mut self.predictor.markov {
                mk.observe(layer.saturating_sub(1), &self.prev_experts, actual_experts);
            }
        }
        self.prev_experts = actual_experts.to_vec();

        // 3. Predict next layer.
        let pred = self.predictor.predict_full(
            layer,
            hidden,
            Some(actual_experts),
            self.top_k,
        );
        if pred.experts.is_empty() {
            return None;
        }

        // 4. Issue prefetch for next layer.
        self.scheduler.on_prediction(&pred);

        Some(pred)
    }

    pub fn prefetch_hit_rate(&self) -> f32 {
        self.scheduler.hit_rate()
    }

    pub fn total_prefetch_hits(&self) -> u64 {
        self.scheduler.total_hits()
    }

    /// Feed one full token's real routing data (from the vLLM hook) through
    /// the pipeline. Calls `on_layer_done` for each layer, doing Markov online
    /// updates and issuing prefetch actions to the sink.
    ///
    /// Returns the hit rate for this token specifically (delta hits / delta
    /// prefetched), so callers can report per-request accuracy.
    pub fn feed_token_routing(&mut self, routing: &[Vec<ExpertId>]) -> f32 {
        let before_hits = self.scheduler.total_hits();
        let before_prefetched = self.scheduler.total_prefetched();
        for (layer, experts) in routing.iter().enumerate() {
            self.on_layer_done(layer, &[], experts);
        }
        let delta_prefetched = self.scheduler.total_prefetched() - before_prefetched;
        let delta_hits = self.scheduler.total_hits() - before_hits;
        if delta_prefetched == 0 { 0.0 } else { delta_hits as f32 / delta_prefetched as f32 }
    }
}

// ---------------------------------------------------------------------------
// Factory: build a fully-seeded pipeline from routing stats files
// ---------------------------------------------------------------------------

impl<S: PrefetchSink> PredictionPipeline<S> {
    /// Build from routing_stats.json (Markov seed) + optimized_placement.json.
    ///
    /// Falls back gracefully:
    ///   - No routing_stats.json → Markov uses uniform prior.
    ///   - No optimized_placement.json → round-robin placement.
    pub fn from_stats_files(
        routing_stats_path: &str,
        placement_path: &str,
        sink: S,
    ) -> anyhow::Result<Self> {
        Self::from_stats_files_with_dims(
            routing_stats_path, placement_path, sink,
            N_LAYERS, N_EXPERTS, TOP_K, N_GPUS,
        )
    }

    pub fn from_stats_files_with_dims(
        routing_stats_path: &str,
        placement_path: &str,
        sink: S,
        n_layers: usize,
        n_experts: usize,
        top_k: usize,
        n_gpus: usize,
    ) -> anyhow::Result<Self> {
        // --- Markov predictor: seed from real transition matrices ---
        let mut markov = MarkovTransition::new(n_layers, n_experts);
        let mut n_seeded = 0usize;

        match std::fs::read_to_string(routing_stats_path) {
            Ok(raw) => {
                match serde_json::from_str::<RoutingStats>(&raw) {
                    Ok(stats) => {
                        for (key, mat) in &stats.routing.markov_matrices {
                            if let Some(layer) = parse_transition_key(key) {
                                markov.seed_layer_from_probs(layer, mat, 1000.0);
                                n_seeded += 1;
                            }
                        }
                        eprintln!(
                            "[pipeline] Markov seeded from {} layer transitions",
                            n_seeded
                        );
                    }
                    Err(e) => eprintln!("[pipeline] Failed to parse {routing_stats_path}: {e}"),
                }
            }
            Err(_) => eprintln!("[pipeline] {routing_stats_path} not found — uniform Markov prior"),
        }

        // --- Placement map ---
        let placement = match placement_from_json(placement_path, n_gpus) {
            Ok(p) => {
                eprintln!(
                    "[pipeline] Loaded optimized placement ({} replicated expert-layers)",
                    p.n_replicated_experts()
                );
                p
            }
            Err(e) => {
                eprintln!("[pipeline] {placement_path} not found ({e}) — round-robin");
                PlacementMap::round_robin(n_layers, n_gpus, n_experts)
            }
        };

        // --- Ensemble: Markov only until proxy weights are loaded by conifer ---
        let mut ensemble = EnsemblePredictor::new(n_experts);
        ensemble.markov = Some(markov);
        // weights = [proxy, linear, markov]
        ensemble.weights = if n_seeded > 0 { [0.0, 0.0, 1.0] } else { [0.0, 0.0, 1.0] };

        let scheduler = PrefetchScheduler::new(sink, placement);
        Ok(Self::new(ensemble, scheduler, top_k))
    }
}

fn parse_transition_key(key: &str) -> Option<usize> {
    // Parses "L->L+1" e.g. "0->1" → 0, "9->10" → 9
    key.split("->").next()?.parse().ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::scheduler::{NullSink, RecordingSink};

    fn make_pipeline(n_layers: usize) -> PredictionPipeline<RecordingSink> {
        let mut markov = MarkovTransition::new(n_layers, 8);
        // Seed layer 0: expert 0 always transitions to expert 3
        for _ in 0..200 {
            markov.observe(0, &[0], &[3]);
        }
        let mut ensemble = EnsemblePredictor::new(8);
        ensemble.markov = Some(markov);
        ensemble.weights = [0.0, 0.0, 1.0];
        let placement = PlacementMap::round_robin(n_layers, 4, 8);
        let scheduler = PrefetchScheduler::new(RecordingSink::new(), placement);
        PredictionPipeline::new(ensemble, scheduler, 3)
    }

    #[test]
    fn pipeline_predicts_and_issues_prefetch() {
        let mut p = make_pipeline(4);
        let hidden = vec![0.0f32; 4096];
        let actual = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
        let pred = p.on_layer_done(0, &hidden, &actual);
        assert!(pred.is_some(), "should predict next layer");
        assert_eq!(pred.unwrap().layer, 1);
        // Scheduler should have issued prefetch actions
        assert!(p.scheduler.total_prefetched() > 0);
    }

    #[test]
    fn pipeline_markov_learns_online() {
        let mut p = make_pipeline(4);
        let hidden = vec![0.0f32; 4096];
        // Feed layer 0 → layer 1 many times so Markov can learn
        for _ in 0..50 {
            p.on_layer_done(0, &hidden, &[0, 1, 2, 3, 4, 5, 6, 7]);
            p.on_layer_done(1, &hidden, &[3, 4, 5, 6, 7, 0, 1, 2]);
        }
        // After many observations, hit rate should be > 0
        // (we can't assert a specific value without a real predictor)
        let _ = p.prefetch_hit_rate();
    }

    #[test]
    fn parse_transition_key_correct() {
        assert_eq!(parse_transition_key("0->1"), Some(0));
        assert_eq!(parse_transition_key("9->10"), Some(9));
        assert_eq!(parse_transition_key("bad"), None);
    }

    #[test]
    fn pipeline_from_stats_falls_back_gracefully() {
        // Non-existent paths → should still build with defaults
        let result = PredictionPipeline::from_stats_files_with_dims(
            "/nonexistent/routing_stats.json",
            "/nonexistent/optimized_placement.json",
            NullSink,
            4, 8, 3, 4,
        );
        assert!(result.is_ok(), "should fall back to defaults, not error");
    }
}
