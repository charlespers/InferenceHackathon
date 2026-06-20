/// Route predictors: given the current hidden state h_L, predict which experts
/// layer L+1 will select before that layer's computation begins.
///
/// All predictors implement `RoutePredictor`. The engine picks one (or an
/// ensemble) via `PredictorConfig`. The `DirectProxy` is immediately usable
/// with zero training; the others improve as routing traces accumulate.

use crate::error::Result;
use crate::routing::types::{ExpertId, ExpertPrediction, PredictorKind};

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// Predict the top-k experts for layer `next_layer`, given the hidden state
/// `hidden` at the output of `current_layer`.
///
/// Implementations must be cheap: prediction must complete well within the
/// FFN compute time of `current_layer` (~60 µs on H200 at B=1) so that
/// the prefetch can be issued before the current layer finishes.
pub trait RoutePredictor: Send + Sync {
    fn predict(
        &self,
        current_layer: usize,
        hidden: &[f32],       // [hidden_size] = [4096] for Qwen3
        top_k: usize,         // = 8 for Qwen3-235B-A22B
    ) -> Result<ExpertPrediction>;

    fn kind(&self) -> PredictorKind;
}

// ---------------------------------------------------------------------------
// Tier 1: DirectProxy
// Applies the next layer's router weight matrix to the current hidden state.
// No training required. Works immediately, accuracy improves in later layers
// where the residual delta is smaller.
//
// Math: h_{L+1} ≈ h_L + delta  (residual stream)
//   =>  scores_{L+1} = h_{L+1} @ W_r_{L+1} ≈ h_L @ W_r_{L+1}
//
// W_r has shape [hidden, n_experts] = [4096, 128] for Qwen3.
// Stored per layer: 94 * 4096 * 128 * 4B ≈ 196 MB total (negligible).
// ---------------------------------------------------------------------------

pub struct DirectProxy {
    /// Router weight matrices: outer index = layer (0..n_layers).
    /// Each matrix is [hidden_size * n_experts], row-major (hidden-major).
    /// i.e. score_for_expert_e = dot(hidden, router_weights[layer][e * hidden_size .. (e+1) * hidden_size])
    /// but we store transposed as [n_experts * hidden_size] for the GEMV layout.
    router_weights: Vec<Vec<f32>>,  // [n_layers][n_experts * hidden_size]
    hidden_size: usize,
    n_experts: usize,
}

impl DirectProxy {
    /// Build from pre-loaded router weight slices.
    ///
    /// `weights_per_layer`: one flat [n_experts * hidden_size] slice per layer,
    /// where the expert-major layout means expert e's weights are at
    /// [e * hidden_size .. (e+1) * hidden_size].
    pub fn new(
        weights_per_layer: Vec<Vec<f32>>,
        hidden_size: usize,
        n_experts: usize,
    ) -> Self {
        assert_eq!(weights_per_layer[0].len(), n_experts * hidden_size);
        Self { router_weights: weights_per_layer, hidden_size, n_experts }
    }

    /// Convenience: build from Qwen3-235B-A22B defaults (sizes only; caller
    /// provides the actual weight data from conifer's weight loader).
    pub fn qwen3_empty(n_layers: usize) -> Self {
        Self {
            router_weights: vec![vec![0.0; 128 * 4096]; n_layers],
            hidden_size: 4096,
            n_experts: 128,
        }
    }

    /// Load one layer's router weights (called by conifer's weight loader).
    pub fn set_layer_weights(&mut self, layer: usize, weights: Vec<f32>) {
        assert_eq!(weights.len(), self.n_experts * self.hidden_size);
        self.router_weights[layer] = weights;
    }

    /// Raw router score: dot(hidden, expert_e_weights).
    /// GEMV: O(hidden_size) per expert, O(hidden_size * n_experts) total.
    /// For Qwen3: 4096 * 128 = 524k muls — sub-microsecond on H200.
    fn score_all(&self, layer: usize, hidden: &[f32]) -> Vec<f32> {
        let w = &self.router_weights[layer];
        (0..self.n_experts)
            .map(|e| {
                let offset = e * self.hidden_size;
                dot(&hidden[..self.hidden_size], &w[offset..offset + self.hidden_size])
            })
            .collect()
    }
}

impl RoutePredictor for DirectProxy {
    fn predict(
        &self,
        current_layer: usize,
        hidden: &[f32],
        top_k: usize,
    ) -> Result<ExpertPrediction> {
        // Predict for the NEXT layer using current hidden state.
        let next_layer = current_layer + 1;
        if next_layer >= self.router_weights.len() {
            return Ok(ExpertPrediction {
                layer: next_layer,
                experts: vec![],
                scores: vec![],
                source: self.kind(),
            });
        }
        let scores = self.score_all(next_layer, hidden);
        let (experts, expert_scores) = top_k_indices(&scores, top_k);
        Ok(ExpertPrediction {
            layer: next_layer,
            experts,
            scores: expert_scores,
            source: self.kind(),
        })
    }

    fn kind(&self) -> PredictorKind { PredictorKind::DirectProxy }
}

// ---------------------------------------------------------------------------
// Tier 2: LearnedLinear
// A trained per-layer matrix P_L: [hidden_size, n_experts] that maps h_L
// directly to predicted scores_{L+1}, absorbing the nonlinear residual path.
//
// Training objective (offline, on routing traces):
//   min_P ||h_L @ P_L - router_scores_{L+1}||^2
//   (or a top-k ranking loss for better calibration)
//
// Initialised as DirectProxy weights (i.e. W_router_{L+1}) so it's useful
// from day 1; fine-tuned as traces accumulate.
//
// Storage: same as DirectProxy (~196 MB). Inference cost: identical (one GEMV).
// ---------------------------------------------------------------------------

pub struct LearnedLinear {
    /// Prediction matrices: outer index = current layer (0..n_layers-1).
    /// P_L maps hidden_size -> n_experts.
    /// Stored as [n_experts * hidden_size], expert-major.
    pred_matrices: Vec<Vec<f32>>,
    hidden_size: usize,
    n_experts: usize,
}

impl LearnedLinear {
    /// Initialise from DirectProxy weights (safe starting point pre-training).
    pub fn from_proxy(proxy: &DirectProxy) -> Self {
        // P_L is initialised to W_router_{L+1}: predict next layer's scores
        // from current hidden state, starting with the identity assumption.
        let n_layers = proxy.router_weights.len();
        let pred_matrices = (0..n_layers.saturating_sub(1))
            .map(|l| proxy.router_weights[l + 1].clone())
            .collect();
        Self {
            pred_matrices,
            hidden_size: proxy.hidden_size,
            n_experts: proxy.n_experts,
        }
    }

    /// Update one layer's prediction matrix from a batch of training examples.
    ///
    /// Uses a single gradient step: P_L <- P_L - lr * grad
    /// where grad = (h_L @ P_L - scores_{L+1})^T @ h_L / batch_size.
    ///
    /// Call this incrementally as routing traces arrive from the real engine.
    pub fn update(
        &mut self,
        layer: usize,
        hidden_batch: &[f32],      // [batch * hidden_size]
        target_scores: &[f32],     // [batch * n_experts]
        batch_size: usize,
        lr: f32,
    ) {
        if layer >= self.pred_matrices.len() { return; }
        let h = hidden_batch;
        let t = target_scores;
        let p = &mut self.pred_matrices[layer];
        let hs = self.hidden_size;
        let ne = self.n_experts;

        // For each expert e, compute grad_e = sum_b (h_b @ p_e - t_be) * h_b
        for e in 0..ne {
            let offset = e * hs;
            for b in 0..batch_size {
                let h_b = &h[b * hs..(b + 1) * hs];
                let t_be = t[b * ne + e];
                let pred: f32 = dot(h_b, &p[offset..offset + hs]);
                let err = (pred - t_be) / batch_size as f32;
                for (i, &hv) in h_b.iter().enumerate() {
                    p[offset + i] -= lr * err * hv;
                }
            }
        }
    }

    fn score_all(&self, layer: usize, hidden: &[f32]) -> Vec<f32> {
        let p = &self.pred_matrices[layer];
        (0..self.n_experts)
            .map(|e| {
                let offset = e * self.hidden_size;
                dot(hidden, &p[offset..offset + self.hidden_size])
            })
            .collect()
    }
}

impl RoutePredictor for LearnedLinear {
    fn predict(
        &self,
        current_layer: usize,
        hidden: &[f32],
        top_k: usize,
    ) -> Result<ExpertPrediction> {
        let next_layer = current_layer + 1;
        if current_layer >= self.pred_matrices.len() {
            return Ok(ExpertPrediction {
                layer: next_layer, experts: vec![], scores: vec![],
                source: self.kind(),
            });
        }
        let scores = self.score_all(current_layer, hidden);
        let (experts, expert_scores) = top_k_indices(&scores, top_k);
        Ok(ExpertPrediction {
            layer: next_layer, experts, scores: expert_scores, source: self.kind(),
        })
    }

    fn kind(&self) -> PredictorKind { PredictorKind::LearnedLinear }
}

// ---------------------------------------------------------------------------
// Tier 3: MarkovTransition
// Learns a per-layer transition matrix T_L: [n_experts, n_experts] where
// T_L[i][j] = P(expert j selected at L+1 | expert i selected at L).
//
// Doesn't need the hidden state — only the current routing decision.
// Useful when hidden states aren't accessible (e.g. wrapping a black-box
// engine) or as a fast pre-filter before the linear predictor.
//
// Training: accumulate routing co-occurrence counts, normalise per row.
// Storage: 94 * 128 * 128 * 4B ≈ 6 MB. Inference: O(top_k * n_experts).
// ---------------------------------------------------------------------------

pub struct MarkovTransition {
    /// Transition counts: [n_layers-1][n_experts * n_experts].
    /// T[l][i * n_experts + j] = count of (expert i at layer l, expert j at l+1).
    counts: Vec<Vec<f32>>,
    n_experts: usize,
}

impl MarkovTransition {
    pub fn new(n_layers: usize, n_experts: usize) -> Self {
        // Initialise with uniform prior (Laplace smoothing).
        Self {
            counts: vec![vec![1.0; n_experts * n_experts]; n_layers.saturating_sub(1)],
            n_experts,
        }
    }

    /// Record one token's routing at consecutive layers.
    pub fn observe(
        &mut self,
        layer: usize,
        experts_l: &[ExpertId],
        experts_l1: &[ExpertId],
    ) {
        if layer >= self.counts.len() { return; }
        let row = &mut self.counts[layer];
        for &e_l in experts_l {
            for &e_l1 in experts_l1 {
                row[e_l as usize * self.n_experts + e_l1 as usize] += 1.0;
            }
        }
    }

    /// Predict next-layer experts by summing transition probabilities from
    /// all currently active experts and taking the top-k.
    fn transition_scores(&self, layer: usize, active_experts: &[ExpertId]) -> Vec<f32> {
        let row = &self.counts[layer];
        let ne = self.n_experts;
        let mut scores = vec![0.0f32; ne];
        for &e in active_experts {
            let offset = e as usize * ne;
            // Row sum for normalisation (lazy: compute once per query).
            let row_sum: f32 = row[offset..offset + ne].iter().sum();
            for j in 0..ne {
                scores[j] += row[offset + j] / row_sum;
            }
        }
        scores
    }
}

// MarkovTransition uses current expert selection, not hidden state.
// We adapt it to the RoutePredictor interface by ignoring `hidden`.
// The caller should use `predict_from_experts` directly when possible.
impl MarkovTransition {
    pub fn predict_from_experts(
        &self,
        current_layer: usize,
        active_experts: &[ExpertId],
        top_k: usize,
    ) -> ExpertPrediction {
        if current_layer >= self.counts.len() {
            return ExpertPrediction {
                layer: current_layer + 1, experts: vec![], scores: vec![],
                source: PredictorKind::MarkovTransition,
            };
        }
        let scores = self.transition_scores(current_layer, active_experts);
        let (experts, expert_scores) = top_k_indices(&scores, top_k);
        ExpertPrediction {
            layer: current_layer + 1,
            experts,
            scores: expert_scores,
            source: PredictorKind::MarkovTransition,
        }
    }
}

// ---------------------------------------------------------------------------
// Ensemble: combines DirectProxy / LearnedLinear / Markov by score fusion.
// Weights each predictor's confidence and takes the top-k by fused score.
// Falls back gracefully as components become available.
// ---------------------------------------------------------------------------

pub struct EnsemblePredictor {
    pub proxy: Option<DirectProxy>,
    pub linear: Option<LearnedLinear>,
    pub markov: Option<MarkovTransition>,
    /// Weights [proxy, linear, markov]. Tuned empirically; linear gets
    /// higher weight once it's been trained on real routing traces.
    pub weights: [f32; 3],
    pub n_experts: usize,
}

impl EnsemblePredictor {
    pub fn new(n_experts: usize) -> Self {
        Self {
            proxy: None, linear: None, markov: None,
            weights: [1.0, 0.0, 0.0],
            n_experts,
        }
    }

    /// Fuse scores from all available predictors.
    fn fuse_scores(
        &self,
        layer: usize,
        hidden: &[f32],
        current_experts: Option<&[ExpertId]>,
    ) -> Vec<f32> {
        let ne = self.n_experts;
        let mut fused = vec![0.0f32; ne];
        let mut total_weight = 0.0f32;

        if let Some(p) = &self.proxy {
            if self.weights[0] > 0.0 && layer + 1 < p.router_weights.len() {
                let scores = p.score_all(layer + 1, hidden);
                let norm = l2_norm(&scores);
                for (i, s) in scores.iter().enumerate() {
                    fused[i] += self.weights[0] * s / norm.max(1e-9);
                }
                total_weight += self.weights[0];
            }
        }

        if let Some(lin) = &self.linear {
            if self.weights[1] > 0.0 && layer < lin.pred_matrices.len() {
                let scores = lin.score_all(layer, hidden);
                let norm = l2_norm(&scores);
                for (i, s) in scores.iter().enumerate() {
                    fused[i] += self.weights[1] * s / norm.max(1e-9);
                }
                total_weight += self.weights[1];
            }
        }

        if let (Some(mk), Some(experts)) = (&self.markov, current_experts) {
            if self.weights[2] > 0.0 && layer < mk.counts.len() {
                let scores = mk.transition_scores(layer, experts);
                let norm = l2_norm(&scores);
                for (i, s) in scores.iter().enumerate() {
                    fused[i] += self.weights[2] * s / norm.max(1e-9);
                }
                total_weight += self.weights[2];
            }
        }

        if total_weight > 0.0 {
            fused.iter_mut().for_each(|s| *s /= total_weight);
        }
        fused
    }

    pub fn predict_full(
        &self,
        current_layer: usize,
        hidden: &[f32],
        current_experts: Option<&[ExpertId]>,
        top_k: usize,
    ) -> ExpertPrediction {
        let scores = self.fuse_scores(current_layer, hidden, current_experts);
        let (experts, expert_scores) = top_k_indices(&scores, top_k);
        ExpertPrediction {
            layer: current_layer + 1,
            experts,
            scores: expert_scores,
            source: PredictorKind::Ensemble,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared math
// ---------------------------------------------------------------------------

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Return top-k indices and their scores, sorted by score descending.
pub(crate) fn top_k_indices(scores: &[f32], k: usize) -> (Vec<ExpertId>, Vec<f32>) {
    let k = k.min(scores.len());
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    // Partial sort: bring top-k to the front.
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top = indexed[..k].to_vec();
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let experts = top.iter().map(|&(i, _)| i as ExpertId).collect();
    let expert_scores = top.iter().map(|&(_, s)| s).collect();
    (experts, expert_scores)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_proxy(n_layers: usize, hidden: usize, n_experts: usize) -> DirectProxy {
        let mut weights = vec![];
        for l in 0..n_layers {
            // Expert e at layer l has weight 1.0 at position e % hidden, 0 elsewhere.
            let mut w = vec![0.0f32; n_experts * hidden];
            for e in 0..n_experts {
                w[e * hidden + (e + l) % hidden] = 1.0;
            }
            weights.push(w);
        }
        DirectProxy::new(weights, hidden, n_experts)
    }

    #[test]
    fn direct_proxy_returns_top_k() {
        let proxy = make_proxy(4, 16, 8);
        let hidden = vec![1.0f32; 16];
        let pred = proxy.predict(0, &hidden, 3).unwrap();
        assert_eq!(pred.experts.len(), 3);
        assert_eq!(pred.layer, 1);
        assert_eq!(pred.source, PredictorKind::DirectProxy);
        // Scores should be descending.
        for w in pred.scores.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }

    #[test]
    fn direct_proxy_last_layer_returns_empty() {
        let proxy = make_proxy(3, 16, 8);
        let hidden = vec![1.0f32; 16];
        let pred = proxy.predict(2, &hidden, 8).unwrap(); // no layer 3
        assert!(pred.experts.is_empty());
    }

    #[test]
    fn learned_linear_updates_decrease_loss() {
        let proxy = make_proxy(4, 16, 8);
        let mut linear = LearnedLinear::from_proxy(&proxy);
        let hidden = vec![0.5f32; 16];
        let target = vec![1.0f32; 8]; // one batch item

        let score_before: f32 = {
            let s = linear.score_all(0, &hidden);
            s.iter().zip(target.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
        };
        linear.update(0, &hidden, &target, 1, 0.01);
        let score_after: f32 = {
            let s = linear.score_all(0, &hidden);
            s.iter().zip(target.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
        };
        assert!(score_after < score_before, "loss should decrease after update");
    }

    #[test]
    fn markov_transition_favours_observed_pairs() {
        let mut mk = MarkovTransition::new(4, 8);
        // Observe: expert 0 at layer 0 always followed by expert 5 at layer 1.
        for _ in 0..100 {
            mk.observe(0, &[0], &[5]);
        }
        let pred = mk.predict_from_experts(0, &[0], 1);
        assert_eq!(pred.experts[0], 5, "should predict expert 5");
    }

    #[test]
    fn top_k_indices_sorted_descending() {
        let scores = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let (experts, _) = top_k_indices(&scores, 3);
        assert_eq!(experts, vec![1, 3, 4]);
    }
}
