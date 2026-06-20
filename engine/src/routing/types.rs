/// Shared types for the routing prediction subsystem.

/// Expert index within a layer (0..n_experts).
pub type ExpertId = u32;

/// A set of predicted expert activations for one layer.
#[derive(Debug, Clone)]
pub struct ExpertPrediction {
    pub layer: usize,
    /// Predicted expert IDs, sorted by confidence descending.
    pub experts: Vec<ExpertId>,
    /// Confidence score for each predicted expert (higher = more confident).
    /// Same length as `experts`.
    pub scores: Vec<f32>,
    /// Which predictor produced this (for telemetry / A/B comparison).
    pub source: PredictorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictorKind {
    /// h_L @ W_router_{L+1}: no training required.
    DirectProxy,
    /// h_L @ P_L where P_L is a trained linear map.
    LearnedLinear,
    /// scores_L @ T_L where T_L is a trained Markov transition.
    MarkovTransition,
    /// Ensemble vote across multiple predictors.
    Ensemble,
}

/// Ground-truth routing outcome for one token at one layer (for training /
/// accuracy tracking).
#[derive(Debug, Clone)]
pub struct RouteOutcome {
    pub layer: usize,
    pub selected_experts: Vec<ExpertId>,
    /// Raw router logits (pre-softmax, pre-topk). Length = n_experts.
    pub logits: Vec<f32>,
}

/// Prediction accuracy summary for one layer over a window of tokens.
#[derive(Debug, Default, Clone)]
pub struct LayerAccuracy {
    pub layer: usize,
    pub n_tokens: usize,
    /// Fraction of predicted experts that appeared in the actual top-k.
    pub hit_rate: f32,
    /// Expected number of correctly prefetched experts per token.
    pub expected_hits: f32,
}
