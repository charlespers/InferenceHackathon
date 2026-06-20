/// Prefetch scheduler: issues expert prefetch requests while the current
/// layer's FFN is computing, using the route predictor's output.
///
/// # The overlap opportunity
///
/// At B=1, one layer's timeline looks like:
///
///   |---attention---|---all2all dispatch---|---expert FFN---|---all2all combine---|
///                                          ^
///                             prediction fires here (h_L is ready after attention)
///                             => issue next-layer's dispatch on a background stream
///
/// If prediction is correct, next-layer's all-to-all arrives (or is in flight)
/// before we need to issue it, hiding the ~5µs collective latency.
///
/// # CUDA stream model
///
/// We use two CUDA streams per GPU:
///   - `compute_stream`: the main forward-pass stream (attention, FFN kernels).
///   - `prefetch_stream`: lower-priority, for early dispatch and weight hints.
///
/// The prefetch_stream runs concurrently with compute_stream on the SM scheduler.
/// For expert weights already in HBM (they always are — we load all 128 upfront)
/// this is a "logical prefetch": we issue a dummy read / L2-warming operation so
/// the cache lines are hot when the compute stream needs them. The real gain is
/// the early all-to-all dispatch: tokens are sent to the right GPU before the
/// compute stream reaches the dispatch call, so the NVLink transfer overlaps.
///
/// # Interface
///
/// This module is CUDA-agnostic (no cudarc imports). The CUDA-specific parts
/// (stream handles, async memcpy hints) are injected via the `PrefetchSink` trait.
/// This makes it testable without a GPU and lets conifer provide the real sink.

use std::collections::VecDeque;

use crate::routing::types::{ExpertId, ExpertPrediction};

// ---------------------------------------------------------------------------
// PrefetchSink: what the scheduler calls when it wants to prefetch.
// Conifer implements this with real CUDA stream ops.
// ---------------------------------------------------------------------------

/// Actions the scheduler issues to the GPU layer.
#[derive(Debug, Clone)]
pub enum PrefetchAction {
    /// Warm the HBM cache lines for this expert's weights on the given GPU.
    /// In practice: issue a low-priority async read of expert_weight_ptr.
    WarmExpert { layer: usize, expert_id: ExpertId, gpu_id: usize },
    /// Issue the all-to-all dispatch token to the GPU hosting this expert,
    /// before the compute stream reaches that layer.
    EarlyDispatch { layer: usize, expert_id: ExpertId, gpu_id: usize },
}

pub trait PrefetchSink: Send + Sync {
    fn submit(&self, action: PrefetchAction);
}

/// No-op sink for testing and dry-run mode.
pub struct NullSink;
impl PrefetchSink for NullSink {
    fn submit(&self, _: PrefetchAction) {}
}

/// Recording sink (captures actions for inspection in tests).
pub struct RecordingSink {
    pub actions: std::sync::Mutex<Vec<PrefetchAction>>,
}
impl RecordingSink {
    pub fn new() -> Self { Self { actions: std::sync::Mutex::new(vec![]) } }
    pub fn drain(&self) -> Vec<PrefetchAction> {
        self.actions.lock().unwrap().drain(..).collect()
    }
}
impl PrefetchSink for RecordingSink {
    fn submit(&self, action: PrefetchAction) {
        self.actions.lock().unwrap().push(action);
    }
}

// ---------------------------------------------------------------------------
// Expert placement map: expert_id -> gpu_id for each layer.
// Defaults to round-robin (expert_id % n_gpus).
// ---------------------------------------------------------------------------

pub struct PlacementMap {
    /// primary[layer][expert_id] = gpu_id.
    data: Vec<Vec<usize>>,
    /// replica[layer][expert_id] = Some(gpu_id) for hot experts with a second copy.
    replicas: Vec<Vec<Option<usize>>>,
    pub n_gpus: usize,
    pub n_experts: usize,
}

impl PlacementMap {
    pub fn round_robin(n_layers: usize, n_gpus: usize, n_experts: usize) -> Self {
        let data = (0..n_layers)
            .map(|_| (0..n_experts).map(|e| e % n_gpus).collect())
            .collect();
        let replicas = vec![vec![None; n_experts]; n_layers];
        Self { data, replicas, n_gpus, n_experts }
    }

    /// Build from pre-computed placement + replica tables.
    /// `placement[layer][expert] = primary_gpu`
    /// `replicas[layer][expert]  = Some(replica_gpu)` or `None`
    pub fn from_tables(
        placement: Vec<Vec<usize>>,
        replicas: Vec<Vec<Option<usize>>>,
        n_gpus: usize,
        n_experts: usize,
    ) -> Self {
        Self { data: placement, replicas, n_gpus, n_experts }
    }

    /// Primary GPU for this expert.
    pub fn gpu_for(&self, layer: usize, expert_id: ExpertId) -> usize {
        self.data[layer][expert_id as usize % self.n_experts]
    }

    /// All GPUs hosting this expert (primary + optional replica).
    pub fn gpus_for(&self, layer: usize, expert_id: ExpertId) -> (usize, Option<usize>) {
        let idx = expert_id as usize % self.n_experts;
        (self.data[layer][idx], self.replicas[layer][idx])
    }

    /// Whether this expert is replicated at this layer.
    pub fn is_replicated(&self, layer: usize, expert_id: ExpertId) -> bool {
        self.replicas[layer][expert_id as usize % self.n_experts].is_some()
    }

    pub fn set(&mut self, layer: usize, expert_id: ExpertId, gpu_id: usize) {
        self.data[layer][expert_id as usize] = gpu_id;
    }

    pub fn set_replica(&mut self, layer: usize, expert_id: ExpertId, gpu_id: usize) {
        self.replicas[layer][expert_id as usize] = Some(gpu_id);
    }

    pub fn n_replicated_experts(&self) -> usize {
        self.replicas.iter().flat_map(|l| l.iter()).filter(|r| r.is_some()).count()
    }
}

// ---------------------------------------------------------------------------
// PrefetchScheduler
// ---------------------------------------------------------------------------

/// Pending prefetch: a prediction that has been issued but not yet confirmed.
struct PendingPrefetch {
    target_layer: usize,
    predicted_experts: Vec<ExpertId>,
    _actual_experts: Option<Vec<ExpertId>>, // filled in when layer completes
}

pub struct PrefetchScheduler<S: PrefetchSink> {
    sink: S,
    placement: PlacementMap,
    /// In-flight prefetch predictions awaiting confirmation.
    pending: VecDeque<PendingPrefetch>,
    /// Running counters for hit tracking (avoid heap alloc on hot path).
    total_prefetched: u64,
    total_hits: u64,
}

impl<S: PrefetchSink> PrefetchScheduler<S> {
    pub fn new(sink: S, placement: PlacementMap) -> Self {
        Self {
            sink,
            placement,
            pending: VecDeque::with_capacity(4), // rarely more than 1-2 in flight
            total_prefetched: 0,
            total_hits: 0,
        }
    }

    /// Call this immediately after attention completes for `current_layer`,
    /// when the hidden state is ready and the route predictor has fired.
    ///
    /// Issues `WarmExpert` and `EarlyDispatch` actions for the predicted
    /// experts at `current_layer + 1`, scheduled on the prefetch stream.
    pub fn on_prediction(&mut self, prediction: &ExpertPrediction) {
        let target_layer = prediction.layer;
        for &expert_id in &prediction.experts {
            let (primary, replica) = self.placement.gpus_for(target_layer, expert_id);
            for &gpu_id in &[Some(primary), replica].into_iter().flatten().collect::<Vec<_>>() {
                self.sink.submit(PrefetchAction::WarmExpert {
                    layer: target_layer, expert_id, gpu_id,
                });
                self.sink.submit(PrefetchAction::EarlyDispatch {
                    layer: target_layer, expert_id, gpu_id,
                });
            }
        }
        self.total_prefetched += prediction.experts.len() as u64;
        self.pending.push_back(PendingPrefetch {
            target_layer,
            predicted_experts: prediction.experts.clone(),
            _actual_experts: None,
        });
    }

    /// Call this when layer `layer` completes and actual expert selection is known.
    /// Updates hit counters; the pending entry is resolved and popped.
    pub fn on_layer_complete(&mut self, layer: usize, actual_experts: &[ExpertId]) {
        if let Some(pending) = self.pending.front_mut() {
            if pending.target_layer == layer {
                let actual_set: std::collections::HashSet<ExpertId> =
                    actual_experts.iter().copied().collect();
                let hits = pending.predicted_experts.iter()
                    .filter(|e| actual_set.contains(e))
                    .count() as u64;
                self.total_hits += hits;
                self.pending.pop_front();
            }
        }
    }

    /// Prefetch hit rate since construction (for telemetry / ensemble weight tuning).
    pub fn hit_rate(&self) -> f32 {
        if self.total_prefetched == 0 { return 0.0; }
        self.total_hits as f32 / self.total_prefetched as f32
    }

    pub fn total_hits(&self) -> u64 { self.total_hits }
    pub fn total_prefetched(&self) -> u64 { self.total_prefetched }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::types::PredictorKind;

    fn make_prediction(layer: usize, experts: Vec<ExpertId>) -> ExpertPrediction {
        ExpertPrediction {
            layer,
            experts,
            scores: vec![1.0; 8],
            source: PredictorKind::DirectProxy,
        }
    }

    #[test]
    fn scheduler_issues_actions_for_each_expert() {
        let sink = RecordingSink::new();
        let placement = PlacementMap::round_robin(10, 8, 128);
        let mut sched = PrefetchScheduler::new(sink, placement);

        let pred = make_prediction(1, vec![0, 16, 32, 48, 64, 80, 96, 112]);
        sched.on_prediction(&pred);

        let actions = sched.sink.drain();
        // 8 experts × 2 actions (WarmExpert + EarlyDispatch), no replicas = 16
        assert_eq!(actions.len(), 16);
    }

    #[test]
    fn scheduler_issues_extra_actions_for_replicated_experts() {
        let sink = RecordingSink::new();
        let mut placement = PlacementMap::round_robin(10, 8, 128);
        // Replicate expert 0 (layer 1) onto GPU 4
        placement.set_replica(1, 0, 4);
        let mut sched = PrefetchScheduler::new(sink, placement);

        let pred = make_prediction(1, vec![0, 16, 32, 48, 64, 80, 96, 112]);
        sched.on_prediction(&pred);

        let actions = sched.sink.drain();
        // expert 0 has replica → 4 actions; 7 others → 2 each = 4 + 14 = 18
        assert_eq!(actions.len(), 18);
    }

    #[test]
    fn hit_rate_perfect() {
        let placement = PlacementMap::round_robin(10, 8, 128);
        let mut sched = PrefetchScheduler::new(NullSink, placement);

        let pred = make_prediction(1, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        sched.on_prediction(&pred);
        sched.on_layer_complete(1, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert!((sched.hit_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hit_rate_zero() {
        let placement = PlacementMap::round_robin(10, 8, 128);
        let mut sched = PrefetchScheduler::new(NullSink, placement);

        let pred = make_prediction(1, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        sched.on_prediction(&pred);
        sched.on_layer_complete(1, &[8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(sched.hit_rate(), 0.0);
    }

    #[test]
    fn hit_rate_partial() {
        let placement = PlacementMap::round_robin(10, 8, 128);
        let mut sched = PrefetchScheduler::new(NullSink, placement);

        let pred = make_prediction(1, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        sched.on_prediction(&pred);
        // 4 of 8 correct
        sched.on_layer_complete(1, &[0, 1, 2, 3, 8, 9, 10, 11]);
        assert!((sched.hit_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn round_robin_placement_correct() {
        let p = PlacementMap::round_robin(1, 8, 128);
        assert_eq!(p.gpu_for(0, 0), 0);
        assert_eq!(p.gpu_for(0, 7), 7);
        assert_eq!(p.gpu_for(0, 8), 0);
        assert_eq!(p.gpu_for(0, 127), 7);
    }
}
