//! Telemetry — fold per-round [`Eagle3RoundStats`] into the running metrics the spec loop reports.
//!
//! The route-aware engine emits an [`Eagle3RoundStats`] per [`crate::spec::eagle3_engine::Eagle3Engine::step`];
//! this accumulates them into the same quantities the box measures on vLLM (mean acceptance length τ,
//! mean verify expert-union, acceptance rate) so the native engine's live numbers are directly
//! comparable to the EAGLE3 measurements (`results/eagle3_redhat/`, τ≈2.7) and to the
//! [`crate::spec::projection`] cost model. This is the server-independent core of the
//! "Eagle3RoundStats → SSE" telemetry wiring: the aggregation lives here and is fully unit-tested; an
//! SSE/HTTP surface just serializes a [`SpecTelemetry::snapshot`].
//!
//! Note on speedup: realized S needs wall-clock per round, which `step` does not time, so this module
//! reports the acceptance/union STATISTICS only and defers the S estimate to the projection model via
//! [`SpecTelemetry::projected_speedup`] (predict S at the realized mean union + a supplied floor).

use crate::spec::eagle3_engine::Eagle3RoundStats;
use crate::spec::projection::{MeasuredAccept, RoundCostModel};
use serde::Serialize;

/// Running accumulation of spec-round outcomes. Cheap to update; clone the [`TelemetrySnapshot`] out.
#[derive(Debug, Default, Clone)]
pub struct SpecTelemetry {
    rounds: u64,
    /// Rounds that actually speculated (n_drafted > 0). A degenerate round (head produced nothing →
    /// caller decodes one token normally) still counts a round but emits no draft/union stats.
    spec_rounds: u64,
    total_accepted: u64,
    total_drafted: u64,
    total_verify_depth: u64,
    total_union: u64,
}

/// Immutable point-in-time view of the accumulated metrics (what an SSE frame carries).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TelemetrySnapshot {
    pub rounds: u64,
    pub spec_rounds: u64,
    /// Mean acceptance length τ = emitted tokens per round = (accepted + 1 bonus)/round. Directly
    /// comparable to vLLM's "Mean acceptance length" (≈2.7 measured). Averaged over ALL rounds (a
    /// degenerate round emits exactly the 1 bonus/normal token), so this is the true per-round yield.
    pub mean_accept_length: f32,
    /// Mean distinct experts the verify read per SPECULATING round — the cost driver route-awareness
    /// minimizes (compare to the projection's union and to EVICT's −32.5% activated experts).
    pub mean_union: f32,
    /// Mean adaptive verify depth over speculating rounds.
    pub mean_verify_depth: f32,
    /// Draft acceptance rate = accepted / drafted over speculating rounds (∈[0,1]).
    pub acceptance_rate: f32,
}

impl TelemetrySnapshot {
    /// Serialize as a Server-Sent-Events frame for the live telemetry stream: `data: {json}\n\n`.
    /// This is the exact payload an SSE/HTTP surface emits per snapshot — the final hop of the
    /// "Eagle3RoundStats → SSE" wiring. Serialization can't fail for these plain scalar fields, so a
    /// failure degrades to an empty object rather than panicking in the hot stream.
    pub fn to_sse_frame(&self) -> String {
        let body = serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string());
        format!("data: {}\n\n", body)
    }
}

impl SpecTelemetry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one round's stats in.
    pub fn record(&mut self, s: &Eagle3RoundStats) {
        self.rounds += 1;
        self.total_accepted += s.n_accepted as u64;
        if s.n_drafted > 0 {
            self.spec_rounds += 1;
            self.total_drafted += s.n_drafted as u64;
            self.total_verify_depth += s.verify_depth as u64;
            self.total_union += s.union_size as u64;
        }
    }

    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// τ = emitted per round = (accepted + 1 bonus)/round, averaged over ALL rounds. Every round
    /// emits exactly one bonus/target token plus its accepted draft prefix, so emitted-per-round is
    /// `total_accepted/rounds + 1`. Zero rounds → 0.
    pub fn mean_accept_length(&self) -> f32 {
        if self.rounds == 0 {
            return 0.0;
        }
        self.total_accepted as f32 / self.rounds as f32 + 1.0
    }

    /// Mean verify union over speculating rounds (0 if none speculated).
    pub fn mean_union(&self) -> f32 {
        if self.spec_rounds == 0 {
            return 0.0;
        }
        self.total_union as f32 / self.spec_rounds as f32
    }

    pub fn mean_verify_depth(&self) -> f32 {
        if self.spec_rounds == 0 {
            return 0.0;
        }
        self.total_verify_depth as f32 / self.spec_rounds as f32
    }

    /// Accepted / drafted over speculating rounds (0 if none speculated).
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafted == 0 {
            return 0.0;
        }
        self.total_accepted as f32 / self.total_drafted as f32
    }

    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            rounds: self.rounds,
            spec_rounds: self.spec_rounds,
            mean_accept_length: self.mean_accept_length(),
            mean_union: self.mean_union(),
            mean_verify_depth: self.mean_verify_depth(),
            acceptance_rate: self.acceptance_rate(),
        }
    }

    /// Convenience: the current snapshot as a Server-Sent-Events frame (see
    /// [`TelemetrySnapshot::to_sse_frame`]).
    pub fn to_sse_frame(&self) -> String {
        self.snapshot().to_sse_frame()
    }

    /// Tie the realized telemetry to the projection cost model: predict the graphs-regime speedup S at
    /// the REALIZED mean union and the supplied residual floor. Lets the live engine report a
    /// projected S from its own measured acceptance, without needing wall-clock timing. Uses the
    /// realized `mean_accept_length` as τ (so emitted reflects what actually happened) and rounds the
    /// realized mean union to the nearest expert. Returns 0 before any speculating round.
    pub fn projected_speedup(
        &self,
        model: &RoundCostModel,
        depth: usize,
        floor_fraction: f32,
        graphs: bool,
    ) -> f32 {
        if self.spec_rounds == 0 {
            return 0.0;
        }
        let measured = MeasuredAccept {
            mean_accept_len: self.mean_accept_length(),
            // first-pos is not separately tracked here; derive a profile from τ alone by attributing
            // the same conditional accept across positions (first_pos = τ-implied). The projection
            // only consumes the profile's emitted total, which we pin to the realized τ, so the
            // speedup is governed by the realized union — the quantity telemetry actually measures.
            first_pos: (self.acceptance_rate()).clamp(0.05, 0.99),
        };
        let union = self.mean_union().round().max(8.0) as u32;
        model.predict_speedup(&measured, depth, union, floor_fraction, graphs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round(n_accepted: usize, n_drafted: usize, verify_depth: usize, union_size: u32) -> Eagle3RoundStats {
        Eagle3RoundStats { n_accepted, n_drafted, verify_depth, union_size }
    }

    #[test]
    fn empty_telemetry_is_all_zero() {
        let t = SpecTelemetry::new();
        let s = t.snapshot();
        assert_eq!(s.rounds, 0);
        assert_eq!(s.mean_accept_length, 0.0);
        assert_eq!(s.mean_union, 0.0);
        assert_eq!(s.acceptance_rate, 0.0);
    }

    #[test]
    fn mean_accept_length_matches_vllm_definition() {
        // Three rounds accepting 2, 1, 3 draft tokens → mean accepted 2.0, +1 bonus = τ 3.0.
        let mut t = SpecTelemetry::new();
        t.record(&round(2, 4, 3, 8));
        t.record(&round(1, 4, 2, 8));
        t.record(&round(3, 4, 4, 16));
        assert!((t.mean_accept_length() - 3.0).abs() < 1e-6, "τ = mean accepted (2) + 1 bonus");
        assert_eq!(t.rounds(), 3);
    }

    #[test]
    fn union_and_depth_average_over_speculating_rounds_only() {
        let mut t = SpecTelemetry::new();
        t.record(&round(2, 4, 3, 8));
        t.record(&round(1, 4, 2, 16));
        // a degenerate (no-draft) round: counts a round + bonus token, but no union/depth/draft stats
        t.record(&round(0, 0, 0, 0));
        assert_eq!(t.snapshot().spec_rounds, 2);
        assert!((t.mean_union() - 12.0).abs() < 1e-6, "(8+16)/2 over the 2 spec rounds");
        assert!((t.mean_verify_depth() - 2.5).abs() < 1e-6);
        // τ over ALL 3 rounds: accepted 2+1+0 = 3 → mean 1.0 + bonus = 2.0
        assert!((t.mean_accept_length() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn acceptance_rate_is_accepted_over_drafted() {
        let mut t = SpecTelemetry::new();
        t.record(&round(3, 4, 4, 8)); // 3 of 4
        t.record(&round(1, 4, 2, 8)); // 1 of 4
        assert!((t.acceptance_rate() - 0.5).abs() < 1e-6, "4 accepted / 8 drafted");
    }

    #[test]
    fn degenerate_round_does_not_divide_by_zero_or_inflate_union() {
        let mut t = SpecTelemetry::new();
        t.record(&round(0, 0, 0, 0));
        let s = t.snapshot();
        assert_eq!(s.rounds, 1);
        assert_eq!(s.spec_rounds, 0);
        assert_eq!(s.mean_union, 0.0);
        assert_eq!(s.acceptance_rate, 0.0);
        // still emits the bonus token each round → τ = 0 accepted + 1 = 1.0
        assert!((s.mean_accept_length - 1.0).abs() < 1e-6);
    }

    #[test]
    fn snapshot_serializes_to_a_valid_sse_frame() {
        let mut t = SpecTelemetry::new();
        t.record(&round(2, 4, 3, 8));
        t.record(&round(1, 4, 2, 16));
        let frame = t.to_sse_frame();
        assert!(frame.starts_with("data: "), "SSE frames start with 'data: '");
        assert!(frame.ends_with("\n\n"), "SSE frames end with a blank line");
        // The payload must be valid JSON carrying the snapshot fields.
        let json = frame.strip_prefix("data: ").unwrap().trim_end();
        let v: serde_json::Value = serde_json::from_str(json).expect("valid JSON payload");
        assert_eq!(v["rounds"], 2);
        assert_eq!(v["spec_rounds"], 2);
        assert!((v["mean_union"].as_f64().unwrap() - 12.0).abs() < 1e-6);
        assert!((v["mean_accept_length"].as_f64().unwrap() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn projected_speedup_uses_realized_union() {
        // Realized small union → higher projected S than a realized large union, holding τ.
        let model = RoundCostModel { head_eager_cost: 0.3, draft_graph_cost: 0.2 };
        let mut small = SpecTelemetry::new();
        let mut large = SpecTelemetry::new();
        for _ in 0..5 {
            small.record(&round(2, 4, 3, 8));   // tight union
            large.record(&round(2, 4, 3, 48));  // wide union, same acceptance
        }
        let s_small = small.projected_speedup(&model, 4, 0.40, true);
        let s_large = large.projected_speedup(&model, 4, 0.40, true);
        assert!(s_small > s_large, "tight realized union projects higher S ({s_small} > {s_large})");
        assert_eq!(SpecTelemetry::new().projected_speedup(&model, 4, 0.40, true), 0.0,
            "no speculating rounds → 0");
    }
}
