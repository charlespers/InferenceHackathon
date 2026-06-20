//! Projection — a **pre-registered** cost model that turns the MEASURED eager EAGLE3 stats into a
//! falsifiable prediction of the CUDA-graphs speedup S, *before* the graphs slot lands.
//!
//! Why this exists: on our box, EAGLE3+RedHat-head measures τ≈2.7 mean acceptance length and
//! first-position accept ≈0.75 (eager, lossless) — yet the *speedup* there is S≈1.0, because every
//! eager launch pays the per-step floor, so a draft chain of `depth` head forwards + one verify
//! roughly costs as many floor-bound steps as it emits tokens. CUDA graphs collapse that floor, and
//! the open question is how much S that buys. This module writes down the round-cost arithmetic
//! transparently (one knob, pinned by the eager S≈1 anchor) so we can PREDICT the graphs S range now
//! and check it against the measurement when `slot_graphs.DONE` arrives.
//!
//! The model is deliberately a transparent function of its inputs — it bakes in no hidden constants.
//! The caller supplies the two floor fractions (eager / graphs) and the verify expert-union sizes;
//! the tests below pin those to our measured/estimated values and assert the predicted ranges. Those
//! assertions ARE the pre-registration: if the graphs slot lands outside them, the model (or an
//! input) is wrong, and that's a finding.
//!
//! Cost unit: one **baseline decode step** = 1.0. By construction `verify_cost(8, F) == 1.0` for any
//! F (a single-token target forward reads one token's top-8 experts and pays the floor), so a plain
//! decode step is 1.0 in every regime and S = emitted / round_cost is directly the tok/s ratio.
//!
//! Reuses [`verify_cost`] and [`emitted`] from [`crate::spec::adaptive_verify`]; this module only adds
//! the acceptance-profile calibration and the draft-cost term. Pure CPU, fully unit-tested.

use crate::spec::adaptive_verify::{emitted, expected_accepted, verify_cost};

/// Measured aggregate acceptance from a real eager EAGLE3 run (RedHat head, FP8 target).
#[derive(Debug, Clone, Copy)]
pub struct MeasuredAccept {
    /// vLLM "Mean acceptance length" τ = emitted tokens per round (accepted draft tokens + 1 bonus).
    /// Measured τ≈2.7 on our box.
    pub mean_accept_len: f32,
    /// First-position accept probability P(target accepts draft token 0). Measured ≈0.75.
    pub first_pos: f32,
}

impl MeasuredAccept {
    /// Per-position **conditional** persistence p such that the geometric chain
    /// `[first_pos, p, p, …]` of length `depth` reproduces the measured `mean_accept_len`.
    ///
    /// Model: position 0 accepts with `first_pos` (the EAGLE3 head is most calibrated at the token it
    /// is directly conditioned on); each subsequent position accepts with a constant conditional `p`.
    /// Then expected accepted = first_pos·(1 + p + … + p^{depth-1}) and emitted = 1 + that. We solve
    /// for `p` by bisection (emitted is monotonic in p). Returns `first_pos`-independent edge cases
    /// cleanly: if the target is already met at p=0 (mean_accept_len ≤ 1 + first_pos) we return 0.
    pub fn persistence(&self, depth: usize) -> f32 {
        let target_accepted = (self.mean_accept_len - 1.0).max(0.0);
        if depth == 0 || self.first_pos <= 0.0 {
            return 0.0;
        }
        // expected_accepted at p: first_pos * sum_{i=0}^{depth-1} p^i
        let acc_at = |p: f32| -> f32 {
            let probs = self.accept_probs_with(p, depth);
            expected_accepted(&probs)
        };
        if acc_at(0.0) >= target_accepted {
            return 0.0; // first_pos alone already explains the (small) accept length
        }
        // p in [0, 0.999]; acc_at(0.999) is the max we can represent here.
        let (mut lo, mut hi) = (0.0f32, 0.999f32);
        if acc_at(hi) < target_accepted {
            return hi; // chain can't reach the measured length at this depth — saturate
        }
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if acc_at(mid) < target_accepted {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// The calibrated per-position accept-prob profile `[first_pos, p, p, …]` of length `depth`.
    pub fn accept_probs(&self, depth: usize) -> Vec<f32> {
        self.accept_probs_with(self.persistence(depth), depth)
    }

    fn accept_probs_with(&self, p: f32, depth: usize) -> Vec<f32> {
        (0..depth)
            .map(|i| if i == 0 { self.first_pos.clamp(0.0, 1.0) } else { p.clamp(0.0, 1.0) })
            .collect()
    }
}

/// Round-cost model. One spec round emits `emitted(profile)` tokens and costs:
///   - EAGER:  `depth · head_eager_cost + verify_cost(union, F_eager)`
///   - GRAPHS: `draft_graph_cost      + verify_cost(union, F_graphs)`
///
/// `head_eager_cost` is the cost of ONE eager draft-head forward in baseline-step units — it is the
/// single free knob, and [`RoundCostModel::pinned_from_eager`] solves it from the measured eager
/// anchor S≈1. `draft_graph_cost` is the whole draft chain's cost once captured into a CUDA graph
/// (the per-position launch floor is paid ~once); it is small and supplied as an estimate.
#[derive(Debug, Clone, Copy)]
pub struct RoundCostModel {
    pub head_eager_cost: f32,
    pub draft_graph_cost: f32,
}

impl RoundCostModel {
    /// Pin `head_eager_cost` so the model reproduces the measured eager speedup `s_eager` (≈1.0).
    ///
    /// Eager round_cost must equal emitted/s_eager. round_cost = depth·head + verify_cost(union_eager,
    /// F_eager) ⇒ head = (emitted/s_eager − verify_cost) / depth. Clamped ≥0 (a negative knob would
    /// mean the verify alone already over-explains the eager cost — then head≈0).
    pub fn pinned_from_eager(
        measured: &MeasuredAccept,
        depth: usize,
        union_eager: u32,
        f_eager: f32,
        s_eager: f32,
        draft_graph_cost: f32,
    ) -> Self {
        let em = emitted(&measured.accept_probs(depth));
        let target_round_cost = em / s_eager.max(1e-3);
        let verify = verify_cost(union_eager, f_eager);
        let head = if depth > 0 {
            ((target_round_cost - verify) / depth as f32).max(0.0)
        } else {
            0.0
        };
        Self { head_eager_cost: head, draft_graph_cost: draft_graph_cost.max(0.0) }
    }

    /// Predicted speedup S = emitted / round_cost for a round of `depth` drafted tokens whose verify
    /// reads `union` distinct experts, at floor fraction `floor_fraction`, in the given regime.
    pub fn predict_speedup(
        &self,
        measured: &MeasuredAccept,
        depth: usize,
        union: u32,
        floor_fraction: f32,
        graphs: bool,
    ) -> f32 {
        let em = emitted(&measured.accept_probs(depth));
        let draft = if graphs {
            self.draft_graph_cost
        } else {
            depth as f32 * self.head_eager_cost
        };
        let round_cost = draft + verify_cost(union, floor_fraction);
        if round_cost > 0.0 {
            em / round_cost
        } else {
            f32::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Our measured eager anchors (RedHat head, FP8 target, lossless).
    fn measured() -> MeasuredAccept {
        MeasuredAccept { mean_accept_len: 2.7, first_pos: 0.75 }
    }

    // Documented floor estimates (NOT measured by this module — they are the model's inputs):
    //   F_eager  ≈ 0.86  (spec_floor_model: eager per-step launch+comms+host floor share)
    //   F_graphs ≈ 0.40  (graphs collapse the floor ~5× (Alyssa); residual host/comms floor share)
    const F_EAGER: f32 = 0.86;
    const F_GRAPHS: f32 = 0.40;
    const DEPTH: usize = 5;

    #[test]
    fn persistence_reproduces_measured_accept_length() {
        let m = measured();
        let probs = m.accept_probs(DEPTH);
        assert_eq!(probs.len(), DEPTH);
        assert!((probs[0] - 0.75).abs() < 1e-6, "position 0 = measured first-pos");
        // emitted (= 1 + expected_accepted) should reproduce the measured τ≈2.7.
        let em = emitted(&probs);
        assert!((em - 2.7).abs() < 0.02, "calibrated emitted {em} ≈ measured 2.7");
        // persistence is a sane conditional prob in (0,1).
        let p = m.persistence(DEPTH);
        assert!(p > 0.3 && p < 0.9, "persistence {p} in plausible range");
    }

    #[test]
    fn eager_anchor_is_reproduced_by_construction() {
        // Pin to S_eager = 1.0 (measured: EAGLE3 eager ≈ baseline eager). Predicting eager back must
        // return ~1.0. union_eager: an EAGLE3 chain's verify touches more than one token's experts;
        // take ~24 (depth-5 chain, partial overlap) as the eager union.
        let m = measured();
        let union_eager = 24;
        let model = RoundCostModel::pinned_from_eager(&m, DEPTH, union_eager, F_EAGER, 1.0, 0.2);
        let s = model.predict_speedup(&m, DEPTH, union_eager, F_EAGER, false);
        assert!((s - 1.0).abs() < 0.05, "eager prediction {s} ≈ pinned 1.0");
        assert!(model.head_eager_cost > 0.0, "a real eager head-forward cost was pinned");
    }

    #[test]
    fn graphs_with_small_union_lands_in_the_predicted_headline_range() {
        // THE PRE-REGISTERED PREDICTION. Graphs collapse the floor; if the verify union stays small
        // (8–12 — consecutive EAGLE3 tokens route similarly, the route-aware thesis), S lands in the
        // literature/Alyssa-consistent 1.8–2.4× band. If the graphs slot lands outside [1.6,2.6],
        // this model or an input (F_graphs, union) is falsified — a finding either way.
        let m = measured();
        let model = RoundCostModel::pinned_from_eager(&m, DEPTH, 24, F_EAGER, 1.0, 0.2);
        let s_small = model.predict_speedup(&m, DEPTH, 10, F_GRAPHS, true);
        assert!(s_small >= 1.6 && s_small <= 2.6, "graphs small-union S={s_small} in [1.6,2.6]");
    }

    #[test]
    fn graphs_speedup_collapses_when_the_union_is_large_route_aware_payoff() {
        // Same regime, but a large verify union (≥48: a naive chain whose tokens route divergently).
        // At low F the routed-expert term dominates the verify, so S falls well below the small-union
        // case — i.e. in the graphs regime the EXPERT UNION is the first-order lever, which is exactly
        // what route-aware verification (adaptive_verify + RouteAwarePolicy) shrinks. Quantifies WHY
        // the route-aware add-on matters precisely in the regime where the headline speedup lives.
        let m = measured();
        let model = RoundCostModel::pinned_from_eager(&m, DEPTH, 24, F_EAGER, 1.0, 0.2);
        let s_small = model.predict_speedup(&m, DEPTH, 10, F_GRAPHS, true);
        let s_large = model.predict_speedup(&m, DEPTH, 48, F_GRAPHS, true);
        assert!(s_large < s_small, "large union S={s_large} < small union S={s_small}");
        assert!(s_small - s_large > 0.3, "route-aware union shrink buys a real S delta (>0.3×)");
    }

    #[test]
    fn graphs_beats_eager_for_the_same_chain() {
        // Holding union fixed, collapsing the floor (eager F → graphs F) strictly raises S — the
        // CUDA-graphs lever. This is the structural claim independent of the exact F_graphs value.
        let m = measured();
        let union = 12;
        let model = RoundCostModel::pinned_from_eager(&m, DEPTH, union, F_EAGER, 1.0, 0.2);
        let s_eager = model.predict_speedup(&m, DEPTH, union, F_EAGER, false);
        let s_graphs = model.predict_speedup(&m, DEPTH, union, F_GRAPHS, true);
        assert!(s_graphs > s_eager, "graphs S={s_graphs} > eager S={s_eager} for the same chain");
    }

    #[test]
    fn floor_effect_on_the_verify_is_union_dependent_min_union_is_8() {
        // Subtlety the model makes precise: the MINIMUM verify union is 8 (a single token's top-8),
        // and verify_cost(8, F) == 1.0 for ALL F. So at the ideal (all-tokens-route-identically)
        // union the residual graphs floor does NOT change the verify cost — graphs' win over eager is
        // the DRAFT-LAUNCH collapse (depth·head → one captured launch), not a verify-floor effect.
        // For union > 8 the routed-expert weight term exceeds one step, so LOWERING the floor exposes
        // it and the verify gets *costlier*. Pinning this prevents the false intuition that "graphs
        // always helps the verify", and shows the union — not the floor — is the graphs-regime lever.
        let m = measured();
        let model = RoundCostModel::pinned_from_eager(&m, DEPTH, 24, F_EAGER, 1.0, 0.2);
        // union == 8: floor-invariant verify → S identical across graphs floors.
        let s_lo = model.predict_speedup(&m, DEPTH, 8, 0.3, true);
        let s_hi = model.predict_speedup(&m, DEPTH, 8, 0.5, true);
        assert!((s_lo - s_hi).abs() < 1e-4, "union=8 verify is floor-invariant (S {s_lo} vs {s_hi})");
        // union > 8: lowering the floor exposes the routed-expert weight → verify costlier → S lower.
        let s_lo_big = model.predict_speedup(&m, DEPTH, 16, 0.3, true);
        let s_hi_big = model.predict_speedup(&m, DEPTH, 16, 0.5, true);
        assert!(s_lo_big < s_hi_big, "union>8: lower floor exposes union, S {s_lo_big} < {s_hi_big}");
    }
}
