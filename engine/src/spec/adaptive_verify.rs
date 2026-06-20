//! Adaptive verification — the EVICT-style "verify only what pays" selector.
//!
//! Companion to `route_aware.rs`. Where `RouteAwarePolicy` shapes *which* tokens the drafter
//! proposes (to keep the expert union small), this module decides *how far down the proposed
//! chain to actually verify*. The verify pass reads the UNION of experts the verified positions
//! touch (a grouped GEMM reads each active expert's weight once — `tools/verify_cost_check.py`),
//! so a longer verify chain buys more expected accepted tokens but costs more expert-weight read.
//! There is an optimal depth; this finds it per round.
//!
//! Validated direction: EVICT (arXiv:2605.00342) reports −32.5% activated experts / −26.6% verify
//! latency / 1.25× over EAGLE-3 on Qwen3-235B-A22B (our exact model), training-free and LOSSLESS —
//! because most of the win is verifying ~75% fewer tokens (a comms/union win), which is exactly the
//! floor-bound regime we measured. Truncating the verify NEVER changes correctness: speculative
//! sampling is lossless for any verify budget; an unverified position just isn't speculated (the
//! target decodes it normally next round). So this is a pure throughput knob.
//!
//! All inputs (per-position accept-prob estimates + predicted expert sets) come from the drafter /
//! `routing::predictor`; this module is the pure-CPU decision and is fully unit-testable.

use crate::routing::types::ExpertId;
use crate::spec::route_aware::ExpertUnion;

const TOPK: f32 = 8.0;
/// Verify weight cost split (from the FP8 verify config / `spec_floor_model.py`): the non-expert
/// term (attention + router) is paid regardless; the routed-expert term scales with union/TOPK.
const NONEXPERT_SHARE: f32 = 0.34;
const ROUTED_SHARE: f32 = 0.66;

/// Verify cost of reading `union_size` distinct experts, in units of one normal decode step.
///
///   cost = F + (1-F)·(NONEXPERT_SHARE + ROUTED_SHARE·union/TOPK)
///
/// `floor_fraction` F = the per-step comms+launch+host floor as a fraction of a decode step
/// (measured ~0.86 eager-bound; → 0 as CUDA graphs / comms fixes remove the launch floor). At high
/// F the union barely matters (verify ≈ one step, spec just amortizes the floor); at low F the
/// union dominates and adaptive verification / route-awareness become first-order.
pub fn verify_cost(union_size: u32, floor_fraction: f32) -> f32 {
    let f = floor_fraction.clamp(0.0, 1.0);
    let weight_units = NONEXPERT_SHARE + ROUTED_SHARE * (union_size as f32 / TOPK);
    f + (1.0 - f) * weight_units
}

/// Expected number of accepted DRAFT tokens when verifying a chain of `accept_probs` to full depth.
/// Standard speculative result: E[accepted] = Σ_i Π_{j≤i} p_j (the accepted run ends at the first
/// reject). Does NOT include the always-emitted bonus token — see [`emitted`].
pub fn expected_accepted(accept_probs: &[f32]) -> f32 {
    let mut total = 0.0;
    let mut prefix = 1.0;
    for &p in accept_probs {
        prefix *= p.clamp(0.0, 1.0);
        total += prefix;
    }
    total
}

/// Tokens EMITTED per round = accepted draft tokens + 1 bonus (the target's resampled token at the
/// first mismatch is always emitted). This is the spec throughput numerator.
pub fn emitted(accept_probs: &[f32]) -> f32 {
    expected_accepted(accept_probs) + 1.0
}

/// The chosen verify budget for one round.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyPlan {
    /// How many leading chain positions to verify (1..=chain length). >=1 always (the bonus token
    /// makes depth 1 worthwhile whenever there's any candidate).
    pub depth: usize,
    /// Expected emitted tokens at that depth (accepted + bonus).
    pub emitted: f32,
    /// Distinct experts the verify reads at that depth.
    pub union_size: u32,
    /// Verify cost at that depth (decode-step units).
    pub verify_cost: f32,
    /// Throughput value = emitted / verify_cost (the quantity maximized).
    pub value: f32,
}

/// EVICT-style adaptive verification: pick the verify depth d ∈ 1..=k that maximizes emitted/cost.
///
/// `accept_probs[i]`     = estimated P(target accepts draft token i | position i is reached),
///                         from the drafter's confidence or a predictor. (Lossless either way —
///                         a bad estimate only costs throughput, never correctness.)
/// `experts_per_pos[i]`  = predicted experts the verify activates for position i (top-k ids <128).
/// `floor_fraction`      = F (see [`verify_cost`]).
///
/// Returns the best `VerifyPlan`. Ties resolve to the SHALLOWER depth (cheaper, fewer experts).
/// Greedy-prefix is optimal here because both emitted(d) and union(d) are monotonic non-decreasing
/// in d, so we just scan all depths and take the peak value.
pub fn adaptive_verify_depth(
    accept_probs: &[f32],
    experts_per_pos: &[Vec<ExpertId>],
    floor_fraction: f32,
) -> Option<VerifyPlan> {
    let k = accept_probs.len().min(experts_per_pos.len());
    if k == 0 {
        return None;
    }
    let mut union = ExpertUnion::new();
    let mut best: Option<VerifyPlan> = None;
    for d in 1..=k {
        union.insert_all(&experts_per_pos[d - 1]); // commit position d-1's experts
        let em = emitted(&accept_probs[..d]);
        let u = union.size();
        let cost = verify_cost(u, floor_fraction);
        let value = if cost > 0.0 { em / cost } else { f32::INFINITY };
        let plan = VerifyPlan { depth: d, emitted: em, union_size: u, verify_cost: cost, value };
        match &best {
            Some(b) if plan.value > b.value => best = Some(plan),
            None => best = Some(plan),
            _ => {} // strictly-greater keeps the shallower depth on ties
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_cost_is_one_step_at_single_token_decode() {
        // union=8 (one token's top-8), F=0 (weight-bound): weight = 0.34 + 0.66*1 = 1.0.
        assert!((verify_cost(8, 0.0) - 1.0).abs() < 1e-6);
        // grows with the union
        assert!(verify_cost(64, 0.0) > verify_cost(8, 0.0));
        // floor-bound: a SMALL union step barely moves cost (tax is on the (1-F) weight term)
        let c8 = verify_cost(8, 0.86);
        let c16 = verify_cost(16, 0.86);
        assert!(c16 - c8 < 0.15, "small union step barely moves cost at F=0.86 ({c8}->{c16})");
        // but a BIG union DOMINATES even floor-bound (Charles: route-aware first-order for the
        // big-tree verify — union~112 >> comms), so adaptive verification still pays there.
        assert!(verify_cost(112, 0.86) > 2.0, "big union dominates even at F=0.86");
    }

    #[test]
    fn expected_accepted_matches_geometric_chain() {
        // all p=1 → accept all k; emitted = k + bonus
        assert!((expected_accepted(&[1.0, 1.0, 1.0]) - 3.0).abs() < 1e-6);
        assert!((emitted(&[1.0, 1.0, 1.0]) - 4.0).abs() < 1e-6);
        // p=0.5 chain: 0.5 + 0.25 + 0.125 = 0.875
        assert!((expected_accepted(&[0.5, 0.5, 0.5]) - 0.875).abs() < 1e-6);
    }

    #[test]
    fn truncates_when_deep_positions_add_fresh_experts_for_little_gain() {
        // Weight-bound (F=0) so the union tax bites. Positions 0,1 high-prob & expert-overlapping;
        // position 2 low-prob and brings 8 FRESH experts → not worth verifying.
        let probs = vec![0.9, 0.85, 0.05];
        let experts = vec![
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![0, 1, 2, 3, 4, 5, 6, 7],         // reuses → union still 8
            vec![64, 65, 66, 67, 68, 69, 70, 71], // fresh 8 → union 16, but p=0.05
        ];
        let plan = adaptive_verify_depth(&probs, &experts, 0.0).unwrap();
        assert_eq!(plan.depth, 2, "should stop before the low-value union-doubling position");
        assert_eq!(plan.union_size, 8);
    }

    #[test]
    fn verifies_deep_when_floor_bound_hides_the_union_tax() {
        // SAME chain, but floor-bound (F=0.86): the union tax barely bites, so extending to the
        // bonus-bearing 3rd position is worth it (emitted rises, cost ~flat).
        let probs = vec![0.9, 0.85, 0.6];
        let experts = vec![
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![8, 9, 10, 11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20, 21, 22, 23],
        ];
        let plan = adaptive_verify_depth(&probs, &experts, 0.86).unwrap();
        assert_eq!(plan.depth, 3, "floor-bound → go deep, union tax is hidden");
    }

    #[test]
    fn always_verifies_at_least_one_and_handles_empty() {
        let one = adaptive_verify_depth(&[0.01], &[vec![0, 1]], 0.0).unwrap();
        assert_eq!(one.depth, 1);
        assert!((one.emitted - (0.01 + 1.0)).abs() < 1e-6); // bonus makes depth-1 always emit ~1
        assert!(adaptive_verify_depth(&[], &[], 0.0).is_none());
    }

    #[test]
    fn route_aware_smaller_union_raises_value_vs_disjoint() {
        // Two equally-accepting depth-3 chains; the route-aware one reuses experts (union 8),
        // the naive one is disjoint (union 24). Weight-bound → route-aware has higher value.
        let probs = vec![0.8, 0.8, 0.8];
        let overlap = vec![vec![0,1,2,3,4,5,6,7]; 3];
        let disjoint = vec![
            vec![0,1,2,3,4,5,6,7],
            vec![8,9,10,11,12,13,14,15],
            vec![16,17,18,19,20,21,22,23],
        ];
        let a = adaptive_verify_depth(&probs, &overlap, 0.0).unwrap();
        let b = adaptive_verify_depth(&probs, &disjoint, 0.0).unwrap();
        assert!(a.value > b.value, "route-aware (union {}) should beat disjoint (union {})", a.union_size, b.union_size);
    }
}
