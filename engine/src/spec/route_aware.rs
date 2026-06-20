//! Route-aware draft candidate selection — the λ-knob mechanism.
//!
//! Implements the design in `docs/route-aware-drafting-design.md` (Charles) and the
//! "unclaimed lever" from `tools/spec_moe_model.py`: re-rank/prune draft candidates to
//! favor ones whose predicted experts **overlap the tree's already-committed expert union**,
//! so the verify pass reads a smaller union of MoE experts.
//!
//! # Why it pays (regime)
//! `tools/verify_cost_check.py`: a correctly-batched spec verify reads each active expert's
//! weight ONCE (grouped GEMM), so its weight cost grows with the UNION of experts the tree
//! touches, saturating toward all 128. For a big tree (W4×D8, M=32) union≈112 ⇒ ~7.66 ms ≫
//! the ~3 ms comms — the union DOMINATES the verify. Shrinking it is **first-order on the
//! comms-bound 1000-path engine** (second-order for plain floor-bound decode, where the union
//! tax falls on only ~14% weight — so keep `lambda` regime-adaptive: ~0 while floor-bound,
//! >0 as the floor falls / the tree grows).
//!
//! # Losslessness
//! This ONLY changes *which* tokens are proposed by the drafter. Speculative-sampling
//! acceptance (`accept::accept_multi_drafter`) corrects for ANY draft distribution, so the
//! emitted output remains distributed exactly as the target. Route-awareness trades a little
//! acceptance (`E[accepted]` falls with λ) for a smaller union — it is never a quality risk,
//! only a throughput trade. The optimum maximizes `E[accepted] / verify_cost(union)`.
//!
//! The predictor (`routing::predictor::RoutePredictor`, e.g. `DirectProxy`) supplies each
//! candidate's predicted expert set ~for free from the residual stream; this module is the
//! pure-CPU *policy* that consumes those predictions, decoupled from how they're produced so
//! it is exhaustively unit-testable without a model or GPU.

use crate::routing::types::ExpertId;
use crate::spec::types::TokenId;

/// Qwen3-235B-A22B has 128 experts → a u128 bitset gives O(1) union/overlap with no alloc.
const MAX_EXPERTS: u32 = 128;

/// One draft candidate the policy ranks: a proposed token, its draft log-prob, and the
/// experts the verify would activate for it (predicted by a `RoutePredictor`).
#[derive(Debug, Clone)]
pub struct Candidate {
    pub token: TokenId,
    pub draft_logprob: f32,
    /// Predicted top-k experts for this candidate (length ~= top_k, e.g. 8). Expert ids < 128.
    pub experts: Vec<ExpertId>,
}

/// The set of experts the draft tree has committed so far this round — what the verify must read.
/// Backed by a u128 bitset (expert ids 0..=127).
#[derive(Debug, Clone, Default)]
pub struct ExpertUnion {
    bits: u128,
}

impl ExpertUnion {
    pub fn new() -> Self {
        Self { bits: 0 }
    }

    /// Add a candidate's experts to the committed union.
    pub fn insert_all(&mut self, experts: &[ExpertId]) {
        for &e in experts {
            if e < MAX_EXPERTS {
                self.bits |= 1u128 << e;
            }
        }
    }

    /// How many of `experts` are already in the union (the part the verify reads "for free").
    pub fn overlap_count(&self, experts: &[ExpertId]) -> u32 {
        experts
            .iter()
            .filter(|&&e| e < MAX_EXPERTS && (self.bits & (1u128 << e)) != 0)
            .count() as u32
    }

    /// Distinct experts committed so far (the union size that drives verify weight cost).
    pub fn size(&self) -> u32 {
        self.bits.count_ones()
    }
}

/// Route-aware candidate selection policy.
///
/// `lambda` is the knob: 0.0 → plain speculation (pick the highest draft-prob token, max
/// acceptance); large → minimize the union (max overlap) at the cost of acceptance.
#[derive(Debug, Clone, Copy)]
pub struct RouteAwarePolicy {
    pub lambda: f32,
}

impl RouteAwarePolicy {
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }

    /// Score = draft_logprob + λ · overlap_fraction(candidate.experts, committed union).
    /// overlap_fraction ∈ [0,1] = |experts ∩ U| / |experts|.
    pub fn score(&self, c: &Candidate, u: &ExpertUnion) -> f32 {
        let ov = if c.experts.is_empty() {
            0.0
        } else {
            u.overlap_count(&c.experts) as f32 / c.experts.len() as f32
        };
        c.draft_logprob + self.lambda * ov
    }

    /// Pick the single best candidate by score, commit its experts to `u`, return its index.
    /// Returns `None` only for an empty candidate set. Ties resolve to the lowest index
    /// (deterministic).
    pub fn select_one(&self, candidates: &[Candidate], u: &mut ExpertUnion) -> Option<usize> {
        let mut best_idx = None;
        let mut best_score = f32::NEG_INFINITY;
        for (i, c) in candidates.iter().enumerate() {
            let s = self.score(c, u);
            if s > best_score {
                best_score = s;
                best_idx = Some(i);
            }
        }
        if let Some(i) = best_idx {
            u.insert_all(&candidates[i].experts);
        }
        best_idx
    }

    /// Greedily select up to `width` candidates for a width-`width` tree level. Each pick
    /// commits its experts to `u` BEFORE the next is scored, so later picks see the growing
    /// union and are pulled toward it (the whole point — the union grows sub-linearly).
    /// Returns selected indices in pick order (no candidate picked twice).
    pub fn select_width(
        &self,
        candidates: &[Candidate],
        width: usize,
        u: &mut ExpertUnion,
    ) -> Vec<usize> {
        let mut chosen: Vec<usize> = Vec::with_capacity(width);
        while chosen.len() < width {
            let mut best_idx = None;
            let mut best_score = f32::NEG_INFINITY;
            for (i, c) in candidates.iter().enumerate() {
                if chosen.contains(&i) {
                    continue;
                }
                let s = self.score(c, u);
                if s > best_score {
                    best_score = s;
                    best_idx = Some(i);
                }
            }
            match best_idx {
                Some(i) => {
                    u.insert_all(&candidates[i].experts);
                    chosen.push(i);
                }
                None => break, // exhausted candidates
            }
        }
        chosen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(token: TokenId, lp: f32, experts: &[ExpertId]) -> Candidate {
        Candidate { token, draft_logprob: lp, experts: experts.to_vec() }
    }

    #[test]
    fn union_tracks_distinct_experts_and_overlap() {
        let mut u = ExpertUnion::new();
        assert_eq!(u.size(), 0);
        u.insert_all(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(u.size(), 8);
        // re-inserting overlapping experts doesn't grow the union
        u.insert_all(&[5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(u.size(), 12);
        // overlap_count sees the 4 shared (9..12 are now in U)
        assert_eq!(u.overlap_count(&[9, 10, 11, 12]), 4);
        assert_eq!(u.overlap_count(&[100, 101]), 0);
    }

    #[test]
    fn union_ignores_out_of_range_expert_ids() {
        let mut u = ExpertUnion::new();
        u.insert_all(&[127, 128, 200]); // only 127 is valid (<128)
        assert_eq!(u.size(), 1);
        assert_eq!(u.overlap_count(&[127, 128]), 1);
    }

    #[test]
    fn lambda_zero_picks_highest_draft_prob() {
        // Plain speculation: ignore experts, pick max logprob → index 1.
        let cands = vec![
            cand(10, -1.0, &[0, 1, 2, 3, 4, 5, 6, 7]),
            cand(11, -0.2, &[64, 65, 66, 67, 68, 69, 70, 71]), // best logprob, disjoint experts
            cand(12, -0.8, &[0, 1, 2, 3, 4, 5, 6, 7]),
        ];
        let policy = RouteAwarePolicy::new(0.0);
        let mut u = ExpertUnion::new();
        u.insert_all(&[0, 1, 2, 3, 4, 5, 6, 7]); // tree already touched 0..7
        assert_eq!(policy.select_one(&cands, &mut u), Some(1));
    }

    #[test]
    fn large_lambda_prefers_overlapping_experts() {
        // Same candidates, but a big λ flips the choice to the fully-overlapping token (index 0),
        // even though index 1 has a higher draft prob.
        let cands = vec![
            cand(10, -1.0, &[0, 1, 2, 3, 4, 5, 6, 7]), // 100% overlap with U
            cand(11, -0.2, &[64, 65, 66, 67, 68, 69, 70, 71]), // 0% overlap, best logprob
            cand(12, -0.8, &[0, 1, 2, 3, 64, 65, 66, 67]), // 50% overlap
        ];
        let policy = RouteAwarePolicy::new(5.0);
        let mut u = ExpertUnion::new();
        u.insert_all(&[0, 1, 2, 3, 4, 5, 6, 7]);
        // index 0: -1.0 + 5*1.0 = 4.0 ; index 1: -0.2 + 0 = -0.2 ; index 2: -0.8 + 5*0.5 = 1.7
        assert_eq!(policy.select_one(&cands, &mut u), Some(0));
    }

    #[test]
    fn route_aware_yields_smaller_final_union_than_plain() {
        // A width-2 selection from candidates where the high-prob picks are expert-disjoint
        // (big union) but slightly-lower-prob picks overlap (small union). Route-aware should
        // produce a strictly smaller committed union than plain (λ=0) for this constructed case.
        let cands = vec![
            cand(1, -0.10, &[0, 1, 2, 3, 4, 5, 6, 7]),       // highest prob, fresh experts
            cand(2, -0.15, &[16, 17, 18, 19, 20, 21, 22, 23]), // 2nd prob, fresh experts (disjoint)
            cand(3, -0.50, &[0, 1, 2, 3, 4, 5, 6, 7]),       // low prob, SAME experts as cand 1
        ];
        // Plain: picks 1 then 2 → union = 16 distinct experts.
        let mut u_plain = ExpertUnion::new();
        let plain = RouteAwarePolicy::new(0.0).select_width(&cands, 2, &mut u_plain);
        assert_eq!(plain, vec![0, 1]);
        assert_eq!(u_plain.size(), 16);

        // Route-aware: picks 1 (fresh) then, with U={0..7}, cand 3 fully overlaps so a big λ
        // pulls it over cand 2 → union stays 8.
        let mut u_ra = ExpertUnion::new();
        let ra = RouteAwarePolicy::new(2.0).select_width(&cands, 2, &mut u_ra);
        assert_eq!(ra, vec![0, 2]);
        assert!(u_ra.size() < u_plain.size(), "route-aware union {} should be < plain {}", u_ra.size(), u_plain.size());
        assert_eq!(u_ra.size(), 8);
    }

    #[test]
    fn select_width_never_picks_same_candidate_twice_and_respects_count() {
        let cands = vec![
            cand(1, -0.1, &[0, 1]),
            cand(2, -0.2, &[2, 3]),
            cand(3, -0.3, &[4, 5]),
        ];
        let mut u = ExpertUnion::new();
        let chosen = RouteAwarePolicy::new(0.0).select_width(&cands, 5, &mut u); // ask for more than available
        assert_eq!(chosen.len(), 3); // capped at candidate count
        let mut sorted = chosen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3); // no duplicates
    }

    #[test]
    fn empty_candidates_select_none() {
        let mut u = ExpertUnion::new();
        assert_eq!(RouteAwarePolicy::new(1.0).select_one(&[], &mut u), None);
        assert!(RouteAwarePolicy::new(1.0).select_width(&[], 3, &mut u).is_empty());
    }
}
