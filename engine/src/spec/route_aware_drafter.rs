//! Route-aware drafter — builds a draft chain that keeps the verify's expert union small.
//!
//! Ties the validated pieces together: at each draft position it asks a [`CandidateSource`] for the
//! top-`width` candidate tokens (each with the draft model's logprob and that token's PREDICTED
//! experts), then uses [`RouteAwarePolicy`] to pick the candidate that best trades draft-probability
//! for overlap with the experts already committed this round — growing a shared [`ExpertUnion`] so
//! later positions reuse loaded experts. The result feeds the normal verify + `accept.rs`; pair it
//! with [`crate::spec::adaptive_verify::adaptive_verify_depth`] to also truncate the verify depth.
//!
//! The expert predictions come from `routing::predictor` (DirectProxy), now measured at ~0.72–0.81
//! top-k accuracy on a real Qwen3 MoE (11.6× random) — i.e. the route signal this lever rides is
//! real. The candidate *tokens* come from the EAGLE3 head; in tests both are mocked behind
//! [`CandidateSource`] so the selection logic is exercised without a model or GPU.
//!
//! Losslessness: this only changes WHICH tokens are proposed; speculative-sampling acceptance is
//! exact for any draft distribution. Route-awareness trades a little acceptance for a smaller union.

use crate::spec::route_aware::{Candidate, ExpertUnion, RouteAwarePolicy};
use crate::spec::types::{DraftProposal, TokenId};

/// Supplies per-position draft candidates. Real impl = EAGLE3 head + route predictor; mock in tests.
///
/// `context`        : confirmed tokens (KV warm up to here).
/// `chain_so_far`   : the route-aware tokens chosen this round so far (the head conditions on these).
/// `width`          : how many candidates to return for this position (top-`width` by draft prob).
pub trait CandidateSource {
    fn candidates(&self, context: &[TokenId], chain_so_far: &[TokenId], width: usize) -> Vec<Candidate>;
}

/// Builds a route-aware draft chain from a [`CandidateSource`] using a [`RouteAwarePolicy`].
pub struct RouteAwareDrafter<S: CandidateSource> {
    pub source: S,
    pub policy: RouteAwarePolicy,
}

impl<S: CandidateSource> RouteAwareDrafter<S> {
    pub fn new(source: S, lambda: f32) -> Self {
        Self { source, policy: RouteAwarePolicy::new(lambda) }
    }

    /// Build a draft chain of up to `depth` tokens, `width` candidates considered per position.
    /// Returns the proposal and the committed expert union (what the verify would read).
    /// Stops early if the source runs out of candidates.
    pub fn draft_chain(
        &self,
        context: &[TokenId],
        depth: usize,
        width: usize,
    ) -> (DraftProposal, ExpertUnion) {
        let mut tokens: Vec<TokenId> = Vec::with_capacity(depth);
        let mut logprobs: Vec<f32> = Vec::with_capacity(depth);
        let mut union = ExpertUnion::new();
        for _ in 0..depth {
            let cands = self.source.candidates(context, &tokens, width);
            match self.policy.select_one(&cands, &mut union) {
                Some(i) => {
                    tokens.push(cands[i].token);
                    logprobs.push(cands[i].draft_logprob);
                }
                None => break, // source exhausted
            }
        }
        (DraftProposal { drafter_id: 0, tokens, logprobs }, union)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::types::ExpertId;

    /// Mock source: at every position offers the same fixed candidate set (ignores context).
    struct FixedSource {
        cands: Vec<Candidate>,
    }
    impl CandidateSource for FixedSource {
        fn candidates(&self, _ctx: &[TokenId], _chain: &[TokenId], _width: usize) -> Vec<Candidate> {
            self.cands.clone()
        }
    }

    fn cand(token: TokenId, lp: f32, experts: &[ExpertId]) -> Candidate {
        Candidate { token, draft_logprob: lp, experts: experts.to_vec() }
    }

    fn three_choices() -> Vec<Candidate> {
        vec![
            cand(10, -0.10, &[0, 1, 2, 3, 4, 5, 6, 7]),        // best prob, "anchor" experts
            cand(11, -0.20, &[64, 65, 66, 67, 68, 69, 70, 71]), // 2nd prob, DISJOINT experts
            cand(12, -0.50, &[0, 1, 2, 3, 4, 5, 6, 7]),        // low prob, SAME experts as cand 10
        ]
    }

    #[test]
    fn lambda_zero_builds_plain_highest_prob_chain() {
        let d = RouteAwareDrafter::new(FixedSource { cands: three_choices() }, 0.0);
        let (prop, union) = d.draft_chain(&[1, 2, 3], 3, 3);
        // λ=0 → always token 10 (highest logprob). All three positions same token/experts → union 8.
        assert_eq!(prop.tokens, vec![10, 10, 10]);
        assert_eq!(union.size(), 8);
    }

    #[test]
    fn route_aware_keeps_union_smaller_than_naive() {
        // Naive (λ=0): pos0 picks 10 (experts 0..7). Subsequent picks also 10 → union 8.
        // Make the naive path expensive instead: a source whose top-prob candidate rotates experts.
        struct RotatingSource;
        impl CandidateSource for RotatingSource {
            fn candidates(&self, _c: &[TokenId], chain: &[TokenId], _w: usize) -> Vec<Candidate> {
                // best-prob candidate brings FRESH experts each step (0..7, 8..15, 16..23 -> union
                // grows); the lower-prob "anchor" candidate always reuses [0..7], which (after pos0
                // commits [0..7]) OVERLAPS the union, so route-awareness can reuse it.
                let base = (chain.len() as u32) * 8; // 0, 8, 16 ... fresh each position
                vec![
                    cand(100 + chain.len() as u32, -0.1, &[base, base+1, base+2, base+3, base+4, base+5, base+6, base+7]),
                    cand(200 + chain.len() as u32, -0.4, &[0, 1, 2, 3, 4, 5, 6, 7]), // anchor (overlaps after pos0)
                ]
            }
        }
        let naive = RouteAwareDrafter::new(RotatingSource, 0.0).draft_chain(&[1], 3, 2);
        let aware = RouteAwareDrafter::new(RotatingSource, 3.0).draft_chain(&[1], 3, 2);
        // naive chases fresh-expert high-prob tokens → big union; aware reuses the anchor → small union.
        assert!(aware.1.size() < naive.1.size(),
            "route-aware union {} should be < naive union {}", aware.1.size(), naive.1.size());
    }

    #[test]
    fn chain_length_respects_depth_and_stops_when_source_empty() {
        let d = RouteAwareDrafter::new(FixedSource { cands: three_choices() }, 0.0);
        assert_eq!(d.draft_chain(&[1], 5, 3).0.tokens.len(), 5);

        struct EmptySource;
        impl CandidateSource for EmptySource {
            fn candidates(&self, _c: &[TokenId], _ch: &[TokenId], _w: usize) -> Vec<Candidate> { vec![] }
        }
        let e = RouteAwareDrafter::new(EmptySource, 1.0).draft_chain(&[1], 4, 2);
        assert!(e.0.tokens.is_empty());
    }
}
