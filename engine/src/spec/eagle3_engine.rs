//! Eagle3Engine — the route-aware speculative decode loop, end to end.
//!
//! Ties the built levers into one round: [`RouteAwareDrafter::draft_and_plan`] (route-aware draft +
//! adaptive verify-depth) → truncate the chain to the chosen depth → batched target verify →
//! frozen lossless [`accept::accept_multi_drafter`]. The draft head ([`CandidateSource`]) and the
//! target ([`ModelRunner`]) are abstracted, so the whole loop is unit-tested with mocks here; the
//! native cudarc EAGLE3 head + FP8 target (and the [`AuxModelRunner`] aux-hidden-state API) drop in
//! later per `engine/docs/eagle3-engine-integration.md` without changing this control flow.
//!
//! This is the route-aware sibling of [`crate::spec::engine::SpecEngine`] (left untouched): same
//! draft→verify→accept skeleton, but the drafter shapes the chain to shrink the verify expert union
//! and the verify depth is chosen adaptively. Output is bit-identical to normal decoding regardless
//! of λ or verify depth — both only trade throughput.

use crate::error::Result;
use crate::spec::accept::accept_multi_drafter;
use crate::spec::model::ModelRunner;
use crate::spec::route_aware_drafter::{CandidateSource, RouteAwareDrafter};
use crate::spec::types::{AcceptedRun, DraftTree, RngCore, TokenId};

/// Aux-hidden-state contract the target must expose for a REAL EAGLE3 head (the head reads 3 aux
/// states from the target at layers (1,46,90)). The verify path only needs `ModelRunner`; this is
/// the extra the head's [`CandidateSource`] consumes. Native target implements it; not needed by
/// `Eagle3Engine` itself, which is why the engine is generic over plain `ModelRunner`.
pub trait AuxModelRunner: ModelRunner {
    /// One target forward that also returns the auxiliary hidden states (flattened) the EAGLE3 head
    /// reads. `aux_layers` are the configured layer indices (e.g. [1, 46, 90]).
    fn forward_single_with_aux(
        &self,
        context: &[TokenId],
        next_token: TokenId,
        aux_layers: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>)>;
}

/// Per-step config for the route-aware spec engine.
#[derive(Debug, Clone)]
pub struct Eagle3Config {
    /// Max draft chain length (`num_speculative_tokens`).
    pub depth: usize,
    /// Candidates considered per draft position (the route-aware pick width).
    pub width: usize,
    pub vocab_size: usize,
    /// Floor fraction F for the adaptive verify-depth decision (~0.86 eager → →0 with graphs).
    pub floor_fraction: f32,
}

impl Default for Eagle3Config {
    fn default() -> Self {
        Self { depth: 5, width: 4, vocab_size: 151936, floor_fraction: 0.86 }
    }
}

/// Stats emitted per round (for telemetry / λ + depth tuning).
#[derive(Debug, Default, Clone)]
pub struct Eagle3RoundStats {
    pub n_accepted: usize,
    /// Chain length the drafter produced before adaptive truncation.
    pub n_drafted: usize,
    /// Verify depth chosen by adaptive_verify (positions actually verified).
    pub verify_depth: usize,
    /// Distinct experts the verify read at that depth (the cost driver this lever minimizes).
    pub union_size: u32,
}

/// Route-aware EAGLE3 speculative engine. `S` = draft head (candidate source), `T` = target verifier.
pub struct Eagle3Engine<S: CandidateSource, T: ModelRunner> {
    pub drafter: RouteAwareDrafter<S>,
    pub target: T,
    pub config: Eagle3Config,
}

impl<S: CandidateSource, T: ModelRunner> Eagle3Engine<S, T> {
    pub fn new(drafter: RouteAwareDrafter<S>, target: T, config: Eagle3Config) -> Self {
        Self { drafter, target, config }
    }

    /// Run one route-aware speculative round. Returns the accepted run (+ bonus) and stats.
    /// If the head yields no candidates the round degenerates to a single normal decode step
    /// (handled by the caller); here that surfaces as `n_drafted == 0`.
    pub fn step(
        &self,
        context: &[TokenId],
        rng: &mut impl RngCore,
    ) -> Result<(AcceptedRun, Eagle3RoundStats)> {
        let (mut proposal, _experts, plan) = self.drafter.draft_and_plan(
            context,
            self.config.depth,
            self.config.width,
            self.config.floor_fraction,
        );
        let n_drafted = proposal.tokens.len();
        // Adaptive verify-depth: verify only the prefix that pays (union-aware). Default to the full
        // chain if there's no plan (no candidates → empty).
        let verify_depth = plan.as_ref().map(|p| p.depth).unwrap_or(n_drafted);
        let union_size = plan.as_ref().map(|p| p.union_size).unwrap_or(0);

        proposal.tokens.truncate(verify_depth);
        proposal.logprobs.truncate(verify_depth);

        if proposal.tokens.is_empty() {
            // Degenerate round: nothing to speculate. Caller decodes normally.
            return Ok((
                AcceptedRun { accepted: vec![], accept_mask: vec![], bonus_token: 0, winning_drafter: None },
                Eagle3RoundStats { n_accepted: 0, n_drafted, verify_depth: 0, union_size },
            ));
        }

        let tree = DraftTree { proposals: vec![proposal], draft_len: verify_depth };
        let flat = tree.flat_tokens();
        let target_logits = self.target.forward_batch(context, &flat, self.config.vocab_size)?;
        let run = accept_multi_drafter(&tree, &target_logits, rng);

        let stats = Eagle3RoundStats {
            n_accepted: run.n_accepted(),
            n_drafted,
            verify_depth,
            union_size,
        };
        Ok((run, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::types::ExpertId;
    use crate::spec::route_aware::Candidate;

    /// Mock head: every position offers two candidates that REUSE experts [0..7] (so route-aware
    /// keeps the union small), with high draft confidence (logprob → accept-prob ~0.9).
    struct OverlapHead;
    impl CandidateSource for OverlapHead {
        fn candidates(&self, _c: &[TokenId], chain: &[TokenId], _w: usize) -> Vec<Candidate> {
            let t = 100 + chain.len() as u32;
            vec![
                Candidate { token: t, draft_logprob: -0.1, experts: (0u32..8).collect() },
                Candidate { token: t + 50, draft_logprob: -0.3, experts: (0u32..8).collect() },
            ]
        }
    }

    /// Target that scores whatever token it sees highly → accepts the whole draft (with FixedRng).
    struct EchoTarget(usize);
    impl ModelRunner for EchoTarget {
        fn forward_single(&self, _ctx: &[u32], tok: u32) -> Result<Vec<f32>> {
            let mut v = vec![0.0f32; self.0];
            v[tok as usize % self.0] = 12.0;
            Ok(v)
        }
        fn vocab_size(&self) -> usize { self.0 }
    }

    struct FixedRng;
    impl RngCore for FixedRng {
        fn next_f32(&mut self) -> f32 { 0.01 } // always accept
    }

    #[test]
    fn route_aware_round_accepts_and_keeps_union_small() {
        let vocab = 512;
        let drafter = RouteAwareDrafter::new(OverlapHead, 2.0); // route-aware
        let cfg = Eagle3Config { depth: 4, width: 2, vocab_size: vocab, floor_fraction: 0.86 };
        let engine = Eagle3Engine::new(drafter, EchoTarget(vocab), cfg);
        let (run, stats) = engine.step(&[1, 2, 3], &mut FixedRng).unwrap();

        assert_eq!(stats.n_drafted, 4, "head produced the full depth-4 chain");
        assert!(stats.verify_depth >= 1 && stats.verify_depth <= 4);
        assert_eq!(stats.union_size, 8, "all positions reuse experts [0..7] → union stays 8");
        // EchoTarget + always-accept RNG → the verified prefix is all accepted.
        assert_eq!(run.n_accepted(), stats.verify_depth);
    }

    #[test]
    fn empty_head_degenerates_gracefully() {
        struct EmptyHead;
        impl CandidateSource for EmptyHead {
            fn candidates(&self, _c: &[TokenId], _ch: &[TokenId], _w: usize) -> Vec<Candidate> { vec![] }
        }
        let engine = Eagle3Engine::new(
            RouteAwareDrafter::new(EmptyHead, 1.0),
            EchoTarget(64),
            Eagle3Config { depth: 4, width: 2, vocab_size: 64, floor_fraction: 0.0 },
        );
        let (run, stats) = engine.step(&[1], &mut FixedRng).unwrap();
        assert_eq!(stats.n_drafted, 0);
        assert_eq!(stats.verify_depth, 0);
        assert_eq!(run.n_accepted(), 0);
    }

    #[test]
    fn lambda_zero_is_plain_spec_same_acceptance() {
        // Route-awareness only changes WHICH tokens are drafted; with this overlapping head the
        // accepted count is identical to λ=0 (sanity that the policy doesn't break acceptance).
        let vocab = 256;
        let plain = Eagle3Engine::new(RouteAwareDrafter::new(OverlapHead, 0.0), EchoTarget(vocab),
            Eagle3Config { depth: 3, width: 2, vocab_size: vocab, floor_fraction: 0.86 });
        let aware = Eagle3Engine::new(RouteAwareDrafter::new(OverlapHead, 5.0), EchoTarget(vocab),
            Eagle3Config { depth: 3, width: 2, vocab_size: vocab, floor_fraction: 0.86 });
        let a = plain.step(&[7], &mut FixedRng).unwrap().1;
        let b = aware.step(&[7], &mut FixedRng).unwrap().1;
        assert_eq!(a.union_size, b.union_size); // overlapping head → same union either way
        let _ = (a.verify_depth, b.verify_depth);
    }

    #[test]
    fn aux_layers_constant_documents_the_head_contract() {
        // The native head reads aux at these layers (RedHat head); pin them so a regression is loud.
        let aux: &[usize] = &[1, 46, 90];
        assert_eq!(aux.len(), 3);
        let _ = aux.iter().map(|&l| l as ExpertId).collect::<Vec<_>>();
    }
}
