use crate::error::Result;
use crate::spec::{
    accept::accept_multi_drafter,
    model::{DrafterPool, ModelRunner},
    types::{AcceptedRun, DraftTree, RngCore, TokenId},
};

/// Configuration for the speculative engine.
#[derive(Debug, Clone)]
pub struct SpecConfig {
    /// Number of tokens each drafter proposes per round.
    pub draft_len: usize,
    /// Vocabulary size (must match both draft and target models).
    pub vocab_size: usize,
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self { draft_len: 8, vocab_size: 151936 }  // Qwen3 vocab
    }
}

/// Statistics emitted per round (maps to `x_telemetry` in the SSE stream).
#[derive(Debug, Default, Clone)]
pub struct RoundStats {
    pub n_accepted: usize,
    pub n_proposed: usize,
    pub winning_drafter: Option<usize>,
}

/// Top-level speculative decoding engine.
///
/// Owns the drafter pool and a reference to the target model. The caller
/// holds the KV context and calls `step` once per generation round.
///
/// # Example flow (one generate call)
/// ```ignore
/// let mut context: Vec<TokenId> = tokenize(prompt);
/// loop {
///     let (run, stats) = engine.step(&context, &mut rng)?;
///     for tok in run.all_tokens() {
///         context.push(tok);
///         emit_token(tok);
///     }
///     if is_eos(run.bonus_token) { break; }
/// }
/// ```
pub struct SpecEngine<D: DrafterPool, T: ModelRunner> {
    pub drafters: D,
    pub target: T,
    pub config: SpecConfig,
}

impl<D: DrafterPool, T: ModelRunner> SpecEngine<D, T> {
    pub fn new(drafters: D, target: T, config: SpecConfig) -> Self {
        Self { drafters, target, config }
    }

    /// Run one speculative decoding round.
    ///
    /// 1. All N drafters propose `draft_len` tokens in parallel.
    /// 2. Target verifies the full tree in one batched forward pass.
    /// 3. Acceptance logic walks the tree and returns the accepted run.
    ///
    /// Returns: (accepted run, round statistics).
    pub fn step(
        &self,
        context: &[TokenId],
        rng: &mut impl RngCore,
    ) -> Result<(AcceptedRun, RoundStats)> {
        // 1. Draft phase: N drafters run in parallel (pool handles scheduling).
        let proposals = self.drafters.draft(context, self.config.draft_len)?;
        let n = proposals.len();
        let k = self.config.draft_len;

        let tree = DraftTree { proposals, draft_len: k };

        // 2. Verify phase: one batched target forward pass over all N*k tokens.
        let flat_tokens = tree.flat_tokens();
        let target_logits = self.target.forward_batch(
            context,
            &flat_tokens,
            self.config.vocab_size,
        )?;

        // 3. Accept phase: pure CPU logic over the logit rows copied from GPU.
        let run = accept_multi_drafter(&tree, &target_logits, rng);

        let stats = RoundStats {
            n_accepted: run.n_accepted(),
            n_proposed: n * k,
            winning_drafter: run.winning_drafter,
        };

        Ok((run, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::types::{DraftProposal, TargetLogits};

    // ---- Minimal test stubs --------------------------------------------------

    struct FixedRng;
    impl RngCore for FixedRng {
        fn next_f32(&mut self) -> f32 { 0.01 } // always accept
    }

    struct EchoTarget(usize); // target that scores the proposed token highly
    impl ModelRunner for EchoTarget {
        fn forward_single(&self, _ctx: &[u32], tok: u32) -> Result<Vec<f32>> {
            let mut v = vec![0.0f32; self.0];
            v[tok as usize] = 10.0;
            Ok(v)
        }
        fn vocab_size(&self) -> usize { self.0 }
    }

    struct SingleDrafter { vocab: usize }
    impl DrafterPool for SingleDrafter {
        fn draft(&self, _ctx: &[u32], draft_len: usize) -> Result<Vec<DraftProposal>> {
            Ok(vec![DraftProposal {
                drafter_id: 0,
                tokens: (0..draft_len as u32).collect(),
                logprobs: vec![-0.5; draft_len],
            }])
        }
        fn n_drafters(&self) -> usize { 1 }
    }

    #[test]
    fn step_accepts_all_when_target_agrees() {
        let cfg = SpecConfig { draft_len: 4, vocab_size: 8 };
        let engine = SpecEngine::new(SingleDrafter { vocab: 8 }, EchoTarget(8), cfg);
        let context: Vec<TokenId> = vec![1, 2, 3];
        let (run, stats) = engine.step(&context, &mut FixedRng).unwrap();
        assert_eq!(stats.n_accepted, 4);
        assert_eq!(run.n_accepted(), 4);
        assert_eq!(run.accepted, vec![0, 1, 2, 3]);
    }
}
