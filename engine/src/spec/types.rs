/// Token vocabulary index.
pub type TokenId = u32;

/// Flat logit vector over the full vocabulary, host-side (f32).
/// On the GPU path this lives in a cudarc CudaSlice<f32>; we copy to host
/// only for the acceptance step (which is cheap at B=1 vocab sizes).
pub type Logits = Vec<f32>;

/// One drafter's proposal for a single round.
#[derive(Debug, Clone)]
pub struct DraftProposal {
    /// Drafter index (0..N).
    pub drafter_id: usize,
    /// Tokens proposed, in order. Length == k (the draft length).
    pub tokens: Vec<TokenId>,
    /// Draft model's log-probabilities for each proposed token.
    /// logprobs[i] = log p_draft(tokens[i] | context + tokens[..i])
    pub logprobs: Vec<f32>,
}

/// The flattened tree of all N drafters' proposals sent to the target for
/// verification in a single batched forward pass.
///
/// Layout: [drafter_0_tok_0, drafter_0_tok_1, ..., drafter_0_tok_{k-1},
///          drafter_1_tok_0, ..., drafter_{N-1}_tok_{k-1}]
/// Total tokens: N * k.
#[derive(Debug, Clone)]
pub struct DraftTree {
    pub proposals: Vec<DraftProposal>,  // length N
    pub draft_len: usize,               // k
}

impl DraftTree {
    pub fn n_drafters(&self) -> usize {
        self.proposals.len()
    }

    /// Flat token sequence for the batched target forward pass.
    pub fn flat_tokens(&self) -> Vec<TokenId> {
        self.proposals.iter().flat_map(|p| p.tokens.iter().copied()).collect()
    }
}

/// Target model's logits for each position in the draft tree, host-side.
///
/// Shape: [N * k, vocab_size] flattened row-major.
/// target_logits[drafter_i * k + pos] is the logit vector at that position.
#[derive(Debug)]
pub struct TargetLogits {
    pub data: Vec<f32>,         // [N*k * vocab]
    pub n_positions: usize,     // N * k
    pub vocab_size: usize,
}

impl TargetLogits {
    /// Log-probability assigned by the target to `token_id` at position `pos`.
    pub fn logprob_at(&self, pos: usize, token_id: TokenId) -> f32 {
        let row = &self.data[pos * self.vocab_size..(pos + 1) * self.vocab_size];
        log_softmax_single(row, token_id as usize)
    }

    /// Sample a token from the target distribution at `pos` using the
    /// adjusted rejection-sampling distribution.
    pub fn sample_adjusted(
        &self,
        pos: usize,
        draft_token: TokenId,
        draft_logprob: f32,
        rng: &mut impl RngCore,
    ) -> TokenId {
        let row = &self.data[pos * self.vocab_size..(pos + 1) * self.vocab_size];
        sample_adjusted_distribution(row, draft_token, draft_logprob, rng)
    }

    /// Greedy sample (argmax) from target at `pos`.
    pub fn greedy_at(&self, pos: usize) -> TokenId {
        let row = &self.data[pos * self.vocab_size..(pos + 1) * self.vocab_size];
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as TokenId)
            .unwrap_or(0)
    }
}

/// Outcome of one speculative decoding round.
#[derive(Debug, Clone)]
pub struct AcceptedRun {
    /// Tokens that passed acceptance (0..=k tokens from drafts).
    pub accepted: Vec<TokenId>,
    /// One additional token sampled directly from the target model
    /// (always produced, even if zero draft tokens were accepted).
    pub bonus_token: TokenId,
    /// Which drafter provided the accepted run (None if zero accepted).
    pub winning_drafter: Option<usize>,
    /// Per-position accept/reject flags (for telemetry).
    pub accept_mask: Vec<bool>,
}

impl AcceptedRun {
    /// All output tokens: accepted draft tokens + bonus.
    pub fn all_tokens(&self) -> impl Iterator<Item = TokenId> + '_ {
        self.accepted.iter().copied().chain(std::iter::once(self.bonus_token))
    }

    pub fn n_accepted(&self) -> usize {
        self.accepted.len()
    }
}

// ---------------------------------------------------------------------------
// Minimal RNG trait so the acceptance logic stays testable without pulling in
// a specific RNG crate. The real implementation will use a fast PRNG.
// ---------------------------------------------------------------------------

pub trait RngCore {
    /// Sample from Uniform(0, 1).
    fn next_f32(&mut self) -> f32;
}

// ---------------------------------------------------------------------------
// Internal math helpers
// ---------------------------------------------------------------------------

/// log p(token_id) from a raw logit row, computed with numerically stable
/// log-softmax.  Only needs the single value, not the full distribution.
pub(crate) fn log_softmax_single(logits: &[f32], idx: usize) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    logits[idx] - max - sum_exp.ln()
}

/// Sample from the adjusted distribution used when a draft token is rejected:
///   p_adj(x) ∝ max(0, p_target(x) - p_draft(x))
/// This preserves the exact target distribution in expectation.
pub(crate) fn sample_adjusted_distribution(
    target_logits: &[f32],
    draft_token: TokenId,
    draft_logprob: f32,
    rng: &mut impl RngCore,
) -> TokenId {
    let vocab = target_logits.len();
    let max = target_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = target_logits.iter().map(|&x| (x - max).exp()).sum();

    // Compute adjusted weights: max(0, p_target - p_draft)
    let draft_prob = draft_logprob.exp();
    let mut weights: Vec<f32> = target_logits
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            let p_t = (l - max - sum_exp.ln()).exp();
            let p_d = if i == draft_token as usize { draft_prob } else { 0.0 };
            (p_t - p_d).max(0.0)
        })
        .collect();

    let total: f32 = weights.iter().sum();
    if total < 1e-9 {
        // Degenerate: fall back to greedy.
        return target_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as TokenId)
            .unwrap_or(0);
    }

    // Normalise and sample.
    weights.iter_mut().for_each(|w| *w /= total);
    let u = rng.next_f32();
    let mut cum = 0.0f32;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if u <= cum {
            return i as TokenId;
        }
    }
    (vocab - 1) as TokenId
}
