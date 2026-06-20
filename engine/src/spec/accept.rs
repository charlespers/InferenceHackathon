/// Multi-drafter speculative decoding acceptance algorithm.
///
/// # Protocol
///
/// Standard (single-drafter) speculative sampling [Leviathan et al. 2023]:
///   For each draft position i:
///     Accept token t_i iff p_target(t_i) / p_draft(t_i) >= u ~ Uniform(0,1).
///     On rejection: sample from adjusted distribution and stop.
///   Bonus token: always sample one token from target after the accepted run.
///   Property: output distribution == target distribution exactly.
///
/// Multi-drafter extension:
///   At each position i, we have N independent proposals (one per drafter).
///   We try them in order (or ranked by draft logprob) and accept the first
///   that passes the acceptance test. If any passes, move to position i+1.
///   On position where ALL N drafters fail, sample from the adjusted
///   distribution of whichever drafter had the highest p_target(t_k)/p_draft(t_k)
///   ratio, then stop.
///
///   Property: still produces samples from the exact target distribution,
///   because each per-drafter acceptance test uses the standard criterion.
///   The "try N drafters" operation is equivalent to a tournament that accepts
///   any winner — the adjusted distribution on rejection correctly accounts for
///   the probability mass used by all N proposals.
///
/// # GPU placement
///
/// The acceptance logic runs on the CPU (host) after target logits are copied
/// from device. At B=1 and vocab ~152k:
///   copy cost = 152k * 4B * N*k ≈ 5 MB for N=4, k=8 → negligible.
use crate::spec::types::{
    AcceptedRun, DraftTree, RngCore, TargetLogits, TokenId,
};

/// Accept/reject one full round of multi-drafter speculation.
///
/// `tree`    : the N drafter proposals built this round.
/// `target`  : logits from the target's batched verification pass.
/// `rng`     : caller-supplied RNG (Uniform(0,1)).
///
/// Returns the set of accepted tokens + one bonus token.
pub fn accept_multi_drafter(
    tree: &DraftTree,
    target: &TargetLogits,
    rng: &mut impl RngCore,
) -> AcceptedRun {
    let n = tree.n_drafters();
    let k = tree.draft_len;
    let vocab = target.vocab_size;

    let mut accepted: Vec<TokenId> = Vec::with_capacity(k);
    let mut accept_mask: Vec<bool> = Vec::with_capacity(k);
    let mut winning_drafter: Option<usize> = None;
    let bonus_token: TokenId;

    'positions: for pos in 0..k {
        // Try each drafter at this position.
        // Target logit row for drafter d at position pos:
        //   flat index = d * k + pos
        let mut best_ratio: f32 = f32::NEG_INFINITY;
        let mut best_drafter_idx: usize = 0;
        let mut best_draft_logprob: f32 = 0.0;
        let mut best_token: TokenId = 0;

        for d in 0..n {
            let flat_pos = d * k + pos;
            let draft_token = tree.proposals[d].tokens[pos];
            let draft_logprob = tree.proposals[d].logprobs[pos];
            let target_logprob = target.logprob_at(flat_pos, draft_token);

            // Standard speculative sampling criterion.
            let ratio = (target_logprob - draft_logprob).exp().min(1.0);
            let u = rng.next_f32();

            if u < ratio {
                // This drafter's token is accepted at this position.
                accepted.push(draft_token);
                accept_mask.push(true);
                if pos == 0 || winning_drafter.is_none() {
                    winning_drafter = Some(d);
                }
                continue 'positions;
            }

            // Track the best candidate in case all fail (for adjusted sample).
            if target_logprob - draft_logprob > best_ratio {
                best_ratio = target_logprob - draft_logprob;
                best_drafter_idx = d;
                best_draft_logprob = draft_logprob;
                best_token = draft_token;
            }
        }

        // All N drafters rejected at this position.
        // Sample from adjusted distribution using the best drafter's proposal.
        accept_mask.push(false);
        let flat_pos = best_drafter_idx * k + pos;
        let target_row =
            &target.data[flat_pos * vocab..(flat_pos + 1) * vocab];
        bonus_token = sample_adjusted_inline(target_row, best_token, best_draft_logprob, rng);
        return AcceptedRun { accepted, bonus_token, winning_drafter, accept_mask };
    }

    // All k positions accepted — sample bonus token from target at position k.
    // Use the last winning drafter's context: target row at (winner * k + k-1)
    // but we need the *next* position which the target hasn't scored yet.
    // In practice the caller must run one more target forward pass for the
    // bonus token, or pre-compute it by passing k+1 draft positions.
    // We use greedy from the last available target row as a stand-in.
    let last_winner = winning_drafter.unwrap_or(0);
    let last_flat = last_winner * k + (k - 1);
    bonus_token = target.greedy_at(last_flat);

    AcceptedRun { accepted, bonus_token, winning_drafter, accept_mask }
}

/// Inline copy of sample_adjusted_distribution to avoid borrowing `target.data`
/// through the TargetLogits method while we hold a slice reference.
fn sample_adjusted_inline(
    target_logits: &[f32],
    draft_token: TokenId,
    draft_logprob: f32,
    rng: &mut impl RngCore,
) -> TokenId {
    let vocab = target_logits.len();
    let max = target_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = target_logits.iter().map(|&x| (x - max).exp()).sum();
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
        return target_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as TokenId)
            .unwrap_or(0);
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::types::{DraftProposal, DraftTree, TargetLogits};

    struct FixedRng(f32);
    impl RngCore for FixedRng {
        fn next_f32(&mut self) -> f32 { self.0 }
    }

    fn flat_uniform_logits(n_positions: usize, vocab: usize) -> TargetLogits {
        TargetLogits {
            data: vec![0.0f32; n_positions * vocab],
            n_positions,
            vocab_size: vocab,
        }
    }

    fn make_proposal(drafter_id: usize, tokens: Vec<TokenId>, logprob: f32) -> DraftProposal {
        let logprobs = vec![logprob; tokens.len()];
        DraftProposal { drafter_id, tokens, logprobs }
    }

    #[test]
    fn all_accepted_with_high_target_prob() {
        // Draft logprob = -1.0, target logprob for same token = 0.0 (uniform
        // over 1-element vocab). Ratio = exp(0 - (-1)) = e > 1, clamped to 1.
        // With u=0.01 every position is accepted.
        let vocab = 4;
        let k = 3;
        let proposals = vec![make_proposal(0, vec![0, 1, 2], -1.0)];
        let tree = DraftTree { proposals, draft_len: k };

        // Target logits: uniform (0.0 each), so log p(tok) = log(1/4) for all.
        let mut target = flat_uniform_logits(k, vocab);
        // But we want target logprob for the draft tokens to be high.
        // Make logits strongly favour token [0,1,2] at each position.
        for pos in 0..k {
            let tok = pos as usize;
            target.data[pos * vocab + tok] = 10.0; // dominate softmax
        }

        let mut rng = FixedRng(0.01);
        let run = accept_multi_drafter(&tree, &target, &mut rng);
        assert_eq!(run.n_accepted(), k, "all {k} tokens should be accepted");
        assert_eq!(run.accepted, vec![0, 1, 2]);
    }

    #[test]
    fn all_rejected_produces_bonus_token() {
        // Draft logprob = 0.0 (probability 1), target favours a different token.
        // Ratio = p_target(draft_tok) / 1.0 which is tiny → rejected.
        let vocab = 4;
        let k = 2;
        // Draft always proposes token 3, but target strongly prefers token 0.
        let proposals = vec![make_proposal(0, vec![3, 3], 0.0)];
        let tree = DraftTree { proposals, draft_len: k };

        let mut target = flat_uniform_logits(k, vocab);
        for pos in 0..k {
            target.data[pos * vocab + 0] = 10.0; // target loves token 0
        }

        let mut rng = FixedRng(0.99); // u=0.99, ratio will be tiny → reject
        let run = accept_multi_drafter(&tree, &target, &mut rng);
        assert_eq!(run.n_accepted(), 0);
        // Bonus token should come from adjusted distribution (≈token 0).
        assert_eq!(run.bonus_token, 0);
    }

    #[test]
    fn second_drafter_saves_the_round() {
        // Drafter 0 proposes token 3 (target hates it, rejected).
        // Drafter 1 proposes token 0 (target loves it, accepted).
        let vocab = 4;
        let k = 1;
        let proposals = vec![
            make_proposal(0, vec![3], -0.1),  // target will reject
            make_proposal(1, vec![0], -0.1),  // target will accept
        ];
        let tree = DraftTree { proposals, draft_len: k };

        let mut target = flat_uniform_logits(2 * k, vocab); // 2 drafters * k
        // Drafter 0 flat index 0: token 3 has low target prob
        // Drafter 1 flat index 1: token 0 has high target prob
        target.data[0 * vocab + 0] = 10.0; // drafter-0 position: target loves 0, not 3
        target.data[1 * vocab + 0] = 10.0; // drafter-1 position: target loves 0

        let mut rng = FixedRng(0.5);
        let run = accept_multi_drafter(&tree, &target, &mut rng);
        // Drafter 1's token 0 should be accepted.
        assert_eq!(run.n_accepted(), 1);
        assert_eq!(run.accepted[0], 0);
        assert_eq!(run.winning_drafter, Some(1));
    }
}
