//! Draft-vocabulary mapping (d2t / t2d) — the exact thing the broken nm-testing conversion got wrong.
//!
//! An EAGLE3 head emits logits over a SMALL draft vocabulary (`draft_vocab_size`, e.g. 64000 for the
//! RedHat head, 32000 for the broken nm-testing one) — not the target's full 151936. Draft index `i`
//! corresponds to target token `d2t[i]`. Two rules the conversion must honour, or acceptance silently
//! collapses (this is what produced the measured accept-length 1.4 vs the correct ~2.7):
//!   1. the candidate TOKEN is `d2t[i]` (map draft-space → target-space), and
//!   2. the `draft_logprob` is the **draft-space** log-softmax at index `i` (NOT a full-vocab softmax,
//!      NOT the raw logit) — speculative acceptance divides p_target/p_draft, so a mis-scaled p_draft
//!      poisons every acceptance test.
//!
//! This module is the correct, tested mapping; the native head plugs its raw draft logits in here to
//! produce `route_aware::Candidate` tokens + logprobs. Pure CPU, no model.

use crate::spec::types::TokenId;

/// Maps an EAGLE3 head's draft vocabulary to/from the target vocabulary.
#[derive(Debug, Clone)]
pub struct DraftVocabMap {
    /// draft index -> target token id. `len == draft_vocab_size`.
    d2t: Vec<TokenId>,
}

impl DraftVocabMap {
    /// `d2t[i]` = the target token id for draft-vocab index `i`. Must be the head's published map.
    pub fn new(d2t: Vec<TokenId>) -> Self {
        Self { d2t }
    }

    pub fn draft_vocab_size(&self) -> usize {
        self.d2t.len()
    }

    /// Top-`m` candidates `(target_token, draft_logprob)` from the head's draft-vocab logits.
    ///
    /// `draft_logits.len()` must equal `draft_vocab_size`. Returns up to `m` candidates sorted by
    /// draft probability (descending). `draft_logprob` is the draft-space log-softmax at that index —
    /// exactly the value speculative acceptance needs as `log p_draft(token)`.
    pub fn top_candidates(&self, draft_logits: &[f32], m: usize) -> Vec<(TokenId, f32)> {
        assert_eq!(
            draft_logits.len(),
            self.d2t.len(),
            "draft logits ({}) must be over the draft vocab ({})",
            draft_logits.len(),
            self.d2t.len()
        );
        if draft_logits.is_empty() || m == 0 {
            return Vec::new();
        }
        // draft-space log-softmax denominator (stable logsumexp)
        let max = draft_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = max
            + draft_logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();

        // indices sorted by logit descending; take top-m
        let mut idx: Vec<usize> = (0..draft_logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| {
            draft_logits[b].partial_cmp(&draft_logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.into_iter()
            .take(m)
            .map(|i| (self.d2t[i], draft_logits[i] - lse)) // log p_draft = logit - logsumexp
            .collect()
    }

    /// Map a target token back to its draft index (t2d), if it's in the draft vocab. O(n) scan; for
    /// the native path this would be a prebuilt reverse table — kept simple/correct here.
    pub fn to_draft_index(&self, token: TokenId) -> Option<usize> {
        self.d2t.iter().position(|&t| t == token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> DraftVocabMap {
        // draft vocab of 4 → target tokens [500, 501, 502, 503]
        DraftVocabMap::new(vec![500, 501, 502, 503])
    }

    #[test]
    fn maps_top_candidate_to_target_token_space() {
        // logits favor draft index 2 → target token 502
        let logits = vec![0.0, 1.0, 5.0, 0.5];
        let cands = map().top_candidates(&logits, 2);
        assert_eq!(cands[0].0, 502, "top candidate token is d2t[2]");
        assert_eq!(cands[1].0, 501, "second is d2t[1]");
    }

    #[test]
    fn logprob_is_draft_space_log_softmax() {
        // uniform logits → each draft prob = 1/4 → logprob = ln(0.25)
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let cands = map().top_candidates(&logits, 4);
        for (_tok, lp) in &cands {
            assert!((lp - (0.25f32).ln()).abs() < 1e-5, "logprob {lp} should be ln(1/4)");
        }
        // log-probs must sum (over the full draft vocab) to ~ -entropy, and each <= 0
        assert!(cands.iter().all(|(_, lp)| *lp <= 1e-6));
        // exp(logprobs) over the whole vocab sums to 1
        let s: f32 = cands.iter().map(|(_, lp)| lp.exp()).sum();
        assert!((s - 1.0).abs() < 1e-5, "draft probs sum to 1 ({s})");
    }

    #[test]
    fn t2d_round_trips_and_rejects_out_of_vocab() {
        let m = map();
        assert_eq!(m.to_draft_index(502), Some(2));
        assert_eq!(m.d2t[m.to_draft_index(502).unwrap()], 502);
        assert_eq!(m.to_draft_index(9999), None); // not in the draft vocab
    }

    #[test]
    fn m_and_empty_edge_cases() {
        let m = map();
        assert!(m.top_candidates(&[1.0; 4], 0).is_empty());
        assert_eq!(m.top_candidates(&[1.0, 2.0, 3.0, 4.0], 10).len(), 4); // capped at vocab
    }
}
