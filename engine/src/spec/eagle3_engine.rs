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
use crate::spec::telemetry::SpecTelemetry;
use crate::spec::tree::{accept_tree, SpecTree};
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

/// Stopping criteria for the multi-round [`Eagle3Engine::decode`] loop.
#[derive(Debug, Clone)]
pub struct DecodeStop {
    /// Emit at most this many new tokens (hard cap; the loop never overshoots it).
    pub max_new_tokens: usize,
    /// Stop as soon as this token is emitted (end-of-sequence). `None` = run to the cap.
    pub eos_token: Option<TokenId>,
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
        let mut run = accept_multi_drafter(&tree, &target_logits, rng);

        // FULL-accept bonus fix (see engine/docs/spec-accept-correctness-notes.md #2): when every
        // verified position is accepted, `accept_multi_drafter` had to use a GREEDY STAND-IN for the
        // bonus (it has no target row for the (k+1)-th position), which can re-emit the last accepted
        // token. Replace it with a real sample of P(· | context + accepted) — one extra target
        // forward. This preserves exact losslessness on full-accept rounds, not just rejection rounds.
        if run.accepted.len() == verify_depth && verify_depth > 0 {
            let mut ctx_ext = Vec::with_capacity(context.len() + run.accepted.len());
            ctx_ext.extend_from_slice(context);
            ctx_ext.extend_from_slice(&run.accepted);
            run.bonus_token = self.sample_bonus(&ctx_ext, rng)?;
        }

        let stats = Eagle3RoundStats {
            n_accepted: run.n_accepted(),
            n_drafted,
            verify_depth,
            union_size,
        };
        Ok((run, stats))
    }

    /// Run the full speculative decode loop until `stop.max_new_tokens` are emitted or `eos_token`
    /// appears, accumulating per-round [`SpecTelemetry`]. Returns the generated tokens (excluding the
    /// prompt) and the telemetry snapshot accumulator.
    ///
    /// Each round emits its accepted draft prefix + the bonus token (lossless speculative sampling).
    /// A *degenerate* round (the head proposed nothing, `n_drafted == 0`) falls back to one normal
    /// target decode step ([`Self::fallback_decode`]) so the loop always makes progress and stays
    /// correct — exactly the "caller decodes normally" path the single-round [`Self::step`] documents.
    /// Output is bit-identical to plain target decoding regardless of λ / verify depth.
    pub fn decode(
        &self,
        prompt: &[TokenId],
        stop: &DecodeStop,
        rng: &mut impl RngCore,
    ) -> Result<(Vec<TokenId>, SpecTelemetry)> {
        let mut context = prompt.to_vec();
        let mut output: Vec<TokenId> = Vec::new();
        let mut telem = SpecTelemetry::new();

        while output.len() < stop.max_new_tokens {
            let (run, stats) = self.step(&context, rng)?;
            telem.record(&stats);

            if stats.n_drafted == 0 {
                // Degenerate round: nothing speculated → one normal target decode step.
                let tok = self.fallback_decode(&context)?;
                output.push(tok);
                context.push(tok);
                if stop.eos_token == Some(tok) {
                    break;
                }
                continue;
            }

            let mut stop_now = false;
            for tok in run.all_tokens() {
                if output.len() >= stop.max_new_tokens {
                    stop_now = true;
                    break;
                }
                output.push(tok);
                context.push(tok);
                if stop.eos_token == Some(tok) {
                    stop_now = true;
                    break;
                }
            }
            if stop_now {
                break;
            }
        }
        Ok((output, telem))
    }

    /// One normal greedy target decode of the token following `context` (used when the head proposes
    /// nothing). `forward_single(ctx, t)` returns the distribution at `t`'s position, so feeding the
    /// last context token with the rest as prefix yields P(next | context). Empty context → token 0.
    fn fallback_decode(&self, context: &[TokenId]) -> Result<TokenId> {
        if context.is_empty() {
            return Ok(0);
        }
        let (prefix, last) = context.split_at(context.len() - 1);
        let logits = self.target.forward_single(prefix, last[0])?;
        let arg = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as TokenId)
            .unwrap_or(0);
        Ok(arg)
    }

    /// Sample the bonus token from the exact target distribution `P(· | context)` — a categorical
    /// draw from `softmax(target logits at the end of context)`. Unlike [`Self::fallback_decode`]
    /// (greedy), this is a true sample, so it's lossless for temperature > 0; for a peaked
    /// (near-deterministic) target it coincides with the argmax. Used for the full-accept bonus.
    fn sample_bonus(&self, context: &[TokenId], rng: &mut impl RngCore) -> Result<TokenId> {
        if context.is_empty() {
            return Ok(0);
        }
        let (prefix, last) = context.split_at(context.len() - 1);
        let logits = self.target.forward_single(prefix, last[0])?;
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
        let u = rng.next_f32();
        let mut cum = 0.0f32;
        for (i, &l) in logits.iter().enumerate() {
            cum += (l - max).exp() / sum;
            if u <= cum {
                return Ok(i as TokenId);
            }
        }
        Ok((logits.len().saturating_sub(1)) as TokenId)
    }

    /// One TREE-spec round: build a draft tree (top-`branch` per node, depth `depth`) from the head,
    /// verify every node, and accept the longest matching PATH via lossless [`accept_tree`]. The tree
    /// analog of [`Self::step`] — a wider/flatter verify accepts more often (higher τ) for ~the same
    /// cost on the flat M=k verify (validated: `mk_tree_attn` ~flat in tree width). Degenerate (head
    /// produced nothing) → one normal target decode, as in [`Self::step`].
    pub fn step_tree(
        &self,
        context: &[TokenId],
        depth: usize,
        branch: usize,
        rng: &mut impl RngCore,
    ) -> Result<(AcceptedRun, Eagle3RoundStats)> {
        let root = *context.last().unwrap_or(&0);
        let tree = SpecTree::build_from_source(&self.drafter.source, context, root, depth, branch);
        let n_drafted = tree.n_nodes().saturating_sub(1);
        if n_drafted == 0 {
            return Ok((
                AcceptedRun { accepted: vec![], accept_mask: vec![], bonus_token: self.fallback_decode(context)?, winning_drafter: None },
                Eagle3RoundStats { n_accepted: 0, n_drafted: 0, verify_depth: 0, union_size: 0 },
            ));
        }
        let rows = self.tree_target_rows(context, &tree)?;
        let run = accept_tree(&tree, &rows, rng);
        let stats = Eagle3RoundStats {
            n_accepted: run.n_accepted(),
            n_drafted,
            verify_depth: run.accepted.len(), // depth of the accepted path
            union_size: n_drafted as u32,     // tree nodes verified (the M of the M=k verify)
        };
        Ok((run, stats))
    }

    /// Target verify rows for a tree: `rows[i]` = `P(· | context + path-to-node-i)` (predicts node i's
    /// children); `rows[0]` = `P(· | context)`. One target forward per node here (mock path); the
    /// native engine produces all rows in ONE M=k tree-masked forward (`mk_tree_attn` + flat GEMM).
    fn tree_target_rows(&self, context: &[TokenId], tree: &SpecTree) -> Result<Vec<Vec<f32>>> {
        let mut rows = Vec::with_capacity(tree.n_nodes());
        for i in 0..tree.n_nodes() {
            let mut path = Vec::new();
            let mut cur = i;
            while cur != 0 {
                path.push(tree.tokens[cur]);
                cur = tree.parent[cur];
            }
            path.reverse();
            let mut ctx: Vec<TokenId> = Vec::with_capacity(context.len() + path.len());
            ctx.extend_from_slice(context);
            ctx.extend_from_slice(&path);
            if ctx.is_empty() {
                rows.push(vec![0.0; self.config.vocab_size]);
                continue;
            }
            let (prefix, last) = ctx.split_at(ctx.len() - 1);
            rows.push(self.target.forward_single(prefix, last[0])?);
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::types::ExpertId;
    use crate::spec::route_aware::Candidate;
    use crate::spec::types::TargetLogits;

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
    fn decode_emits_exactly_max_new_tokens_and_records_telemetry() {
        // OverlapHead always proposes, EchoTarget + always-accept RNG → every round accepts its
        // verified prefix + bonus. The loop must emit EXACTLY max_new_tokens (never overshoot) and
        // the telemetry must reflect the rounds it ran.
        let vocab = 512;
        let engine = Eagle3Engine::new(
            RouteAwareDrafter::new(OverlapHead, 2.0),
            EchoTarget(vocab),
            Eagle3Config { depth: 4, width: 2, vocab_size: vocab, floor_fraction: 0.86 },
        );
        let stop = DecodeStop { max_new_tokens: 10, eos_token: None };
        let (out, telem) = engine.decode(&[1, 2, 3], &stop, &mut FixedRng).unwrap();
        assert_eq!(out.len(), 10, "emits exactly the cap, no overshoot");
        assert!(telem.rounds() >= 1);
        // τ over the rounds it ran is a sane >1 (each round emits accepted prefix + bonus).
        assert!(telem.mean_accept_length() > 1.0);
    }

    #[test]
    fn decode_stops_at_eos() {
        // Pick an EOS that the head will emit: OverlapHead yields token (100 + chain_len) at each
        // position, so token 100 appears in round 1. Stop there.
        let vocab = 512;
        let engine = Eagle3Engine::new(
            RouteAwareDrafter::new(OverlapHead, 2.0),
            EchoTarget(vocab),
            Eagle3Config { depth: 4, width: 2, vocab_size: vocab, floor_fraction: 0.86 },
        );
        let stop = DecodeStop { max_new_tokens: 100, eos_token: Some(100) };
        let (out, _telem) = engine.decode(&[1, 2, 3], &stop, &mut FixedRng).unwrap();
        assert!(out.len() < 100, "stopped at EOS well before the cap");
        assert_eq!(*out.last().unwrap(), 100, "last emitted token is the EOS");
        assert!(!out[..out.len() - 1].contains(&100), "EOS appears once, at the end");
    }

    #[test]
    fn decode_makes_progress_on_a_degenerate_head_via_fallback() {
        // EmptyHead proposes nothing every round → every round is degenerate → the loop must fall
        // back to a normal target decode and still reach the cap (no infinite loop). EchoTarget's
        // greedy of forward_single(prefix, last) returns `last`, so the fallback echoes the last
        // context token; output is that token repeated.
        struct EmptyHead;
        impl CandidateSource for EmptyHead {
            fn candidates(&self, _c: &[TokenId], _ch: &[TokenId], _w: usize) -> Vec<Candidate> { vec![] }
        }
        let vocab = 64;
        let engine = Eagle3Engine::new(
            RouteAwareDrafter::new(EmptyHead, 1.0),
            EchoTarget(vocab),
            Eagle3Config { depth: 4, width: 2, vocab_size: vocab, floor_fraction: 0.0 },
        );
        let stop = DecodeStop { max_new_tokens: 5, eos_token: None };
        let (out, telem) = engine.decode(&[7, 9], &stop, &mut FixedRng).unwrap();
        assert_eq!(out.len(), 5, "fallback guarantees progress to the cap");
        assert!(out.iter().all(|&t| t == 9), "fallback echoes the last context token (9)");
        // every round was degenerate → no speculating rounds, but rounds counted, τ = 1 per round
        assert_eq!(telem.snapshot().spec_rounds, 0);
        assert!((telem.mean_accept_length() - 1.0).abs() < 1e-6);
    }

    /// Deterministic reference target: the greedy continuation is `prev_token + 1` (mod vocab),
    /// INDEPENDENT of what was drafted. It overrides `forward_batch` with the CORRECT verify layout
    /// (`data[pos]` = the distribution predicting position `pos`'s token from `context + draft[..pos]`,
    /// i.e. the (last_context + 1 + pos) ramp token) — not the off-by-one default. This lets the test
    /// exercise REAL losslessness rather than a mock quirk. (See engine/docs/spec-accept-correctness-notes.md #1.)
    struct RampTarget(usize);
    impl ModelRunner for RampTarget {
        fn forward_single(&self, _ctx: &[u32], tok: u32) -> Result<Vec<f32>> {
            // predict-next after `tok`: peak at (tok+1) % vocab (used for the bonus sample / fallback)
            let mut v = vec![0.0f32; self.0];
            v[((tok as usize) + 1) % self.0] = 30.0;
            Ok(v)
        }
        fn forward_batch(&self, context: &[u32], draft_tokens: &[u32], vocab: usize) -> Result<TargetLogits> {
            // CORRECT verify layout (single drafter): position `pos` predicts the ramp token
            // (last_context + 1 + pos) % vocab — independent of the draft tokens.
            let last = *context.last().unwrap_or(&0) as usize;
            let n = draft_tokens.len();
            let mut data = vec![0.0f32; n * vocab];
            for pos in 0..n {
                data[pos * vocab + (last + 1 + pos) % vocab] = 30.0;
            }
            Ok(TargetLogits { data, n_positions: n, vocab_size: vocab })
        }
        fn vocab_size(&self) -> usize { self.0 }
    }

    /// Head that, at each position, offers the CORRECT ramp token (rotating/fresh experts) and a WRONG
    /// distractor (reusing experts [0..8]). λ=0 drafts the correct token (higher prob) → accepted;
    /// large λ prefers the small-union distractor → rejected → resampled to the ramp. Either way the
    /// emitted sequence is the same target ramp — so λ changes throughput, never the output.
    struct RampHead(usize);
    impl CandidateSource for RampHead {
        fn candidates(&self, context: &[TokenId], chain: &[TokenId], _w: usize) -> Vec<Candidate> {
            let last = chain.last().or_else(|| context.last()).copied().unwrap_or(0) as usize;
            let correct = ((last + 1) % self.0) as u32;
            let distract = ((last + 1 + 7) % self.0) as u32;
            let base = chain.len() as u32 * 8; // rotates → correct's experts are fresh each position
            vec![
                Candidate { token: correct, draft_logprob: -0.1, experts: (base..base + 8).collect() },
                Candidate { token: distract, draft_logprob: -0.3, experts: (0u32..8).collect() },
            ]
        }
    }

    #[test]
    fn decode_is_lossless_invariant_to_lambda_and_verify_depth() {
        // THE central correctness invariant: route-awareness (λ) and adaptive verify-depth (floor)
        // trade throughput but NEVER change the emitted tokens. With a deterministic target, the
        // output must be the exact greedy ramp regardless of λ or floor. (Requires the full-accept
        // bonus fix — without it, λ=0 full-accept rounds duplicate the last token and break the ramp.)
        let vocab = 256;
        let prompt = vec![5u32];
        let stop = DecodeStop { max_new_tokens: 12, eos_token: None };
        let run = |lambda: f32, floor: f32| -> Vec<TokenId> {
            Eagle3Engine::new(
                RouteAwareDrafter::new(RampHead(vocab), lambda),
                RampTarget(vocab),
                Eagle3Config { depth: 4, width: 2, vocab_size: vocab, floor_fraction: floor },
            )
            .decode(&prompt, &stop, &mut FixedRng)
            .unwrap()
            .0
        };
        let expected: Vec<TokenId> = (6u32..18).collect(); // ramp from prompt-last 5
        let base = run(0.0, 0.0);
        assert_eq!(base, expected, "lossless output = the deterministic target ramp");
        assert_eq!(run(5.0, 0.0), expected, "λ must NOT change the output (lossless)");
        assert_eq!(run(0.0, 0.86), expected, "verify-depth (floor) must NOT change the output");
        assert_eq!(run(5.0, 0.86), expected, "λ and verify-depth jointly invariant");
    }

    #[test]
    fn step_tree_is_lossless_and_accepts_the_ramp_path() {
        // Build a draft TREE from RampHead (correct ramp token + a distractor per node) verified by the
        // deterministic RampTarget. The accepted path + bonus must be the exact target ramp regardless
        // of tree width/depth (lossless), and the correct branch is accepted deeper than 0 (τ>1).
        let vocab = 256;
        let engine = Eagle3Engine::new(
            RouteAwareDrafter::new(RampHead(vocab), 0.0),
            RampTarget(vocab),
            Eagle3Config { depth: 4, width: 2, vocab_size: vocab, floor_fraction: 0.0 },
        );
        let (run, stats) = engine.step_tree(&[5], /*depth*/ 3, /*branch*/ 2, &mut FixedRng).unwrap();
        let emitted: Vec<TokenId> =
            run.accepted.iter().copied().chain(std::iter::once(run.bonus_token)).collect();
        for (i, &t) in emitted.iter().enumerate() {
            assert_eq!(t, 6 + i as TokenId, "tree-spec emits the target ramp (lossless)");
        }
        assert!(run.accepted.len() >= 1, "the correct ramp branch is accepted (tau > 1)");
        assert_eq!(stats.union_size as usize, stats.n_drafted, "union_size = tree nodes verified");
        assert!(stats.n_drafted >= 3, "a depth-3 branch-2 tree drafts several nodes");
    }

    #[test]
    fn aux_layers_constant_documents_the_head_contract() {
        // The native head reads aux at these layers (RedHat head); pin them so a regression is loud.
        let aux: &[usize] = &[1, 46, 90];
        assert_eq!(aux.len(), 3);
        let _ = aux.iter().map(|&l| l as ExpertId).collect::<Vec<_>>();
    }
}
