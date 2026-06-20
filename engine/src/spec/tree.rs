//! Tree drafting + LOSSLESS tree acceptance — the spec lever that actually pays at B=1.
//!
//! Single-level chain spec (`accept.rs`) drafts one path of `k` tokens. A *tree* drafts MANY candidate
//! continuations at once (each node branches), and the target verifies the WHOLE tree in one batched
//! forward — and because the M=k tensor-core GEMM verify is FLAT in the number of nodes up to the 16-wide
//! fp8 tile (Charles, `spec_verify_forward_gemm.cu` T16/T1≈1.0), a *wider* tree costs ~the same as a chain
//! but accepts more often → higher τ → higher speedup. So trees are the way to push τ on the flat verify;
//! a second drafting level ("decoder for the decoder") is NOT (the drafter is already a 1-layer head).
//!
//! Acceptance here is the exact SpecInfer/EAGLE-style **residual multi-candidate** rule, applied per node:
//! try the node's children in draft-prob order; accept the first that passes `u < min(1, p_resid(x)/q)`;
//! on each rejection subtract that child's draft mass from the residual and renormalize; if all children
//! reject, the bonus is a sample from the final residual. This is LOSSLESS for ANY tree shape — the
//! emitted path + bonus is an exact sample from the target distribution (validated below: with a
//! deterministic target the output is the target's greedy ramp regardless of tree topology).
//!
//! Contract with the verify forward (`engine/docs/spec-accept-correctness-notes.md`): `target_rows[i]` =
//! `P(· | context + path-to-node-i)` — the distribution PREDICTING node i's children (the token after
//! node i). Row 0 is the root (= last confirmed token) and predicts the first-level draft tokens. The
//! native flat M=k tree verify (with a tree-attention mask so each node attends only to its ancestors)
//! produces exactly these rows; pure CPU + mockable here.

use crate::spec::types::{AcceptedRun, RngCore, TokenId};

/// A speculative draft TREE. Node 0 is the root = the last confirmed context token (a sentinel that is
/// never emitted). Every other node carries a drafted token + its draft logprob (given the path to its
/// parent). `children[i]` are the indices whose `parent == i`.
#[derive(Debug, Clone)]
pub struct SpecTree {
    pub tokens: Vec<TokenId>,     // [n_nodes]; tokens[0] = root sentinel (last ctx token)
    pub parent: Vec<usize>,       // [n_nodes]; parent[0] = 0
    pub draft_logprob: Vec<f32>,  // [n_nodes]; draft_logprob[0] unused
    pub children: Vec<Vec<usize>>,// [n_nodes] adjacency (derived)
}

impl SpecTree {
    /// New tree rooted at `root_token` (the last confirmed context token).
    pub fn new(root_token: TokenId) -> Self {
        Self {
            tokens: vec![root_token],
            parent: vec![0],
            draft_logprob: vec![0.0],
            children: vec![Vec::new()],
        }
    }

    /// Add a child `token` (with draft logprob `lp`) under `parent`; returns the new node index.
    pub fn add_child(&mut self, parent: usize, token: TokenId, lp: f32) -> usize {
        let idx = self.tokens.len();
        self.tokens.push(token);
        self.parent.push(parent);
        self.draft_logprob.push(lp);
        self.children.push(Vec::new());
        self.children[parent].push(idx);
        idx
    }

    pub fn n_nodes(&self) -> usize {
        self.tokens.len()
    }

    /// Build a degenerate single-path tree (a chain) — for chain/tree equivalence.
    pub fn chain(root_token: TokenId, tokens: &[TokenId], logprobs: &[f32]) -> Self {
        let mut t = SpecTree::new(root_token);
        let mut cur = 0;
        for (tok, lp) in tokens.iter().zip(logprobs.iter()) {
            cur = t.add_child(cur, *tok, *lp);
        }
        t
    }

    /// Build a draft TREE from a [`CandidateSource`] (the EAGLE3 head): from the root (last context
    /// token), expand the top-`branch` candidates at each node down to `depth` levels (BFS). Each edge
    /// carries the head's candidate token + its draft logprob; `draft_logprob` must be the draft-space
    /// log-softmax (the d2t/τ-1.4 fix — `DraftVocabMap` already produces it). The result feeds the M=k
    /// tree verify + [`accept_tree`]. This is the tree analog of `RouteAwareDrafter::draft_chain`.
    ///
    /// Total nodes ≈ 1 + branch + branch² + … + branch^depth; cap `branch`/`depth` to stay ≤ the flat
    /// verify tile (≤16-wide, Charles). Stops a path early if the source returns no candidates.
    pub fn build_from_source<S: crate::spec::route_aware_drafter::CandidateSource>(
        source: &S,
        context: &[TokenId],
        root_token: TokenId,
        depth: usize,
        branch: usize,
    ) -> SpecTree {
        let mut t = SpecTree::new(root_token);
        let mut frontier: Vec<(usize, Vec<TokenId>)> = vec![(0, Vec::new())];
        for _ in 0..depth {
            let mut next: Vec<(usize, Vec<TokenId>)> = Vec::new();
            for (node, path) in frontier {
                let cands = source.candidates(context, &path, branch);
                for c in cands.into_iter().take(branch) {
                    let child = t.add_child(node, c.token, c.draft_logprob);
                    let mut p = path.clone();
                    p.push(c.token);
                    next.push((child, p));
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        t
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Sample a token index from a probability vector using `rng`.
fn sample_probs(p: &[f32], rng: &mut impl RngCore) -> TokenId {
    let total: f32 = p.iter().sum();
    let u = rng.next_f32() * if total > 0.0 { total } else { 1.0 };
    let mut cum = 0.0;
    for (i, &w) in p.iter().enumerate() {
        cum += w;
        if u <= cum {
            return i as TokenId;
        }
    }
    (p.len().saturating_sub(1)) as TokenId
}

/// Lossless tree acceptance. `target_rows[i]` = logits for `P(· | context + path-to-node-i)` (predicts
/// node i's children). Returns the accepted path tokens (longest verified prefix) + one bonus token.
pub fn accept_tree(
    tree: &SpecTree,
    target_rows: &[Vec<f32>],
    rng: &mut impl RngCore,
) -> AcceptedRun {
    let mut accepted: Vec<TokenId> = Vec::new();
    let mut accept_mask: Vec<bool> = Vec::new();
    let mut node = 0usize; // root

    loop {
        let kids = &tree.children[node];
        // Leaf (or root with no drafts): the whole path was accepted → bonus is a pure target sample.
        if kids.is_empty() {
            let bonus = sample_probs(&softmax(&target_rows[node]), rng);
            return AcceptedRun { accepted, bonus_token: bonus, winning_drafter: None, accept_mask };
        }

        // Children sorted by draft prob (desc) — try the most-likely draft first.
        let mut order = kids.clone();
        order.sort_by(|&a, &b| {
            tree.draft_logprob[b]
                .partial_cmp(&tree.draft_logprob[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Residual distribution from this node's verify row.
        let mut p = softmax(&target_rows[node]);
        let mut descended: Option<usize> = None;

        for &c in &order {
            let x = tree.tokens[c] as usize;
            let q = tree.draft_logprob[c].exp(); // draft prob of this child token
            let px = if x < p.len() { p[x] } else { 0.0 };
            let a = if q > 0.0 { (px / q).min(1.0) } else { 1.0 };
            if rng.next_f32() < a {
                descended = Some(c);
                break;
            }
            // Reject: remove this child's draft mass from the residual and renormalize.
            if x < p.len() {
                p[x] = (p[x] - q).max(0.0);
            }
            let s: f32 = p.iter().sum();
            if s > 1e-12 {
                for v in p.iter_mut() {
                    *v /= s;
                }
            }
        }

        match descended {
            Some(c) => {
                accepted.push(tree.tokens[c]);
                accept_mask.push(true);
                node = c; // descend into the accepted child
            }
            None => {
                // All children rejected → bonus from the final residual (exact target sample).
                accept_mask.push(false);
                let bonus = sample_probs(&p, rng);
                return AcceptedRun { accepted, bonus_token: bonus, winning_drafter: None, accept_mask };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedRng(f32);
    impl RngCore for FixedRng {
        fn next_f32(&mut self) -> f32 {
            self.0
        }
    }

    /// One-hot-ish logit row peaked at `tok`.
    fn peak(vocab: usize, tok: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab];
        v[tok] = 30.0;
        v
    }

    /// Deterministic "ramp" target: P(·|context+path-to-i) peaks at tokens[i]+1 (greedy next = +1),
    /// INDEPENDENT of tree shape. One row per node.
    fn ramp_rows(tree: &SpecTree, vocab: usize) -> Vec<Vec<f32>> {
        (0..tree.n_nodes())
            .map(|i| peak(vocab, (tree.tokens[i] as usize + 1) % vocab))
            .collect()
    }

    #[test]
    fn leaf_only_root_emits_bonus_from_target() {
        // Tree with no drafts (root only) → emit the target's next token (here ramp: 5 → 6).
        let t = SpecTree::new(5);
        let rows = ramp_rows(&t, 64);
        let run = accept_tree(&t, &rows, &mut FixedRng(0.01));
        assert!(run.accepted.is_empty());
        assert_eq!(run.bonus_token, 6, "bonus = target greedy next (5→6)");
    }

    #[test]
    fn chain_tree_accepts_matching_ramp_path() {
        // Chain 6→7→8 under root 5; ramp target accepts all three, bonus = 9.
        let t = SpecTree::chain(5, &[6, 7, 8], &[-0.1, -0.1, -0.1]);
        let rows = ramp_rows(&t, 64);
        let run = accept_tree(&t, &rows, &mut FixedRng(0.01));
        assert_eq!(run.accepted, vec![6, 7, 8], "accepts the full ramp chain");
        assert_eq!(run.bonus_token, 9, "bonus continues the ramp");
    }

    #[test]
    fn build_from_source_makes_a_branch_depth_tree() {
        use crate::spec::route_aware::Candidate;
        use crate::spec::route_aware_drafter::CandidateSource;
        // Mock head: returns `branch` candidates per node; tokens depend on the path length so the
        // tree is non-degenerate. draft_logprob is a draft-space-style value (<=0).
        struct MockHead;
        impl CandidateSource for MockHead {
            fn candidates(&self, _ctx: &[TokenId], chain: &[TokenId], width: usize) -> Vec<Candidate> {
                let base = 100 + 10 * chain.len() as u32;
                (0..width as u32)
                    .map(|i| Candidate { token: base + i, draft_logprob: -0.1 - 0.1 * i as f32, experts: vec![] })
                    .collect()
            }
        }
        let t = SpecTree::build_from_source(&MockHead, &[1, 2, 3], /*root*/ 5, /*depth*/ 2, /*branch*/ 2);
        // nodes = 1 (root) + 2 (level1) + 4 (level2) = 7
        assert_eq!(t.n_nodes(), 7, "branch=2 depth=2 -> 1+2+4 nodes");
        // root has 2 children; each level-1 node has 2 children
        assert_eq!(t.children[0].len(), 2, "root branches into 2");
        for &c in &t.children[0] {
            assert_eq!(t.children[c].len(), 2, "each level-1 node branches into 2");
        }
        // level-1 tokens are the head's first-position candidates (path len 0 -> base 100)
        let l1: Vec<TokenId> = t.children[0].iter().map(|&c| t.tokens[c]).collect();
        assert_eq!(l1, vec![100, 101], "level-1 tokens = head top-2 at the root");
        // it composes with accept_tree (no panic) and is lossless against a ramp target
        let rows = ramp_rows(&t, 256);
        let run = accept_tree(&t, &rows, &mut FixedRng(0.01));
        for (i, &tok) in run.accepted.iter().chain(std::iter::once(&run.bonus_token)).enumerate() {
            assert_eq!(tok, 6 + i as TokenId, "built tree still emits the target ramp (lossless)");
        }
    }

    #[test]
    fn tree_accepts_the_correct_branch_among_siblings() {
        // Root 5; two children at level 1: a WRONG token (40, higher draft prob) and the RIGHT ramp
        // token (6, lower draft prob). The residual rule must reject 40 and accept 6.
        let mut t = SpecTree::new(5);
        let _wrong = t.add_child(0, 40, -0.1); // higher draft prob, tried first
        let right = t.add_child(0, 6, -0.5); // the ramp token
        let _g = t.add_child(right, 7, -0.2); // extend the correct branch
        let rows = ramp_rows(&t, 64);
        let run = accept_tree(&t, &rows, &mut FixedRng(0.01));
        assert_eq!(run.accepted, vec![6, 7], "tree picks the correct branch (6) over the wrong sibling (40)");
        assert_eq!(run.bonus_token, 8);
    }

    #[test]
    fn tree_lossless_invariant_to_topology() {
        // THE losslessness invariant: for a deterministic target, the emitted tokens (accepted path +
        // bonus) are the target ramp [6,7,8,...] regardless of how the draft TREE is shaped.
        let vocab = 128;
        // shape A: a wide-then-deep tree containing the ramp + many wrong branches
        let mut a = SpecTree::new(5);
        a.add_child(0, 50, -0.1); // wrong
        let a1 = a.add_child(0, 6, -0.2); // ramp
        a.add_child(a1, 60, -0.1); // wrong
        let a2 = a.add_child(a1, 7, -0.3); // ramp
        a.add_child(a2, 8, -0.2); // ramp
        // shape B: a chain that is ALL WRONG (no ramp tokens) → rejects immediately, bonus = ramp
        let b = SpecTree::chain(5, &[90, 91, 92], &[-0.1, -0.1, -0.1]);
        // shape C: just the correct ramp chain
        let c = SpecTree::chain(5, &[6, 7], &[-0.4, -0.4]);

        let run_a = accept_tree(&a, &ramp_rows(&a, vocab), &mut FixedRng(0.01));
        let run_b = accept_tree(&b, &ramp_rows(&b, vocab), &mut FixedRng(0.01));
        let run_c = accept_tree(&c, &ramp_rows(&c, vocab), &mut FixedRng(0.01));

        // Every emitted token must equal the ramp (5+1+i), regardless of topology → lossless.
        let emitted = |r: &AcceptedRun| -> Vec<TokenId> {
            r.accepted.iter().copied().chain(std::iter::once(r.bonus_token)).collect()
        };
        for r in [&run_a, &run_b, &run_c] {
            let e = emitted(r);
            for (i, &tok) in e.iter().enumerate() {
                assert_eq!(tok, 6 + i as TokenId, "emitted token {i} must be ramp {} (got {tok})", 6 + i as u32);
            }
        }
        // And the tree containing more ramp depth accepts MORE (higher τ) than the all-wrong tree.
        assert!(run_a.accepted.len() > run_b.accepted.len(),
            "tree with the ramp branch accepts deeper ({}) than the all-wrong tree ({})",
            run_a.accepted.len(), run_b.accepted.len());
    }

    #[test]
    fn wider_tree_raises_expected_acceptance() {
        // τ proxy: a 1-child chain with a wrong token accepts 0; a 2-child tree where one child is right
        // accepts 1 — the wider tree raises acceptance for the SAME target (the point of trees).
        let vocab = 64;
        let chain = SpecTree::chain(5, &[40], &[-0.1]);            // wrong only
        let mut tree = SpecTree::new(5);
        tree.add_child(0, 40, -0.1);                               // wrong
        tree.add_child(0, 6, -0.5);                                // right (ramp)
        let rc = accept_tree(&chain, &ramp_rows(&chain, vocab), &mut FixedRng(0.01));
        let rt = accept_tree(&tree, &ramp_rows(&tree, vocab), &mut FixedRng(0.01));
        assert_eq!(rc.accepted.len(), 0, "chain with wrong token accepts nothing");
        assert_eq!(rt.accepted.len(), 1, "wider tree catches the correct token");
        // both still lossless: emitted[0] == ramp 6
        assert_eq!(rc.bonus_token, 6);
        assert_eq!(*rt.accepted.first().unwrap(), 6);
    }
}
