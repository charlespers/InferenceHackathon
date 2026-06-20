//! tree_shape.rs — optimal tree shape (branch × depth) for spec drafting on a FLAT verify.
//!
//! The measured payoff (results/mk_tree_attn/K2_FLATNESS_AB.md): the M=k verify attention on tensor
//! cores is FLAT in node count M (≈1.0–1.1× M=1 up to a flatness ceiling), where the warp-shuffle K2
//! scaled ~4×. With a flat verify, *adding tree width is ~free on the verify step* — so the lever is to
//! pick the (branch, depth) that maximizes emitted-tokens / round-cost within the node budget.
//!
//! Model (one spec round = draft a tree, verify it once, accept the longest verified path):
//!   - per-position single-candidate accept prob `p` = [`MeasuredAccept::persistence`] (b=1 chain prob),
//!     first level uses `first_pos`. A node with `branch` candidate children advances iff ≥1 child is
//!     accepted: `level_advance_prob(p, b) = 1-(1-p)^b` (OPTIMISTIC independence approx of EAGLE3's
//!     lossless residual sibling acceptance — documented; b=1 reduces EXACTLY to the chain).
//!   - emitted = 1 + Σ_levels Π advance — trees accept more often per level → higher τ.
//!   - round_cost = depth·head_level_cost (the tree is built level-by-level, one batched head forward per
//!     level) + verify_flat_cost (CONSTANT in node count while nodes ≤ node_budget — the flat-verify win).
//!   - speedup = emitted / round_cost.
//! The optimizer searches (b,d) with `tree_nodes(b,d) ≤ node_budget` and returns the best shape.

use crate::spec::projection::MeasuredAccept;

/// P(≥1 of `branch` candidates accepted), each accepted with per-position prob `p`. b=1 → p.
/// Optimistic (independence) approximation of EAGLE3 residual sibling acceptance.
pub fn level_advance_prob(p: f32, branch: usize) -> f32 {
    let p = p.clamp(0.0, 1.0);
    if branch == 0 {
        return 0.0;
    }
    1.0 - (1.0 - p).powi(branch as i32)
}

/// Expected accepted depth (levels advanced) for a `branch`×`depth` tree. emitted = 1 + this.
/// Level 0 advances from `first_pos`, deeper levels from persistence `p`. b=1 reduces to the chain
/// `expected_accepted([first_pos, p, p, …])`.
pub fn tree_expected_accepted(first_pos: f32, p: f32, branch: usize, depth: usize) -> f32 {
    let mut total = 0.0;
    let mut prefix = 1.0;
    for i in 0..depth {
        let base = if i == 0 { first_pos } else { p };
        prefix *= level_advance_prob(base, branch);
        total += prefix;
    }
    total
}

/// Verified nodes in a full `branch`-ary tree of `depth` levels (excludes the committed root token):
/// branch + branch² + … + branch^depth. Saturates at `u64::MAX` to avoid overflow on huge shapes.
pub fn tree_nodes(branch: usize, depth: usize) -> u64 {
    let mut total: u64 = 0;
    let mut level: u64 = 1;
    for _ in 0..depth {
        level = level.saturating_mul(branch as u64);
        total = total.saturating_add(level);
    }
    total
}

/// Cost inputs (decode-step units: a plain decode forward = 1.0).
#[derive(Clone, Copy, Debug)]
pub struct TreeCostModel {
    /// FLAT verify cost — constant in node count while nodes ≤ `node_budget` (the flat-verify win).
    pub verify_flat_cost: f32,
    /// Draft cost per tree level (one batched EAGLE3 head forward per level).
    pub head_level_cost: f32,
    /// Flatness ceiling: max verified nodes before the verify stops being flat (measured/conservative).
    pub node_budget: u64,
}

/// A scored tree shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeShape {
    pub branch: usize,
    pub depth: usize,
    pub emitted: f32,
    pub nodes: u64,
    pub speedup: f32,
}

impl TreeCostModel {
    /// Score one (branch, depth) shape against the measured accept profile.
    pub fn evaluate(&self, m: &MeasuredAccept, branch: usize, depth: usize) -> TreeShape {
        let p = m.persistence(depth);
        let emitted = 1.0 + tree_expected_accepted(m.first_pos, p, branch, depth);
        let nodes = tree_nodes(branch, depth);
        let draft = depth as f32 * self.head_level_cost;
        let round_cost = draft + self.verify_flat_cost;
        let speedup = if round_cost > 0.0 { emitted / round_cost } else { 0.0 };
        TreeShape { branch, depth, emitted, nodes, speedup }
    }

    /// Search (branch ∈ 1..=max_branch, depth ∈ 1..=max_depth) for the speedup-maximizing shape whose
    /// node count fits `node_budget`. Returns the best shape (chain b=1,d=1 always fits as a fallback).
    pub fn optimal(&self, m: &MeasuredAccept, max_branch: usize, max_depth: usize) -> TreeShape {
        let mut best = self.evaluate(m, 1, 1);
        for branch in 1..=max_branch.max(1) {
            for depth in 1..=max_depth.max(1) {
                let nodes = tree_nodes(branch, depth);
                if nodes > self.node_budget {
                    continue;
                }
                let s = self.evaluate(m, branch, depth);
                if s.speedup > best.speedup {
                    best = s;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::adaptive_verify::{emitted, expected_accepted};

    fn measured() -> MeasuredAccept {
        MeasuredAccept { mean_accept_len: 2.7, first_pos: 0.75 }
    }

    #[test]
    fn branch_one_reduces_to_chain() {
        // b=1 tree must reproduce the chain expected_accepted([first_pos, p, p, …]) exactly.
        let m = measured();
        let depth = 5;
        let p = m.persistence(depth);
        let tree_ea = tree_expected_accepted(m.first_pos, p, 1, depth);
        let chain_probs: Vec<f32> = (0..depth)
            .map(|i| if i == 0 { m.first_pos } else { p })
            .collect();
        let chain_ea = expected_accepted(&chain_probs);
        assert!((tree_ea - chain_ea).abs() < 1e-5, "b=1 tree {tree_ea} == chain {chain_ea}");
    }

    #[test]
    fn wider_tree_accepts_more() {
        // A branchier tree advances more often per level → strictly more emitted than the chain.
        let m = measured();
        let chain = 1.0 + tree_expected_accepted(m.first_pos, m.persistence(4), 1, 4);
        let tree = 1.0 + tree_expected_accepted(m.first_pos, m.persistence(4), 4, 4);
        assert!(tree > chain, "tree emitted {tree} > chain emitted {chain}");
        // sanity: chain emitted ≈ the calibrated τ at this depth.
        let cal = emitted(&measured().accept_probs(4));
        assert!((chain - cal).abs() < 1e-4, "chain {chain} == calibrated {cal}");
    }

    #[test]
    fn advance_prob_monotonic_in_branch() {
        let p = 0.5;
        assert!((level_advance_prob(p, 1) - p).abs() < 1e-6);
        assert!(level_advance_prob(p, 2) > level_advance_prob(p, 1));
        assert!(level_advance_prob(p, 8) > level_advance_prob(p, 2));
        assert!(level_advance_prob(p, 100) < 1.0 + 1e-6);
    }

    #[test]
    fn tree_nodes_geometric_and_budget_respected() {
        assert_eq!(tree_nodes(2, 3), 2 + 4 + 8); // 14
        assert_eq!(tree_nodes(1, 5), 5); // chain
        let m = measured();
        let model = TreeCostModel { verify_flat_cost: 1.0, head_level_cost: 0.1, node_budget: 16 };
        let best = model.optimal(&m, 8, 8);
        assert!(best.nodes <= 16, "optimal nodes {} within budget", best.nodes);
        assert!(best.speedup >= model.evaluate(&m, 1, 1).speedup, "optimal ≥ chain fallback");
    }

    #[test]
    fn flat_verify_favors_width_cheap_draft_favors_depth() {
        let m = measured();
        // Flat, cheap verify + non-trivial draft per level: width (more emitted per level, ~free verify)
        // should be chosen over a deep narrow chain at equal node budget.
        let model = TreeCostModel { verify_flat_cost: 1.0, head_level_cost: 0.25, node_budget: 24 };
        let best = model.optimal(&m, 6, 6);
        assert!(best.branch >= 2, "flat verify + per-level draft cost favors width, got b={}", best.branch);
        // The optimal tree must beat the best CHAIN (b=1) under the same budget/costs.
        let mut best_chain = model.evaluate(&m, 1, 1);
        for d in 1..=6 {
            let s = model.evaluate(&m, 1, d);
            if s.speedup > best_chain.speedup { best_chain = s; }
        }
        assert!(best.speedup > best_chain.speedup,
            "tree speedup {} > best chain {}", best.speedup, best_chain.speedup);
    }
}
