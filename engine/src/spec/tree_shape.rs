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

/// MEASURED K2 verify-cost ratio vs M=1 as a function of tree node count (ctx 4096, tensor-core verify,
/// results/mk_tree_attn/tc_attn_probe_result.txt — the flatness CEILING run). Flat to ~M=16, then the TC
/// tiles saturate the SMs and the verify re-acquires M-scaling. Use this instead of a hard `node_budget`
/// cliff so the optimizer trades tree width against the REAL degradation.
pub const CEILING_CTX4096: &[(u64, f32)] = &[
    (1, 1.00), (8, 1.02), (16, 1.09), (24, 1.16), (32, 1.24), (48, 1.42), (64, 1.64),
];
/// ctx 8192 degrades sooner (longer per-warp draft-self + bigger score GEMM).
pub const CEILING_CTX8192: &[(u64, f32)] = &[
    (1, 1.00), (8, 1.05), (16, 1.14), (24, 1.21), (32, 1.38), (48, 1.76), (64, 2.21),
];

/// Piecewise-linear verify-cost ratio at `nodes` from a measured `(nodes, ratio)` curve. Clamps below the
/// first point; linearly extrapolates past the last (so very wide trees keep getting penalized).
pub fn verify_ratio(nodes: u64, curve: &[(u64, f32)]) -> f32 {
    if curve.is_empty() {
        return 1.0;
    }
    if nodes <= curve[0].0 {
        return curve[0].1;
    }
    for w in curve.windows(2) {
        let (n0, r0) = w[0];
        let (n1, r1) = w[1];
        if nodes <= n1 {
            let t = (nodes - n0) as f32 / (n1 - n0) as f32;
            return r0 + t * (r1 - r0);
        }
    }
    // extrapolate from the last segment
    let (n0, r0) = curve[curve.len() - 2];
    let (n1, r1) = curve[curve.len() - 1];
    let slope = (r1 - r0) / (n1 - n0) as f32;
    r1 + slope * (nodes - n1) as f32
}

impl TreeCostModel {
    /// Calibrated to our regime: flat verify cost ≈ 1.0 decode-step, EAGLE3-head draft per level ≈ 0.1.
    /// Pair with a CEILING_* curve via [`optimal_measured`]. node_budget is a hard cap (set generous;
    /// the curve does the real trading).
    pub fn calibrated() -> Self {
        TreeCostModel { verify_flat_cost: 1.0, head_level_cost: 0.1, node_budget: 256 }
    }

    /// Score using the MEASURED verify-cost curve: round_cost = depth·head + verify_flat_cost·ratio(nodes).
    pub fn evaluate_measured(&self, m: &MeasuredAccept, branch: usize, depth: usize, curve: &[(u64, f32)]) -> TreeShape {
        let p = m.persistence(depth);
        let emitted = 1.0 + tree_expected_accepted(m.first_pos, p, branch, depth);
        let nodes = tree_nodes(branch, depth);
        let draft = depth as f32 * self.head_level_cost;
        let round_cost = draft + self.verify_flat_cost * verify_ratio(nodes, curve);
        let speedup = if round_cost > 0.0 { emitted / round_cost } else { 0.0 };
        TreeShape { branch, depth, emitted, nodes, speedup }
    }

    /// Optimal (branch, depth) using the measured degradation curve (no hard cliff — the curve penalizes
    /// over-wide trees). Returns the speedup-maximizing shape.
    pub fn optimal_measured(&self, m: &MeasuredAccept, max_branch: usize, max_depth: usize, curve: &[(u64, f32)]) -> TreeShape {
        let mut best = self.evaluate_measured(m, 1, 1, curve);
        for branch in 1..=max_branch.max(1) {
            for depth in 1..=max_depth.max(1) {
                if tree_nodes(branch, depth) > self.node_budget {
                    continue;
                }
                let s = self.evaluate_measured(m, branch, depth, curve);
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
    fn verify_ratio_interpolates_and_extrapolates() {
        let c = CEILING_CTX4096;
        assert!((verify_ratio(1, c) - 1.00).abs() < 1e-6);
        assert!((verify_ratio(16, c) - 1.09).abs() < 1e-6);
        let mid = verify_ratio(20, c); // between (16,1.09) and (24,1.16)
        assert!(mid > 1.09 && mid < 1.16, "interp {mid}");
        assert!(verify_ratio(128, c) > 1.64, "extrapolates past last point");
        // monotonic non-decreasing
        assert!(verify_ratio(32, c) >= verify_ratio(16, c));
    }

    #[test]
    fn measured_optimal_is_a_modest_tree_in_the_flat_regime() {
        // With the real ceiling curve, the optimizer should pick a tree that stays in/near the flat
        // regime (nodes within ~the ceiling), beat the chain, and not run away to huge width.
        let m = measured();
        let model = TreeCostModel::calibrated();
        let best = model.optimal_measured(&m, 8, 8, CEILING_CTX4096);
        assert!(best.branch >= 2, "trees pay: branch {} >= 2", best.branch);
        assert!(best.nodes <= 64, "stays near the flat regime: nodes {}", best.nodes);
        // beats the best chain under the same measured-cost model
        let mut best_chain = model.evaluate_measured(&m, 1, 1, CEILING_CTX4096);
        for d in 1..=8 {
            let s = model.evaluate_measured(&m, 1, d, CEILING_CTX4096);
            if s.speedup > best_chain.speedup { best_chain = s; }
        }
        assert!(best.speedup > best_chain.speedup, "tree {} > chain {}", best.speedup, best_chain.speedup);
    }

    #[test]
    fn ctx8192_curve_penalizes_width_more_than_ctx4096() {
        // Longer context degrades sooner -> the optimal tree should be no wider (fewer nodes) at ctx8192.
        let m = measured();
        let model = TreeCostModel::calibrated();
        let b4 = model.optimal_measured(&m, 8, 8, CEILING_CTX4096);
        let b8 = model.optimal_measured(&m, 8, 8, CEILING_CTX8192);
        assert!(b8.nodes <= b4.nodes, "ctx8192 nodes {} <= ctx4096 nodes {}", b8.nodes, b4.nodes);
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
