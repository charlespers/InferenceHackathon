/// Load-balanced expert placement optimizer.
///
/// Reads real activation counts (from routing_analysis) and produces a
/// placement map that minimises per-step GPU load imbalance:
///
///  1. Greedy bin-pack: sort experts by activation count desc, assign each
///     to the least-loaded GPU. Separates hot experts onto different GPUs.
///  2. Hot-expert replication: experts firing > threshold × layer mean get
///     a replica on a second GPU, halving their contribution to any one GPU.
///
/// The output feeds `PlacementMap::from_tables()` directly.

use serde::{Deserialize, Serialize};

use crate::routing::scheduler::PlacementMap;

// ---------------------------------------------------------------------------
// JSON schema for routing_stats.json (produced by tools/routing_analysis.py)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct RoutingStats {
    pub routing: RoutingCore,
}

#[derive(Deserialize)]
pub struct RoutingCore {
    pub activation_counts: Vec<Vec<u32>>, // [n_layers][n_experts]
}

// ---------------------------------------------------------------------------
// JSON schema for optimized_placement.json
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct OptimizedPlacement {
    /// placement[layer_str][expert_str] = primary_gpu
    pub placement: std::collections::HashMap<String, std::collections::HashMap<String, usize>>,
    /// replicas[layer_str][expert_str] = replica_gpu (only for hot experts)
    pub replicas: std::collections::HashMap<String, std::collections::HashMap<String, usize>>,
    pub stats: PlacementStats,
}

#[derive(Serialize, Deserialize)]
pub struct PlacementStats {
    pub n_layers: usize,
    pub n_experts: usize,
    pub n_gpus: usize,
    pub round_robin_mean_imbalance: f64,
    pub optimized_mean_imbalance: f64,
    pub improvement_pct: f64,
    pub total_replicated_expert_layers: usize,
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Greedy bin-pack: sort experts by count descending, assign each to the
/// least-loaded GPU. Returns `placement[expert] = gpu`.
fn greedy_balanced(counts: &[u32], n_gpus: usize) -> Vec<usize> {
    let n_experts = counts.len();
    let mut order: Vec<usize> = (0..n_experts).collect();
    order.sort_unstable_by(|&a, &b| counts[b].cmp(&counts[a]));

    let mut gpu_load = vec![0u64; n_gpus];
    let mut placement = vec![0usize; n_experts];

    for e in order {
        let g = gpu_load
            .iter()
            .enumerate()
            .min_by_key(|&(_, &l)| l)
            .map(|(i, _)| i)
            .unwrap_or(0);
        placement[e] = g;
        gpu_load[g] += counts[e] as u64;
    }
    placement
}

/// Return `replicas[expert] = Some(replica_gpu)` for hot experts.
/// "Hot" = activation count > `threshold_mult` × layer mean.
fn find_replicas(
    counts: &[u32],
    placement: &[usize],
    n_gpus: usize,
    threshold_mult: f64,
) -> Vec<Option<usize>> {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut replicas = vec![None; counts.len()];
    if total == 0 {
        return replicas;
    }
    let mean = total as f64 / counts.len() as f64;
    let threshold = (threshold_mult * mean) as u32;

    // Build current GPU load from primary placement
    let mut gpu_load = vec![0u64; n_gpus];
    for (e, &c) in counts.iter().enumerate() {
        gpu_load[placement[e]] += c as u64;
    }

    // Process hot experts hottest-first
    let mut hot: Vec<usize> = (0..counts.len())
        .filter(|&e| counts[e] > threshold)
        .collect();
    hot.sort_unstable_by(|&a, &b| counts[b].cmp(&counts[a]));

    for e in hot {
        let primary = placement[e];
        let replica_g = (0..n_gpus)
            .filter(|&g| g != primary)
            .min_by_key(|&g| gpu_load[g])
            .unwrap_or((primary + 1) % n_gpus);
        replicas[e] = Some(replica_g);
        let half = counts[e] as u64 / 2;
        gpu_load[replica_g] += half;
        gpu_load[primary] = gpu_load[primary].saturating_sub(half);
    }
    replicas
}

fn compute_imbalance(
    counts: &[u32],
    placement: &[usize],
    replicas: &[Option<usize>],
    n_gpus: usize,
) -> f64 {
    let mut gpu_load = vec![0.0f64; n_gpus];
    for (e, &c) in counts.iter().enumerate() {
        let c = c as f64;
        if let Some(rep) = replicas[e] {
            gpu_load[placement[e]] += c * 0.5;
            gpu_load[rep] += c * 0.5;
        } else {
            gpu_load[placement[e]] += c;
        }
    }
    let total: f64 = gpu_load.iter().sum();
    if total == 0.0 {
        return 1.0;
    }
    let mean = total / n_gpus as f64;
    gpu_load.iter().cloned().fold(f64::NEG_INFINITY, f64::max) / mean
}

fn rr_imbalance(counts: &[u32], n_gpus: usize) -> f64 {
    let mut gpu_load = vec![0u64; n_gpus];
    for (e, &c) in counts.iter().enumerate() {
        gpu_load[e % n_gpus] += c as u64;
    }
    let total: u64 = gpu_load.iter().sum();
    if total == 0 {
        return 1.0;
    }
    let mean = total as f64 / n_gpus as f64;
    *gpu_load.iter().max().unwrap() as f64 / mean
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Build an optimized placement from real activation counts.
/// Returns `(PlacementMap, OptimizedPlacement)` — the map is ready to use
/// in the engine; the struct can be serialized to JSON for persistence.
pub fn optimize(stats: &RoutingStats, n_gpus: usize) -> (PlacementMap, OptimizedPlacement) {
    let counts = &stats.routing.activation_counts;
    let n_layers = counts.len();
    let n_experts = counts.first().map(|l| l.len()).unwrap_or(128);

    let mut primary_table: Vec<Vec<usize>> = Vec::with_capacity(n_layers);
    let mut replica_table: Vec<Vec<Option<usize>>> = Vec::with_capacity(n_layers);

    let mut json_placement = std::collections::HashMap::new();
    let mut json_replicas: std::collections::HashMap<
        String,
        std::collections::HashMap<String, usize>,
    > = std::collections::HashMap::new();

    let mut rr_total = 0.0f64;
    let mut opt_total = 0.0f64;
    let mut total_replicated = 0usize;

    for (layer, layer_counts) in counts.iter().enumerate() {
        rr_total += rr_imbalance(layer_counts, n_gpus);

        let placement = greedy_balanced(layer_counts, n_gpus);
        let replicas = find_replicas(layer_counts, &placement, n_gpus, 2.0);

        opt_total += compute_imbalance(layer_counts, &placement, &replicas, n_gpus);

        // JSON maps
        let mut layer_map = std::collections::HashMap::new();
        let mut layer_rep_map = std::collections::HashMap::new();
        for e in 0..n_experts {
            layer_map.insert(e.to_string(), placement[e]);
            if let Some(r) = replicas[e] {
                layer_rep_map.insert(e.to_string(), r);
                total_replicated += 1;
            }
        }
        json_placement.insert(layer.to_string(), layer_map);
        if !layer_rep_map.is_empty() {
            json_replicas.insert(layer.to_string(), layer_rep_map);
        }

        primary_table.push(placement);
        replica_table.push(replicas);
    }

    let mean_rr = rr_total / n_layers as f64;
    let mean_opt = opt_total / n_layers as f64;
    let improvement_pct = (1.0 - mean_opt / mean_rr) * 100.0;

    let map = PlacementMap::from_tables(primary_table, replica_table, n_gpus, n_experts);
    let out = OptimizedPlacement {
        placement: json_placement,
        replicas: json_replicas,
        stats: PlacementStats {
            n_layers,
            n_experts,
            n_gpus,
            round_robin_mean_imbalance: (mean_rr * 1000.0).round() / 1000.0,
            optimized_mean_imbalance: (mean_opt * 1000.0).round() / 1000.0,
            improvement_pct: (improvement_pct * 10.0).round() / 10.0,
            total_replicated_expert_layers: total_replicated,
        },
    };
    (map, out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_counts(n: usize, val: u32) -> Vec<u32> {
        vec![val; n]
    }

    #[test]
    fn greedy_balanced_spreads_hot_experts() {
        // One expert fires 1000x, others fire 1x. It should land on its own GPU.
        let mut counts = vec![1u32; 128];
        counts[0] = 1000;
        let placement = greedy_balanced(&counts, 8);
        // Expert 0 goes to some GPU; no other expert should share that GPU
        // if they can be spread (they can — 127 experts / 7 GPUs).
        let hot_gpu = placement[0];
        let sharing = counts
            .iter()
            .enumerate()
            .filter(|&(e, _)| e != 0 && placement[e] == hot_gpu)
            .count();
        // With 128 experts / 8 GPUs = 16 per GPU; hottest goes first so it
        // picks least loaded; others fill in. At most 15 others on same GPU.
        assert!(sharing <= 15);
    }

    #[test]
    fn greedy_balanced_uses_all_gpus() {
        let counts = uniform_counts(128, 10);
        let placement = greedy_balanced(&counts, 8);
        let mut gpu_counts = vec![0usize; 8];
        for &g in &placement {
            gpu_counts[g] += 1;
        }
        // Every GPU should get exactly 16 experts
        assert!(gpu_counts.iter().all(|&c| c == 16));
    }

    #[test]
    fn find_replicas_marks_hot_experts() {
        let mut counts = vec![1u32; 128];
        counts[0] = 1000; // way above 2x mean
        let placement = greedy_balanced(&counts, 8);
        let replicas = find_replicas(&counts, &placement, 8, 2.0);
        assert!(replicas[0].is_some(), "hot expert should be replicated");
        // Cold experts should not be replicated
        let n_replicated = replicas.iter().filter(|r| r.is_some()).count();
        assert!(n_replicated < 10, "only hot experts replicated, got {}", n_replicated);
    }

    #[test]
    fn optimize_reduces_imbalance() {
        // Hot experts are multiples of 8 (0,8,16,...,120) — all map to GPU 0
        // under round-robin, creating a ~7.5x imbalance. Optimizer spreads them.
        let layer_counts: Vec<u32> = (0..128u32)
            .map(|e| if e % 8 == 0 { 100 } else { 1 })
            .collect();
        let stats = RoutingStats {
            routing: RoutingCore { activation_counts: vec![layer_counts.clone()] },
        };
        let rr = rr_imbalance(&layer_counts, 8);
        let (_, out) = optimize(&stats, 8);
        assert!(
            out.stats.optimized_mean_imbalance < rr,
            "optimized ({}) should beat round-robin ({})",
            out.stats.optimized_mean_imbalance,
            rr
        );
    }
}
