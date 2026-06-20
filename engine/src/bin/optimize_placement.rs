/// CLI: build optimized expert placement from routing stats.
///
/// Usage:
///   cargo run --release --bin optimize-placement -- \
///       /alloc/data/routing_stats.json \
///       /alloc/data/optimized_placement.json

use std::fs;
use std::path::PathBuf;

use engine::routing::optimizer::{optimize, RoutingStats};

const N_GPUS: usize = 8;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let stats_path = PathBuf::from(
        args.get(1).map(String::as_str).unwrap_or("/alloc/data/routing_stats.json"),
    );
    let out_path = PathBuf::from(
        args.get(2)
            .map(String::as_str)
            .unwrap_or("/alloc/data/optimized_placement.json"),
    );

    eprintln!("Loading {}", stats_path.display());
    let raw = fs::read_to_string(&stats_path)?;
    let stats: RoutingStats = serde_json::from_str(&raw)?;

    let n_layers = stats.routing.activation_counts.len();
    let n_experts = stats.routing.activation_counts.first().map(|l| l.len()).unwrap_or(0);
    eprintln!("  {} layers × {} experts × {} GPUs", n_layers, n_experts, N_GPUS);

    let (_map, result) = optimize(&stats, N_GPUS);

    let s = &result.stats;
    eprintln!("\nResults:");
    eprintln!("  Round-robin mean imbalance : {:.3}x", s.round_robin_mean_imbalance);
    eprintln!("  Optimized  mean imbalance  : {:.3}x", s.optimized_mean_imbalance);
    eprintln!("  Improvement                : {:.1}%", s.improvement_pct);
    eprintln!("  Hot experts replicated     : {} expert-layer pairs", s.total_replicated_expert_layers);

    // Show top 10 hottest replicated experts
    let counts = &stats.routing.activation_counts;
    let mut hot_pairs: Vec<(u32, usize, usize, usize, usize)> = Vec::new(); // (count, layer, expert, primary, replica)
    for (layer_str, rep_map) in &result.replicas {
        let layer: usize = layer_str.parse()?;
        for (e_str, &replica) in rep_map {
            let e: usize = e_str.parse()?;
            let count = counts[layer][e];
            let primary = result.placement[layer_str][e_str];
            hot_pairs.push((count, layer, e, primary, replica));
        }
    }
    hot_pairs.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    eprintln!("\n  Top 10 replicated experts (hottest first):");
    for (count, layer, expert, primary, replica) in hot_pairs.iter().take(10) {
        eprintln!(
            "    L{:3} E{:3}: {:5} activations  GPU {} + replica GPU {}",
            layer, expert, count, primary, replica
        );
    }

    fs::write(&out_path, serde_json::to_string(&result)?)?;
    eprintln!("\nSaved → {}", out_path.display());
    Ok(())
}
