# EP placement for B=1: optimize per-step busiest-rank, not average load

A correction + extension for the team's placement optimizer (`engine/src/routing/optimizer.rs`,
`tools/placement_optimizer.py`). The optimizer is sound, but its **objective is the throughput one**; at
B=1 the right objective is different, and getting it right is what claws back the EP penalty I measured
(EP8 94 vs TP8 261 tok/s in the team's own model).

## Why placement matters here at all
Pure TP8 has **zero** expert imbalance (every GPU reads 8/8 of each active expert's column-slice) and is
the latency winner. **But the FP8 vLLM path is forced onto EP** (`--tensor-parallel-size 8` crashes:
`192 % 128`, see `gpu-agent-experiments.md` E1). On the EP path, placement is the only knob on the
busiest-rank penalty — so it's worth getting exactly right *for as long as we serve via EP*.

## The objective mismatch
The optimizer's `greedy_balanced` sorts experts by **total activation count** and balances the **sum** of
counts per GPU (→ `round_robin_mean_imbalance` vs `optimized_mean_imbalance`). That minimizes the **long-run
average** load per GPU — the **throughput** metric.

At **B=1** the per-token latency is set by the **busiest GPU on that single token**:
```
per_step_busiest = max over GPUs of  ( # of THIS token's 8 experts placed on that GPU )
B=1 latency objective:  minimize  E_token[ per_step_busiest ]      # NOT mean aggregate load
```
These differ because per-step busiest depends on **within-token co-activation** (which experts land in the
*same* token's top-8), while marginal counts don't see it. Two experts with moderate individual counts that
**frequently co-fire** can be greedily placed on the same GPU → they collide every time they co-fire,
inflating per-step busiest — invisible to the average-load metric.

- Uniform routing, 8 experts → 8 GPUs: E[busiest] = 2.597 (balls-in-bins) regardless of placement.
- Zipf-skewed routing: marginal-count balancing (current) helps by spreading *hot* experts → pulls E[max]
  from ~3.37 toward ~2.6. Good, but it leaves the **co-activation** collisions on the table.

## What to change
1. **Add a B=1 metric to `PlacementStats`:** `e_step_busiest` = mean over routing traces of
   `max_g |{token's experts on g}|`, for round-robin vs optimized. This is the number that predicts B=1
   TPOT; report it alongside the average imbalance. (Compute from per-token expert sets — see data note.)
2. **Co-activation-aware placement:** minimize per-step busiest by spreading experts that **co-fire within a
   token**, not just hot ones. Objective: a graph-partition / min-conflict placement where edge weight =
   co-activation frequency (place high-co-activation pairs on different GPUs). Greedy seed + local swaps that
   reduce `e_step_busiest`.
3. **Replication for B=1 (their hot-replica idea, sharpened):** a replicated expert can be served from
   *either* GPU, so the per-step assignment can pick the *less-loaded* copy for **this** token → directly
   lowers per-step busiest. Prioritize replicating experts that most often cause collisions, not just the
   highest-count ones.

## The novel synergy: predictive replica selection (route-prediction × placement)
`predictor.rs` DirectProxy predicts a token's 8 experts *before* the layer runs. With replicas, the engine
can then **choose, per token, which replica of each predicted expert to use so the 8 land 1-per-GPU** — a
bipartite matching that pushes per-step busiest toward **1.0** (the ideal), not the static-placement 2.6.
This is a real B=1 latency lever that *only* exists because route prediction + replication are both present:
spend a little HBM on replicas, use the predictor to balance each token. Worth modeling: even halving the
EP busiest factor (2.6→~1.5) is ~1.5× on the EP expert term.

## Data note
`optimizer.rs` reads `routing.activation_counts[n_layers][n_experts]` — **marginal** counts only. The B=1
metric + co-activation placement need the **within-token joint** (a co-activation matrix or per-token expert
sets) which `tools/routing_analysis.py` can emit. `routing_stats.json` already has `markov_matrices`
(token-to-token transitions) — useful for the *predictor*, but per-step placement needs the *within-token*
co-activation, a different statistic.

## Honest bound
Even an optimal EP placement (predictive replica selection → busiest ≈ 1.5, heavy replication → →1) does not
beat pure TP8 (busiest = 1, no all-to-all, fewer collectives). **Placement is the EP-path mitigation; the
real fix is a block-64 FP8 requant (or the cudarc engine) that lets us serve pure TP8.** Until then,
co-activation-aware placement + predictive replica selection is the highest-leverage EP improvement, and the
`e_step_busiest` metric is how to measure it.
