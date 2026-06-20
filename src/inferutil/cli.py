"""Pretty-print the roofline / latency analysis.

Run:  python -m inferutil            (defaults: Qwen3-235B-A22B on 8xH100)
      python -m inferutil --gpu H200-SXM-141GB
"""

from __future__ import annotations

import argparse

from .hardware import GPUS, Cluster
from .model import QWEN3_235B
from .latency import decode_latency
from .speculative import sweep as spec_sweep, memory_feasibility, expected_accepted_multi
from .routing import (CoActGraph, ingest_telemetry, greedy_partition,
                      round_robin_placement, placement_stats, TokenDAG)


def _b(n: int) -> str:
    return f"{n/1e9:.1f}B"


def _gb(n: float) -> str:
    return f"{n/1e9:.1f} GB"


def hardware_summary(cluster) -> str:
    g = cluster.gpu
    TB = 1e12
    TFLOP = 1e12
    GB = 1e9
    out = [
        f"== HARDWARE: {cluster.n_gpus}x {g.name} ==",
        f"  HBM per GPU  : {g.hbm_bytes/GB:.0f} GB    "
        f"total: {cluster.total_hbm/GB:.0f} GB",
        f"  HBM BW       : {g.hbm_bw/TB:.2f} TB/s per GPU  "
        f"(aggregate: {cluster.aggregate_hbm_bw/TB:.2f} TB/s)",
        f"  BF16 compute : {g.bf16_flops/TFLOP:.1f} TFLOP/s per GPU",
        f"  FP8  compute : {g.fp8_flops/TFLOP:.1f} TFLOP/s per GPU",
        f"  NVLink BW    : {g.nvlink_bw/GB:.0f} GB/s unidirectional (NVLink4)",
        f"  Collective   : ~{g.collective_latency_s*1e6:.0f} µs assumed launch+sync "
        f"latency per collective (NVSwitch fabric, B=1 payloads)",
    ]
    return "\n".join(out)


def model_summary(cfg) -> str:
    out = [f"== MODEL: {cfg.name} ==",
           f"  layers={cfg.n_layers}  hidden={cfg.hidden}  "
           f"heads={cfg.n_heads}(q)/{cfg.n_kv_heads}(kv) head_dim={cfg.head_dim}",
           f"  experts={cfg.n_experts} top_k={cfg.top_k} "
           f"moe_inter={cfg.moe_inter} shared={cfg.n_shared_experts}",
           f"  total params : {_b(cfg.total_params)}   "
           f"(attn {_b(cfg.n_layers*cfg.attn_params)}, "
           f"experts {_b(cfg.n_layers*cfg.moe_params_per_layer)}, "
           f"embed {_b(cfg.embed_params)})",
           f"  active/token : {_b(cfg.active_params)}   "
           f"({cfg.active_params/cfg.total_params*100:.1f}% of total)"]
    return "\n".join(out)


def memory_budget(cfg, cluster) -> str:
    out = [f"== MEMORY BUDGET: {cluster.n_gpus}x {cluster.gpu.name} "
           f"= {_gb(cluster.total_hbm)} HBM ==",]
    for name, b in (("bf16", 2), ("fp8", 1)):
        w = cfg.total_params * b
        out.append(f"  weights @{name:>4}: {_gb(w):>9}  "
                   f"({w/cluster.total_hbm*100:4.1f}% HBM, "
                   f"{_gb(w/cluster.n_gpus)}/gpu)")
    out.append("  KV cache (bf16, GQA 4 kv-heads):")
    for s in (4096, 32768, 131072):
        kv = s * cfg.kv_bytes_per_token(2)
        out.append(f"     seq {s:>7}: {_gb(kv):>8} / sequence")
    return "\n".join(out)


def latency_table(cfg, cluster) -> str:
    out = ["== B=1 DECODE LATENCY (per token, lower bound) =="]
    header = (f"  {'plan':<8}{'dtype':<6}{'seq':>7}  {'weight':>8}{'kv':>7}"
              f"{'compute':>8}{'comms':>7}{'TOTAL':>8}{'tok/s':>8}{'imbal':>7}")
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    configs = [
        ("floor", 2, 32768), ("tp", 2, 32768), ("ep", 2, 32768),
        ("hybrid", 2, 4096), ("hybrid", 2, 32768), ("hybrid", 2, 131072),
        ("hybrid", 1, 32768),
    ]
    for plan, dtype, seq in configs:
        r = decode_latency(cfg, cluster, plan=plan, dtype_bytes=dtype,
                           seq_len=seq).as_row()
        out.append(f"  {r['plan']:<8}{r['dtype']:<6}{r['seq']:>7}  "
                   f"{r['weight_ms']:>7.2f}{r['kv_ms']:>7.2f}"
                   f"{r['compute_ms']:>8.3f}{r['comms_ms']:>7.2f}"
                   f"{r['total_ms']:>7.2f}{r['tok_per_s']:>8.1f}"
                   f"{r['expert_imbalance']:>7.2f}")
    return "\n".join(out)


def takeaways(cfg, cluster) -> str:
    bf16 = decode_latency(cfg, cluster, plan="hybrid", seq_len=32768)
    fp8 = decode_latency(cfg, cluster, plan="hybrid", dtype_bytes=1, seq_len=32768)
    ideal = decode_latency(cfg, cluster, plan="hybrid", seq_len=32768,
                           ideal_routing=True)
    tp = decode_latency(cfg, cluster, plan="tp", seq_len=32768)
    ep = decode_latency(cfg, cluster, plan="ep", seq_len=32768)
    lines = [
        "== TAKEAWAYS (8xH100, hybrid TP-attn + EP-experts) ==",
        f"  - PLAN CHOICE: naive EP ({ep.total_s*1e3:.2f}ms) is SLOWER than plain "
        f"TP ({tp.total_s*1e3:.2f}ms) at B=1. With {cfg.top_k} active experts over "
        f"{cluster.n_gpus} GPUs the busiest does {ep.imbalance:.1f}x its share while "
        f"others idle. EP only wins once routing imbalance is fixed -> that is the "
        f"prize expert prediction/placement unlocks.",
        f"  - Compute is ~{bf16.compute_s/bf16.total_s*100:.1f}% of the budget: "
        f"decode is MEMORY-BANDWIDTH bound. Tuning FLOPs ~ wasted effort.",
        f"  - Weight reads dominate ({bf16.weight_read_s/bf16.total_s*100:.0f}%). "
        f"FP8 weights ~halve them: {bf16.total_s*1e3:.2f}ms -> {fp8.total_s*1e3:.2f}ms "
        f"({bf16.tokens_per_s:.0f} -> {fp8.tokens_per_s:.0f} tok/s).",
        f"  - Expert imbalance costs {(bf16.total_s/ideal.total_s-1)*100:.0f}%: "
        f"busiest GPU runs {bf16.imbalance:.2f}x the ideal expert count. "
        f"Smart placement / replication of hot experts is real latency.",
        f"  - Per-layer collective latency = {bf16.comms_s*1e3:.2f}ms "
        f"({bf16.comms_s/bf16.total_s*100:.0f}%): {cfg.n_layers} layers x small "
        f"all-to-all. Fusing/overlapping comms matters more than payload size.",
        f"  - KV reads grow with context ({bf16.kv_read_s*1e3:.2f}ms @32k): "
        f"at long context this rivals weights. KV-cache compression pays off.",
        "",
        "  Optimization priority (latency, B=1):",
        "    1. FP8 (or lower) weights        -> ~halves the dominant term",
        "    2. Expert placement / prediction -> kills imbalance + hides transfer",
        "    3. Comms fusion / overlap        -> reclaims per-layer collective tax",
        "    4. KV compression (long context) -> caps the attention term",
        "    5. B=1 GEMV / attention kernels   -> hit the BW roofline, not FLOPs",
    ]
    return "\n".join(lines)


def speculative_report() -> str:
    out = ["== MULTI-DRAFTER SPECULATIVE DECODING =="]

    mem = memory_feasibility(use_fp8_target=True)
    out.append(f"  Memory (fp8 target): {mem['target_weight_gb']} GB used, "
               f"{mem['hbm_headroom_gb']} GB free -> fits {mem['max_drafters']}x "
               f"Qwen3-1.7B drafters on-chip")

    # E[accepted] table for α=0.7, varying N and k
    alpha = 0.7
    out.append(f"\n  E[tokens accepted per round] @ α={alpha} (single-drafter baseline):")
    out.append(f"  {'N drafters':<12} {'k=4':>6} {'k=8':>6} {'k=16':>6}  "
               f"{'k=4 speedup':>12} {'k=8 speedup':>12}")
    out.append("  " + "-" * 66)
    for n in [1, 2, 4, 8]:
        vals = [expected_accepted_multi(alpha, k, n) for k in (4, 8, 16)]
        # speedup = e_acc / (drafter_cost + 1.0); drafter_cost=0.05 (parallel)
        speedups = [round(v / (0.05 + 1.0), 2) for v in vals[:2]]
        out.append(f"  {n:<12} {vals[0]:>6.2f} {vals[1]:>6.2f} {vals[2]:>6.2f}  "
                   f"{speedups[0]:>12.2f} {speedups[1]:>12.2f}")

    out.append(f"\n  Key insight: at B=1 the verify pass is BW-bound and reads the "
               f"same active weights regardless of N*k batch size (up to ~32 tokens).")
    out.append(f"  N=4 drafters with k=8 turns α=0.70 per-token acceptance into "
               f"~{expected_accepted_multi(alpha, 8, 4):.1f} accepted tokens/round "
               f"(vs {expected_accepted_multi(alpha, 8, 1):.1f} single-drafter).")
    out.append(f"  Drafters run in parallel, so wall-clock drafter cost ≈ constant.")
    return "\n".join(out)


def routing_report() -> str:
    out = ["== EXPERT PLACEMENT: GRAPH PARTITION vs ROUND-ROBIN =="]

    # Simulate uniform routing (worst case for graph partition — no structure to exploit)
    # and a skewed routing (tokens cluster around a subset of experts)
    import random
    rng = random.Random(42)
    n_experts, n_layers, n_gpus, top_k = 128, 94, 8, 8
    n_tokens = 500

    graph_uniform = CoActGraph(n_experts=n_experts, n_layers=n_layers)
    graph_skewed = CoActGraph(n_experts=n_experts, n_layers=n_layers)

    # Build a "hot cluster": experts 0-31 are 4x more likely (simulates topic specialisation)
    hot = set(range(32))

    for _ in range(n_tokens):
        prev_experts: list[int] = []
        for layer in range(n_layers):
            # uniform routing
            u_experts = rng.sample(range(n_experts), top_k)
            graph_uniform.add_token_step(layer, u_experts)

            # skewed routing: draw from hot cluster with 4x weight
            weights = [4 if e in hot else 1 for e in range(n_experts)]
            total_w = sum(weights)
            pop = list(range(n_experts))
            s_experts: list[int] = []
            pool = list(zip(pop, weights))
            for _ in range(top_k):
                r = rng.uniform(0, sum(w for _, w in pool))
                cum = 0.0
                for i, (e, w) in enumerate(pool):
                    cum += w
                    if r <= cum:
                        s_experts.append(e)
                        pool.pop(i)
                        break
            graph_skewed.add_token_step(layer, s_experts)

            if prev_experts:
                graph_uniform.add_cross_layer(layer - 1, prev_experts, u_experts)
                graph_skewed.add_cross_layer(layer - 1, prev_experts, s_experts)
            prev_experts = u_experts  # uniform for both cross-layer

    rr = round_robin_placement(n_gpus, n_experts, n_layers)
    greedy_u = greedy_partition(graph_uniform, n_gpus, n_experts, n_layers)
    greedy_s = greedy_partition(graph_skewed, n_gpus, n_experts, n_layers)

    for label, graph, gp in [("uniform routing", graph_uniform, greedy_u),
                              ("skewed routing (hot 32/128 experts)", graph_skewed, greedy_s)]:
        rr_stats = placement_stats(graph, rr, "round-robin")
        gp_stats = placement_stats(graph, gp, "graph-partition")
        out.append(f"\n  {label}:")
        out.append(f"    round-robin    cross-GPU fraction: {rr_stats['cross_gpu_fraction']:.3f}")
        out.append(f"    graph-part.    cross-GPU fraction: {gp_stats['cross_gpu_fraction']:.3f}  "
                   f"({(rr_stats['cross_gpu_fraction']-gp_stats['cross_gpu_fraction'])/rr_stats['cross_gpu_fraction']*100:.1f}% reduction)")

    # DAG / prefetch schedule example
    dag = TokenDAG()
    for layer in range(0, 5):
        dag.add_layer(layer, rng.sample(range(n_experts), top_k))
    sched = dag.prefetch_schedule(rr)
    out.append(f"\n  Prefetch schedule (first 3 layers, round-robin placement):")
    for step in sched[:3]:
        out.append(f"    compute L{step['compute_layer']} | prefetch L{step['prefetch_layer']} "
                   f"experts→GPUs {step['prefetch_experts'][:3]}... "
                   f"({step['n_gpus_touched']} GPUs touched, cross={step['cross_gpu']})")

    out.append(f"\n  Key insight: skewed routing (topic-clustered tokens) lets graph "
               f"partition cut cross-GPU comms significantly. Uniform routing offers "
               f"less — but real LLM traffic is rarely uniform. Measure on the real "
               f"engine with actual token distributions to see the true gain.")
    out.append(f"  The prefetch schedule above shows how topo-sort enables overlapping "
               f"L+1 dispatch with L compute, hiding all-to-all latency behind FFN time.")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="MoE B=1 inference latency model")
    ap.add_argument("--gpu", default="H100-SXM-80GB", choices=list(GPUS))
    ap.add_argument("--n-gpus", type=int, default=8)
    args = ap.parse_args(argv)

    cfg = QWEN3_235B
    cluster = Cluster(gpu=GPUS[args.gpu], n_gpus=args.n_gpus)
    print(hardware_summary(cluster))
    print()
    print(model_summary(cfg))
    print()
    print(memory_budget(cfg, cluster))
    print()
    print(latency_table(cfg, cluster))
    print()
    print(takeaways(cfg, cluster))
    print()
    print(speculative_report())
    print()
    print(routing_report())


if __name__ == "__main__":
    main()
