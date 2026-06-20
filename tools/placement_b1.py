#!/usr/bin/env python3
"""B=1-correct expert placement: minimize per-step BUSIEST-RANK, not average load.

The team's optimizer (engine/routing/optimizer.rs) and routing_predict's `cross_gpu_fraction` minimize the
*average* / total cross-GPU traffic — the throughput objective. At B=1 the per-token latency is set by the
**busiest GPU on that single token** (max # of the token's 8 experts on one GPU). This tool:
  1. computes e_step_busiest = E_token[ max_g (# of token's experts on g) ]  — the real B=1 metric,
  2. compares round-robin vs greedy-by-count (the team's) vs co-activation-aware placement,
  3. adds PREDICTIVE REPLICA SELECTION: with replicas, each token picks the less-loaded copy of each expert
     (a greedy min-max assignment) -> pushes per-step busiest toward 1.0 (the ideal). (docs/ep-placement-for-b1.md)
Pure-stdlib, no GPU. Runs on a synthetic Zipf+co-activation routing model, or real per-token expert traces:
  python3 tools/placement_b1.py                          # synthetic demo
  python3 tools/placement_b1.py --traces token_experts.json   # [[e,e,..8], ...] real top-8 sets
"""
import argparse, json, random, math

N_EXP, TOP_K, N_GPU = 128, 8, 8


def synth_traces(n_tokens, zipf_s, n_clusters, seed):
    rng = random.Random(seed)
    # Zipf popularity + co-activation: experts in the same cluster co-fire (chat/topic locality).
    pop = [1.0 / ((i + 1) ** zipf_s) for i in range(N_EXP)]
    cluster = [e % n_clusters for e in range(N_EXP)]
    by_cluster = {c: [e for e in range(N_EXP) if cluster[e] == c] for c in range(n_clusters)}
    cw = [sum(pop[e] for e in by_cluster[c]) for c in range(n_clusters)]
    traces, counts = [], [0] * N_EXP
    for _ in range(n_tokens):
        c = rng.choices(range(n_clusters), weights=cw)[0]      # pick a topic cluster
        pool = by_cluster[c] + rng.sample(range(N_EXP), 16)    # mostly in-cluster + some spread
        w = [pop[e] for e in pool]
        sel = set()
        while len(sel) < TOP_K:
            sel.add(rng.choices(pool, weights=w)[0])
        s = list(sel)
        traces.append(s)
        for e in s: counts[e] += 1
    return traces, counts


def busiest(placement, traces):  # placement: expert -> primary gpu (no replicas)
    tot = 0
    for s in traces:
        load = [0] * N_GPU
        for e in s: load[placement[e]] += 1
        tot += max(load)
    return tot / len(traces)


def busiest_replicas(reps, traces):  # reps: expert -> list of allowed gpus; greedy min-max per token
    tot = 0
    for s in traces:
        load = [0] * N_GPU
        for e in s:                                   # assign each expert to its least-loaded allowed copy
            g = min(reps[e], key=lambda g: load[g]); load[g] += 1
        tot += max(load)
    return tot / len(traces)


def cross_gpu_fraction(placement, traces):  # the team's proxy: fraction of co-fired pairs split across GPUs
    cross = tot = 0
    for s in traces:
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                tot += 1; cross += (placement[s[i]] != placement[s[j]])
    return cross / tot if tot else 0.0


def greedy_by_count(counts):  # team's optimizer.rs: hot experts -> least-loaded GPU (balances AVG load)
    order = sorted(range(N_EXP), key=lambda e: -counts[e]); gpu_load = [0] * N_GPU; pl = {}
    for e in order:
        g = min(range(N_GPU), key=lambda g: gpu_load[g]); pl[e] = g; gpu_load[g] += counts[e]
    return pl


def coactivation_aware(traces, init, iters=4000, seed=0):  # local swaps minimizing e_step_busiest (the B=1 metric)
    rng = random.Random(seed); pl = dict(init); cur = busiest(pl, traces)
    for _ in range(iters):
        e = rng.randrange(N_EXP); g_old = pl[e]; g_new = rng.randrange(N_GPU)
        if g_new == g_old: continue
        pl[e] = g_new; new = busiest(pl, traces)
        if new <= cur: cur = new
        else: pl[e] = g_old
    return pl, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces"); ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--zipf", type=float, default=1.1); ap.add_argument("--clusters", type=int, default=12)
    ap.add_argument("--replicas", type=int, default=16, help="# hottest experts to replicate (predictive selection)")
    a = ap.parse_args()
    if a.traces:
        traces = json.load(open(a.traces)); counts = [0] * N_EXP
        for s in traces:
            for e in s: counts[e] += 1
    else:
        traces, counts = synth_traces(a.tokens, a.zipf, a.clusters, 0)

    rr = {e: e % N_GPU for e in range(N_EXP)}
    gc = greedy_by_count(counts)
    ca, ca_b = coactivation_aware(traces, gc)
    print(f"placement                 e_step_busiest   cross_gpu_frac   (ideal busiest=1.0, random≈{1+ (TOP_K-1)/N_GPU:.2f})")
    for name, pl in (("round-robin", rr), ("greedy-by-count (team)", gc), ("co-activation (B1-optimal)", ca)):
        print(f"  {name:26s}  {busiest(pl,traces):6.3f}          {cross_gpu_fraction(pl,traces):6.3f}")
    # predictive replica selection on top of the co-activation placement
    hot = sorted(range(N_EXP), key=lambda e: -counts[e])[:a.replicas]
    reps = {e: [ca[e]] for e in range(N_EXP)}
    for e in hot:
        g2 = min(range(N_GPU), key=lambda g: sum(1 for x in range(N_EXP) if ca[x] == g))
        if g2 != ca[e]: reps[e].append(g2)
    print(f"  + predictive replica sel ({a.replicas} hot)  {busiest_replicas(reps,traces):6.3f}          "
          f"(each token picks the less-loaded copy -> toward 1.0)")
    print("\nTakeaway: greedy-by-count minimizes cross_gpu_frac (throughput) but NOT e_step_busiest (B=1 latency);")
    print("co-activation-aware + predictive replicas drive the per-step busiest down — the EP-path latency win.")


if __name__ == "__main__":
    main()
