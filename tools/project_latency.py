"""Project B=1 decode latency from measured routes -> the demo speedup table.

Consumes routing_predict.json (raw per-token routes + predictor numbers) and
drives inferutil's roofline with the *measured* per-token GPU imbalance under
each strategy, so the speedup figures are anchored on real routing, not theory.

Levers modeled (note: 'prefetch-to-hide-transfer' is intentionally NOT here —
at B=1 all expert weights are HBM-resident, so it's a no-op; replication helps
*imbalance*, not transfer):
  - placement (jminding greedy_balanced) + hot-expert replication  -> imbalance
  - FP8 weights                                                    -> bytes
  - n-gram speculative decode (alpha sweep, jminding speculative.py) -> tokens/pass

Usage:
    PYTHONPATH=src python3 tools/project_latency.py routing_predict.json
"""

from __future__ import annotations

import json
import sys

from inferutil.hardware import H100_SXM, Cluster
from inferutil.latency import decode_latency
from inferutil.model import QWEN3_235B as CFG
from inferutil.speculative import expected_accepted_multi

N_GPUS = 8


# --- jminding's placement algorithm (tools/placement_optimizer.py), reused ---
def greedy_balanced(counts, n_gpus=N_GPUS):
    order = sorted(range(len(counts)), key=lambda e: -counts[e])
    load = [0.0] * n_gpus
    place = [0] * len(counts)
    for e in order:
        g = min(range(n_gpus), key=lambda g: load[g])
        place[e] = g
        load[g] += counts[e]
    return place


def find_replicas(counts, place, n_gpus=N_GPUS, threshold_mult=2.0):
    total = sum(counts)
    if total == 0:
        return {}
    thr = threshold_mult * (total / len(counts))
    load = [0.0] * n_gpus
    for e, c in enumerate(counts):
        load[place[e]] += c
    reps = {}
    for e in sorted([e for e in range(len(counts)) if counts[e] > thr],
                    key=lambda e: -counts[e]):
        prim = place[e]
        rg = min((g for g in range(n_gpus) if g != prim), key=lambda g: load[g])
        reps[e] = rg
        load[rg] += counts[e] * 0.5
        load[prim] -= counts[e] * 0.5
    return reps


def per_token_max_experts(routes, n_layers, place=None, reps=None):
    """Mean over (token, layer) of the busiest GPU's chosen-expert count among
    the 8 active experts -- the quantity that sets B=1 EP latency."""
    vals = []
    for tok in routes:
        for layer, experts in enumerate(tok):
            if not experts:
                continue
            counts = [0] * N_GPUS
            deferred = []
            for e in experts:
                if reps and layer in reps and e in reps[layer]:
                    deferred.append((place[layer][e], reps[layer][e]))
                elif place is not None:
                    counts[place[layer][e]] += 1
                else:
                    counts[e % N_GPUS] += 1
            for prim, rep in deferred:  # replicated: serve from the lighter GPU
                counts[prim if counts[prim] <= counts[rep] else rep] += 1
            vals.append(max(counts))
    return sum(vals) / len(vals) if vals else float(CFG.top_k) / N_GPUS


def build_placement(routes, n_layers, n_experts):
    counts = [[0] * n_experts for _ in range(n_layers)]
    for tok in routes:
        for layer, experts in enumerate(tok):
            for e in experts:
                counts[layer][e] += 1
    place = {L: greedy_balanced(counts[L]) for L in range(n_layers)}
    reps = {}
    for L in range(n_layers):
        r = find_replicas(counts[L], place[L])
        if r:
            reps[L] = r
    return place, reps


def row(label, bd, baseline_tps, comms_override=None):
    total = bd.total_s if comms_override is None else (
        bd.weight_read_s + bd.kv_read_s + bd.compute_s + comms_override)
    tps = 1.0 / total
    return {
        "strategy": label,
        "weight_ms": round(bd.weight_read_s * 1e3, 2),
        "comms_ms": round((comms_override if comms_override is not None else bd.comms_s) * 1e3, 2),
        "kv_ms": round(bd.kv_read_s * 1e3, 2),
        "total_ms": round(total * 1e3, 2),
        "tok_per_s": round(tps, 1),
        "imbalance": round(bd.imbalance, 2),
        "speedup_vs_tp": round(tps / baseline_tps, 2),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "routing_predict.json"
    d = json.load(open(path))
    routes = d.get("routes")
    if not routes:
        print("ERROR: routing_predict.json has no raw 'routes' (old script "
              "version). Rerun the current tools/routing_predict.py.")
        sys.exit(1)
    n_layers = CFG.n_layers
    cl = Cluster(H100_SXM, N_GPUS)
    ctx = d.get("config", {}).get("ctx", 8192)

    place, reps = build_placement(routes, n_layers, CFG.n_experts)
    me_rr = per_token_max_experts(routes, n_layers, place=None)
    me_pl = per_token_max_experts(routes, n_layers, place=place)
    me_rep = per_token_max_experts(routes, n_layers, place=place, reps=reps)
    print(f"measured per-token busiest-GPU expert count (ideal = {CFG.top_k/N_GPUS:.2f}):")
    print(f"  round-robin {me_rr:.2f} | placement {me_pl:.2f} | "
          f"+replication {me_rep:.2f}")

    base = decode_latency(CFG, cl, plan="tp", dtype_bytes=2, seq_len=ctx)
    base_tps = base.tokens_per_s
    rows = [row("TP8 bf16 (baseline)", base, base_tps)]

    def hybrid(dt, me):
        return decode_latency(CFG, cl, plan="hybrid", dtype_bytes=dt,
                              seq_len=ctx, tp=N_GPUS, ep=N_GPUS,
                              measured_max_experts=me)
    rows.append(row("EP8 bf16 naive", hybrid(2, None), base_tps))
    rows.append(row("EP8 bf16 +placement", hybrid(2, me_pl), base_tps))
    rows.append(row("EP8 bf16 +placement+replication", hybrid(2, me_rep), base_tps))
    rows.append(row("EP8 FP8 +placement+replication", hybrid(1, me_rep), base_tps))

    # + n-gram spec decode on top of the FP8 row: divides weight+comms by E[acc].
    fp8 = hybrid(1, me_rep)
    print("\n  + n-gram spec-decode sensitivity (k=4, 1 drafter) on the FP8 row:")
    for alpha in (0.4, 0.6, 0.8):
        e_acc = max(expected_accepted_multi(alpha, 4, 1), 1.0)
        tps = fp8.tokens_per_s * e_acc
        print(f"    alpha={alpha}: E[acc]={e_acc:.2f} tok/pass -> "
              f"{tps:.0f} tok/s ({tps/base_tps:.2f}x vs TP8)")

    print("\n  strategy                              weight  comms   kv   total  tok/s  imb  x")
    for r in rows:
        print(f"  {r['strategy']:<36} {r['weight_ms']:>5}  {r['comms_ms']:>5} "
              f"{r['kv_ms']:>4} {r['total_ms']:>6} {r['tok_per_s']:>6} "
              f"{r['imbalance']:>4} {r['speedup_vs_tp']:>4}")

    out = {"measured_max_experts": {"round_robin": me_rr, "placement": me_pl,
                                    "placement_replication": me_rep},
           "rows": rows, "ctx": ctx}
    json.dump(out, open("latency_projection.json", "w"), indent=2)
    print("\n-> latency_projection.json")


if __name__ == "__main__":
    main()
