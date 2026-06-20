#!/usr/bin/env python3
"""Floor-aware spec-decode go/no-go — extends tools/spec_moe_model.py with the FLOOR fraction.

spec_moe_model.py uses verify_cost = 0.34 + 0.66·(union/8) — the WEIGHT-units cost (assumes the verify cost
is the expert-weight read, i.e. floor=0 / weight-bound). It concludes naive big trees lose on MoE and you
need route-awareness. But the MEASURED engine is FLOOR-bound (floor ≈ 86% of a step: 188 all-reduces @16µs +
launch + host; weight only 14% — overhead-attribution.md). The verify pays the floor ONCE and the union tax
falls on the 14% weight term, so in real-time units:

    verify_cost(real) = F + (1-F)·(0.34 + 0.66·union/8)        # F = floor fraction of a decode step
    speedup           = E[accepted] / verify_cost(real)

At F=0 this reproduces spec_moe_model.py (weight-bound). At the measured F≈0.86 the union tax barely bites →
LARGE k wins WITHOUT route-awareness. This reverses the go/no-go in the floor-bound regime and makes the
optimal k regime-dependent (large now; shrink to 2-3 as the floor is fixed). See spec-decode-floor-bound.md.
"""
E, TOPK = 128, 8


def expected_accepted(alpha, k, n):           # tokens EMITTED/round incl. bonus (Leviathan); +1 vs the old form
    p = 1.0 - (1.0 - alpha) ** n              # (validate_routing_model.py §4; LOOP-A fixed spec_moe_model to match)
    return float(k + 1) if p >= 1.0 else (1.0 - p ** (k + 1)) / (1.0 - p)


def union(positions, overlap=0.0):            # Charles; route-aware overlap shrinks the union
    eff = positions * (1.0 - overlap)
    return E * (1.0 - ((E - TOPK) / E) ** max(eff, 1.0))


def verify_cost(positions, F, overlap=0.0):   # FLOOR-AWARE (the new bit)
    weight_units = 0.34 + 0.66 * (union(positions, overlap) / TOPK)   # spec_moe_model.py's weight cost
    return F + (1.0 - F) * weight_units


def main():
    cfgs = [(2, 1), (3, 1), (4, 1), (8, 1), (4, 2), (8, 2)]
    for F, tag in ((0.0, "F=0.00  (weight-bound = spec_moe_model.py)"),
                   (0.5, "F=0.50  (floor half-fixed)"),
                   (0.86, "F=0.86  (MEASURED floor-bound, bf16-TP8 today)")):
        print(f"\n=== {tag} ===  speedup = E[acc] / verify_cost(real);  >1 = win")
        print(f"  {'cfg':10s} {'E[acc]':>7} {'vc(naive)':>10} {'spd':>6}   {'vc(aware.45)':>12} {'spd':>6}")
        best = (0, "")
        for k, n in cfgs:
            ea = expected_accepted(0.7, k, n)
            pos = k * n
            vc0 = verify_cost(pos, F, 0.0); s0 = ea / vc0
            vc1 = verify_cost(pos, F, 0.45); s1 = ea / vc1
            tag0 = "WIN" if s0 > 1 else "loss"; tag1 = "WIN" if s1 > 1 else "loss"
            print(f"  k={k} N={n}     {ea:7.2f} {vc0:10.2f} {s0:5.2f} {tag0:4s} {vc1:12.2f} {s1:5.2f} {tag1}")
            if max(s0, s1) > best[0]: best = (max(s0, s1), f"k={k} N={n}")
        print(f"  -> best: {best[1]} at {best[0]:.2f}x")
    print("\nTakeaway: spec_moe_model.py's 'naive big trees lose, need route-awareness' is the WEIGHT-BOUND")
    print("(F=0) conclusion. At the measured F=0.86, naive k=8 already WINS (~2x) — the verify-tax is on the")
    print("14% weight term, not the 86% floor. So: run big-tree n-gram NOW; shrink k + add route-awareness")
    print("(spec_moe_model.py) as the floor falls toward weight-bound. Optimal k is regime-dependent.")


if __name__ == "__main__":
    main()
