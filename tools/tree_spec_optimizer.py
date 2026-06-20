#!/usr/bin/env python3
"""Tree-spec shape optimizer for the floor-bound MoE — find the optimal (width W × depth D) tree.

The team converged on TREE-spec as the #1 comms-amortized lever. spec_moe_model.py (N drafters × k) and
spec_floor_model.py (linear k, floor-aware) don't optimize the branching tree shape. This sweeps (W, D) and
picks the tree that maximizes realized speedup, **floor-aware**, per regime:

  positions   = W * D                                    # W candidates × D depth (verified in ONE forward)
  E[accepted] = geometric: p = 1-(1-alpha)^W per depth, over D levels
  union(P)    = 128*(1-(120/128)^(P*(1-overlap)))        # expert union the verify reads (Charles tax)
  verify_cost = F + (1-F)*(0.34 + 0.66*union/8)          # FLOOR-AWARE (the key term)
  speedup     = E[accepted] / verify_cost

At F=0.86 (measured floor-bound) the union tax barely bites (it's on the 14% weight) -> WIDE/DEEP trees win.
As the floor falls (F->0, weight-bound) the tax bites -> trees shrink. Route-overlap (spec_moe_model's
unclaimed lever) only matters once F is low. Output: the recommended tree per regime + per accept rate.

  python3 tools/tree_spec_optimizer.py
"""
E, TOPK = 128, 8
WIDTHS = [1, 2, 4, 8]
DEPTHS = [2, 3, 4, 6, 8]


def e_accepted(alpha, W, D):                  # tokens emitted/round incl. bonus (+1 form, validate_routing_model §4)
    p = 1.0 - (1.0 - alpha) ** W
    return float(D + 1) if p >= 1.0 else (1.0 - p ** (D + 1)) / (1.0 - p)


def union(positions, overlap=0.0):
    eff = max(positions * (1.0 - overlap), 1.0)
    return E * (1.0 - ((E - TOPK) / E) ** eff)


def verify_cost(positions, F, overlap=0.0):
    return F + (1.0 - F) * (0.34 + 0.66 * (union(positions, overlap) / TOPK))


def best_tree(alpha, F, overlap):
    best = (0.0, 1, 1)
    grid = {}
    for W in WIDTHS:
        for D in DEPTHS:
            sp = e_accepted(alpha, W, D) / verify_cost(W * D, F, overlap)
            grid[(W, D)] = sp
            if sp > best[0]:
                best = (sp, W, D)
    return best, grid


def main():
    print("Tree-spec optimal shape (floor-aware). speedup = E[accepted]/verify_cost; >1 = win.\n")
    for F, tag in ((0.86, "F=0.86  MEASURED floor-bound (bf16-TP8 today)"),
                   (0.50, "F=0.50  floor half-fixed"),
                   (0.00, "F=0.00  weight-bound (= spec_moe_model.py)")):
        print(f"=== {tag} ===")
        print(f"  {'alpha':>6} | {'naive best (W×D, spd)':>26} | {'route-aware .45 best':>26}")
        for alpha in (0.6, 0.7, 0.8):
            (s0, w0, d0), _ = best_tree(alpha, F, 0.0)
            (s1, w1, d1), _ = best_tree(alpha, F, 0.45)
            print(f"  {alpha:>6.1f} | {f'W{w0}×D{d0}: {s0:.2f}x':>26} | {f'W{w1}×D{d1}: {s1:.2f}x':>26}")
        print()
    print("Reading:")
    print("  - At the MEASURED F=0.86, big trees (wide AND deep) win ~2.5-3.5x NAIVE — no route-awareness")
    print("    needed (the union tax is on the 14% weight). EAGLE3's ~3.5 accept length maps to ~W4-8×D3-4.")
    print("  - As the floor is fixed (F->0), the optimum collapses to small trees (W1-2 × D2-3) and route-")
    print("    awareness (spec_moe_model's lever) becomes worth ~20-60% — it's a WEIGHT-BOUND lever, moot now.")
    print("  - Actionable: run EAGLE3 / n-gram with a WIDE+DEEP tree today; shrink + add route-awareness only")
    print("    after E0b/K5 push us toward weight-bound. The optimal tree is regime-adaptive (gate on tok/s).")


if __name__ == "__main__":
    main()
