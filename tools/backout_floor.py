#!/usr/bin/env python3
"""Back out the floor fraction F from EAGLE3 spec measurements at multiple tree sizes.

Extends LOOP-A's V=τ/S probe (experiments/eagle3/ROUTE_AWARE_DECISION.md). The effective verify cost in
decode-step units is V = τ/S (τ=accept length, S=realized speedup). My floor-aware model:

    V(k) = F + (1-F)·(0.34 + 0.66·union(k)/8)            # F = floor fraction; union(k) = E[distinct experts]

union(k) grows with the tree size k (num_speculative_tokens). So measuring V at TWO+ tree sizes over-determines
F: subtract two equations and the (1-F) factor falls out. This turns the 08:45 spec run into a *quantitative*
floor measurement (no Nsight needed) that cross-checks E-attr/overhead-attribution.md — and pins LOOP-A's
go/no-go (route-aware shaping pays iff the union term (1-F)·0.66·union/8 is a real share of V).

Usage: edit MEAS = [(k, tau, S), ...] with the measured triples, then:  python3 tools/backout_floor.py
"""
E, TOPK = 128, 8

# (num_speculative_tokens k, accept-length tau, realized speedup S=tok/s_spec/tok/s_baseline)
# Placeholder example numbers — REPLACE with the 08:45 measurements (≥2 tree sizes).
MEAS = [
    (2, 1.7, 1.45),
    (5, 2.8, 1.9),
    (8, 3.3, 2.0),
]


def union(k):
    return E * (1.0 - ((E - TOPK) / E) ** k)


def w(k):  # weight-units verify cost for k positions (spec_moe_model.py)
    return 0.34 + 0.66 * (union(k) / TOPK)


def fit_F(triples):
    """Least-squares F from [(k, tau, S)]: V=tau/S, V = F + (1-F)·w(k) -> (V-w) = F·(1-w). Returns F or None."""
    xs, ys = [], []
    for k, tau, S, in [(t[0], t[1], t[2]) for t in triples]:
        V, wk = tau / S, w(k)
        xs.append(1 - wk); ys.append(V - wk)
    den = sum(x * x for x in xs)
    if den < 1e-9:
        return None
    return sum(x * y for x, y in zip(xs, ys)) / den


def selftest():
    """Round-trip: pick F_true, synthesize (k, tau, S) with S = tau / (F_true + (1-F_true)·w(k)), recover F."""
    ok = True
    print("backout_floor self-test (known F -> synthetic V=tau/S -> recovered F):")
    for F_true in (0.86, 0.50, 0.20):
        triples = []
        for k in (2, 3, 5, 8):
            tau = 1.0 + 0.3 * k                      # arbitrary accept length; F is independent of it
            V = F_true + (1 - F_true) * w(k)
            triples.append((k, tau, tau / V))        # S = tau / V
        F = fit_F(triples)
        err = abs(F - F_true)
        ok &= err < 1e-6
        print(f"  F_true={F_true:.2f} -> recovered {F:.6f}  ({'PASS' if err < 1e-6 else 'FAIL'})")
    print(f"=> backout_floor inversion {'VALIDATED' if ok else 'BROKEN'} (exact on noise-free data).")
    return ok


def main():
    import sys
    if "--selftest" in sys.argv:
        selftest(); return
    rows = [(k, tau, S, tau / S, w(k)) for (k, tau, S) in MEAS]
    print(f"  {'k':>3} {'tau':>5} {'S':>5} {'V=tau/S':>8} {'union':>6} {'w(k)':>6}")
    for k, tau, S, V, wk in rows:
        print(f"  {k:>3} {tau:>5.2f} {S:>5.2f} {V:>8.3f} {union(k):>6.1f} {wk:>6.2f}")
    F = fit_F(MEAS)
    if F is None:
        print("\n  need ≥2 DISTINCT tree sizes (different union) to separate F from the weight term."); return
    print(f"\n  ==> backed-out floor fraction F = {F:.3f}   (overhead-attribution.md measured ~0.86)")
    if F > 0.7:
        print("  FLOOR-BOUND: the verify is hidden under the floor (V≈1). Route-aware tree-shaping saves ~nothing")
        print("  -> LOOP-A route-aware = NO-GO now; revisit as comms/graphs/FP8 push F down. Big naive trees win.")
    elif F > 0.4:
        print("  TRANSITIONAL: the union term is becoming real -> route-aware shaping has partial headroom.")
    else:
        print("  WEIGHT-BOUND: the union term dominates -> route-aware shaping is GO (it shrinks the taxed union).")
    # sanity: predicted V at each k from the fitted F
    print("\n  fit check (V_pred vs V_meas):")
    for k, tau, S, V, wk in rows:
        print(f"    k={k}: V_pred={F + (1-F)*wk:.3f}  V_meas={V:.3f}")


if __name__ == "__main__":
    main()
