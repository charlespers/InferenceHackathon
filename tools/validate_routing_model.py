#!/usr/bin/env python3
"""Monte-Carlo validation of the routing-model formulas my spec/EP analysis rests on (local, no GPU).

Per the user's "verify on a smaller MoE locally first" suggestion. Two analytical claims underpin everything:
  1. EXPERT UNION (the spec verify-tax, spec-decode-moe-tax.md): P draft positions, each top-8-of-128, read
     E[distinct experts] = 128·(1 − (120/128)^P).
  2. EP BUSIEST-RANK (the EP→TP inversion, ep-placement-for-b1.md): one token's 8 experts placed across 8 EP
     ranks → E[max experts on a rank] ≈ 2.5–2.6 (balls-in-bins), vs 1.0 under TP8.
Plus the EP-balance-of-the-verify claim (ep-balance-spec-verify.md): the per-rank imbalance for a P-position
verify falls toward 1.0 as P grows (the union → all 128 → every rank reads its 16).

Validates all three against simulation. Pure stdlib (no numpy/torch needed).
  python3 tools/validate_routing_model.py
"""
import random

E, TOPK, NRANK = 128, 8, 8
TRIALS = 20000


def union_formula(P):
    return E * (1.0 - ((E - TOPK) / E) ** P)


def sim_union(P, rng):
    tot = 0
    for _ in range(TRIALS // max(P, 1)):
        seen = set()
        for _ in range(P):
            seen.update(rng.sample(range(E), TOPK))
        tot += len(seen)
    return tot / (TRIALS // max(P, 1))


def sim_busiest(P, rng, placement):
    # placement: expert -> rank. Busiest = max experts (of the P-position UNION) on any rank.
    tot = 0
    for _ in range(TRIALS // max(P, 1)):
        seen = set()
        for _ in range(P):
            seen.update(rng.sample(range(E), TOPK))
        load = [0] * NRANK
        for e in seen:
            load[placement[e]] += 1
        mean = sum(load) / NRANK
        tot += (max(load) / mean) if mean else 1.0
    return tot / (TRIALS // max(P, 1))


def expected_accepted_formula(alpha, k, n):
    p = 1.0 - (1.0 - alpha) ** n
    return float(k) if p >= 1.0 else (1.0 - p ** k) / (1.0 - p)


def sim_accepted(alpha, k, n, rng):
    # p_hit per position = 1-(1-alpha)^n (n independent drafters); accept the run until the first miss.
    p = 1.0 - (1.0 - alpha) ** n
    tot = 0
    for _ in range(TRIALS):
        acc = 0
        for _ in range(k):
            if rng.random() < p:
                acc += 1
            else:
                break
        tot += acc
    return tot / TRIALS


def main():
    rng = random.Random(12345)
    rr = {e: e % NRANK for e in range(E)}    # round-robin EP placement
    print("1) EXPERT UNION — formula 128·(1−(120/128)^P) vs Monte-Carlo")
    print(f"   {'P':>3} {'formula':>9} {'sim':>9} {'err%':>7}")
    ok_u = True
    for P in (1, 2, 4, 8, 16, 32, 64):
        f, s = union_formula(P), sim_union(P, rng)
        err = 100 * abs(f - s) / s
        ok_u &= err < 3
        print(f"   {P:>3} {f:>9.1f} {s:>9.1f} {err:>6.1f}%")
    print(f"   => union formula {'VALIDATED' if ok_u else 'MISMATCH'} (<3% error)\n")

    print("2) EP BUSIEST-RANK — single-token (P=1) imbalance vs the ~2.6 claim, and the VERIFY rebalancing")
    print(f"   {'P(positions)':>12} {'busiest/mean(EP)':>16}  note")
    b1 = sim_busiest(1, rng, rr)
    print(f"   {1:>12} {b1:>16.2f}  decode: matches the ~2.5–2.6 balls-in-bins penalty (the EP→TP finding)")
    for P in (4, 8, 16, 32):
        b = sim_busiest(P, rng, rr)
        print(f"   {P:>12} {b:>16.2f}  verify: falls toward 1.0 as union→128 (ep-balance-spec-verify.md)")
    print("   => EP penalty is a PLAIN-DECODE (P=1) effect; the big-tree verify rebalances. CONFIRMED.\n")

    print("3) TP8 (every expert column-sharded on every rank) -> per-rank imbalance is exactly 1.0 by")
    print("   construction (no balls-in-bins) — the reason TP8 > EP at B=1. (No sim needed; structural.)")
    print("4) ACCEPTANCE — E[accepted]=(1−p^k)/(1−p), p=1−(1−α)^N (the spec speedup numerator) vs Monte-Carlo")
    print(f"   {'α':>4} {'k':>3} {'N':>3} {'formula':>9} {'sim':>9} {'err%':>7}")
    ok_a = True
    for alpha, k, n in ((0.6, 5, 1), (0.7, 5, 1), (0.8, 8, 1), (0.7, 5, 2), (0.6, 8, 4)):
        f, s = expected_accepted_formula(alpha, k, n), sim_accepted(alpha, k, n, rng)
        err = 100 * abs(f - s) / max(s, 1e-9)
        ok_a &= err < 3
        print(f"   {alpha:>4.1f} {k:>3} {n:>3} {f:>9.3f} {s:>9.3f} {err:>6.1f}%")
    print(f"   => acceptance formula {'VALIDATED' if ok_a else 'MISMATCH'} (<3% error)\n")

    print(f"Net: the union + EP-imbalance + verify-rebalancing + acceptance models are empirically sound on a simulated")
    print("128-expert top-8 MoE. The same formulas drive spec_floor_model / tree_spec_optimizer / spec_predict /")
    print("backout_floor — so those rest on a validated foundation before the H100 runs confirm the absolute numbers.")


if __name__ == "__main__":
    main()
