"""Route-aware (multi-)speculative decoding model for the comms-bound 235B MoE.

Synthesizes three results into one go/no-go model:
  - Charles (docs/spec-decode-moe-tax.md): verifying N*k draft positions reads the
    UNION of experts they touch -> verify_cost = 0.34 + 0.66*(E[union]/8). Naive big
    trees lose on MoE.
  - jminding (inferutil.speculative): E[accepted tokens/round] for k draft, N drafters.
  - djamoils (route data): consecutive tokens share ~45% of their top-8 experts
    (persistence 44.6%). If the DRAFT is route-aware (tokens chosen/biased to overlap
    in expert space), the verify union shrinks -> the tax collapses -> wider/multi
    speculation becomes net-positive. This is the unclaimed lever.

Why it matters at B=1: decode is COMMS-bound (~188 serial all-reduces). Spec-decode
verifies E[accepted] tokens with ONE set of collectives, so speedup ~ E[accepted] /
verify_cost. Route-awareness is what lets us push E[accepted] up without verify_cost
exploding on the MoE.

Usage: python3 tools/spec_moe_model.py
"""
from __future__ import annotations

E = 128          # experts
TOPK = 8         # active per token
ROUTED_BYTE_SHARE = 0.66   # routed experts' share of decode bytes (Charles)


def expected_accepted(alpha: float, k: int, n: int) -> float:
    """Tokens EMITTED per round (the spec throughput numerator), N independent drafters, tree verify.
    p_hit = 1-(1-alpha)^N per position. Leviathan: emitted = (1-p^{k+1})/(1-p) — INCLUDES the always-
    emitted bonus token (the target's resampled token at the first mismatch). The old (1-p^k)/(1-p)
    omitted the bonus and understated speedup by ~p^k (37% at k=1, ~3-6% at k=5-8) — bug caught by
    Charles' Monte-Carlo (validate_routing_model.py, 2026-06-20)."""
    p = 1.0 - (1.0 - alpha) ** n
    if p >= 1.0:
        return float(k + 1)
    return (1.0 - p ** (k + 1)) / (1.0 - p)


def union_naive(positions: int) -> float:
    """E[distinct experts] over `positions` independent top-8-of-128 draws (Charles)."""
    return E * (1.0 - ((E - TOPK) / E) ** positions)


def union_route_aware(positions: int, overlap: float) -> float:
    """Route-aware union: each extra drafted position, being route-correlated with
    the run, adds only (1-overlap)*TOPK NEW experts on average (vs TOPK independent).
    overlap=0 -> ~naive-ish upper bound; overlap=1 -> all positions share one set."""
    if positions <= 1:
        return float(TOPK)
    u = TOPK + (positions - 1) * TOPK * (1.0 - overlap)
    return min(u, union_naive(positions))   # never worse than independent


def verify_cost(union: float) -> float:
    return (1.0 - ROUTED_BYTE_SHARE) + ROUTED_BYTE_SHARE * (union / TOPK)


def speedup(alpha: float, k: int, n: int, overlap: float) -> tuple[float, float, float]:
    e_acc = expected_accepted(alpha, k, n)
    u = union_route_aware(n * k, overlap)
    vc = verify_cost(u)
    return e_acc / vc, e_acc, vc


def main():
    # alpha: per-token draft acceptance. n-gram on code ~0.5-0.7; small draft model ~0.6-0.8.
    print("=== route-aware (multi-)spec on Qwen3-235B MoE (comms-bound) ===")
    print("speedup ~ E[accepted] / verify_cost ;  >1 = net win over plain decode\n")
    for alpha in (0.5, 0.7):
        print(f"alpha={alpha}:  (naive overlap=0  vs  route-aware overlap=0.45 measured)")
        print(f"  {'cfg':<12}{'E[acc]':>7}{'  naive: vc  spd':>18}{'  aware: vc  spd':>18}")
        for (k, n) in [(2, 1), (3, 1), (4, 1), (8, 1), (2, 2), (4, 2), (2, 4)]:
            sp0, e0, vc0 = speedup(alpha, k, n, 0.0)
            sp1, e1, vc1 = speedup(alpha, k, n, 0.45)
            tag = f"k={k} N={n}"
            flag0 = "WIN" if sp0 > 1 else "loss"
            flag1 = "WIN" if sp1 > 1 else "loss"
            print(f"  {tag:<12}{e1:>7.2f}   {vc0:>5.2f} {sp0:>5.2f} {flag0:<4}"
                  f"   {vc1:>5.2f} {sp1:>5.2f} {flag1}")
        print()
    # Headline: best config under each regime
    best = {}
    for label, ov in (("naive", 0.0), ("route-aware(0.45)", 0.45)):
        rows = [(speedup(0.7, k, n, ov)[0], k, n)
                for (k, n) in [(2, 1), (3, 1), (4, 1), (8, 1), (2, 2), (4, 2), (2, 4)]]
        sp, k, n = max(rows)
        best[label] = (sp, k, n)
        print(f"best @alpha=0.7 {label:<18}: speedup {sp:.2f}x  at k={k} N={n}")
    gain = best["route-aware(0.45)"][0] / best["naive"][0]
    print(f"\nroute-awareness lifts the best achievable spec speedup by "
          f"{(gain-1)*100:.0f}% and unlocks wider/multi trees that naive MoE spec can't use.")
    print("Caveat: needs a drafter whose tokens are route-correlated with the target "
          "(self-spec or a draft trained/biased to match routes); overlap=0.45 is the "
          "token-to-token measured ceiling, achievable overlap TBD on real drafts.")
    print("\n*** REGIME CAVEAT (read before trusting the above) ***")
    print("This model is WEIGHT-BOUND (floor=0): verify_cost scales fully with the expert")
    print("union. The MEASURED engine is ~86% FLOOR-bound, so the union tax falls on only the")
    print("~14% weight term and the route-aware lift SHRINKS toward marginal — see")
    print("charles-work tools/spec_floor_model.py (floor-aware) which reverses 'big trees lose'.")
    print("The lever is therefore MEASUREMENT-GATED: decide on V=tau/S from the FP8 EAGLE3 slot")
    print("(experiments/eagle3/ROUTE_AWARE_DECISION.md). V~1 => floor hides the tax => NO-GO;")
    print("V>=1.3 & rising with tree size => GO. Do not implement route-aware shaping on this")
    print("weight-bound model alone.")


if __name__ == "__main__":
    main()
