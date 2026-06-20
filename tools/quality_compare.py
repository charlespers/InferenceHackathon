"""Compare two quality_probe outputs -> the adaptive-top-k quality gate.

Both runs are greedy (temperature 0), so an identical engine gives identical text.
Divergence measures the quality cost of dropping experts. Reports, per prompt and
overall: word-level common-prefix agreement and exact-match rate. Ship gate (per
docs/b1-latency-architecture.md avenue #10): adaptive is acceptable if outputs stay
close (high agreement) on the confident-token regime we target.

Usage:
    python3 tools/quality_compare.py q_baseline.json q_adaptive.json [--out gate.json]
"""
from __future__ import annotations
import argparse, json


def prefix_agreement(a: str, b: str) -> float:
    """Word-level longest-common-prefix fraction (of the baseline length)."""
    wa, wb = a.split(), b.split()
    if not wa:
        return 1.0 if not wb else 0.0
    i = 0
    while i < len(wa) and i < len(wb) and wa[i] == wb[i]:
        i += 1
    return i / len(wa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("adaptive")
    ap.add_argument("--out", default="quality_gate.json")
    a = ap.parse_args()
    base = json.load(open(a.baseline))["completions"]
    adpt = json.load(open(a.adaptive))["completions"]
    shared = [p for p in base if p in adpt]

    rows, agrees, exacts = [], [], 0
    for p in shared:
        ba, ad = base[p], adpt[p]
        ag = prefix_agreement(ba, ad)
        ex = (ba.strip() == ad.strip())
        exacts += int(ex)
        agrees.append(ag)
        rows.append({"prompt": p[:50], "agreement": round(ag, 3), "exact": ex})

    mean_ag = sum(agrees) / len(agrees) if agrees else 0.0
    exact_rate = exacts / len(shared) if shared else 0.0
    verdict = ("SHIP (outputs ~identical)" if mean_ag > 0.97
               else "REVIEW (some drift)" if mean_ag > 0.85
               else "FAIL (large drift)")
    print(f"=== adaptive-top-k quality gate ({len(shared)} prompts) ===")
    print(f"  mean word-prefix agreement : {mean_ag*100:.1f}%")
    print(f"  exact-match rate           : {exact_rate*100:.1f}%")
    print(f"  verdict                    : {verdict}")
    for r in sorted(rows, key=lambda r: r["agreement"])[:5]:
        print(f"    {r['agreement']*100:5.1f}%  {'=' if r['exact'] else '~'}  {r['prompt']}")
    out = {"n": len(shared), "mean_agreement": mean_ag, "exact_rate": exact_rate,
           "verdict": verdict, "rows": rows}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
