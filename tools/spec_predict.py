#!/usr/bin/env python3
"""Unified B=1 spec-decode speedup predictor — the one model that combines every refinement.

Consolidates spec_floor_model.py (floor-aware verify cost) + tree_spec_optimizer.py (W×D tree) and adds the
three refinements scattered across docs:
  - EAGLE3 DRAFT COST (eagle3-draft-tp.md): draft = draft_ms_per_step × D; draft_tp=8 ⇒ ~0.11ms/step,
    draft_tp=1 ⇒ ~0.6ms/step, n-gram ⇒ 0. Added to the denominator (free-draft models omit it).
  - TEMPERATURE (spec-in-production.md): temp>0 lowers acceptance (greedy α → ~0.8α at temp 0.7).
  - EP-BALANCE (ep-balance-spec-verify.md): on EP, the big-tree verify reads the full union → every rank
    reads all its experts → the imbalance factor on the weight term falls toward 1.0 as union→128 (on TP it's
    flat 1.0). So a big tree is CHEAPER on EP than the naive union model says.

    speedup = E[accepted] / (verify_cost + draft_cost)
    verify_cost = F + (1-F)·(0.34 + 0.66·(union/8)·ep_factor)
    draft_cost  = draft_ms_per_step · D / decode_step_ms

Usage:
  python3 tools/spec_predict.py                       # sweep the headline configs
  python3 tools/spec_predict.py --F 0.86 --alpha 0.7 --layout ep --draft-tp 8 --temp 0.0
"""
import argparse

E, TOPK, NRANK = 128, 8, 8


def expected_accepted(alpha, W, D):
    # tokens EMITTED/round incl. the always-emitted bonus = (1−p^{D+1})/(1−p) (Leviathan; validate_routing_model
    # §4). LOOP-A fixed spec_moe_model to match; this is the consistent +1 form. (Was (1−p^D)/(1−p), ~p^D low.)
    p = 1.0 - (1.0 - alpha) ** W
    return float(D + 1) if p >= 1.0 else (1.0 - p ** (D + 1)) / (1.0 - p)


def union(positions):
    return E * (1.0 - ((E - TOPK) / E) ** positions)


def ep_factor(u, layout):
    """Weight-term imbalance multiplier (busiest-rank / mean). TP: 1.0 (column-sharded, balanced). EP: the u
    union-experts are a subset of 128 spread over 8 ranks (16 slots each) -> hypergeometric busiest/mean,
    which decays from ~2.5 (u≈8) to 1.0 as u→128 (finite-population correction: at u=128 every rank reads all
    16). The form below is FITTED+VALIDATED against Monte-Carlo (validate_routing_model.py §2): predicts
    2.36/1.65/1.26/1.13 at u=8/29/82/112 vs sim 2.54/1.69/1.25/1.12. (Replaces the old crude linear decay,
    which over-penalized EP at mid-unions — caught by the local validation.)"""
    if layout != "ep" or u >= E:
        return 1.0
    return 1.0 + 1.5 * (7.0 * (E - u) / (127.0 * max(u, 1.0))) ** 0.5


def draft_ms_per_step(mode, draft_tp):
    if mode in ("ngram", "lookahead"):
        return 0.0     # n-gram = free lookup; lookahead = the model verifies its own Jacobi window (no draft net)
    # EAGLE3 1B head ~2GB bf16: /1 ⇒ ~0.60ms read; /8 ⇒ ~0.075ms + ~0.03ms all-reduce ≈ 0.11ms (eagle3-draft-tp.md)
    return 0.11 if draft_tp == 8 else 0.60


def predict(F, alpha, W, D, layout, mode, draft_tp, temp, decode_step_ms=11.67):
    a = alpha * (1.0 - 0.28 * (temp > 0) * (temp / 0.7))    # temp 0.7 ⇒ ~0.72α (spec-in-production.md)
    a = max(0.05, min(a, 0.99))
    ea = expected_accepted(a, W, D)
    pos = W * D
    u = union(pos)
    verify = F + (1.0 - F) * (0.34 + 0.66 * (u / TOPK) * ep_factor(u, layout))
    draft = draft_ms_per_step(mode, draft_tp) * D / decode_step_ms
    return ea / (verify + draft), {"alpha_eff": round(a, 3), "E_acc": round(ea, 2),
                                   "union": round(u, 1), "verify": round(verify, 3), "draft": round(draft, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--F", type=float, default=0.86); ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--layout", choices=["tp", "ep"], default="tp")
    ap.add_argument("--mode", choices=["eagle3", "ngram", "lookahead"], default="eagle3")
    ap.add_argument("--draft-tp", type=int, default=8); ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()
    print(f"F={a.F} alpha={a.alpha} layout={a.layout} mode={a.mode} draft_tp={a.draft_tp} temp={a.temp}")
    print(f"  {'tree W×D':10} {'speedup':>8}   detail")
    best = (0, "")
    for W, D in [(1, 3), (1, 5), (1, 8), (2, 5), (4, 5), (4, 8), (8, 8)]:
        sp, det = predict(a.F, a.alpha, W, D, a.layout, a.mode, a.draft_tp, a.temp)
        print(f"  W{W}×D{D:<7} {sp:>7.2f}x   {det}")
        if sp > best[0]:
            best = (sp, f"W{W}×D{D}")
    print(f"  -> best: {best[1]} at {best[0]:.2f}x")
    # the actionable contrasts
    print("\nContrasts (best speedup):")
    for tag, kw in [("EAGLE3 draft_tp=8", dict(mode="eagle3", draft_tp=8)),
                    ("EAGLE3 draft_tp=1", dict(mode="eagle3", draft_tp=1)),
                    ("n-gram (free draft)", dict(mode="ngram", draft_tp=8)),
                    ("lookahead (draft-free, α=0.55)", dict(mode="lookahead", draft_tp=8, alpha=0.55)),
                    ("temp 0.7 (product)", dict(mode="eagle3", draft_tp=8, temp=0.7))]:
        kw2 = dict(layout=a.layout, F=a.F, alpha=a.alpha); kw2.update({k: v for k, v in kw.items()})
        # HONEST tree set: EAGLE3/n-gram can spread WIDTH (independent drafters/candidates); lookahead's Jacobi
        # window is a sequential n-gram match with NO independent width, so cap it to W=1 (depth-only) — else
        # the tree model over-credits it. This is why lookahead's honest floor-bound τ is ~1.5–2.5, not ~3.
        trees = [(1, 5), (1, 8)] if kw2.get("mode") == "lookahead" else [(1, 5), (4, 5), (4, 8), (8, 8)]
        b = max(predict(kw2["F"], kw2["alpha"], W, D, kw2["layout"], kw2.get("mode", "eagle3"),
                        kw2.get("draft_tp", 8), kw2.get("temp", 0.0))[0]
                for W, D in trees)
        print(f"  {tag:24} {b:.2f}x")


if __name__ == "__main__":
    main()
