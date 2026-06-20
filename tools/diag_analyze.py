#!/usr/bin/env python3
"""Analyze the 3-arm graphs DIAGNOSTIC (tools/slot_graphs_diag.sh) — separates "does CUDA-graphs
speed up plain B=1 decode" from "spec speedup under graphs", and reads out the verify expert union.

Reads from --dir (default /alloc/data/eagle3_diag):
  m_baseline_eager.json    decode tok/s, plain decode, eager
  m_baseline_graphs.json   decode tok/s, plain decode, graphs
  m_eagle3_graphs.json     decode tok/s, EAGLE3 spec, graphs (capture-sizes fixed)
  metrics_eagle3_graphs.txt  Prometheus scrape -> accept-length tau
  parity_gate_graphs.json  lossless verdict (eagle3 vs baseline_graphs greedy)

Reports:
  S_graph_plain = base_graphs / base_eager   -> does graphs help PLAIN decode? (the foundational claim)
  S_spec        = eagle3_graphs / base_graphs -> spec speedup once graphs is truly engaged
  S_e2e         = eagle3_graphs / base_eager   -> end-to-end spec+graphs vs eager
  V = tau / S_spec  (verify cost in decode-step units) and the back-solved verify expert union
  route-aware go/no-go (V~1 floor-bound NO-GO; V>>1 union-taxed GO), vs the bf16-best 85.7 tok/s.

Mirrors engine/src/spec/projection.rs (verify_cost / back_solve_graphs_union) so the box analysis and
the engine model agree. Stdlib-only.
"""
from __future__ import annotations
import argparse, json, os, re

BF16_BEST_TOK_S = 85.7          # team's best non-spec (bf16-TP8); FP8 is ~25% slower at B=1
TOPK = 8.0
NONEXPERT_SHARE, ROUTED_SHARE = 0.34, 0.66


def load(path):
    try:
        return json.load(open(path))
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}", "_path": path}


def tok_s(m):
    if not isinstance(m, dict):
        return None
    if m.get("decode_tok_s") is not None:
        return m["decode_tok_s"]
    return (m.get("median") or {}).get("decode_tok_s")


def accept_len(metrics_text):
    """tau = 1 + accepted/drafts from a Prometheus scrape (names drift; match loosely)."""
    if not metrics_text:
        return None
    vals = {}
    for line in metrics_text.splitlines():
        m = re.match(r"([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
        if m:
            try:
                vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(2))
            except ValueError:
                pass

    def find(*subs):
        for k, v in vals.items():
            if all(s in k.lower() for s in subs):
                return v
        return None

    mean_al = find("acceptance", "length") or find("mean", "accept")
    if mean_al:
        return mean_al
    acc, drafts = find("spec_decode", "accepted") or find("accepted", "token"), \
                  find("spec_decode", "num_drafts") or find("spec_decode", "drafts")
    if acc is not None and drafts:
        return 1.0 + acc / drafts
    return None


def back_solve_union(tau, s_spec, f, draft_graph_cost=0.2):
    """Invert verify_cost: given measured spec S and floor f, recover the verify expert union.
    Mirrors RoundCostModel::back_solve_graphs_union. None if floor-saturated or non-physical."""
    if tau is None or s_spec in (None, 0) or (1.0 - f) < 1e-3:
        return None
    verify = tau / s_spec - draft_graph_cost
    if verify <= 0:
        return None
    weight_units = (verify - f) / (1.0 - f)
    return max(8.0, 8.0 * (weight_units - NONEXPERT_SHARE) / ROUTED_SHARE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/alloc/data/eagle3_diag")
    ap.add_argument("--f-graphs", type=float, default=0.40, help="residual graphs floor estimate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = a.dir

    be = tok_s(load(os.path.join(d, "m_baseline_eager.json")))
    bg = tok_s(load(os.path.join(d, "m_baseline_graphs.json")))
    eg = tok_s(load(os.path.join(d, "m_eagle3_graphs.json")))
    mpath = os.path.join(d, "metrics_eagle3_graphs.txt")
    tau = accept_len(open(mpath).read() if os.path.exists(mpath) else "")
    parity = load(os.path.join(d, "parity_gate_graphs.json"))

    def ratio(n, dn):
        return round(n / dn, 3) if (n and dn) else None

    s_graph_plain = ratio(bg, be)
    s_spec = ratio(eg, bg)
    s_e2e = ratio(eg, be)
    v = round(tau / s_spec, 3) if (tau and s_spec) else None
    union = back_solve_union(tau, s_spec, a.f_graphs)

    print("=== GRAPHS DIAGNOSTIC FINDINGS ===")
    print(f"  tok/s:  baseline_eager={be}  baseline_graphs={bg}  eagle3_graphs={eg}")
    print(f"  S_graph_plain (graphs/eager, plain decode) = {s_graph_plain}   "
          f"-> {'graphs HELPS plain decode' if (s_graph_plain or 0) > 1.15 else 'graphs ~flat/worse on plain decode'}")
    print(f"  S_spec (eagle3/baseline, graphs)           = {s_spec}")
    print(f"  S_e2e (eagle3 graphs / baseline eager)     = {s_e2e}")
    print(f"  tau (accept length)                        = {tau}")
    if v is not None:
        verdict = "floor-bound -> route-aware NO-GO" if v < 1.3 else "union-taxed -> route-aware GO"
        print(f"  V = tau/S_spec                             = {v}   ({verdict})")
    if union is not None:
        print(f"  back-solved verify union (f={a.f_graphs})           = {round(union,1)} experts "
              f"({'tight ~route-aware-friendly' if union < 16 else 'wide -> route-aware headroom'})")
    if eg:
        print(f"  EAGLE3-absolute vs bf16-best {BF16_BEST_TOK_S}: {round(eg/BF16_BEST_TOK_S,3)}x "
              f"(FP8 target carries the ~25% B=1 handicap)")
    pv = parity.get("identical", parity.get("match", parity.get("lossless")))
    print(f"  lossless parity (eagle3 vs baseline_graphs greedy): {pv}  {parity.get('_error','')}")

    if a.out:
        json.dump({"baseline_eager": be, "baseline_graphs": bg, "eagle3_graphs": eg,
                   "S_graph_plain": s_graph_plain, "S_spec": s_spec, "S_e2e": s_e2e,
                   "tau": tau, "V": v, "back_solved_union": union, "parity": pv},
                  open(a.out, "w"), indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
