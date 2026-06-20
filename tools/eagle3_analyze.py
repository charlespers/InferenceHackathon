"""Analyze an EAGLE3 slot run -> speedup, accept-length, the V=tau/S floor probe,
and the route-aware go/no-go verdict. Writes FINDINGS to stdout + a JSON summary.

Inputs (from /alloc/data/eagle3/ produced by tools/slot_eagle3.sh), MODE in {eager,graphs}:
  m_eagle3_<MODE>.json     measure_baseline output for EAGLE3
  m_baseline_<MODE>.json   measure_baseline output for the matched baseline
  metrics_eagle3_<MODE>.txt  Prometheus /metrics scrape (spec counters -> accept-length)
  parity_gate_<MODE>.json  quality_compare verdict (lossless check)

Definitions:
  S (speedup)         = tok/s(EAGLE3) / tok/s(baseline)        [matched mode]
  tau (accept-length) = 1 + accepted_tokens / num_drafts        [tokens emitted per target fwd]
  V (verify cost)     = tau / S   in units of one normal decode step
    V ~ 1   -> floor-bound verify: the N*k batch is hidden under the per-step floor;
               EAGLE3 over-delivers (speedup ~ tau) and the MoE union tax barely bites
               => route-aware tree-shaping is NO-GO in this regime.
    V >> 1  -> the verify weight/union term is real (verify costs >1 step) => the union
               IS large enough to tax => route-aware shaping has headroom (GO).

Usage:
  python3 tools/eagle3_analyze.py --dir /alloc/data/eagle3 --mode eager --out FINDINGS.json
"""
from __future__ import annotations
import argparse, json, os, re

# Reference: the team's BEST measured non-spec baseline is bf16-TP8 = 85.7 tok/s (Alyssa,
# docs/config-sweep.md), NOT FP8. FP8 is ~25% SLOWER at B=1 (FP8+EP 64.5, FP8-otf 69.0) —
# overhead-dominated regime + dequant cost. EAGLE3's head verifier is pinned to the FP8 target,
# so EAGLE3 MUST run on FP8: its matched baseline is FP8 (clean spec speedup S), but the real-world
# win is EAGLE3-absolute vs this bf16 best. Report BOTH so FP8's handicap isn't hidden.
BF16_BEST_TOK_S = 85.7


def load_json(path):
    try:
        return json.load(open(path))
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}", "_path": path}


def tok_s(measure):
    """Pull decode_tok_s from a measure_baseline.py output (median or top-level)."""
    if not isinstance(measure, dict):
        return None
    if measure.get("decode_tok_s") is not None:
        return measure["decode_tok_s"]
    med = measure.get("median") or {}
    return med.get("decode_tok_s")


def parse_accept_length(metrics_text):
    """Recover accept-length tau from a Prometheus /metrics scrape.

    vLLM v1 exposes spec counters; names have drifted across versions, so match
    loosely. tau = 1 + accepted/drafts (each target forward emits accepted drafts
    + 1 bonus token). Returns (tau, detail-dict)."""
    if not metrics_text:
        return None, {"reason": "no metrics text"}
    vals = {}
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        try:
            vals[name] = vals.get(name, 0.0) + float(val)
        except ValueError:
            pass

    def find(*subs):
        for k, v in vals.items():
            kl = k.lower()
            if all(s in kl for s in subs):
                return k, v
        return None, None

    # accepted tokens
    ka, accepted = find("spec_decode", "accepted")
    if ka is None:
        ka, accepted = find("accepted", "token")
    # number of drafts / draft steps (denominator)
    kd, drafts = find("spec_decode", "num_drafts")
    if kd is None:
        kd, drafts = find("spec_decode", "drafts")
    # explicit acceptance-length / mean if vLLM exposes it directly
    km, mean_al = find("acceptance", "length")
    if km is None:
        km, mean_al = find("mean", "accept")

    detail = {"matched": {k: v for k, v in vals.items()
                          if "spec" in k.lower() or "accept" in k.lower() or "draft" in k.lower()}}
    if mean_al:
        detail["source"] = km
        return float(mean_al), detail
    if accepted is not None and drafts not in (None, 0):
        tau = 1.0 + accepted / drafts
        detail["source"] = f"1 + {ka}/{kd}"
        return tau, detail
    detail["reason"] = "no usable accepted/drafts counters found"
    return None, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/alloc/data/eagle3")
    ap.add_argument("--mode", default="eager", choices=["eager", "graphs"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d, mode = a.dir, a.mode

    m_e = load_json(os.path.join(d, f"m_eagle3_{mode}.json"))
    m_b = load_json(os.path.join(d, f"m_baseline_{mode}.json"))
    parity = load_json(os.path.join(d, f"parity_gate_{mode}.json"))
    metrics_path = os.path.join(d, f"metrics_eagle3_{mode}.txt")
    metrics_text = open(metrics_path).read() if os.path.exists(metrics_path) else ""

    ts_e, ts_b = tok_s(m_e), tok_s(m_b)
    S = (ts_e / ts_b) if (ts_e and ts_b) else None          # clean spec speedup (matched FP8)
    win_vs_bf16 = (ts_e / BF16_BEST_TOK_S) if ts_e else None  # real-world win vs the team's best
    tau, tau_detail = parse_accept_length(metrics_text)     # MEASURED emitted/step (bonus included)
    V = (tau / S) if (tau and S) else None

    # route-aware go/no-go (decision rule from ROUTE_AWARE_DECISION.md)
    if V is None:
        verdict = "INCONCLUSIVE — missing tok/s or accept-length; inspect raw files"
    elif V < 1.3:
        verdict = (f"NO-GO (V={V:.2f}<1.3): floor-bound verify, MoE union tax hidden — "
                   "route-aware shaping wouldn't pay in this regime. Kill it honestly.")
    else:
        verdict = (f"GO-candidate (V={V:.2f}>=1.3): verify costs >1 step, union tax is real — "
                   "measure V vs num_speculative_tokens; if rising, build union-capped tree.")

    parity_ok = None
    if isinstance(parity, dict) and "exact_rate" in parity:
        parity_ok = parity.get("exact_rate", 0) >= 0.999

    summary = {
        "mode": mode,
        "tok_s_eagle3": ts_e, "tok_s_baseline": ts_b,
        "speedup_S_vs_fp8_matched": round(S, 3) if S else None,
        "eagle3_abs_vs_bf16_best": round(win_vs_bf16, 3) if win_vs_bf16 else None,
        "bf16_best_tok_s": BF16_BEST_TOK_S,
        "accept_length_tau": round(tau, 3) if tau else None,
        "verify_cost_V": round(V, 3) if V else None,
        "lossless_parity": parity.get("verdict") if isinstance(parity, dict) else None,
        "parity_exact_ok": parity_ok,
        "route_aware_verdict": verdict,
        "tau_detail": tau_detail,
    }

    print("=== EAGLE3 slot analysis (mode={}) ===".format(mode))
    print(f"  baseline tok/s : {ts_b}")
    print(f"  EAGLE3   tok/s : {ts_e}")
    print(f"  speedup  S     : {summary['speedup_S_vs_fp8_matched']}  (vs matched FP8 baseline, {mode})")
    print(f"  EAGLE3 vs bf16 : {summary['eagle3_abs_vs_bf16_best']}x  (abs tok/s vs team best 85.7; FP8 is a ~25% handicap)")
    print(f"  accept-len tau : {summary['accept_length_tau']}   [{tau_detail.get('source','?')}]")
    print(f"  verify cost V  : {summary['verify_cost_V']}   (= tau/S; ~1 floor-bound, >1 tax real)")
    print(f"  lossless       : {summary['lossless_parity']}  (exact_ok={parity_ok})")
    print(f"  ROUTE-AWARE    : {verdict}")
    if summary["speedup_S_vs_fp8_matched"] is None:
        print("  !! speedup missing — check m_*.json for failed runs (server didn't serve?)")

    if a.out:
        json.dump(summary, open(a.out, "w"), indent=2)
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
