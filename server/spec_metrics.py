"""Surface the REAL speculative-decoding accept rate in the console.

`VLLMBackend` currently emits `x_summary.spec_accept_rate: 0.0` (a placeholder). The real rate lives in vLLM's
Prometheus `/metrics` spec counters (the same source `tools/eagle3_analyze.py` uses). These are CUMULATIVE
counters, so the per-turn rate is the DELTA across one request: scrape `/metrics` before and after the stream.

This is an isolated, tested helper the console can adopt (it does not edit the active VLLMBackend). Wiring:

    before = fetch_metrics(VLLM_URL)
    ... stream the turn ...
    after  = fetch_metrics(VLLM_URL)
    st = accept_stats(before, after)          # {"accept_rate", "tau", "accepted", "drafts"}
    x_summary["spec_accept_rate"] = st["accept_rate"]
    x_summary["spec_tau"]         = st["tau"]

Counter names have drifted across vLLM versions, so matching is loose (substring), matching eagle3_analyze.py.
See docs/console-telemetry-spec.md (the spec panel) and docs/spec-in-production.md (accept-rate varies with temp).
"""
from __future__ import annotations
import re
import urllib.request


def _scalars(metrics_text: str) -> dict:
    vals: dict[str, float] = {}
    for line in (metrics_text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        try:
            vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(2))
        except ValueError:
            pass
    return vals


def _find(vals: dict, *subs):
    for k, v in vals.items():
        kl = k.lower()
        if all(s in kl for s in subs):
            return v
    return None


def parse_spec_counters(metrics_text: str) -> dict:
    """Pull cumulative {accepted, drafts} from a /metrics scrape (loose name match). 0.0 if absent."""
    vals = _scalars(metrics_text)
    accepted = _find(vals, "spec_decode", "accepted")
    if accepted is None:
        accepted = _find(vals, "accepted", "token")
    drafts = _find(vals, "spec_decode", "num_drafts")
    if drafts is None:
        drafts = _find(vals, "spec_decode", "draft")
    return {"accepted": accepted or 0.0, "drafts": drafts or 0.0}


def accept_stats(before_text: str, after_text: str) -> dict:
    """Per-turn accept rate + accept-length tau from before/after cumulative scrapes.

    accept_rate = Δaccepted / Δdrafts (fraction of drafted tokens accepted, 0..1).
    tau         = 1 + Δaccepted / Δdrafts (tokens emitted per target forward).
    Returns zeros (accept_rate 0.0, tau 1.0) when there are no drafts in the window (spec off / no draft fired).
    """
    b, a = parse_spec_counters(before_text), parse_spec_counters(after_text)
    d_acc = max(a["accepted"] - b["accepted"], 0.0)
    d_drf = max(a["drafts"] - b["drafts"], 0.0)
    if d_drf <= 0:
        return {"accept_rate": 0.0, "tau": 1.0, "accepted": d_acc, "drafts": d_drf}
    rate = d_acc / d_drf
    return {"accept_rate": round(min(rate, 1.0), 4), "tau": round(1.0 + rate, 3),
            "accepted": d_acc, "drafts": d_drf}


def cumulative_accept(metrics_text: str) -> dict:
    """Running-average accept rate from a SINGLE scrape (after the stream). Low-latency console path:
    no before-scrape, so it adds nothing to TTFT/inter-token — just one post-stream call before x_summary.
    It's the average over all requests since server start (fine for a live demo gauge)."""
    c = parse_spec_counters(metrics_text)
    if c["drafts"] <= 0:
        return {"accept_rate": 0.0, "tau": 1.0, "accepted": c["accepted"], "drafts": c["drafts"]}
    rate = c["accepted"] / c["drafts"]
    return {"accept_rate": round(min(rate, 1.0), 4), "tau": round(1.0 + rate, 3),
            "accepted": c["accepted"], "drafts": c["drafts"]}


def fetch_metrics(vllm_url: str, timeout: float = 2.0) -> str:
    """Scrape {vllm_url}/metrics; return '' on any failure (caller treats as no-spec-data)."""
    try:
        with urllib.request.urlopen(f"{vllm_url}/metrics", timeout=timeout) as r:
            return r.read().decode(errors="replace")
    except Exception:
        return ""
