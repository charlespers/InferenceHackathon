# Console telemetry spec — make the B=1 optimization visible in the deliverable

The console (the hackathon deliverable) should *show* the levers this session found, so a viewer sees where
the time goes, which regime they're in, which lever to pull, and how close to the ceiling. This is the
`x_telemetry` / `x_summary` contract extension + the UI panels, grounded in the final understanding
(supersedes the early `b1-latency-architecture.md` §6 sketch).

## `x_summary` (per-turn) — add
```jsonc
{
  "tpot_ms": 11.67, "ttft_ms": 777, "decode_tok_per_s": 85.7,
  "floor_breakdown_ms": { "weight": 1.6, "comms": 3.0, "kernel": 4.0, "host": 2.5, "kv": 0.0 }, // sums≈tpot
  "regime": "floor-bound",            // floor-bound | weight-bound — decides the next lever
  "pct_of_ceiling": 4.3,              // decode_tok_per_s / ~2000 (fp8+spec absolute ceiling)
  "pct_of_roofline": 6.9,             // decode_tok_per_s / weight-only roofline at this precision/layout
  "spec": { "accept_rate": 0.0, "tau": 1.0, "realized_speedup": 1.0, "tree": "none" },
  "layout": "tp8", "weight_dtype": "bf16", "kv_dtype": "bf16",
  "busiest_rank": 1.0,                // per-step busiest expert count (EP only; 1.0 under TP8)
  "collective_us": 16.0              // measured per-collective latency feeding the comms bar
}
```

## `x_telemetry` (per-token) — keep + add
Existing: `experts[]`, `t_ms`, `spec{proposed,accepted}`. Add `bytes_moved{weight,kv}`, `comms_ms`,
`busiest_rank_bytes` vs `mean_rank_bytes` (the EP-imbalance proof — collapses 2.6→1.0 under TP8).

## UI panels (the story, top to bottom)
1. **Floor bar** (the headline) — stacked `weight / comms / kernel / host` ms per token, with the **ceiling
   line** and a `pct_of_ceiling` readout. This is `overhead-attribution.md` made visual: a viewer instantly
   sees it's ~95% overhead, not bytes. Color the dominant slice.
2. **Regime chip** — `floor-bound` (→ "fix the floor: spec + comms + kernels") vs `weight-bound` (→ "now
   quant/route-aware pays"). The single most useful "what do I do next" signal (`interpretation-playbook.md`).
3. **Spec panel** — live `accept_rate` + `τ` + `realized_speedup`. **Wiring (the field is currently hardcoded
   0.0 in `VLLMBackend`):** `server/spec_metrics.py` → `cumulative_accept(fetch_metrics(VLLM_URL))` after the
   stream (after-only = no TTFT hit) gives the real `{accept_rate, tau}` from vLLM's `/metrics`. With a **red
   flag when realized < 1.0**
   (the MoE verify-tax trap, `spec-decode-moe-tax.md`) and the tree shape. Accept-rate is *the* knob a user
   watches — it varies with temperature + content (`spec-in-production.md`).
4. **Ladder gauge** — current vs cheap-wins (~508) vs +kernels (~750) vs ceiling (~2000), so the headroom is
   visible (`absolute-ceiling.md`): "you are at 4% of physics."
5. **EP-imbalance heatmap** (when layout=ep) — `busiest_rank` per GPU; the TP8 toggle flattens it to 1.0
   (`ep-placement-for-b1.md`). The clearest visual of the EP→TP inversion.
6. **TTFT vs decode split** — TTFT 777ms / N·TPOT, with a "prefix-cache hit" indicator (turn-2 → ~10ms),
   showing TTFT is 34% of perceived latency and the cheapest fix (`ttft-analysis.md`).

## Why this matters for the deliverable
The console isn't just a chat — it's the **instrument** that makes the optimization legible: a viewer (or a
judge) sees the floor decomposition, the regime, the spec accept-rate, and the ceiling-fraction live, and
understands *why* the levers are what they are. The mock (`server/mock_engine.py`) can synthesize these fields
from `tools/latency_budget.py` + the measured constants so the console demos the full story GPU-free; the
`RealEngineBackend` maps the live engine's hooks into the same contract (`predict_matrix.py` /
`latency_budget.py` are the calibration). Every panel above has a backing doc — the console becomes the
visual index of `b1-optimization-atlas.md`.
