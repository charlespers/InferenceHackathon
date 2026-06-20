# results/ — shared B=1 experiment outputs

Result JSONs from on-box runs, committed so the whole team can see them (not just
SSH summaries). Slot runs auto-push their JSON here (see `tools/` + the slot runner).

| file | what | run |
|---|---|---|
| `routing_predict_early.json` | per-token route stats: persistence 44.6% (token→token top-8 overlap, ≫6% chance). Early run, pre-DirectProxy. | superseded by router_mass |
| `router_mass.json` | router softmax concentration → adaptive-top-k headroom (avg-k, expert-byte savings, per-p sweep) | 06:45 slot |
| `baseline.json` | live vLLM B=1 TTFT/TPOT/decode tok/s, % of roofline | when vLLM serving in-slot |

Adaptive-top-k optimization (the active deliverable) lives in `experiments/adaptive_topk/`.
