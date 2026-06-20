# The absolute single-user ceiling on 8×H100 — where physics stops you

The proven levers project ~750 tok/s. But what's the *fundamental* limit, and how much headroom is left after
the cheap wins? This computes it from the bandwidth roofline (the "stopping rule" from
`b1-latency-architecture.md` §G: stop optimizing when TPOT approaches `bytes_per_token / usable_HBM_BW`).

## Weight-only rooflines (e=1, comms=0, no overhead — the pure physics)
`decode_latency(plan="floor")` on 8×H100 (8 × 3.35 TB/s = 26.8 TB/s usable agg, 22B active):
| precision | weight read | **roofline tok/s** |
|---|---|---|
| bf16 | 1.61 ms | **618** |
| fp8 | 0.80 ms | **1236** |
| int4 | 0.40 ms | **2457** |

These assume perfect kernels (e=1), zero comms, zero overhead — the unreachable physics ceiling for *plain*
decode at each precision.

## The absolute ceiling = roofline × the optimal spec multiplier
At the weight-bound limit (floor fixed → F=0), `tree_spec_optimizer.py` says the optimal route-aware tree gives
**~1.5–1.7×** (spec amortizes the read-once non-expert/attention weight; the expert union caps it). So:
| stack | absolute ceiling |
|---|---|
| **fp8 + optimal spec** | **~1850–2100 tok/s** |
| **int4 + optimal spec** | **~3700–4200 tok/s** |
| bf16 + optimal spec | ~930–1050 tok/s |

(Beyond that: B200 — ~1.43× HBM BW → ~2900 fp8+spec; or more GPUs, but TP>8 crosses the NVSwitch domain and
the comms-latency cliff kills it. SRAM machines don't transfer to MoE — `b1-latency-architecture.md` §13.)

## Where we are on the ladder
| | tok/s | % of fp8+spec ceiling (~2000) |
|---|---|---|
| current (bf16-TP8, real, overhead-bound) | **85.7** | **~4%** |
| proven cheap wins (prefix+spec+TP8+LL comms) | ~508 | ~25% |
| + K5 kernels + tuned comms | ~750 | ~37% |
| **absolute fp8+spec ceiling** | **~2000** | 100% |
| absolute int4+spec ceiling | ~3900 | (2×) |

## Reading it — the proven levers are necessary but not sufficient
- **85.7 → 750 (~9×)** is the cheap+kernel+comms work (this session's plan). Big, but it's only **~37% of the
  ceiling.**
- **750 → ~2000 (the last ~2.7×)** is the *hard* residual: driving kernel efficiency e→1 (the K5 work is at
  0.46; perfect is 1.0), comms→0 on the critical path (device-initiated NVSHMEM in-graph, `comms_floor.md`),
  the B=1 fast-path removing the last fixed overhead (`b1-fast-path-design.md`), and the optimal regime-adaptive
  spec tree. This is the cudarc engine's territory — a custom B=1 engine is what closes the e=0.46→1 and the
  comms→0 gaps that vLLM structurally can't.
- **2000 → 3900** is **int4 experts** — but it's the *last* lever (weight is 14% while floor-bound; it only
  doubles the ceiling once everything else is at the roofline, and carries an accuracy gate).

## The stopping rule (when to quit)
Stop when TPOT approaches `bytes_per_token / usable_HBM_BW` at the chosen precision — at that point you're
physics-limited and only **moving fewer bytes (int4, more spec) or more/faster HBM (B200)** helps. Today at
85.7 tok/s we're at **~4% of physics** — nowhere near the wall; the entire gap is overhead + comms + the
inefficient B=1 GEMV + no spec, all of which the plan removes. The ceiling says: **the prize is ~20× (to ~2000),
the cheap wins get the first ~9×, and the custom engine + int4 get the rest.**
