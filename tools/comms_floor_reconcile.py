#!/usr/bin/env python3
"""Reconcile the per-collective comms latency C against the MEASURED TPOT — which C is self-consistent?

LOOP-C, 2026-06-20. Charles's E0 (docs/onbox-collective-latency-e0.md) measured stock-NCCL small-message
all-reduce at ~35 us on-box (nccl-tests), 7x the team model's 5 us and >2x the ladder's 16 us. E0's headline
is "comms is the DOMINANT floor term (188 x 35us = 6.6ms)". That would rewrite the ladder
(tools/ladder_to_1000.py: comms=3.0ms @16us, overhead=7.0ms). Before the team rebuilds projections on 35us,
this tool ADVERSARIALLY CHECKS it against two OTHER independent on-box measurements:

  (M1) TPOT  = 11.67 ms   (bf16-TP8, B=1, greedy, ctx 512 — the real 85.7 tok/s)
  (M2) e_eng ~ 0.16-0.19  (vLLM whole-model achieved BW fraction at B=1 — K5/overhead-attribution candidate 2)

The B=1 decode step is SERIAL (no comms/compute overlap exists yet — that's the whole point of the
exact-overlap lever). So:                       TPOT = T_compute + T_comms + T_host
  - T_compute = weight_floor / e_eng     (decode is BW-bound; attn+MoE kernel time ~ weight-read time)
  - T_comms   = n_coll * C
  - T_host    = residual launch/python/sample not hidden by CUDA graphs (>= 0)

Given M1 and a hypothesized C, the residual R = TPOT - n_coll*C - kv must cover T_compute + T_host. Since
T_host >= 0, the hypothesis REQUIRES the kernels to run at e >= weight_floor / R. If that required-e exceeds
the independently-measured e_eng (M2), the C hypothesis is INCONSISTENT — it can't fit the measured TPOT
without the kernels being far more efficient than they actually are. This collapses the 7x C-spread into the
one (or two) corners that are self-consistent, and names the single trace (E-attr Nsight NCCL-slice) that
picks between them.

  python3 tools/comms_floor_reconcile.py
"""

# --- measured / physics inputs (sourced, not guessed) ---
TPOT_MS      = 11.67   # M1: measured bf16-TP8 B=1 greedy TPOT (overhead-attribution.md, atlas)
N_COLL       = 188     # 2 all-reduces/layer x 94 layers, pure TP8 (no EP) — the stock decode config
KV_MS        = 0.07    # kv read at short ctx (negligible; ladder)
# weight-read floor at e=1: active 20.9B params x 2B (bf16) / 8 (TP) / 3.35 TB/s peak  (comms_floor.md)
ACTIVE_B     = 20.9e9
WEIGHT_FLOOR_MS = ACTIVE_B * 2 / 8 / 3.35e12 * 1e3   # ~1.56 ms
E_MEAS_LO, E_MEAS_HI = 0.16, 0.19  # M2: measured vLLM whole-model achieved-BW fraction at B=1 (K5)

# the three C hypotheses in play across the team's docs
HYPOTHESES = [
    ("5 us  (team model guess)",          5.0,  "src/inferutil/hardware.py original collective_latency_s"),
    ("16 us (ladder / comms_floor)",     16.0,  "nccl-tests microbench, taken as the in-engine custom-AR proxy"),
    ("35 us (E0 stock NCCL ring)",       35.0,  "nccl-tests standalone-launched stock ring all-reduce"),
]

def required_e(C_us):
    """Min kernel efficiency the C-hypothesis forces, to fit the measured TPOT (T_host>=0 => e >= wf/R)."""
    comms_ms = N_COLL * C_us / 1e3
    R = TPOT_MS - comms_ms - KV_MS          # budget left for compute + host
    if R <= 0:
        return comms_ms, R, float("inf")    # comms alone exceeds TPOT — impossible
    return comms_ms, R, WEIGHT_FLOOR_MS / R

print(f"weight-read floor (e=1)  : {WEIGHT_FLOOR_MS:.2f} ms")
print(f"measured TPOT            : {TPOT_MS:.2f} ms   over {N_COLL} collectives")
print(f"measured vLLM e (M2)     : {E_MEAS_LO:.2f}-{E_MEAS_HI:.2f}")
print()
print(f"{'C hypothesis':32} {'comms ms':>8} {'R(comp+host)':>13} {'req. e>=':>9}  verdict")
print("-" * 92)
for label, C, _src in HYPOTHESES:
    comms_ms, R, req_e = required_e(C)
    if req_e == float("inf"):
        verdict = "IMPOSSIBLE (comms alone > TPOT)"
    elif req_e <= E_MEAS_HI:
        verdict = f"CONSISTENT (req e {req_e:.2f} <= measured {E_MEAS_HI:.2f})"
    elif req_e <= E_MEAS_HI * 1.25:
        verdict = f"MARGINAL (req e {req_e:.2f} ~ {req_e/E_MEAS_HI:.1f}x measured)"
    else:
        verdict = f"INCONSISTENT (req e {req_e:.2f} = {req_e/E_MEAS_HI:.1f}x measured)"
    print(f"{label:32} {comms_ms:8.2f} {R:13.2f} {req_e:9.2f}  {verdict}")

# What C does the measured e actually imply? (invert: T_compute = wf/e, then C = (TPOT - kv - T_compute)/n_coll,
# with T_host=0 as the upper bound on C — any host time only lowers the implied C.)
print()
print("Inverting M2 (measured e) -> the implied in-engine C (T_host=0 upper bound):")
for e in (E_MEAS_LO, (E_MEAS_LO + E_MEAS_HI) / 2, E_MEAS_HI):
    t_comp = WEIGHT_FLOOR_MS / e
    c_implied_us = (TPOT_MS - KV_MS - t_comp) / N_COLL * 1e3
    print(f"  e={e:.2f} -> T_compute={t_comp:5.2f}ms -> in-engine C <= {c_implied_us:5.1f} us")

print("""
READING:
- 35 us is INCONSISTENT with the measured TPOT+e: it forces vLLM's kernels to run ~1.6-1.9x more efficient
  than the K5-measured ~0.16-0.19. So 35 us is NOT the engine's effective AR — it's the STOCK-NCCL ring
  measured STANDALONE (nccl-tests launches each collective fresh; CUDA-graph decode amortizes that launch).
  E0's structural conclusion (env-tuning dead; lever is fused-AR+norm / one-shot / overlap) STANDS and is right;
  only its *magnitude* ("comms is THE dominant floor in the engine") overstates the in-engine reality.
- The self-consistent corner is C ~ 16 us (req e 0.18 ~ measured) -> comms ~3.0ms ~26% of TPOT (the ladder's
  number). The engine's graph-captured custom one-shot AR (vLLM uses it at 8KB << 256KB cutoff, comms_floor.md)
  is in this regime, NOT the 35us stock-ring regime.
- HONEST CONSEQUENCE FOR LOOP-C's OWN LEVER: this REDUCES the apparent comms prize (35->~16us). Exact-overlap /
  NVLS hide ~3ms, not ~6.6ms. The smaller, correct number is what the ladder should use; don't bank 6.6ms.
- THE ONE RESOLVER: E-attr (Nsight `nccl_sum` on ~20 decode steps) reads the engine's REAL per-step NCCL time
  directly -> picks the corner in a single trace. Until then, ladder C=16us (not 35) is the defensible value.
""")
