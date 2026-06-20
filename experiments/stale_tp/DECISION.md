# LOOP-C stale-TP probe — pre-written GO/NO-GO decision (fill numbers, then execute)

**Purpose.** When `run_stale_probe.sh` finishes (results in `/alloc/data/stale_tp/` →
`origin/loopc-results`), read the parity gates and follow the branch below **without
re-deliberating**. The gate tool (`tools/quality_compare.py`) emits per comparison:
`mean_agreement` (word-prefix LCP fraction vs exact), `exact_rate` (full-string match),
`verdict` (its own 0.97/0.85 bands — informational; **use the stale-TP thresholds below**).

> The comparison is **each sweep point vs `q_exact.json`** (same engine, greedy temp 0,
> so an unperturbed run is byte-identical). Divergence = the quality cost of staleness.

---

## 1. SANITY GATES first (if these fail, the numbers are meaningless — fix, re-run)

| check | where | must be | if not |
|---|---|---|---|
| all-reduce count | `vllm_stale.log` `observed_calls_per_pass` | **== 188** (2×94) | wrong `STALE_TP_PERIOD`; layer alignment off → set PERIOD to observed, re-run |
| exact reproduces baseline | `q_exact.json` vs the team's 85.7 bf16 greedy (or self-consistent) | sensible text, no garbage | engine/config broken |
| **probe actually perturbs** | `parity_lyr_local_k2.json` | **must DEGRADE** (mean_agreement clearly <0.9) | substitution is a no-op → bug: check ctl reload took effect (`[stale_tp] ctl reload` in log), decode_only, that wrapper is on the hot path |

`lyr_local_k2` is the **control**: returning the un-reduced local partial *must* wreck
output (it's 1/8 of the sum). If it doesn't, the hook isn't biting — do NOT trust any GO.

---

## 2. The decision — keyed on `parity_lyr_proxy_k2.json` (the core N4 hypothesis)

Let **A2 = mean_agreement**, **E2 = exact_rate** for `lyr_proxy_k2`; **A4** for `lyr_proxy_k4`.

### ✅ GO (strong / novel positive)  —  A2 ≥ 0.99 AND E2 ≥ 0.70  AND A4 ≥ 0.95  AND control degraded
No-retrain, K-layer stale-TP preserves quality. **This contradicts the literature prior
(Ladder/Kog needed retraining) → a genuinely novel result.** Actions, in order:
1. Post the headline to `danielAgentScheduling.md` + commit results to `origin/loopc-results`.
2. Build the **real overlap kernel** (N1 path): on non-refresh layers, issue the AR on a
   side stream / skip-and-reuse, overlapping it with the next layer's weight read; reuse the
   cached reduced value. Fold into Charles's megakernel. Expected (ceiling model): comms
   hidden → toward roofline, stacking with the multimem one-shot.
3. Re-probe at K=3 and the **temporal** point to map the tolerance frontier; pick max K that
   holds A≥0.99.
4. Measure real decode tok/s of the kernel vs 85.7 (separate slot) — the actual win number.

### 🟡 CONDITIONAL GO  —  A2 ∈ [0.90, 0.99) OR (A2 high but E2 low, i.e. late drift) OR holds at K=2 not K=4
Quality is close but drifts (small per-layer error accumulates over the sequence). Don't kill —
**try the cheap recovery first**, all no-retrain:
1. **Error-feedback** (the key idea): accumulate the staleness residual `e = (true − stale)` and
   add it back at the next refresh layer (like EF in gradient compression). Add an `ef` policy to
   `stale_tp.py`, re-probe `lyr_ef_k2/k4`. Often turns drift→near-lossless for free.
2. **Attention-exact / MLP-stale**: stale only slot 1 (post-MoE), keep slot 0 (post-attn) exact
   (attention is more sensitive). Add a `slot_mask` knob, re-probe.
3. **Smaller K only** (K=2) + the above. If A≥0.99 emerges → promote to GO branch.
4. If still stuck at ~0.9 → treat as NO-GO.

### ❌ NO-GO  —  A2 < 0.90 (even with error-feedback + attention-exact)
Confirms the literature prior: **no-retrain runtime stale-TP is lossy for B=1 MoE decode.**
Honest KILL of the *runtime-only* variant. Then:
1. Document the kill in `research/n4_speculative_stale_tp.md` §6 with the numbers (this is a
   real, citable negative result — the open question the literature left, now answered).
2. **Pivot to the LOSSLESS lever (no quality gate):** *exact deferred-overlap* — overlap each
   layer's **exact** NVLS all-reduce with the next layer's weight-stream, deferring only as far
   as the true data dependency allows. Needs no staleness, so no quality risk; lives in the N1
   megakernel. Hand the comms-floor win to Charles's multimem one-shot (lever 2) + this overlap.
3. State plainly: Ladder-Residual-style staleness *works at B=1* but **requires retraining** →
   out of hackathon scope; don't chase it.

---

## 3. Cross-checks that sharpen the verdict
- **`lyr_proxy_k4` vs `k2`**: how fast does quality fall with K? Steep fall → low tolerance
  (NO-GO-ish even if K=2 passes). Flat → strong GO, push K higher for more comms hidden.
- **`tmp_proxy_k2` (temporal)**: if it holds but layer doesn't (or vice-versa), the *kind* of
  staleness that's tolerable tells us which kernel to build (within-token vs across-token).
- **per-prompt `rows`**: which prompts break first? Code/math (low entropy, sharp routing) vs
  prose. If only hard-reasoning prompts break, a conditional deployment may still pay.
- **`m_exact.json` tok/s**: confirms the probe ran the real bf16-TP8 engine (~85 tok/s); the
  probe itself adds overhead (it still does real reduces on refresh layers) so it is NOT the
  speed number — only the quality gate is.

---

## 4. Results table (fill from `parity_*.json`)
| point | mean_agreement | exact_rate | tool verdict | note |
|---|---|---|---|---|
| lyr_proxy_k2 | | | | **the decision driver** |
| lyr_proxy_k4 | | | | tolerance vs K |
| lyr_local_k2 | | | | **control — must degrade** |
| lyr_proxy_k8 | | | | far point |
| tmp_proxy_k2 | | | | across-token variant |

**Branch taken:** ______  →  **next action:** ______
