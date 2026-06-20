# Exact deferred-overlap — LOOP-C pivot (lossless comms-floor lever)

**Date:** 2026-06-20 · **Author:** djamoils (LOOP-C) · **Status:** design + handoff to Charles's kernel
**Pivot from:** `research/n4_speculative_stale_tp.md` (runtime stale-TP = NO-GO, measured 0.000–0.025)

> **Why this, why now.** Stale/predicted TP is dead (substituting the all-reduce from local info
> flips the router → gibberish; retraining is out of scope — weeks + 3.76 TB optimizer state vs 640 GB).
> But the *overlap ceiling* stale-TP pointed at is reachable **losslessly**: don't substitute the
> collective, **overlap the EXACT collective with the next op's HBM weight-stream.** Same ~roofline
> prize (`tools/stale_tp_ceiling.py`), zero quality risk, no retraining.

## 1. The mechanism (and the honest B=1 subtlety)

At B=1 the per-layer cost is **weight-read time** (the GEMV ≈ the HBM read; compute ≈ 0). The TP
all-reduce (NVLink/NVSwitch) and the weight read (HBM) use **different hardware paths** → they *can*
run concurrently. The catch (`comms_floor.md` §3): the all-reduce output is the **activation** the next
op consumes, so the next GEMV's *compute* must wait for it. But the next op's **weight LOAD from HBM
does not depend on the activation** — only the final multiply does. So:

- **Overlap = issue the NVLS all-reduce while streaming the next matmul's weights from HBM.**
  Hidden per collective = `min(AR_latency, next_weight_read)`. The multiply itself (tiny at B=1) waits
  for the reduced activation, then runs against already-resident weights.

This is **lossless** — the exact collective still runs; only its *latency* is hidden behind a read
that was going to happen anyway.

**Why it's a kernel feature, not a config (the real constraint).** `comms_floor.md` §3 is right that
stock vLLM can't do this at B=1: async-TP needs chunked GEMMs (M≫1), and there's no "prefetch this
layer's weights" call. The way to actually overlap an 8 KB NVLS reduction against a weight stream at
B=1 is a **persistent megakernel** (MPK / Charles's K6) that software-pipelines at SM granularity:
some SMs run the multimem in-switch reduction while others stream the next weight tile. So **this lever
lives inside Charles's megakernel + NVLS kernel** — LOOP-C's role is the schedule + the ceiling, not a
separate from-scratch kernel.

## 2. What can overlap at B=1 (dependency map, per layer)

Two collectives/layer (TP=8): AR-A after attention o_proj, AR-M after MoE down-proj.

| collective | produces | next consumer | overlap target (HBM read, activation-independent) |
|---|---|---|---|
| **AR-A** (post-attn) | attn output → added to residual | same layer's MLP/MoE gate+up | **prefetch MoE gate/up expert weights** for the routed experts while AR-A is in flight |
| **AR-M** (post-MoE) | layer output residual | next layer's QKV proj | **prefetch layer L+1's QKV weights** while AR-M is in flight |

At B=1 the cover (one op's weight read) is ~8.3 µs at fp8 / ~16.6 µs at bf16 per layer (the active
slice). So an NVLS reduction at **C ≤ ~4 µs (fp8)** is *fully* hidden — see ceiling table below.

## 2b. Concrete megakernel schedule (implementable — for Charles's K6/NVLS)

The schedule structure is **fixed by the data dependencies** above (independent of the feasibility
question the research `wf_8e6331d8-e91` is settling). It is a 2-stage software pipeline inside the
persistent kernel, with the SMs partitioned into two disjoint sets so there is no occupancy contention:

- **COMMS set (small, ~2–8 SMs):** issues the in-switch all-reduce via **`multimem.ld_reduce` +
  `multimem.st`** (NVLS) over NVLink. Tiny SM footprint — TokenWeave shows the multimem AR needs only
  2–8 SMs. It signals completion via a flag in shared/global memory.
- **STREAM/COMPUTE set (the rest, ~124–130 SMs):** runs the current op's GEMV against resident weights,
  AND **prefetches the NEXT op's weights** from HBM via **Hopper TMA (`cp.async.bulk`)** into a second
  buffer. Pure HBM bandwidth op; uses the memory pipe, not NVLink.

**Per-op pipeline (the overlap):**
```
op N:  [GEMV compute on resident W_N]      (tiny @ B=1)
       └─ depends on AR_{N-1} result ──────────────────┐
   ║ CONCURRENT on disjoint SM sets ║                   │
   COMMS:   AR_N (multimem, NVLink) ───────────┐        │
   STREAM:  cp.async prefetch W_{N+1} (HBM) ───┴── cover│
op N+1: waits on max(AR_N, prefetch W_{N+1}) then GEMV ─┘
```
Critical path per op = **`tiny_compute + max(AR_latency, weight_prefetch)`** instead of the serial
`AR + weight_read + compute`. Hidden = `min(AR, weight_prefetch)`. Since `weight_prefetch ≈ 8.3 µs/layer`
(fp8) and a realistic NVLS AR is ~2–4 µs, the AR hides entirely → critical path collapses to the weight
stream → **fp8 weight roofline (~1280)**.

**The hazard that does NOT overlap (be honest):** the AR *output* is the GEMV's *activation input*, so
the GEMV **compute** of op N+1 must wait for AR_N. That's fine — at B=1 the GEMV compute is ~0; only the
*weight read* is large, and that's the part we prefetch. So we hide the AR behind the *next* op's weight
read, never behind its compute (that's the distinction from FLUX, which needs compute to hide behind and
collapses at B=1).

**Buffering:** double-buffer the active-op weight tiles in SMEM/registers (standard cp.async pipelining).
No need to stage a whole layer — only the next op's first tiles need to be in flight when its turn comes.
L2 (50 MB) is irrelevant; the cover is the in-flight HBM *transfer*, not a resident copy.

**OPEN feasibility questions (being answered by `wf_8e6331d8-e91`, fold in when it lands):**
1. Do `multimem.ld_reduce` (NVLink) and `cp.async.bulk` (HBM) actually run **concurrently**, or do they
   contend (shared mem controllers / copy engines / SM scheduler)? — decides the *hidden fraction*.
2. Realistic 8 KB NVLS AR latency on 8×H100 NVSwitch — is it ≤ the ~4 µs cover, or barrier-bound ~16 µs?
3. Does any published megakernel (MPK, TileLink, Triton-distributed) demonstrate comms↔**memory** overlap
   (not comms↔compute) at M=1? — the precedent that makes this more than a paper design.
If (1) shows contention or (2) shows C ≫ cover, the hidden fraction drops and this lever is partial, not
total — I will temper §3/§5b accordingly.

## 3. The prize (from `tools/stale_tp_ceiling.py`; the overlap math is identical for exact)

The ceiling tool's "+overlap" column is **lossless here** (the mechanism is the same; only correctness
differs from the killed staleness variant):

| per-collective C | fp8 comms exposed | tok/s (fp8) |
|---|---|---|
| 16 µs (today) | 3.01 ms | 257 |
| 7 µs (multimem one-shot) | 0.54 ms | 706 |
| **≤4 µs (Charles's NVLS, fully hidden)** | **~0** | **~1218 (roofline)** |

So **fp8 + exact deferred-overlap + Charles's NVLS → ~roofline**, losslessly. This is the same number
Charles cited for the stale path (`docs/path-to-1000.md`), now **without** the quality gamble.

## 4. Stacking (honest, sub-multiplicative — per Charles `660ef9f`)

- **Spec-decode (EAGLE3)** amortizes the *collective count* across accepted tokens.
- **Exact deferred-overlap + NVLS** hides the *per-collective latency* that remains.
  These attack different factors and compose — but as the regime flips floor→weight-bound (comms
  hidden), the optimal spec-tree shrinks; don't multiply naively. Re-fit with Charles's
  `tree_spec_optimizer` + `backout_floor` once comms is hidden.

## 5. LOOP-C deliverables for this lever
1. **This design + the SM-pipelining schedule** (§2 dependency map → which weights to prefetch per
   collective) — hand to Charles for the K6/NVLS kernel.
2. **`tools/stale_tp_ceiling.py`** — quantifies the prize and the C-threshold (≤~4 µs at fp8) that
   Charles's NVLS must hit; re-runnable at bf16/fp8.
3. **A correctness gate**: exact deferred-overlap is lossless *by construction* (the collective is
   unchanged), so the gate is a numerical-identity check (token-exact vs baseline), not a quality
   sweep — much cheaper than the staleness probe. Reuse `tools/quality_compare.py` for parity == 1.0.

## 5b. Roadmap impact — exact-overlap RELAXES the NVLS make-or-break (≤1 µs → ≤~4 µs)

`docs/path-to-1000.md` calls **NVLS ≤1 µs "non-negotiable"** because in its budget the comms term is
*added*: `0.78 (fp8 weight) + 0.19 (comms@1µs) ≈ 0.97 ms → 1033`. At a *realistic* NVLS (2–4 µs) that
budget gives 0.38–0.75 ms comms → **744–865 tok/s, short of 1000** for plain decode (the doc then needs
small-tree spec to claw back to ~1170).

**Exact deferred-overlap changes the arithmetic: comms is HIDDEN, not added.** The exact NVLS runs on a
few SMs *concurrent with* the fp8 weight-stream on the rest (megakernel/MPK SM-pipelining). The cover is
the per-collective slice of the 0.78 ms weight read ≈ **~4 µs/collective**. So:

| | comms on critical path | plain fp8 decode | needs spec for 1000? |
|---|---|---|---|
| NVLS only (roadmap) @3 µs | +0.56 ms (added) | 0.78+0.56 = 1.34 ms → **744** | yes (→~1170 w/ small tree) |
| **NVLS + exact-overlap** @3 µs | **~0 (hidden, C<cover)** | **0.78 ms → ~1280** | **no — roofline already** |

**Net for the team:** exact-overlap turns the NVLS requirement from a heroic **≤1 µs** into the
*realistic* **≤~4 µs** (fit under the weight cover), reaches **lossless ~1280 with plain fp8 decode**
(no spec needed), and spec then *stacks* on top. It is the **lossless replacement** for the now-dead
stale-TP "hide-it" lever (`n4_speculative_stale_tp.md` §6: stale/predicted measured 0.000–0.025). Same
~roofline destination the roadmap credited to stale-TP, **without the quality gate.** The dependency:
the megakernel must pipeline NVLS-on-some-SMs vs weight-streaming-on-others (MPK does exactly this) —
so this lives in Charles's K6/NVLS kernel, and it makes his make-or-break *easier*, not harder.

## 6. Open questions / next
- Confirm vLLM/megakernel can issue an NVLS multimem reduction on a subset of SMs concurrent with a
  weight-stream at B=1 (Charles's `measure_collective.sh` + K6 wiring).
- Measure real `C` for the multimem one-shot on this 8×H100 NVSwitch (does it beat 16 µs → reach ≤4 µs?).
- If C can't get below the fp8 cover (~4 µs), the lever still pays partially (706 tok/s at 7 µs) — quantify.
</content>
