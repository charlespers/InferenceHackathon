# Results reaction 05 — stale/proxy-TP is DEAD (measured); exact deferred-overlap is the lossless comms lever

LOOP-C measured the staleness probe and **killed all runtime stale/predicted-TP**, then pivoted to a *lossless*
comms-hide (`research/exact_deferred_overlap.md`) and handed the kernel schedule to me. Reacting + integrating.

## The kill (and an honest correction of my own proposal)
- **stale-reuse TP: parity 0.000** at K=2 (gibberish from token 1). My **router-flip mechanism note was right**
  (stale hidden → top-8 flips → wrong experts).
- **predicted-proxy (my DirectProxy proposal): 0.025 — also dead.** LOOP-C's generalization is correct and I
  was wrong: **any *local-info* predictor (DirectProxy included) cannot recover the cross-rank SUM** — it gets
  the magnitude right, direction wrong → router flips. It's an **information barrier, not tuning.** So
  `route-aware-drafting`'s DirectProxy is fine for *drafting* (where errors are caught by the verify) but **not**
  for substituting an all-reduce. Retire the "DirectProxy → proxy-TP" idea. (Retraining = out of scope.)

## The lossless pivot — exact deferred-overlap (this is now THE comms lever)
Don't *substitute* the collective — **overlap the EXACT NVLS all-reduce with the next op's HBM weight stream**
(different hardware paths: NVLink vs HBM). The next GEMV's *multiply* waits for the reduced activation, but its
*weight LOAD* doesn't — so the all-reduce latency hides behind a read that happens anyway. **Lossless, zero
quality risk, no retraining.** LOOP-C's schedule (which I'm folding into the NVLS/megakernel):
- **AR-A** (post-attn) → prefetch this layer's routed **MoE gate/up** expert weights while it's in flight.
- **AR-M** (post-MoE) → prefetch **layer L+1's QKV** weights while it's in flight.
- Hidden per collective = `min(AR_latency, next_weight_read)`. At fp8 the per-layer weight cover is **~8.3 µs**,
  so an all-reduce at **C ≤ ~4 µs is FULLY hidden → comms → ~0 → ~roofline (~1218)**; at the 16 µs barrier it
  hides ~8 µs → comms ~halved (3.0 → ~1.5 ms).

## Revised comms strategy (replaces reaction-04's stale-TP line)
The comms levers are now exactly two, **lossless, stackable, both living in the kernel**:
1. **multimem in-switch reduce** (`nvls_allreduce.cu`) — drive C from 16 µs toward ~4 µs. The make-or-break
   (`measure_collective.sh`). *Now doubly important:* C≤4 µs is the threshold at which deferred-overlap *fully*
   hides the comms.
2. **exact deferred-overlap** (megakernel SM-pipelining, LOOP-C's schedule) — hide C behind the weight stream.
- **Together:** C→4 µs (lever 1) + overlap (lever 2) → comms hidden → **~roofline ~1218, LOSSLESS.**
- **If C stays 16 µs:** overlap still halves it (3.0→1.5 ms) → with spec, `ladder_to_1000.py --C 8 --ncoll 188`
  → ~1000-ish. So **even without sub-4 µs, deferred-overlap + spec is a real lossless path.**

## What changes upstream
- `path-to-1000.md` / atlas: the comms-HIDE path is **exact deferred-overlap (lossless)**, NOT stale/proxy-TP
  (dead). int4 still dead. The lossless ceiling rises from ~870 toward ~1000–1218 *because the comms can be
  hidden losslessly* (it couldn't before — I'd only had spec-amortization + the dead stale-TP).
- The ladder's effective C is now **`min(C, weight_cover)` after overlap** — at C=16 µs use `--C 8`; at C≤4 µs
  use `--C 0`-ish (`--stale-tp` flag now means "comms hidden", relabel to `--overlap`).
- **My kernel work absorbs LOOP-C's schedule:** the NVLS reduce + the weight-prefetch pipeline are one megakernel
  feature (`megakernel-b1.md` Stage 5 / K6). LOOP-C owns the schedule + ceiling; I own the kernel.

## Net
**The comms is hideable LOSSLESSLY after all — not by faking the all-reduce (dead) but by overlapping the exact
one with the weight stream.** So 1000 lossless is back on (~1000–1218), gated on: the multimem in-switch C
(≤4 µs → full hide) + the deferred-overlap kernel + the EAGLE3 spec multiplier. The make-or-break is unchanged
in name (the NVLS C) but now decides *full* vs *half* comms hiding, both lossless.
