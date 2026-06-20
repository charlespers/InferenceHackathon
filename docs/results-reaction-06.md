# Results reaction 06 — TP=8 B=1 is OCCUPANCY-STARVED (33.8 tok/s sharded bench); spec is what reaches the roofline

The team's custom graphed sharded decode (`decode_sharded_nvshmem.cu`) benched **eager 13.0 → graphed 33.8 tok/s**
and surfaced a decisive measured finding that I'm integrating into the 1000 path.

## The finding
- **CUDA graphs work** (eager 13.0 → 33.8; compute-only 71→24 ms — the launch overhead is real and graph-killable).
- **But TP=8 sharding does NOT help B=1 latency:** graphed sharded (33.8) ≈ single-GPU proxy (30.9). Per-GPU the
  sharded kernels hit **118 GB/s = 3.5% of peak** vs single-GPU 859 GB/s (26%) — **the 8× data reduction is
  offset by ~6× worse per-GPU efficiency** because the sharded slices are too small to saturate an H100
  (occupancy-starved). Custom sharded (33.8 fp8) sits *below* vLLM's mature TP=8 (85.7 bf16). One-shot AR = 30 µs.

## How this lands on my 1000 analysis — mostly CONFIRMS, with two corrections
**Confirms `why-spec-wins.md`:** the B=1 GEMV is fundamentally inefficient (here: 3.5% per-GPU). That's exactly
why **spec is the lever** — the **batched verify** (W×D positions) is a *grouped GEMM* with enough per-GPU work
to saturate the SMs, so it reaches the roofline where plain B=1 decode can't. **Spec fixes the occupancy-starvation
AND amortizes the comms — same mechanism, doubly important now.**

**Correction 1 — my ladder's "fp8-K5 at e→1" is realized in the VERIFY, not plain decode.** Rung 3 (weight→0.78 ms
at roofline) assumed a kernel reaching peak BW. At TP=8 B=1 the *plain-decode* GEMV is occupancy-starved (K5 hits
58% MBU *in isolation* but the full sharded step is 3.5% per-GPU — the per-layer per-GPU slices are tiny). So the
roofline weight is reached by the **batched spec verify** (more per-GPU work), not by plain decode or the draft.
The ladder is right *for the spec-verify path*; plain-decode rungs are optimistic.

**Correction 2 — TP degree is now a live lever (LOOP-C's `tp_degree_model.py`).** Fewer GPUs (TP=2/4) = larger
per-GPU work = better occupancy, at the cost of less weight sharding. The team's measurement says TP=8 is
occupancy-bound, so **TP=2/4 may be faster for the non-spec parts (the EAGLE3 draft, the plain-decode fallback).**
The spec *verify* is fine on TP=8 (it's batched), but the *draft* (1-layer, B=1) is occupancy-starved on TP=8 →
this is *another* argument for `draft_tp` tuning (not just the bandwidth argument in `eagle3-draft-tp.md`).

## Honest recalibration of the prize
The team's blunt read — "from vLLM's real 85.7, even spec ×3.8 → ~325; 700 needs spec + TP=2/4 + MoE overlap" —
is the **plain-engine** view. My higher numbers (~900–1280) assume the floor is *removed* (graphs + fast-path +
the spec-batched verify reaching the roofline + deferred-overlap hiding comms). **Both can be true:** spec on the
*current occupancy-limited* engine → ~325; spec on a *floor-removed + occupancy-fixed-by-batching* engine →
~900–1280. The gap between them is exactly the engine work (graphs + the megakernel + the batched-verify kernel).
**So 1000 is a genuine stretch and hinges on the same things — but the occupancy finding says the win comes from
BATCHING (spec verify) + deferred-overlap, NOT from making the tiny B=1 GEMV faster (it can't saturate at TP=8).**

## Updated one-liner for the levers
- **Spec (batched verify)** = the lever: fixes occupancy (grouped GEMM) + amortizes comms. Reaches the roofline.
- **Deferred-overlap** (lossless) = hides the comms floor. **fp8-K5-e→1** = realized *in the verify*.
- **TP degree** = tune it (TP=2/4 for the occupancy-starved draft/plain-decode; TP=8 fine for the batched verify).
- **Dead:** stale/proxy-TP, int4, and "make the plain B=1 GEMV reach the roofline at TP=8" (occupancy-capped).
