# Weight-prefetch comms overlap — ⚠️ PARKED (not validated, real bug found, no trustworthy number)

**Date:** 2026-06-20 · **Author:** Alyssa · **Status:** PARKED — correctness-clean, performance unproven
**Parent:** `research/exact_deferred_overlap.md` (the design this implements); pivots from Charles's
"comms-overlap measured but needs persistent megakernel -> parked" (`acfaf05`)

## TL;DR

Implemented the smaller-scope alternative to the (independently falsified, `1eaf819`) persistent
megakernel: hide each AR's NVLink latency behind a **weight touch** of the next segment's weights — a
read with no data dependency on the AR's result, so it's lossless by construction. Wired it directly
into `decode_step_tp8.cu` (the real engine, not a proxy) behind `USE_WEIGHT_PREFETCH` (default 0,
existing measured numbers untouched). **Correctness passed every run. Performance did not — the final,
correctly-wired version produced a 2.9 tok/s result on a config that should be ~110, a clear regression
artifact, not a measured property of the idea.** Do not bank this lever in either direction (proven or
disproven) until it's re-investigated with Nsight.

## The mechanism (unchanged from the design doc, not in question)

AR(L)'s result feeds the next segment's *activation*-dependent compute (a real dependency — can't be
removed). It does **not** feed that segment's *weight* read (fixed, read-only, no dependency at all). So
while the AR runs on the main stream, a second stream concurrently touches (reads, discards) the next
segment's weights into L2/cache. By the time the AR completes, the weight should be cache-resident
instead of HBM-cold. This part of the idea is sound and matches `exact_deferred_overlap.md`'s design —
the problem below is entirely in my specific implementation, not the underlying mechanism.

## What got built

- `kernels/decode_step_tp8.cu`: `RankState` gained `prefetch_stream` / `prefetch_sink` / `prefetch_fork`
  / `prefetch_join`. `touch_weights_kernel` does a grid-strided XOR-read of a weight buffer into a sink
  (defeats dead-code elimination, no real compute). `touch_segB_weights` (Wgate/Wgu_pack/Wd_pack) fires
  concurrently with AR#1; `touch_segA_weights` (Wqkv/Wo) fires concurrently with AR#2. All gated behind
  `#if USE_WEIGHT_PREFETCH` (default 0) in `enqueue_tp8_layer` — the function actually captured into the
  team's best-measured path ("full NCCL-in-graph").
- `kernels/overlap_prefetch.cu`: an isolated, standalone microbench (correctness gate comparing K1's
  output bit-for-bit with vs without the touch + an isolated K1 cold/warm timing split). Never got a
  result — see below.
- `bench/run_overlap_prefetch.sh`, `bench/run_weight_prefetch_ab.sh`: build+run harnesses with the
  team's `gpu.lock` protocol (refuse to run if the box is held / too little free memory) and an A/B
  compile of the real engine with the flag on vs off.

## What happened, honestly, in order

| attempt | what was wrong | result |
|---|---|---|
| 1. Isolated microbench (`overlap_prefetch.cu`) | build script forgot `-lcublasLt -lcublas -lcuda` (the file includes `decode_step_tp8.cu`, which needs them under `USE_GEMM=1`) | **compile failed**, no signal |
| 2. Real-engine A/B, v1 | patched `replay_tp8_step_kgraph` — a slower fallback path, **not** the one the benchmark actually reports as best ("full NCCL-in-graph", a different function, `enqueue_tp8_step`/`enqueue_tp8_layer`) | my code never executed in the measured path; 108.4 vs 108.6 tok/s, a non-result |
| 3. Real-engine A/B, v2 (re-targeted) | forked `prefetch_stream` off the capturing stream right before each AR, but never joined it back | **CUDA graph capture failed**: `"capturing stream has unjoined work"` on every rank; fell back to a slower path (94.8 tok/s) |
| 4. Real-engine A/B, v3 (join added) | added `cudaEventRecord`/`cudaStreamWaitEvent` join-back right after each AR | **capture succeeded, correctness PASSED** (`TOL=8e-02`), but: baseline 112.3 tok/s vs prefetch-build **2.9 tok/s** on the "full NCCL-in-graph" path — and the prefetch build's own *eager* baseline also cratered (38.2 vs 96.1 tok/s for the unmodified build in the same run) |

Run 4's correctness gate passing rules out wrong answers. The performance collapse is not explained —
plausibly the same two event objects (`prefetch_fork`/`prefetch_join`) being reused across all 188
AR points inside one captured graph introduces serialization or scheduling pathology under 8-rank
concurrent capture that doesn't show up as a capture error, just as catastrophic slowdown. This was
**not root-caused** — it would need an Nsight Systems timeline of the replayed graph to see whether the
touch kernels are actually overlapping the AR or serializing behind/in front of it unexpectedly.

## Why this is parked, not killed

Unlike the team's other dead-ends this session (int4, two different dequant strategies, both genuinely
ALU-bound and conclusively measured slower; stale-TP, a hard quality kill at 0% token agreement), this
lever has **no clean negative result** — only an implementation bug that produced a nonsensical number.
The underlying mechanism (overlap AR with an activation-independent weight read) is still believed sound
per `exact_deferred_overlap.md`'s design and NVLS's measured 3.84µs sitting comfortably under the
~8.3µs/layer weight-read cover. What's missing is a clean, profiled measurement.

## If anyone picks this back up

1. Profile with Nsight Systems (not just wall-clock tok/s) on the `USE_WEIGHT_PREFETCH=1` build to see
   whether the touch kernels are actually concurrent with the ARs in the replayed graph, or serialized.
2. Consider per-layer (not 2 reused) fork/join events — 188 reuses of 2 event handles inside one captured
   graph is untested territory; dedicated events per AR site would rule out an event-aliasing artifact.
3. Re-run the isolated `overlap_prefetch.cu` microbench with the corrected build flags (`-lcublasLt
   -lcublas -lcuda`) — it isolates K1's own cold-vs-warm duration without the full 94-layer graph capture,
   and would be a cheaper, cleaner way to first confirm the mechanism in isolation before re-wiring it
   into the captured per-token graph.
4. Do not re-attempt without (1) — re-running the same A/B blind risks repeating the same inconclusive
   noise rather than learning anything new.
