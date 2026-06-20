# NVLS → TP8 decode step: projected win + integration plan — 2026-06-20

Two measured results now compose into the comms win on the **working** TP=8 step:
- `decode_step_tp8.cu` (`5f1150f`): real TP=8 sharded decode, **70.1 tok/s** (= 14.27 ms/token), using
  **188 NCCL all-reduces/token** (2/layer × 94 + head).
- `nvls_ar.cu`: NVLS in-switch all-reduce, **C = 3.84 µs** (validated) vs NCCL **~33 µs** (E0).

## Projection (rigorous — comms is exactly linear in the measured per-collective C)
| | per-collective C | comms/token (188×) | TPOT | tok/s |
|---|---:|---:|---:|---:|
| baseline (NCCL) | 33 µs | 6.20 ms | 14.27 ms | **70.1** (measured) |
| **+ NVLS swap** | 3.84 µs | 0.72 ms | 8.79 ms | **~114** (1.6×) |
| + deferred-overlap (comms→0) | hidden | ~0 | ~8.07 ms | **~124** |
| + CUDA-graph (k6, kills ~1.1 ms/tok launch) | — | — | ~6.9 ms | **~145** |
| + spec (floor-aware big trees, ~3×) | — | — | — | **~300–450** |
| toward ~1000 | also needs K5 kernel e: 0.28 → ~0.6–1.0 (the compute floor) | | | |

Compute time (TPOT − NCCL comms) = 14.27 − 6.20 = **8.07 ms/token** is the invariant; every row above
shrinks comms and/or launch around it. The NVLS swap alone is a **~1.6× single-stream win**, free of any
accuracy cost (it's the identical all-reduce, just in-switch). *(Baseline 70.1 from `d5c227c`; live re-run
in progress to confirm the exact comms fraction and refine the table.)*

## Integration plan (minimal-risk, NCCL stays the default)
1. **mc_setup** (reuse `nvls_ar.cu`): allocate the two per-layer all-reduced activation buffers
   (`[HIDDEN]`, 16 KB) in multicast-mapped symmetric memory, once, outside the hot loop.
2. **`--nvls` flag** in `decode_step_tp8.cu`: replace each `ncclAllReduce(partial→full)` with a
   `multimem.ld_reduce`+`multimem.st` all-reduce on the MC buffer. NCCL path stays default → the 70 tok/s
   baseline is never at risk; A/B is one flag. Keep the file's existing cross-rank correctness check.
3. **Cross-rank barrier:** NCCL gave this implicitly; NVLS needs an explicit one. v1 = host
   `cudaStreamSynchronize` across the 8 rank-streams per collective (simple, adds host overhead). v2 =
   in-kernel flag barrier so the reduce composes inside a **single CUDA-graph capture (k6)** with no
   per-collective host return — this is what unlocks the deferred-overlap and the ~145+ rows.
4. **Deferred-overlap:** run the reduce on ~2–8 SMs while the rest `cp.async`-stream the next op's fp8
   weights (NVLink vs HBM = different paths). At C=3.84 µs < ~4.3 µs weight-cover the reduce fully hides.

## Why this is the right next step
Comms was the dominant floor term (E0: 6.6 ms, ~75% of TPOT) and env-tuning is dead (E0b). NVLS is the
only thing that moves it, and it stacks multiplicatively with the already-validated levers
(route-prefetch 0.718, floor-aware spec). The swap is a clean, accuracy-free 1.6×; overlap+graph+spec
take it from there toward the ~1000 path.
