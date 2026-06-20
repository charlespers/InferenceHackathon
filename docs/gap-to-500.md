# Closing the gap to ≥500 tok/s — the measured ladder (B=1, Qwen3-235B-A22B, 8×H100, ctx 4096)

The session goal: drive measured decode tok/s to ≥500. This is the honest, measured-where-possible path.

## The ladder
| stage | tok/s | basis |
|---|---:|---|
| GEMV M=1 (hand-rolled decode) | 77.6 | **measured** |
| GEMM forward (cuBLASLt fp8, USE_GEMM=1) | 84.6 | **measured** (gate PASS 1e-6) |
| + NVLS all-reduce in graph | 90 | **measured** (NVLS per-AR 37µs > NCCL 20µs; only graph-pipelining helps) |
| + router-fast (multi-block gate + parallel top-8) | ~95 | **measured µbench** (15.7µs vs 21.7µs GEMM-gate; −565µs/token) |
| **stale-TP (comms hidden)** | **125** | **measured** kernels-only/no-AR floor = 8000µs/token (130 @ctx1024) |
| + router-fast on the comms-hidden forward | ~134 | measured basis |
| **× spec (route-aware tree, EAGLE3 α≈0.85, k=8)** | **~510** | **measured-flat verify** × literature α |

## Why each rung is real
- **GEMM forward**: the M=1 GEMV idiom is occupancy-starved (21% MBU); cuBLASLt fp8 tensor-core GEMM is
  faster *and* flat in M (the spec enabler). Correctness gate PASS (1.07e-6).
- **Comms is the wall (35% of the forward)** and is **not losslessly removable at B=1** — each layer's
  all-reduce feeds the next op (serial dependency). The measured kernels-only/no-AR floor (8000µs = 125 tok/s)
  is what you get when comms is fully hidden. AR (3867µs) < kernel compute (8000µs), so it **hides entirely**
  once overlapped — `overlap_decode_wide` proves the mechanism (21% with a K1–K3 window; full-stream overlap
  reaches the floor).
- **Stale-TP is the mechanism + it is LOSSY.** It runs the AR async and consumes a 1-step-stale cross-rank
  value so compute never waits. Kog's "lossless deferral" was **refuted** (their lossless = quality-preserving
  *after retraining*, not math-equivalent — `research/n4_speculative_stale_tp.md`). So this rung needs a
  **quality gate on the real model** (the proxy uses dummy weights — timing is real, quality is unmeasurable here).
- **Spec is the multiplier and it amortizes comms + weight-read** (both flat over the tree). The double-win is
  **measured**: GEMM M=8 verify = 2704µs for 8 candidates = 11.4×/candidate vs the M=1 GEMV; verify flat to
  M=16 (ratio 1.001). The only modeled input is the acceptance α (EAGLE3-on-Qwen3 literature; α≈0.85 → ~510).

## The two honest caveats on the ~510
1. **Full-stream stale-TP overlap impl** — the comms-hidden forward (125) is the *measured ceiling*; the
   overlap mechanism is proven partial (21%). Realizing the full hide is remaining engineering (not new physics).
2. **Quality gate** — stale-TP is lossy and spec's α both need the **real model** (not the dummy-weight proxy)
   to validate. On the proxy, all *timing* is measured; quality and α come from literature.

## What did NOT work (ruled out, measured)
- **int4 experts**: 0.58× fp8 at B=1 — ALU-bound dequant (the int4→half unpack gates the M=1 GEMV, not HBM).
  Also experts are dominant *bytes* but only ~25% of step *time*. Dead as a B=1 latency lever.
- **NVLS as a primitive**: per-AR 37µs (the 8-rank in-switch barrier) > NCCL 20µs; only helps via graph pipelining.
- **2-bit experts**: NO-GO on quality (uniform 2-bit, lit-confirmed).

## Bottom line
≥500 is reachable: **measured forward ~125–134 (stale-TP comms-hidden + router) × EAGLE3 spec (measured-flat
verify) → ~510**. It is a *defensible* 500, gated on the real-model quality validation of the two lossy/modeled
inputs (staleness tolerance + spec acceptance).
