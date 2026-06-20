# On-box kernel measurements (H100, sm_90a) — E4 — 2026-06-20

Run on the box via the SSH window (nvcc 12.6, single H100). Validates the blueprint's kernel levers
(§6.1 fused grouped-GEMV MoE, §1 int4) with **real numbers + correctness**.

## K5 fused-MoE expert kernel (gate+up / down, fp8) — `kernels/k5_microbench.cu`
`./k5bench 264 1024 3350` (CTAs=264, block=1024, peak 3350 GB/s):

| stage | ms/call | GB/s | e (frac of peak) |
|-------|--------:|-----:|----:|
| reference (scalar, 8 CTAs) | 17.52 | 8.6 | 0.003 |
| warp gate+up (A, 101 MB) | 0.106 | 949 | 0.283 |
| warp down (B, 50 MB) | 0.054 | 929 | 0.277 |
| **warp total (A+B, 151 MB)** | **0.160** | **942** | **0.281** |

- **Correct:** max_rel = 3.9e-5 vs scalar reference. 109× over scalar.
- **e = 0.281 at default params** (vs the team's *tuned* e=0.46) → tuning headroom. The blueprint's
  TMA-async + wgmma + in-mainloop-dequant path (Machete/Marlin-style) is how e→~1.0.
- MoE-only decode ×94 layers: 15.1 ms (warp) — i.e. the expert term alone, at this e, is a big chunk
  of TPOT; lifting e is a direct floor reduction.

## int4 vs fp8 expert GEMV — `kernels/k5_int4_bench.cu`  ⚠️ refines the blueprint's "int4 is the main dial"
`./i4bench 3350`:

| precision | ms/call | GB/s | e |
|-----------|--------:|-----:|----:|
| fp8 (winner) | 0.168 | 900 | 0.269 |
| int4 | 0.308 | 245 | 0.073 |

- **int4 unpack is correct** (max_rel = 8.4e-8 vs CPU ref).
- **int4 is 0.55× fp8 — SLOWER, not the 2× ideal.** The naive nibble-unpack is **issue-bound**: it eats
  the byte win and then some (int4 245 GB/s vs fp8 900).
- **Implication:** the blueprint's "INT4 ≈ another 2×" is a *bandwidth* ceiling that **only materializes
  with a Hopper-tuned dequant kernel** (TMA async loads + wgmma + dequant in the mainloop). The naive
  W4 GEMV *loses* to fp8 on H100. So int4 is the main dial **iff the kernel is right** — otherwise fp8 is
  the better weight precision today. This is the E4 open question ("does the unpack eat the byte win?")
  answered: **yes, for the naive kernel.** Prioritize the Machete/Marlin-style W4 path before banking the
  int4 ceiling.

## Ties to the floor-bound thesis
Even the fp8 fused kernel runs at e≈0.27–0.28 (not 1.0) at default tuning → kernel inefficiency is itself
a floor contributor alongside comms/launch. Sequence: CUDA graphs (k6) + comms fusion + **K5 e→1 (tune)**
to drop the floor, *then* the weight-precision ceiling (and only with a proper W4 kernel) becomes visible.
