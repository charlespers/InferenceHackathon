# The whole-engine MBU ceiling at B=1 — the megakernel is the GATING precondition, not one lever

**LOOP-C, 2026-06-20.** Adversarial validation of the most load-bearing number in every 1000-projection:
the assumed **58–80% MBU → 700–1024 tok/s** compute. Tool: `tools/whole_engine_mbu_ceiling.py`. No GPU.
**Finding: the un-fused engine is LATENCY-bound at B=1 — its compute ceiling is ~100–325 tok/s regardless
of MBU tuning — because ~7–9 ms of the measured 10.3 ms kernels-floor is fixed per-op latency (router +
per-head attn norms + small ops × 94 layers) that MBU cannot touch. The megakernel (fusing that floor away)
is therefore the PRECONDITION for 1000, not one lever among several.**

## The conflation (two different MBUs)
`squeeze-to-700.md §1a` computes *"compute @58% MBU = 1.41 ms = 709 tok/s"* by applying the **K5 expert
kernel's** MBU (45.7→58% measured) to the **whole 2.74 GB byte budget**. But:

| MBU | value | what it is |
|---|---|---|
| per-isolated-kernel (K5) | 58% measured | the expert GEMV alone, in a microbench |
| **whole-engine effective** | **7.9%** (= 0.82 ms byte floor ÷ **10.3 ms measured** kernels-floor, 5f1150f) | the real engine |

The projection silently assumes the whole engine runs at the big kernel's MBU. It does not — and the reason
is structural, not un-tuned kernels.

## Why: most of the compute is LATENCY-bound, not BW-bound (and MBU can't help it)
Decomposing the **measured** 10.3 ms kernels-floor:

| part | ms | scales with MBU? | source |
|---|---|---|---|
| BW-bound (experts + attn weights + lm_head) @ e=0.58 | 1.41 | **yes** | byte floor 0.82 / 0.58 |
| **router (K4)** — 24 µs × 94 | **2.26** | **NO** (0.52 MB GEMV; bytes need 0.16 µs → ~0.7% MBU, pure dispatch/occupancy latency) | MEASURED, 5f1150f |
| K1 attn prologue (Qwen3 per-head q/k-norm ×68 + RoPE + small projs) + norms + ~6 small ops/layer × 94 + inter-kernel gaps | ~6.6 | **NO** (few bytes, many small latency-bound ops) | est. (K1 "44%", 5f1150f) |

The latency-bound part is **fixed ms** — it does not convert to MBU. So pushing the experts to e=1 removes
only ~0.6 ms; the engine stays ~9.7 ms → **103 tok/s**. **Reaching 58–80% *whole-engine* MBU is impossible
by kernel tuning** — the latency floor dominates the denominator.

## The ceilings (compute-only)
| scenario | ms | tok/s |
|---|---|---|
| today (e=0.58, full latency floor) | 10.30 | 97 |
| **MBU tuning ONLY** (experts e→1, floor stays) | 9.71 | **103** |
| **CONSERVATIVE measured-only** (byte floor + ONLY the measured router) | 3.08 | **325** |
| **MEGAKERNEL** fuses the latency floor away (e→1) | 0.82 | **1222** |

**Robustness:** the sensitivity sweep shows even the most generous case — counting *only* the measured
router latency and zeroing everything else — caps un-fused compute at **~325 tok/s**, still far below the
projected 709. At the realistic full floor it's ~100–200. The conclusion does not depend on the (estimated)
K1/norms split; the **measured router alone** already breaks the 709 projection.

## What it means for 1000 (compute + comms × spec)
| path | tok/s |
|---|---|
| un-fused (latency floor) + NVLS, no spec | 91 |
| un-fused + NVLS + **spec ×3** | **272** |
| **MEGAKERNEL** (latency→0, e→1) + NVLS, no spec | 650 |
| **MEGAKERNEL + NVLS + spec ×2** | **1300** |

**So "kernels + comms alone → 650–1024" is unreachable by MBU+comms tuning** (the un-fused ceiling is
~100–325, latency-bound), and **even perfect MBU + NVLS + spec ×3 stays ~300 without fusion.** The megakernel
is the one lever that moves the floor, because it is the only thing that removes per-op latency.

## Reframes
1. **The megakernel is the gating precondition for 1000, not one lever among many.** Its value is collapsing
   the ~7–9 ms per-op **latency** floor (router + norms + small ops × 94) — which is **~3× the comms term** it
   also helps. This *broadens* Alyssa's recent finding (460bba4: graphs-already-banked → megakernel's value is
   "per-collective barrier cost specifically"): the barrier cost is real but it's the *smaller* part; the
   bigger part is the whole per-kernel latency floor across all 94 layers' small ops.
2. **"1000 needs spec" → "1000 needs the MEGAKERNEL first; NVLS + spec stack on the ~1280 BW base it unlocks."**
   Order matters: fuse the latency floor → then MBU/NVLS/spec are the story; before that, they can't reach ~300.
3. **The team's own measured 76.2 tok/s is consistent** with this (10.3 ms kernels latency-bound + ~2.8 ms
   comms), and the optimistic "1024 @ 80% MBU + overlap" corner is reachable **only via fusion**, which the
   projections labeled "MBU" — this note relabels it correctly so the critical path is honest.

## Caveat (measured vs estimated)
MEASURED: byte floor 0.82 ms, kernels-floor 10.3 ms, router 24 µs×94 = 2.26 ms, K5 e. ESTIMATED: the K1/norms
split of the remaining latency floor (anchored on the "K1 44%" profile note). The headline survives on
measured-only evidence (router-alone bound = 325 tok/s). The clean confirmation is an Nsight per-kernel
timeline on the real engine (E-attr) splitting kernel-busy-at-BW from per-op latency/gaps — the same trace
that resolves the overhead fork.

## CONFIRMED by the 352-tok/s single-binary (commit 52813ed) — fusion collapses the floor, as predicted
Charles's `spec_step_e2e.cu` measured a **single-binary 8-GPU cuBLASLt fp8 forward at 352 tok/s plain**
(2.34 ms) vs the vLLM Python engine's **76 tok/s** — a **4.6× jump from going single-binary/fused**. This is
*exactly* this note's thesis: the gating issue is the per-op LATENCY floor, and removing it (compiled binary,
cuBLASLt, no per-op Python/launch) unlocks the BW regime. **The 352 proxy explicitly models only the GEMM
panels (qkv/o/gate-up/down ×94 + lm_head) + a K2 constant (500 µs) — it OMITS the router and the per-op
norms**, i.e. the very latency-floor ops this note identified. They're omitted because in a single binary they
*are* latency-negligible — which confirms the router's 2.26 ms in vLLM was per-op-launch overhead, not inherent
compute. So 76 → 352 IS "the latency floor collapsing."
- **Reconciliation, not refutation:** vLLM 76 (latency-bound, un-fused) and single-binary 352 (floor removed)
  are the two ends of this note's argument. 352 ≈ 1.84 ms GEMMs (cuBLASLt e≈0.45) + 0.5 ms K2; with comms
  added (~1 ms NVLS) it's ~260 plain — still above the ~200 the note said only fusion can cross.
- **Honest caveats on the 352 (so it isn't over-banked):** (a) dummy-weight timing proxy; (b) it OMITS the
  router + per-op norms — verify they actually fuse/overlap in the real single binary and don't re-add a floor;
  (c) cuBLASLt GEMMs at e≈0.45 → MBU tuning still has headroom toward the ~1280 BW ceiling; (d) the spec
  1048–1485 rides on this and re-confirms spec-as-GEMM flat-in-k. Net: the path is **single-binary/megakernel
  (collapse the floor, 76→~260–352) → MBU tune (→ ~1280) → NVLS + spec** — fusion first, exactly as argued.

## Is the forward→430 gate reachable? YES, but with ZERO slack — and K2 is the next floor (2026-06-20)
The native M=k spec result (bbc82a7) pinned the path: spec needs the single-forward at **~430 tok/s (2.1 ms)**
to clear 1000 (→938–1476 at τ). Bounding the forward floor from the MEASURED pieces (byte 0.818 ms @e=1; K2
flash-decode **0.50 ms** measured constant, MBU-immune; NVLS comms 0.72 ms):

| scenario | forward ms | tok/s |
|---|---|---|
| today-ish (e=0.45, router/norms floor ~7 ms unfused, NVLS) | 10.04 | 100 |
| megakernel fuses router/norms (lat→0), e=0.45, NVLS | 3.04 | 329 |
| **+ comms overlapped into kernel (comms→0)** | **2.32** | **431** ✓ |
| + MBU climb e=0.65 / 0.85 | 1.76 / 1.46 | 569 / 684 ✓ |

**430 is reachable — but only when THREE things compound at once:** (1) the megakernel fuses the router/norms
per-op latency floor to ~0, (2) comms is **overlapped into the kernel** to ~0 (not merely NVLS-fast), and
(3) cuBLASLt e≥0.45. Miss any one and it slips (fuse-lat-but-no-comms-overlap = 329, misses 430). The MBU
climb (e→0.85) is the only source of margin.

**NEW lever surfaced — K2 flash-decode is the emerging floor.** At the 2.1 ms target the 0.5 ms K2 is **24%**
of the budget (byte read is 39%). K2 is **NOT a GEMM** (attention over the KV cache) → MBU-immune and latency-
bound at B=1 short ctx; it's currently a **hardcoded constant** in `spec_step_e2e.cu` (`MK2_US=500`), not
optimized. **After router/norms, K2 is the next thing to fuse/overlap** — and because it's MBU-immune, only
fusion/overlap (not kernel-MBU tuning) reduces it. Recommend: fold K2 into the megakernel's SM schedule (it can
overlap the expert weight-stream like the NVLS reduce does) and re-measure; if K2 stays a separate 0.5 ms
launch, it caps the forward at ~684 even at e=0.85+comms→0, which still clears 430 but eats the margin spec
needs. (Open question for the on-box megakernel run: does K2 fuse, or is it a hard 0.5 ms?)
