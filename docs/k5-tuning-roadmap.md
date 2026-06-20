# K5 e=0.46→1 roadmap — the biggest residual lever (the 750→2000 stretch)

`absolute-ceiling.md`: the cheap+comms+spec wins get ~750 tok/s (~37% of the ~2000 ceiling); the hard residual
is driving kernel efficiency e→1 (K5 is at 0.46; perfect is 1.0) + comms→0 + the fast-path. This is the K5 path
to e→1 — the candidate optimizations in likely-impact order, each analyzable from the kernel structure + the
roofline, with how to measure on the next GPU slot. (Measured baseline: `k5_experts_warp.cu`, e=0.459, 264
CTAs × 1024 threads, warp-per-row + split-K + 128-bit fp8 loads + smem-x + hoisted scale + fp8x2→half2.)

## The diagnosis: e=0.46 at B=1 GEMV ⇒ HBM latency isn't fully hidden
A B=1 expert GEMV is pure memory streaming (AI≈1). 46% of peak almost always means **insufficient
memory-level parallelism (MLP)** — not enough in-flight loads to cover HBM latency (~500ns). The fix is more
concurrent outstanding loads, not more FLOPs. The levers below all increase MLP or cut non-overlapped work.

## Candidate optimizations (priority = expected Δe)
1. **`cp.async` double/triple-buffering (the big one).** Prefetch the next weight tile into smem with
   `cp.async.cg.shared.global` while the current tile is consumed → the HBM latency overlaps compute instead
   of stalling the warp. At B=1 this is *the* lever (it directly raises MLP). Expect the largest single jump
   (0.46 → ~0.65–0.75). Stage 2–3 buffers; tune the stage count to the smem budget.
2. **Occupancy / launch shape.** 1024 threads/CTA may be register-limited (few CTAs resident → low MLP). Sweep
   {256, 512} threads/CTA × more CTAs, and check `--ptxas-options=-v` register count vs the 64K/SM file →
   target ≥ 2–3 *resident* CTAs/SM so loads from different warps overlap. (Higher occupancy = more in-flight
   HBM requests = the same MLP goal as #1.)
3. **Wider vectorized loads end-to-end.** Confirm every global load is 128-bit (`uint4`/`float4`) for *both*
   the fp8 weights and the partial-sum I/O; a single 32-bit load path anywhere caps the achieved BW. The
   dequant should consume `uint4`→ 8×fp8 → 4×half2 in registers.
4. **Split-K reduction cost.** The cross-split reduction (atomics or a 2nd pass) is non-overlapped tail work.
   Measure its share; if >5%, switch to a tree reduction in smem or tune the split count down (fewer splits =
   less reduction, but watch MLP). Split-K helps coalescing but its reduction is pure overhead — find the knee.
5. **Dequant overlap.** The fp8x2→half2 convert is on the load→compute path; ensure it's issued so it overlaps
   the *next* tile's `cp.async` (it's cheap ALU, must not serialize behind the load). With #1 this is free.
6. **L2 is useless here (don't chase it).** At B=1 each expert weight is read exactly once (no reuse) and 22B
   ≫ 50MB L2 → no residency win. Don't waste effort on L2 hints; the game is HBM→smem MLP.

## How to measure each (next GPU slot, `k5_microbench.cu`)
- Report `e` = achieved DRAM BW / peak (the bench already does). Add **Nsight Compute** `gld_efficiency`,
  `dram__throughput.pct_of_peak`, `sm__warps_active` (occupancy), and `stall_long_scoreboard` (the HBM-latency
  stall — #1 should crush this).
- A/B each lever in isolation; keep the winner. Expected ladder: cp.async (#1) → occupancy (#2) → vectorization
  (#3) should reach **e ≈ 0.75–0.85**; the last 0.85→1.0 is diminishing (tail effects, the reduction).
- **Stop rule:** when `dram__throughput.pct_of_peak` > ~85% you're physics-limited — `e` won't improve and the
  next tok/s comes from *fewer bytes* (fp8→int4) or *spec*, not the kernel (`absolute-ceiling.md`).

## Where this sits
e=0.46→0.85 turns the K5 line of the budget (`latency_budget.py --eff`) from ~750 toward the ceiling, and it's
the part vLLM's generic `fused_moe` (e≈0.16) can't give — the reason the custom kernel (and the cudarc engine
hosting it) earns its place. It's GPU-bound work (must be measured); this roadmap is the plan for the slot, not
a claim. The **verify** path wants the *batched* grouped-GEMM variant instead (`why-spec-wins.md`) — a separate
kernel, tuned for AI≫1, where tensor-core MFU (not HBM MLP) is the target.
