# k6 deferred-overlap — the exactness gate (token-identical test + invariant)

**LOOP-C, 2026-06-20.** The deferred-overlap megakernel (`kernels/k6_overlap_decode.cu`, my schedule) claims
the comms-hide is **LOSSLESS** — token-identical to serial, because the EXACT NVLS all-reduce still runs, just
concurrently with the next weight stream. That claim needs a gate before the team banks "lossless." This
specifies the **exactness invariant** and the **test methodology**, and shows Alyssa's flagged `expert_gemv`
dequant-scale gap is a **correctness** bug (breaks losslessness), not just a compile error. Supports
Charles/Alyssa's k6 wiring — it does not touch their kernel; it's the test they run when it compiles.

## The key insight that sets the test: overlap changes SCHEDULING, not ARITHMETIC
Deferred-overlap only changes *when* the all-reduce runs relative to the weight `cp.async` — it does **not**
change the reduction algorithm (same `multimem.ld_reduce`), the operand order, or the dequant. Therefore the
correct expectation is **bit-exact equality** with the non-overlapped version, **not** a tolerance. Any
deviation is a real bug (a race or a wrong dequant), not benign fp reordering. This is sharper than the
kernel's current TODO ("parity vs the bf16 reference"), which would conflate two unrelated things:
- **fp8 vs bf16** → shows fp8 *quantization* error (expected, ~1e-2; not what we're testing).
- **fp8-overlap vs fp8-serial** → must be **BIT-EXACT (0 ULP)**; this isolates the overlap correctness. ← the gate.

So the reference is the **fp8-serial** path (same kernels, `N_REDUCE_BLOCKS` reduce done *before* the dependent
op, no concurrency), not the bf16 model.

## The exactness invariant (three conditions; all must hold)
Let `AR(L)` = the post-attn / post-MoE all-reduce of layer L, and `DEP(L)` = the first op that reads its result
(the MoE gate/up `expert_gemv`, or next-layer QKV).

- **(C1) Read-after-write across the barrier.** Every block that reads the reduced activation must run *after*
  the `grid.sync()` that follows `multimem_allreduce_8kb`. In the skeleton this holds (the `grid.sync()` sits
  between the reduce and `expert_gemv`). **Gate:** no dependent read may be hoisted before that barrier — verify
  no compiler/manual reordering moves an `act` read above it (a `__threadfence_block` is *not* sufficient; the
  reduce is grid-wide, so only `grid.sync()` orders it).
- **(C2) smem lifetime / WAR.** `stream_weight_tile` (the overlapped `cp.async` for the *next* op) writes the
  same `smem` that `expert_gemv` reads. The `cp.async.wait_group` + block barrier must complete before the GEMV
  reads smem, AND the prefetch must not clobber smem still feeding the *current* reduce/compute. With one `smem`
  arena and double-buffering absent, this is the easiest place to get a silent partial-read. **Gate:** distinct
  (or correctly ping-ponged) smem regions for in-flight prefetch vs in-use operands.
- **(C3) Arithmetic identity — THIS is Alyssa's gap.** The overlapped `expert_gemv` must apply the *identical*
  fp8 per-row dequant as the serial reference. k6 declares `expert_gemv(const half* x, const void* w_smem,
  half* y, int rows, int k)` — **no `scales` parameter** — but k5's fp8 math (`deq2()` in
  `k5_experts_pipelined.cu`) needs `const half* scales` (per-row block scale). **fp8 weights with no scale =
  wrong magnitude** (Alyssa, `k6_device_functions.cu:60-72`). A build that drops `scales` *compiles and runs*
  but produces wrong numbers → it would pass a "does it run / does it not NaN" check and **fail the bit-exact
  gate.** So C3 is exactly why the gate must be bit-exact, and why the missing `scales` is load-bearing for the
  losslessness claim, not a casting nit. **Required fix (per Alyssa, confirmed here):** widen k6's `expert_gemv`
  extern decl + call sites to pass `const half* scales` — not optional.

## The test (ready to run when k6 compiles; single-GPU first, then TP8)
1. **Build two variants from the same source:** `k6_overlap` (concurrency on) and `k6_serial` (force the reduce
   to finish before any dependent op — e.g. `N_REDUCE_BLOCKS = gridDim` for that phase, or a compile flag that
   serializes reduce→sync→compute). Identical weights, identical input activation, fixed RNG off (greedy).
2. **Dump `act` after every layer** (or at least layers {0, 1, 47, 93}) for both variants.
3. **Assert bit-exact:** `max |act_overlap − act_serial| == 0` (0 ULP). Non-zero ⇒ a C1/C2 race or a C3 dequant
   mismatch — bisect by layer and by phase (post-attn AR vs post-MoE AR).
4. **Separately**, compare `k6_serial` (fp8) vs the **bf16 HF reference** for the *expected* fp8 quant error
   (≤~1e-2 relative, greedy-token-identical over ≥64 tokens) — this validates the fp8 path itself, distinct from
   the overlap. Two tests, two references; don't merge them.
5. **TP8:** repeat across all 8 ranks; the existing cross-rank check (decode_step_tp8.cu: reduced == sum over
   ranks) covers the AR value; this gate adds that the *overlapped schedule* preserves it bit-exactly.

## Liveness caveat (separate from output correctness)
`grid.sync()` requires a **cooperative launch** with the whole grid co-resident in one wave. If
`N_REDUCE_BLOCKS + stream/compute blocks` exceeds the SM occupancy for the chosen smem/registers, the grid
barrier **deadlocks** (hang, not a wrong answer). Verify `cudaOccupancyMaxActiveBlocksPerMultiprocessor ×
N_SM ≥ gridDim` before trusting any timing. This is the constraint behind keeping `N_REDUCE_BLOCKS` small.

## Net
The losslessness claim is checkable and the check is simple and strong: **fp8-overlap must equal fp8-serial bit
for bit.** That single assertion catches all three failure modes (C1 race, C2 partial-read, C3 missing dequant
scale). Until it passes, "lossless deferred-overlap" is a design intent, not a measured fact — same discipline
as the stale-TP kill. When it passes on-box, the comms-hide is *proven* lossless and can be banked.
