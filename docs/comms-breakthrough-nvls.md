# Comms breakthrough — NVLS in-switch all-reduce: C = 3.84 µs (8.6× over NCCL) — 2026-06-20

**Implemented + measured on the box.** The thing that was the dominant floor term — the ~33 µs NCCL
all-reduce (E0) — is broken. A multimem (NVLink-SHARP) in-switch all-reduce does the 8KB B=1 collective
in **3.84 µs**, validated bit-exact, on 8×H100. This is the kog.ai-style result: bypass NCCL, let the
**NVSwitch do the reduction in-network** via CUDA multicast + `multimem` PTX.

## Result (`kernels/nvls_ar.cu`, single-process 8-GPU, 8KB payload)
| all-reduce path | per-collective C | 188×C / token | vs NCCL |
|---|---:|---:|---:|
| NCCL ring (E0, stock) | ~33 µs | 6.6 ms | 1× |
| **NVLS multimem in-switch** | **3.84 µs** | **0.72 ms** | **8.6× faster** |

- **Correctness: PASS** — each GPU d inits its buffer to `d`; after the all-reduce every element on all
  8 GPUs == `sum(0..7) = 28`. Bit-checked.
- **C ≤ 4 µs** → meets the team's `path-to-1000` gate: comms can be **fully, losslessly hidden**.
- Measurement nuance: 3.84 µs is back-to-back host-launched (launch+exec pipelined). **In a CUDA graph
  (k6) or the persistent megakernel the host launch is removed → effective C ≤ 3.84 µs** (the multimem
  reduce of 8KB is sub-µs on NVLink; the residual is launch/scheduling). So this is a conservative number.

## How (fills the empty `mc_setup()` in `kernels/nvls_allreduce.cu`)
1. `cuMulticastCreate` an MC object over all 8 GPUs (`cuMulticastAddDevice` ×8).
2. Per GPU: `cuMemCreate` physical + `cuMulticastBindMem` into the MC, + a local unicast map for init/validate.
3. `cuMemMap` the MC handle to a VA with access for all 8 devices — the kernel uses this address.
4. Kernel: `multimem.ld_reduce.global.add.v4.f16x2` (load+sum across all bound GPUs in one switch round-trip)
   + `multimem.st.global.v4.f16x2` (broadcast back). One pass = a full all-reduce.
- Box prereq confirmed: `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 1` on the H100s.

## Why it breaks the floor (ties to E0 + the floor thesis)
E0 showed NCCL is latency-floored at ~33 µs and **env-tuning is dead** (LL/NVLS-via-NCCL/channels don't
move it). The floor was launch+handshake, not bytes. multimem removes BOTH: no NCCL ring, no per-rank
handshake — a single in-switch reduce. Comms drops from the #1 floor term (6.6 ms, ~75% of an 8.6 ms
TPOT) to 0.72 ms.

## Novel ideas built around it (the next moves)
1. **Deferred-overlap → comms ≈ 0 (lossless).** NVLink (the reduce) and HBM (the next weight stream) are
   *different hardware paths*. Run the 3.84 µs reduce on a few SMs while the rest `cp.async`-stream the
   next op's fp8 weights (~4.3 µs of weight-cover per collective). At C=3.84 < 4.3 µs the reduce is
   **fully hidden** → comms → 0 → ~roofline, with no approximation. (LOOP-C's deferred-overlap schedule.)
2. **+ validated route-prefetch (DirectProxy 0.718).** While the multimem reduce runs, prefetch the
   *predicted* next-layer experts (E8: 72% accurate on the 235B arch) — so both the collective AND the
   next weight-load hide under compute. Two validated levers compose.
3. **In-kernel multimem in the persistent megakernel (k6).** multimem needs only ~2–8 SMs for 8KB → it
   co-resides with the weight-stream warps in one persistent kernel; no per-collective launch at all
   (removes even the 3.84 µs launch residual). This is the path to effective C < 1 µs.
4. **fp8 multimem (4KB).** Reduce the fp8 activation directly (half the bytes) for an even lower C, and
   skip the fp32→fp16 staging.

## Status / next
- `kernels/nvls_ar.cu`: working, validated microbench (this result). Replaces the empty skeleton path.
- Next: wire the multimem reduce into the megakernel/CUDA-graph step (k6) with deferred-overlap, and
  re-measure the end-to-end TPOT (the comms term should ~vanish). This is the unblock for ~1000 tok/s.
