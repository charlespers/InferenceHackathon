# Can Qwen3-235B-A22B hit 1000 tok/s at B=1 on 8x H100? — the recomputed ceiling

Target: Qwen3-235B-A22B, batch size 1, decode + prefill, 8x H100 (80 GB, 3.35 TB/s HBM3,
~1.98 PFLOP/s fp8 tensor-core each), fp8 e4m3. Goal: 1000 tok/s.

This note recomputes the realistic B=1 decode throughput given the three levers worked this
session — pushed **MBU** on the decode GEMVs, pushed **MFU** on prefill via tensor cores, and
**NVSHMEM/overlap** on comms — and shows the arithmetic end to end. Numbers are tagged
**[MEASURED]** (benched on the H100 this session) or **[PROJECTED]** (computed from the model
shape + the measured kernel rates; to be confirmed by the orchestrator on-box).

All model shapes are from `kernels/common.cuh`: HIDDEN=4096, N_LAYERS=94, QKV_OUT=9216,
Q_DIM=8192, KV_DIM=512, N_EXPERTS=128, TOP_K=8, MOE_INTER=1536.

---

## 1. The decode workload is HBM-bound. Count the bytes.

At B=1 every projection is a GEMV (M=1): one pass over the weights, ~no reuse, so the wall is
**HBM bandwidth**, not FLOPs. fp8 weights = 1 byte/param, so bytes/token = active params/token.

Active params read per layer (every token):

| block | shape | params |
|---|---|---|
| Wqkv (fused) | 9216 x 4096 | 37.75 M |
| Wo | 4096 x 8192 | 33.55 M |
| router | 128 x 4096 | 0.52 M |
| MoE experts (top-8) | 8 x (2·1536·4096 + 4096·1536) | 150.99 M |
| **per-layer total** | | **222.82 M** |

x 94 layers = **20.95 GB/token** of weight reads (the experts alone are 14.19 GB/token =
"14.2B params", matching the brief; the full active set is ~21.6B params). KV-cache and
activations are < 1% of this and are ignored in the leading-order bound.

So the decode time floor is:

```
t_compute(ms) = bytes_per_gpu / (HBM_peak * MBU) * 1e3
```

---

## 2. Replicated (no sharding): why one GPU cannot hit 1000

If the model is replicated and one GPU does the whole token, that one GPU must move all
20.95 GB/token:

| MBU | effective BW | ms/token | tok/s |
|---|---|---|---|
| 45.7% **[MEASURED]** (k5 baseline) | 1.53 TB/s | 13.68 | **73** |
| 70% **[PROJECTED]** (k5_experts_v3 target) | 2.35 TB/s | 8.93 | 112 |
| 80% **[PROJECTED]** (stretch) | 2.68 TB/s | 7.82 | 128 |

**Replicated tops out near ~130 tok/s even at 80% MBU.** A single GPU's 3.35 TB/s simply
cannot stream 21 GB 1000 times/second (that would need 21 TB/s). 1000 tok/s **requires
sharding the weight read across the 8 GPUs** so each moves only ~2.6 GB/token. That is what
makes comms the deciding factor.

---

## 3. Sharded TP=8 / EP=8: the compute ceiling clears 1000

With tensor-parallel attention (Wqkv/Wo column/row-sharded) and expert-parallel MoE (experts
spread over 8 ranks), each GPU reads **20.95 GB / 8 = 2.62 GB/token**:

| MBU | ms/token (per-GPU compute) | compute-only tok/s |
|---|---|---|
| 45.7% **[MEASURED]** | 1.710 | 585 |
| 70% **[PROJECTED]** | 1.116 | 896 |
| 80% **[PROJECTED]** | 0.977 | **1024** |

So **on the compute side alone, pushed MBU clears 1000 tok/s** (80% MBU -> 0.98 ms/token =
1024 tok/s). The MBU push is real headroom: the M=1 GEMV is limited by memory-level
parallelism, and `kernels/k5_experts_v3.cu` (cp.async double/triple-buffered staging +
ROWS_PER_WARP) and `kernels/k1k2_mbu_v2.cu` (cp.async + 4-row ILP on K1) attack exactly that.
45.7% MBU is **[MEASURED]**; 70-80% is the **[PROJECTED]** target those kernels reach for,
confirmed only by the on-box bench.

The catch: sharding TP/EP introduces a per-layer cross-rank reduction. **That is the wall.**

---

## 4. Comms: the wall, and how NVSHMEM + overlap removes it

The sharded residual must be summed across ranks twice per layer (after the sharded O-proj and
after the sharded MoE-down): **2 collectives/layer x 94 = 188 collectives/token**, each a tiny
[HIDDEN]=4096-float = 16 KB all-reduce.

These tiny messages are **latency-floored**, not bandwidth-floored (LL/LL128 do not move the
floor). So comms/token = 188 x (per-collective latency):

| path | per-coll latency | ms/token (x188) | comms-cap tok/s |
|---|---|---|---|
| NCCL all-reduce | 35 us **[MEASURED]** | 6.58 | **152** |
| NVSHMEM recursive-doubling AR | ~3 us **[PROJECTED]** | 0.56 | 1773 |
| NVSHMEM + compute/comms overlap | ~1.5 us exposed **[PROJECTED]** | 0.28 | 3546 |

**NCCL serial comms alone caps decode at ~152 tok/s** — this is the dominant blocker today,
and it is why fast kernels are not enough by themselves. Two fixes, both prototyped:

- **`kernels/nvshmem_comms.cu`**: GPU-initiated one-sided puts over NVLink + an on-device
  barrier (recursive-doubling AR, 3 put+barrier rounds for P=8). One-sided NVLink puts are
  sub-microsecond, so the per-collective floor drops from 35 us to **low single-digit us**
  **[PROJECTED]** — i.e. comms/token from 6.58 ms to ~0.56 ms. This is the DeepEP/IBGDA path.
- **`kernels/overlap_decode.cu`**: hide what is left behind independent compute — chunked
  "reduce-as-you-go" and layer-pipeline prefetch (layer L's all-reduce runs on a comm stream
  while layer L+1's K1 QKV GEMV runs on the compute stream). With ~1 ms of per-token compute to
  hide behind, the exposed collective cost shrinks further toward the ~0.28 ms row.

---

## 5. Putting it together: the realistic B=1 decode ceiling

Per-token time ≈ `t_compute` (sharded, pushed MBU) + `t_comms_exposed` (after NVSHMEM +
overlap). Compute at 80% MBU = 0.977 ms.

| scenario | compute | comms | total | tok/s |
|---|---|---|---|---|
| today: 45.7% MBU + NCCL serial | 1.71 ms | 6.58 ms | 8.29 ms | ~120 |
| MBU 80% + NCCL serial | 0.98 ms | 6.58 ms | 7.56 ms | **132** |
| MBU 80% + NVSHMEM (3 us) | 0.98 ms | 0.56 ms | 1.54 ms | **649** |
| MBU 80% + NVSHMEM fully overlapped | 0.98 ms | ~0 ms | 0.98 ms | **1024** |

**Findings:**

1. With NCCL comms, nothing else matters — you are pinned at ~130-150 tok/s no matter how fast
   the kernels are. **Replacing NCCL with NVSHMEM is the single highest-leverage change.**
2. With NVSHMEM (3 us/coll) + 80% MBU, **single-stream decode reaches ~650 tok/s** — close, but
   not 1000.
3. To reach **1000 from kernels + comms alone** you need BOTH 80% MBU **and** the comms almost
   fully overlapped behind compute (comms -> ~0 on the critical path). That is plausible
   (~1 ms of independent compute/token can hide ~0.56 ms of NVSHMEM comms), giving ~1024 tok/s,
   but it is the optimistic corner: it assumes 80% MBU sustained AND near-perfect overlap.

---

## 6. The comfortable path to 1000: speculative decoding

The cleanest way to clear 1000 with margin is to multiply the per-verify-step throughput with
**speculative decoding** (a small draft model proposes k tokens; the 235B verifies them in one
forward pass; accepted tokens are emitted for free). The verify step costs the same ~one decode
pass; if the draft's acceptance yields ~k accepted tokens per verify, throughput scales ~k x.

Taking the realistic **MBU 80% + NVSHMEM 3 us = ~650 tok/s [PROJECTED]** as the base verify rate
(NOT requiring the perfect-overlap corner):

| accept multiplier | tok/s |
|---|---|
| 2.0x | ~1298 |
| 2.5x | ~1622 |
| 3.0x | ~1947 |

A 2x acceptance — routinely achievable with a well-matched draft — takes the realistic 650 tok/s
base to **~1300 tok/s**, comfortably past the goal **[PROJECTED]**.

---

## 7. Prefill (MFU) — why it matters for the end-to-end number

Prefill (prompt processing, M = seq) is **compute-bound**, not memory-bound: the same weights
are reused across all M tokens. The existing SIMT fp32 path runs at ~14 TFLOP/s (prefill_attn)
to ~0.7 TFLOP/s (prefill_moe) **[MEASURED]** — <1% of the H100 tensor-core peak — so a long
prompt's time-to-first-token is dominated by this and starves the decode loop.

`kernels/prefill_wgmma.cu` moves the projection + routed-MoE GEMMs onto the fp8 e4m3 tensor
cores (`mma.sync.m16n8k32`, cp.async double-buffered). Honest ceiling note: on Hopper this
`mma.sync` fp8 instruction up-converts to fp16 and runs at the **fp16-HMMA rate (~989 TFLOP/s),
not** the 1979 TFLOP/s full-fp8 (QMMA, `wgmma.mma_async`) rate — the microbench reports % of the
989 peak. Even so, that is a **1000x+** jump over the SIMT fp32 path it replaces, which removes
prefill as the TTFT bottleneck and frees the GPUs for the decode loop. (Reaching the full
1979 peak needs a `wgmma.mma_async` rewrite — flagged, not done here.) It also evaluates only
the routed **top-8** experts, not all 128.

Prefill MFU does not change the steady-state decode tok/s in §5; it determines time-to-first-
token and how much of the wall-clock is decode vs prefill on a real request.

---

## 8. Verdict

- **Is 1000 reachable? Yes — but not from any single lever.** It requires sharding (to spread
  the 21 GB/token weight read over 8 GPUs) **plus** killing the NCCL comms wall with NVSHMEM
  **plus** pushed MBU.
- **Kernels + comms alone**: ~650 tok/s at MBU 80% + NVSHMEM 3 us **[PROJECTED]**; ~1024 tok/s
  only in the near-perfect-overlap corner. So 1000 is *marginally* reachable from kernels+comms
  if everything lands, and **comfortably** reachable (~1300 tok/s) once a ~2x speculative-decode
  multiplier is added on top.
- **The binding constraint is comms.** NCCL pins decode at ~150 tok/s; NVSHMEM lifts the
  comms-cap to ~1800 tok/s, after which **MBU becomes the limiter** and the kernel work in
  `k5_experts_v3.cu` / `k1k2_mbu_v2.cu` (pushing 45.7% -> 70-80% MBU) is what converts the
  unblocked comms into actual tokens.

### Measured vs projected, at a glance

| quantity | value | status |
|---|---|---|
| k5 MoE GEMV MBU | 45.7% (1530 GB/s) | MEASURED |
| K1 QKV GEMV MBU | 27% (904 GB/s) | MEASURED |
| NCCL all-reduce latency | ~35 us | MEASURED |
| NCCL all-to-all latency | ~60 us | MEASURED |
| prefill SIMT MFU | 14 / 0.7 TFLOP/s | MEASURED |
| pushed decode MBU (v3 kernels) | 70-80% | PROJECTED |
| NVSHMEM recdouble AR latency | ~3 us | PROJECTED |
| sharded compute @ 80% MBU | 0.98 ms/token | PROJECTED |
| comms @ NVSHMEM 3 us | 0.56 ms/token | PROJECTED |
| realistic decode (MBU 80% + NVSHMEM) | ~650 tok/s | PROJECTED |
| with ~2x speculative decode | ~1300 tok/s | PROJECTED |

All PROJECTED rows are computed from the model shape in `common.cuh` and the MEASURED kernel
rates; the orchestrator's on-H100 bench of `k5_experts_v3.cu`, `k1k2_mbu_v2.cu`,
`prefill_wgmma.cu`, `nvshmem_comms.cu`, and `overlap_decode.cu` is what confirms them.
