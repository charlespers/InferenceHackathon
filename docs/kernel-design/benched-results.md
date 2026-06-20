# Benched results — Qwen3-235B-A22B B=1 on 8×H100 (real, this session)

All numbers below are **measured on the H100 box** via `nvcc`-compiled CUDA microbenchmarks
(`cudaEvents`, CPU-fp32 reference checks) or the real vLLM server on `:8001` — **not** the
`:8000` demo mock. HBM peak = 3350 GB/s/GPU. Anything marked *projected* is arithmetic, not measured.

## Decode kernels — MBU (% of 3.35 TB/s)
| kernel | MBU | GB/s | correctness | note |
|---|---|---|---|---|
| K5 fp8 MoE (baseline) | 45.7% | 1530 | PASS | warp-per-row coalesced fp8 |
| **K5 v3 (cp.async, R=4, STAGES=2)** | **58.1%** | **1947** | PASS | +27% via pipelined loads + 4 rows/warp |
| **lm_head (warp+argmax)** | **55.4%** | 1855 | PASS (argmax matches) | 8.3× over naive |
| O-proj / K3 | 38.9% | 1302 | PASS | |
| K1 prologue | 27.0% | 904 | PASS | 12× over original; v2 regressed (798) |
| K2 flash-decode | ~3% | 95 | PASS | overhead-bound (4 MB KV read) |
| int4 MoE v2 (n7-fixed) | — | 461 (raw) | PASS | **169 µs vs fp8 98 µs — unpack-bound, NOT a win yet** |

## Prefill — MFU (tensor cores)
| kernel | MFU | TFLOP/s | note |
|---|---|---|---|
| prefill SIMT (old) | <1% fp16-TC | 7–14 | no tensor cores |
| **prefill wgmma proj** | **15.5% fp16-TC** | 153.6 | mma.sync fp16; ~7.8% of fp8-QMMA peak — more headroom with fp8 wgmma |

## Fused single-GPU decode (full 21.96 GB model on ONE GPU)
8.4 → 14.1 (K1 fix) → **30.9 tok/s** (K2 fix). Capped at ~153 (full model on one card); a **kernel-integration test, not a deployment number.**

## Comms — THE WALL (measured NCCL, 8 GPU, small messages)
| collective (16 KB, 8 PE) | latency | note |
|---|---|---|
| NCCL all-reduce | **35 µs** | latency-floored; LL/LL128 don't help |
| NCCL all-to-all | **60 µs** | |
| **NVSHMEM recdouble all-reduce** | **49 µs** | correct (maxerr 1.4e-6); *worse* than NCCL (barrier-per-round) |
| **NVSHMEM put+barrier all-to-all** | **17 µs** | correct; **3.5× faster than NCCL** — the EP lever |
| overlap (scheme C) | hides **14%** | B=1 has ~no independent compute to hide behind |

188 collectives/token (2/layer × 94). **Best measured comms = NVSHMEM all-to-all 17 µs × 188 = 3.2 ms/token → EP sharded B=1 comms-capped at ~310 tok/s.** NCCL all-reduce path is worse (~90–150). All over the **~1 ms** budget that 1000 tok/s needs.

- **NVSHMEM is now BUILT + RUNNING** (8 PEs, NVLink P2P): `pip install --target nvidia-nvshmem-cu12` (version-matched to the 12.6 `nvcc`) + unversioned-`.so` symlinks + MPI bootstrap (`mpirun -np 8`, `NVSHMEM_REMOTE_TRANSPORT=none` to skip the absent IB). The custom recdouble all-reduce and put-barrier all-to-all both validate. The NVSHMEM *library* collective (`nvshmemx_*_reduce_block`) fails `collective_launch` occupancy on this build — bypassed.
- **TP8 / EP-MoE NCCL shards deadlock** (single-process multi-rank ordering) → no end-to-end sharded tok/s measured yet.

## Baselines (saved / real)
vLLM fp8/EP8 **65.8 tok/s**, bf16/TP8 **85.7**, naive transformers 3.5. fp8 roofline ~1240.

## Verdict on 1000 tok/s (honest)
**Not achievable on the current working stack.** Compute is healthy (sharded ~1.4 ms/token at K5's 58% MBU), but **comms is the wall**: NCCL caps the sharded B=1 decode at ~75–150 tok/s and only 14% is overlappable. Reaching 1000 requires, stacked:
1. **Fewer / cheaper collectives** — NVSHMEM now works but its best (all-to-all, 17 µs) still gives 3.2 ms/token over 188 collectives (~310 tok/s cap). 1000 needs comms < ~1 ms, i.e. **< ~5 µs/collective (NVLS hardware multicast) OR halving the collective count** (1 reduce/layer, or grouped/fused all-to-all) — not just a faster primitive.
2. **A sharded decode that actually runs** — the NCCL shards currently deadlock; the NVSHMEM path (proven for the collective) needs wiring into the full decode.
3. **Speculative decode** (×2–3) — amortizes *both* compute and comms over accepted tokens; the only multiplier past the bandwidth roofline. (The in-repo `spec_decode_bench` models the verify pass as γ+1× cost — wrong for memory-bound B=1, where verify ≈ 1×; corrected, spec gives ≈E[accepted]× ≈ 2.77× at α=0.7,γ=4.)
4. **int4 experts** with a half2-FMA unpack (current unpack is ALU-bound, loses to fp8).

**Real wins this session:** the kernels (K5 58% MBU, lm_head 55%, prefill 15.5% TC, K1 12×/K2 6×) and a quantified comms wall. The end-to-end 1000 is an integration + toolchain effort beyond this window.

## Squeeze round (bottleneck attack — measured, mostly negative, but decisive)
| lever | result | verdict |
|---|---|---|
| **In-kernel NVSHMEM all-reduce** (persistent kernel, no per-collective launch) | **51.75 µs vs 55.10 µs host** = **1.06×** | comms is **barrier-bound, not launch-bound** — in-kernel does NOT break the wall. Floor ≈ 17 µs/collective = one 8-GPU NVLink `barrier_all`. |
| **int4-v3 half2-FMA unpack** | 168 µs = **0.58× fp8** (PASS) | still unpack-ALU-bound on the half2 path; **int4 ruled out** at B=1. |
| **spec verify-in-one-pass** | forward scales ~linearly with draft rows (192→850 µs/layer for M=1→5) | the bench **didn't batch the weight read** — a modeling bug, not a refutation. Real batched verify is flat → spec amortizes (team's EAGLE3 measures ~3.8×). Needs a correctly-batched verify kernel. |
| **megakernel decode** | cg::grid.sync + collective_launch combo unverified + a Q-layout bug | risky/broken — not benched. |

**Refined path to 700 (post-squeeze):** comms can't go below ~17 µs/collective (barrier), so the levers are (1) **EP all-to-all** (17 µs, 1 barrier) not TP recdouble (51 µs, 3 barriers); (2) **halve collectives** 188→94 (1/layer); (3) **correctly-batched speculative decode** (÷~2.77–3.8 — the dominant multiplier, owned by the team's EAGLE3). Arithmetic: 94 × 17 µs = 1.6 ms comms ÷ 2.77 (spec) + ~0.5 ms compute ≈ **1.1 ms → ~900 tok/s** — reachable, but gated on the EP-sharded decode running end-to-end + the batched verify. Neither int4 nor in-kernel-launch-elision is on the path; spec + EP + fewer collectives is.
