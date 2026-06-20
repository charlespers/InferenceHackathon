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

## First REAL end-to-end sharded decode (benched, 8 GPUs, NVSHMEM)
`kernels/decode_sharded_nvshmem.cu` — host-driven 8-PE TP shard, one-shot NVSHMEM all-reduce (validated maxerr 2.2e-6), 94 layers, latency proxy.
```
sharded full step:  28,136 µs/token = 35.5 tok/s  (123.6 GB/s/GPU, 3.7% peak)
  compute-only:     22,464 µs (80%)  <- LAUNCH-OVERHEAD-bound (~850 un-graphed host launches)
  all-reduces (189): 5,672 µs (20%)  = 30 µs/AR (one-shot = 2 barriers; not the 17µs single-barrier hope)
```
**+ CUDA-graph capture (benched):** eager 13.0 → **graphed 33.8 tok/s** (graph 61.5% faster; compute-only 71→24 ms — launch overhead killed, captured graph_A=6 nodes attn + graph_B=4 nodes MoE, replayed 94×, collectives stay host-launched). The graph WORKS — but the graphed sharded number (33.8) ≈ the single-GPU proxy (30.9), which is the **decisive finding**:

**TP=8 sharding does NOT help B=1 latency.** Per-GPU the sharded kernels run at **118 GB/s (3.5% peak)** vs the single-GPU full-model step's 859 GB/s (26%) — the ~8× data reduction is offset by ~6× worse per-GPU efficiency (slices too small to saturate an H100: occupancy-starved). Plus the 5.7 ms comms floor. This is why my custom **sharded** number (33.8 fp8) sits *below* vLLM's mature TP=8 (85.7 bf16): the kernels win in isolation (K5 58% MBU vs ~11%) but don't translate end-to-end — vLLM handles B=1 shard sizes + NCCL + graphs better. **Implication: 700 is a genuine stretch** — from vLLM's real 85.7, even spec ×3.8 → ~325; reaching 700 needs spec + larger per-GPU work (TP=2/4 not 8, or batching) + exceptional MoE expert-overlap. B=1 single-stream on a 235B MoE is fundamentally latency-bound.

**(superseded) earlier verdict:** sharding alone barely beats the 30.9 single-GPU proxy — because the host-driven step is **launch-bound**, not bandwidth-bound (per-GPU read 3.48 GB ideal = ~2 ms vs measured 22 ms compute). The fix is the same CUDA-graph capture that took the single-GPU step 8.4→30.9 (kill the ~850 launches) → compute → ~2–3 ms → step ~8 ms → **~125 tok/s**; then spec (×~2.8) → ~350; + comms/collective reduction → toward 700. The one-shot AR is **30 µs** (2 barriers), so 189× = 5.7 ms is a hard comms floor until collectives are cut or fused into the graph.

**Refined path to 700 (post-squeeze):** comms can't go below ~17 µs/collective (barrier), so the levers are (1) **EP all-to-all** (17 µs, 1 barrier) not TP recdouble (51 µs, 3 barriers); (2) **halve collectives** 188→94 (1/layer); (3) **correctly-batched speculative decode** (÷~2.77–3.8 — the dominant multiplier, owned by the team's EAGLE3). Arithmetic: 94 × 17 µs = 1.6 ms comms ÷ 2.77 (spec) + ~0.5 ms compute ≈ **1.1 ms → ~900 tok/s** — reachable, but gated on the EP-sharded decode running end-to-end + the batched verify. Neither int4 nor in-kernel-launch-elision is on the path; spec + EP + fewer collectives is.
