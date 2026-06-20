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
| collective | latency | note |
|---|---|---|
| NCCL all-reduce (8–128 KB) | **35 µs** | latency-floored; LL/LL128 don't help |
| NCCL all-to-all (8–32 KB) | **60 µs** | |
| overlap (scheme C) | hides **14%** | B=1 has ~no independent compute to hide behind |

188 collectives/token (2/layer × 94) → **6.6–13 ms/token of comms → TP=8 sharded B=1 comms-capped at ~75–150 tok/s, regardless of kernel speed.**

- **TP8 / EP-MoE NCCL shards deadlock** (single-process multi-rank ordering) → no end-to-end sharded tok/s measured yet.
- **NVSHMEM** (the sub-µs GPU-initiated comms fix) is installed (cu13) but **won't build** — the cu13 device lib won't link with the box's CUDA-12.6 `nvcc`, and the cu13 `nvcc` finds cu12 headers. Needs a version-matched toolkit (`nvshmem-cu12`, or a clean cu13 toolchain).

## Baselines (saved / real)
vLLM fp8/EP8 **65.8 tok/s**, bf16/TP8 **85.7**, naive transformers 3.5. fp8 roofline ~1240.

## Verdict on 1000 tok/s (honest)
**Not achievable on the current working stack.** Compute is healthy (sharded ~1.4 ms/token at K5's 58% MBU), but **comms is the wall**: NCCL caps the sharded B=1 decode at ~75–150 tok/s and only 14% is overlappable. Reaching 1000 requires, stacked:
1. **Low-latency comms** — NVSHMEM/DeepEP sub-5 µs (blocked by the cu12/cu13 toolchain; an env fix, not a kernel fix).
2. **A sharded decode that actually runs** — the NCCL shards currently deadlock.
3. **Speculative decode** (×2–3) — amortizes *both* compute and comms over accepted tokens; the only multiplier past the bandwidth roofline. (The in-repo `spec_decode_bench` models the verify pass as γ+1× cost — wrong for memory-bound B=1, where verify ≈ 1×; corrected, spec gives ≈E[accepted]× ≈ 2.77× at α=0.7,γ=4.)
4. **int4 experts** with a half2-FMA unpack (current unpack is ALU-bound, loses to fp8).

**Real wins this session:** the kernels (K5 58% MBU, lm_head 55%, prefill 15.5% TC, K1 12×/K2 6×) and a quantified comms wall. The end-to-end 1000 is an integration + toolchain effort beyond this window.
