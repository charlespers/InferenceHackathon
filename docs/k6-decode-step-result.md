# k6 / fused decode step — first on-box run (H100, 2026-06-20)

**Status: WORKS.** `kernels/decode_step.cu` (K1→K2→K3→K4→K5 ×94 + final-norm + lm_head + argmax),
captured into ONE CUDA graph by k6 (`kernels/k6_graph_capture.cu`), **compiles (nvcc 12.6, sm_90a) and
runs on an H100.** Build+run: `bash bench/run_decode_step.sh [ctx] [iters]` (auto-picks a free GPU).

**Why it didn't exist before:** `decode_step.cu` was complete and correctly wired, but it was never in
the build — `bench/compile_kernels.sh` only compile-checked the individual benches, so the full step
was never compiled or run, and `k6_graph_capture.cu` was an orphaned stub. Fixed (commit `2c5d470`):
k6 finished + wired into the step; `decode_step.cu` added to the compile-check; `run_decode_step.sh`
added. Verified on-box via the SSH GPU window.

## Result (ctx 4096, 150 iters, GPU 0; LATENCY/launch-overhead PROXY — 1 layer's weights reused ×94)
- captured graph: **755 nodes** instantiated; replays with a single launch.
- graph replay: **30.7 tok/s, 674 GB/s, 20.1% of HBM peak**; eager: 29.7 tok/s, 651 GB/s.
- **per-launch dispatch ≈ 1.71 µs**; 661 launches/token ⇒ **~1.13 ms/token pure launch overhead the
  graph collapses** (graph only 3.4% faster here because the proxy is BW-bound, not launch-bound).

## Why this matters (E3 CUDA-graph lever, floor-bound thesis)
Real engine TPOT ≈ 8.6 ms with ~7 ms unmodeled overhead (floor-bound). This run isolates the **pure
kernel-launch component: ~1.13 ms/token (661 × 1.71 µs)** — a CUDA graph over the real per-token kernel
chain caps at ~1.1 ms, i.e. **~16% of the 7 ms floor**. The rest of the floor is host/sampling/comms,
**not** raw dispatch — so graph capture is necessary but not sufficient; pair it with the host/comms
work (`E-attr`). This quantifies the ceiling of the E3 CUDA-graph lever.

**Proxy caveat:** the produced token id is not numerically faithful (dummy weights reused ×94); the
kernel chain, launch count, grid/block shapes and per-token read volume (21.96 GB full single-GPU) are
real. Numeric validation vs HF transformers is a separate on-box step. Raw log on the box:
`/root/decode_step_result.txt`.
