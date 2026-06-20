# Reacting to the first real data (`config-sweep.md`) — comms-bound, EP→TP confirmed

The team's slot benchmarks produced the first real numbers. Running them through the interpretation playbook
reorders the levers. **Bottom line: we are comms-bound (measured), the EP→TP inversion is confirmed on
hardware, and the biggest single lever is now NCCL comms tuning — not quantization.**

## What the data says
| measurement | value | implication |
|---|---|---|
| bf16 TP8 (no EP) | **85.7 tok/s**, TPOT 11.67 ms, TTFT 777 ms | current best; 16% of the optimistic floor |
| fp8 TP8+EP8 | **64.5 tok/s**, TPOT 15.51 ms | *slower* than bf16-TP8 → **EP→TP inversion, measured** (EP penalty swamps fp8) |
| naive transformers | 289 ms/tok | vLLM is ~25× faster; the floor of "do nothing" |
| **nccl-tests** (this is E0) | all-reduce@8 **≈16µs**, all-to-all@8 ≈10µs, all-reduce@2 ≈6.5µs | **comms-bound** — 3.2× the model's 5µs |
| real routing imbalance | **5–8×** (L17·E78: 770 vs ~62) | EP is even worse than the uniform 2.6× → TP8 / co-activation placement matter more |

## Three findings, cross-checked against my analysis
1. **EP→TP inversion — confirmed on hardware.** fp8+EP8 (64.5) < bf16+TP8 (85.7), attributed to EP
   imbalance. Matches the spec, the kernel measurement, *and* `latency.py`. The bf16+EP8 de-confounding run
   (`run_bench3.sh`) will isolate it cleanly (predicted: bf16+EP8 ≈ 88 tok/s floor at 16µs → well below
   bf16+TP8). **Serve TP8.**
2. **E0 resolved = comms-bound.** Real all-reduce ≈16µs. Recalibrating `latency.py` with 16µs (vs 5µs):
   | config | floor @5µs | floor @**16µs** | @8µs (LL) | @4µs (one-shot) |
   |---|---|---|---|---|
   | bf16 TP8 | 390 | **216** | 320 | 421 |
   | fp8 TP8 | 570 | **262** | 431 | **638** |
   At 16µs, **comms (3.0 ms) ≫ weight (0.8–1.6 ms)** for TP8 — comms is THE term. (`predicted-tok-s-matrix.md`
   regenerated at 16µs.)
3. **A ~7 ms unmodeled-overhead gap.** Real bf16-TP8 = 11.67 ms, but the model at 16µs predicts 4.63 ms →
   **~7 ms is vLLM kernel inefficiency + launch + sampling** (whole-model util ~16% vs my K5 kernel's e=0.46).
   This is exactly the surface the K5 kernels (100× over naive), tighter CUDA graphs, and the serving
   fast-path attack.

## Reprioritized levers (data-grounded)
1. **NCCL comms tuning — NEW #1.** The 16µs all-reduce is the measured bottleneck. `NCCL_PROTO=LL`/`LL128`,
   one-shot/two-shot all-reduce, `NCCL_P2P_LEVEL=NVL`, few channels → target 16µs → ~4–8µs. Model says
   bf16-TP8 floor 216→320–421; fp8-TP8 262→431–638. **These are 🔲 in `config-sweep.md` (COMM rows) — run
   them next.** Biggest bang, no requant, no kernel work.
2. **fp8 + TP8 — the prize cell, currently blocked.** TP8 (no EP penalty) + fp8 (½ weight bytes) is the best
   physics, but `--tensor-parallel-size 8` on the block-128 FP8 ckpt crashes (192%128). **Unblock with a
   block-64 FP8 requant** (192/64=3) → fp8-TP8 floor 262 (16µs) → 638 (tuned comms). The single highest cell.
3. **Kernel / overhead reduction — the 7 ms gap.** vLLM's default B=1 MoE kernels run at ~16% util; the K5
   work (`kernels/k5_experts_warp.cu`, e=0.46, 100× over scalar) + `--enforce-eager` vs graphs (E3) + fused
   sampling/detok are how to close it. Long-term: the cudarc Rust engine calling tuned kernels.
4. **Spec decode (n-gram, k=2–3).** Orthogonal multiplier on whatever baseline; honor the MoE verify-tax
   (`spec-decode-moe-tax.md`). Comms-bound makes the verify pass's comms count too → small trees doubly.

## Path
bf16+TP8 (85.7, today) → **+NCCL comms tuning** (→ ~140–200 tok/s, no requant) → **fp8+TP8 via block-64
requant** (→ higher ceiling) → **+kernel/overhead** (close the 7 ms) → **+n-gram spec**. The de-confounding
bf16+EP8 run finishes the layout/precision A/B and nails the EP penalty.

## Notes for the team
- The **5–8× measured imbalance** (vs 2.6× uniform) means EP is even worse than modeled → reinforces TP8 and
  makes co-activation-aware placement (`ep-placement-for-b1.md`) the right EP mitigation if EP is ever forced.
- Set `src/inferutil/hardware.py: collective_latency_s = 16e-6` (or per-collective: 16µs ar@8 / 10µs a2a@8 /
  6.5µs ar@2) so `latency.py` predictions match reality; re-run `predict_matrix.py`.
- This supersedes the comms half of `next-levers-research.md` (which guessed ~1.5µs): real is 16µs, so comms
  tuning jumps from a ~1.05× footnote to the **#1 lever**.
