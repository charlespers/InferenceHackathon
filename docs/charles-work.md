# charles-work — B=1 latency for Qwen3-235B-A22B (8×H100) · coordination branch

Async channel between the **planning agent** (no GPU) and the **GPU agent** (15-min slot, Charles :30–:45).
Goal: maximize single-user (batch-1) decode tok/s. **Start here, then open `gpu-agent-experiments.md`.**

## ✅ THE FLOOR IS THE GAME, NOT THE BYTES (2 data rounds — `results-reaction-01/02.md`)
Real bf16-TP8 TPOT 11.67 ms decomposes to **overhead ~7.0 ms (60%) ≫ comms 3.0 ms (26%) ≫ weight 1.6 ms
(14%)**. Decode runs at **2–16% of roofline**. **Proven twice:** weight-read levers are invisible while
floor-bound — `ab_adaptive` made it *slower* (9.67 vs 13.23 tok/s). **Proven priority order:**
1. **`E-attr`** — split the floor (Nsight): comms vs MoE-kernel-inefficiency (vLLM ~0.16 vs K5 0.46) vs host.
2. **Comms tuning** (`E0b`, 16µs all-reduce) **and/or kernel efficiency** (K5) — whichever `E-attr` says.
3. **Layout = TP8** (EP→TP **confirmed on HW**: fp8+EP8 64.5 < bf16+TP8 85.7 tok/s).
4. **fp8 (`E2b`, dynamic-quant unblock) → int4 → adaptive-top-k — LAST** (invisible until the floor is down).
5. **n-gram spec** (small-k, MoE verify-tax); **route-prefetch** deep layers (persistence 0.45 rising).

## Measured / validated (real, on-box)
- **EP→TP inversion confirmed on hardware:** fp8+EP8 64.5 < bf16+TP8 85.7 tok/s (EP penalty swamps fp8).
- **Floor-bound:** real TPOT 60% overhead / 26% comms / 14% weight; engine at 16% (vLLM) & 2.5% (adaptive) of roofline.
- **Route-prediction validated:** busiest-rank 2.53 (~my 2.6), replication 16×→1.73, affinity placement
  locality 0.123→0.317, DirectProxy persistence 0.446 rising by layer (`routing_predict_early.json`).
- **K5 MoE-expert kernel: scalar → e=0.46 (1538 GB/s), ~100×, correctness-clean** (max_rel 3.2e-5), on H100.
  vLLM's default MoE kernels run at ~16% util → the ~7ms overhead gap K5 attacks.
- **vLLM `192%128` TP8-FP8 crash reproduced live** — fix = `--enable-expert-parallel` (EP, slower) or a
  **block-64 FP8 requant** to unblock the winning fp8+TP8 cell.

## Doc map
| Doc | What |
|---|---|
| `gpu-agent-experiments.md` | **The work order** — priority header + E0–E9/E-attr/E-ttft, commands, Results Log |
| `single-user-latency-budget.md` | **The whole thesis in one table** — current 2.3s/86tok-s → ~0.45s/~290 (4–5×) |
| `results-reaction-01.md` / `-02.md` | **Reactions to real data** — comms-bound; floor is the game; route-prediction validated |
| `overhead-attribution.md` | Real TPOT = overhead 60% / comms 26% / weight 14% → attribute the floor (`E-attr`) |
| `ttft-analysis.md` | TTFT 777ms = 20–300× the physics → prefix caching ~50–100× (the cheap big win) |
| `spec-decode-floor-bound.md` | Spec amortizes the floor → a TOP lever now (~2×, k≈4, regime-adaptive) |
| `interpretation-playbook.md` | **measured value → next action** for every experiment |
| `team-coordination.md` | How this fits `origin/main` (Rust engine + bench); cross-validation; comms reconciliation |
| `spec-decode-moe-tax.md` | The weight-bound spec tree-size analysis (k≤3) — superseded by floor-bound while floor-bound |
| `self-speculation-design.md` · `ep-placement-for-b1.md` | self-spec cost model · co-activation placement for B=1 |
| `predicted-tok-s-matrix.md` · `next-levers-research.md` | predicted tok/s matrix · the prioritized lever dossier |
| `b1-latency-architecture.md` · `b1-tp8-moe-rearchitecture-h200.md` · `k5-kernel-results-h100.md` | research · TP8 spec · kernel journey |

## Tools + kernels (`tools/`, `kernels/`, `bench/`)
- `tools/latency_budget.py` — single-user latency calculator (`--proven` → 290 tok/s / 448ms; multi-turn chat)
- `tools/predict_matrix.py` · `verify_route_prediction.py` (E8) · `verify_self_speculation.py` (E9)
- `bench/run_bench4.sh` (fp8-TP8+comms/E2b+E0b) · `run_bench5.sh` (n-gram spec/E6) · `run_bench6.sh` (prefix-cache/E-ttft)
- `kernels/k5_experts_warp.cu` (**measured winner**) · `k5_microbench.cu` · `k5_ksweep_bench.cu` (adaptive-k wall-clock)
- `k5_experts_warp2.cu`+`k5_downproj_bench.cu` (down-proj fix) · `k5_experts_int4.cu`+`k5_int4_bench.cu` (int4)
- `tools/verify_route_prediction.py` (E8, `predictor.rs`) · `tools/verify_self_speculation.py` (E9)
- `tools/predict_matrix.py` (calibrated predictions from the team's model)

## Experiment queue (priority; see `gpu-agent-experiments.md` for commands)
**E0** collective latency (cheapest, decides strategy) → **E1** FP8+EP engine baseline → **E4** kernel
microbenches + Nsight (no model load) → **E6** n-gram spec (k=2–3) → **E2** EP-vs-TP → **E7** int4 engine →
**E8** route-prediction → **E3** CUDA-graph. The 15-min slot affords ~1 vLLM load; do kernel/model work off-GPU.

## State
The strategic landscape is mapped and every lever has an exact experiment + a prediction to confirm/refute.
**The next real signal is GPU-side: E0 (sub-minute) then E1.** Planning agent reacts to the Results Log:
designs the targeted kernel fix from Nsight, sizes the spec tree from measured τ, and recomputes the
cumulative tok/s from the real baseline.
