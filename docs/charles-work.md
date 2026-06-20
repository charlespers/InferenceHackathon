# charles-work — B=1 latency for Qwen3-235B-A22B (8×H100) · coordination branch

Async channel between the **planning agent** (no GPU) and the **GPU agent** (15-min slot, Charles :30–:45).
Goal: maximize single-user (batch-1) decode tok/s. **Start here, then open `gpu-agent-experiments.md`.**

## The one number that decides everything → run **E0 first**
B=1 decode is bandwidth-bound, but is it **weight**-bound or **comms**-bound? The team's model assumes
`collective_latency_s=5µs` → comms 0.94 ms dominates; my estimate (~1.5µs tuned) → weight dominates.
**`nccl-tests` measures the real all-reduce latency in seconds, no model load** (E0). It decides whether
the #1 lever is route-prefetch/spec (comms-bound) or int4/kernels (weight-bound). Everything keys off it.

## Measured / validated (real, on-box where noted)
- **K5 MoE-expert kernel: scalar → e=0.46 (1538 GB/s), ~100×, correctness-clean** (max_rel 3.2e-5), on H100.
  A/B split: gate/up 0.49, down 0.405 (the weak link).
- **EP→TP inversion, triple-confirmed:** my spec + my measured kernel + the team's own `latency.py` model.
  Their model at my e=0.46: **TP8 261 vs EP8 94 tok/s** (2.8×).
- **vLLM `192%128` TP8-FP8 crash reproduced live** — exactly as the spec predicted; fix = `--enable-expert-parallel`.

## Doc map
| Doc | What |
|---|---|
| `gpu-agent-experiments.md` | **The work order** — E0–E8, exact commands, go/no-go signals, Results Log |
| `team-coordination.md` | How this fits `origin/main` (Rust engine + bench); cross-validation; comms reconciliation |
| `next-levers-research.md` | Prioritized, vetted levers (engine baseline → n-gram spec → int4 → down-proj) |
| `spec-decode-moe-tax.md` | Why their `SpecConfig draft_len=8` loses on the MoE; use k≈2–3 (for `engine/spec/`) |
| `b1-latency-architecture.md` | The 15-avenue first-principles research (H100 canonical) |
| `b1-tp8-moe-rearchitecture-h200.md` | The TP8 MoE re-architecture spec (numbers ÷1.433 for this H100) |
| `k5-kernel-results-h100.md` | The measured kernel optimization journey |

## Kernel files (`kernels/`)
- `k5_experts.cu` (reference) · `k5_experts_warp.cu` (**measured winner**) · `k5_microbench.cu` (repro)
- `k5_experts_warp2.cu` + `k5_downproj_bench.cu` (down-proj occupancy fix, for E4)
- `k5_experts_int4.cu` + `k5_int4_bench.cu` (int4 byte lever — does the unpack eat the 2×? → E4)
- `tools/verify_route_prediction.py` (validates the team's `predictor.rs` on a real MoE → E8)

## Experiment queue (priority; see `gpu-agent-experiments.md` for commands)
**E0** collective latency (cheapest, decides strategy) → **E1** FP8+EP engine baseline → **E4** kernel
microbenches + Nsight (no model load) → **E6** n-gram spec (k=2–3) → **E2** EP-vs-TP → **E7** int4 engine →
**E8** route-prediction → **E3** CUDA-graph. The 15-min slot affords ~1 vLLM load; do kernel/model work off-GPU.

## State
The strategic landscape is mapped and every lever has an exact experiment + a prediction to confirm/refute.
**The next real signal is GPU-side: E0 (sub-minute) then E1.** Planning agent reacts to the Results Log:
designs the targeted kernel fix from Nsight, sizes the spec tree from measured τ, and recomputes the
cumulative tok/s from the real baseline.
