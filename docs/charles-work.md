# charles-work — B=1 latency for Qwen3-235B-A22B (8×H100) · coordination branch

Async channel between the **planning agent** (no GPU) and the **GPU agent(s)** (slot schedule; LOOP-A runs
EAGLE3 :45–:00). Goal: maximize single-user (batch-1) decode tok/s. **Start here → `b1-optimization-atlas.md`
(every lever) → `gpu-agent-experiments.md` (the work order).**

## ✅ THE CONVERGENT ANSWER: spec decode (EAGLE3 tree-spec) is the #1 lever
Independent team research (`research/seriality_breaking · depth_reduction · comms_floor`) converged on what my
floor-amortization predicted: **spec amortizes the floor → the top lever; the structural comms levers don't pay
at B=1; depth-reduction loses to off-the-shelf EAGLE3.** Why (`why-spec-wins.md`): one batched verify replaces
τ serial B=1 decodes — floor paid once (÷τ), expert GEMV→grouped-GEMM, KV read once. **First real numbers land
from LOOP-A's 08:45 EAGLE3 slot** (τ, S, V, F → `eagle3-results-playbook.md`).

## ✅ THE FLOOR IS THE GAME, NOT THE BYTES (3 data rounds — `results-reaction-01/02.md`, `overhead-attribution.md`)
Real bf16-TP8 TPOT 11.67 ms = **overhead ~7.0 ms (60%) ≫ comms 3.0 ms (26%) ≫ weight 1.6 ms (14%)**; engine at
2–16% of roofline. **Proven twice:** weight levers are invisible while floor-bound (`ab_adaptive` regressed).
**Ceiling** (`absolute-ceiling.md`): current 85.7 tok/s = **~4% of physics**; cheap wins → ~508 (EAGLE3 τ=3.5);
+kernels/comms → ~754; absolute fp8+spec ceiling ~2000 (int4 ~3900). **Prize ~20×.**

## Measured / validated (real, on-box)
- **EP→TP inversion confirmed:** fp8+EP8 64.5 < bf16+TP8 85.7 tok/s — BUT it's a *plain-decode* finding;
  **the big-tree spec verify BALANCES EP** (`ep-balance-spec-verify.md`), so FP8+EP+big-tree is fine.
- **Floor-bound:** 60% overhead / 26% comms / 14% weight; collective latency measured **16µs** (not 5µs).
- **Route-prediction validated:** busiest 2.53 (~my 2.6), affinity locality 0.123→0.317, persistence 0.446↑.
- **K5 MoE kernel: scalar → e=0.46, ~100×, correctness-clean** on H100 (vLLM fused_moe ~0.16). Roadmap to e→1:
  `k5-tuning-roadmap.md` (cp.async MLP).
- **vLLM `192%128` TP8-FP8 crash** → `--quantization fp8` (dynamic) / EP / block-64 requant.

## Doc map (the convergent-answer + floor framework)
| Doc | What |
|---|---|
| `b1-optimization-atlas.md` | **THE MAP** — every lever (mine + team), status, the one execution sequence |
| `gpu-agent-experiments.md` | The work order — priority header + E0–E9/E-attr/E-ttft + Results Log |
| `absolute-ceiling.md` | current 85.7 = 4% of physics; prize ~20× (→2000); what each rung needs |
| `why-spec-wins.md` · `spec-decode-floor-bound.md` | spec escapes the B=1 regime · floor-amortization (the reversal of "big trees lose") |
| `eagle3-draft-tp.md` · `spec-in-production.md` | **draft_tp=8 not 1** (6× draft) · greedy-bench vs temp-0.7 product (τ ~2.5) |
| `ep-balance-spec-verify.md` · `route-aware-drafting-design.md` | big-tree verify balances EP · the route-aware mechanism (weight-bound lever) |
| `eagle3-results-playbook.md` | **decision tree for the 08:45 data** (V→F→go/no-go→next lever) |
| `overhead-attribution.md` · `seriality-breaking-notes.md` | the floor split (E-attr) · measured-16µs + spec=amortizer |
| `single-user-latency-budget.md` · `ttft-analysis.md` · `long-context-chat.md` | the thesis table (EAGLE3 → ~508-754) · prefix-cache TTFT · KV/spec long-ctx |
| `fixed-overhead-floor.md` · `b1-fast-path-design.md` · `vllm-b1-config.md` | per-step host tax · the cudarc decode loop · **the B=1 launch checklist** |
| `console-telemetry-spec.md` | make the optimization visible in the deliverable (the panels + x_summary) |
| `interpretation-playbook.md` · `ep-placement-for-b1.md` · `self-speculation-design.md` | measured→action · co-activation placement · self-spec |

## Tools + kernels (`tools/`, `kernels/`, `bench/`, `server/`)
- **Spec model suite:** `spec_predict.py` (**unified**: floor+tree+draft+temp+EP) · `spec_floor_model.py`
  (floor-aware) · `tree_spec_optimizer.py` (W×D optimum) · `backout_floor.py` (**F from V=τ/S**, LOOP-A deployed it)
- `latency_budget.py` (`--proven`, `--spec-tau`, `--ctx-sweep`) · `predict_matrix.py` · `placement_b1.py`
  (per-step-busiest placement) · `verify_route_prediction.py` · `verify_self_speculation.py`
- **Console (deliverable):** `server/spec_metrics.py` (real accept rate from /metrics) · `server/optimization_telemetry.py`
  (floor-breakdown/regime/ceiling-% for x_summary, wired into `mock_engine`) · fixed the stale server tests (145 green)
- `bench/run_eagle3.sh` (**EAGLE3 k-sweep, draft_tp=8, on LOOP-A's venv**) · `run_bench_best.sh` (stacked cheap wins) ·
  `run_bench4/5/6.sh` · `bench/measure.py --temperature` (product τ)
- `kernels/k5_experts_warp.cu` (**measured winner, e=0.46**) + the microbench/sweep/int4/downproj suite

## Collaboration (`danielAgentScheduling.md` notes)
LOOP-A (djamoils) runs EAGLE3; **agreed split** — LOOP-A: FP8+EP + parity + route-aware tree-shaping; me:
bf16 over-delivery + tree optimizer + kernel. LOOP-A **adopted** my draft_tp=8, EP-balance (→ big tree on EP),
and `backout_floor.py` (the 2-point k-sweep). They post τ/S/V/F from 08:45; the **bf16-vs-FP8 ΔF decides the
route-aware lever**.

## State
Framework complete and team-converged on spec. **Next real signal: LOOP-A's 08:45 EAGLE3 (τ, S, V, F).** On
arrival: run `backout_floor.py` (F), confirm/deny the over-delivery prediction (~2.5–3×), set route-aware
go/no-go, size the tree (`spec_predict.py`), update the atlas row 2 with the measured multiplier, and pick the
next decode lever per `eagle3-results-playbook.md`.
