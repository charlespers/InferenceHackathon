# danielAgentScheduling

Coordination doc for djamoils' parallel optimization loops sharing the 8×H100 box.
**Both loops MUST read this (and the live box copy `/alloc/data/danielAgentScheduling.md`)
before any GPU launch, and update the Slot log when they acquire/release the GPUs.**

## Loops
| loop | avenue | branch | owns (dirs/files) | vLLM port |
|---|---|---|---|---|
| **LOOP-A** (adaptive-topk) | confidence-adaptive top-k expert reduction | `djamoils-work` | `experiments/adaptive_topk/`, `tools/{router_mass,measure_baseline,project_latency,routing_predict,routing_predict,slot_ab_adaptive,slot_runner,quality_probe,quality_compare,moe_kernel_microbench}.py` | **8077** |
| **LOOP-B** (kv-fp8) | KV-cache FP8 quantization | `djamoils-kvquant` | `experiments/kv_fp8/`, `tools/kv_*.py` | **8088** |

Never edit the other loop's files/branch. Merge clean pieces to `main`; rebase onto `origin/main` before pushing.

## GPU slot protocol (the box is the shared resource)
1. djamoils owns the **:45–:00 UTC** slot. Only launch models in that window (or when explicitly cleared early). Other people own the other 45 min.
2. **Atomic serialize** (the two loops can't both hold the box — each needs all 8 GPUs):
   - Acquire: `mkdir /alloc/data/gpu.lock` — success ⇒ you hold it; write `<loop> <UTC>` into `/alloc/data/gpu.lock/holder`. Failure ⇒ another loop holds it (treat as stale if `holder` mtime > 20 min, then take over).
   - Also require `nvidia-smi` min-free > 65000 MB (catches teammates' runs).
   - Release: `rmdir /alloc/data/gpu.lock` when done (and on abort).
3. After a run, append to the Slot log below **and** the box copy.

## Slot plan / requests
- **LOOP-A** wants the next free slot for the FP8+EP+CUDA-graphs A/B (2 vLLM launches, ~10 min) — the end-goal experiment (real 235B/8×H100 B=1 number + adaptive-k delta + dominant term).
- **LOOP-B** wants a slot for the KV-fp8 A/B (baseline vs `--kv-cache-dtype fp8` at ctx 128/2k/8k/32k).
- These can't run together. Convention: **alternate slots** — whoever doesn't hold `gpu.lock` waits for the next slot. LOOP-A has priority on the immediate next slot (07:45) since its experiment is armed.

## Notes between loops (append; newest first)
<!-- leave findings/requests/warnings for the other loop here -->
- **LOOP-A → LOOP-B:** Heads-up on regime: the team's real 8×H100 vLLM decode is
  **comms-bound (~85 tok/s bf16+TP8, ~16µs all-reduce)**, NOT weight-bound. Our byte
  levers (your KV-fp8, my adaptive-topk) may show little e2e wall-clock win there — so
  measure the **dominant term** (TTFT/TPOT vs roofline) first, and lean on **long
  context** where KV reads grow (that's where KV-fp8 should actually pay). Working
  launch form: `python3 -m vllm.entrypoints.openai.api_server --model <m>
  --tensor-parallel-size 8 --enable-expert-parallel --served-model-name qwen3 --port 8088`.
  Box can't `git push` (no GitHub auth) — pull results + commit locally. results/* is
  gitignored → `git add -f`.

## Slot log (append; newest first)
<!-- format: <UTC> LOOP-X: acquired/released + what ran + result file -->
- (pending) LOOP-A: A/B armed for 07:45 (pid via /alloc/data/slot.pid).
