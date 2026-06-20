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
- **Charles → LOOP-A (EAGLE3), TIME-SENSITIVE for the 08:45 slot:** in the EAGLE3 `--speculative-config`, use
  **`"draft_tensor_parallel_size": 8`, NOT 1**. INTEGRATION.md's `draft_tp=1` ("sharding a 1-layer head is
  pure overhead") is *throughput* intuition; at **B=1 the draft is bandwidth-bound** — the 1B head's ~2GB read
  on one GPU is ~0.6ms/step × num_spec_tokens ≈ **~3ms of draft per round** (comparable to the verify floor!).
  TP8-sharding the head reads 0.25GB/GPU + a ~32µs all-reduce ≈ **6× faster**, *and* avoids gathering the 3 aux
  hidden states (already TP8-sharded to match the target). `draft_tp=1` caps the win ~2.5×; **`draft_tp=8`
  restores ~3×.** Reasoning + draft-cost model: `docs/eagle3-draft-tp.md` (charles-work). Two more for the run:
  measure τ at **temperature 0.7** (the product) not just greedy (accept-rate ~2.2–2.8 at temp>0,
  `docs/spec-in-production.md`); and a **WIDE+DEEP tree wins in this floor-bound regime** (W4–8×D3–4, not small —
  `tools/tree_spec_optimizer.py`). If the head pins `draft_tp=1`, expect ~2.5× and free n-gram is competitive on
  repetitive prompts.
- **LOOP-A → CHARLES (2026-06-20 08:0x UTC) — de-dup EAGLE3:** Your `bench/run_eagle3.sh`
  skips on the box because system vLLM=0.10.1. **I've solved that prereq:** isolated venv
  `/alloc/data/eagle3-venv` (vLLM 0.11.0, own torch2.8) + converted head cached. Your script
  can run today by using that interpreter (`/alloc/data/eagle3-venv/bin/python -m vllm...`
  instead of `python3`). **Proposed split to avoid double-spending GPU slots:** I (LOOP-A,
  :45–:00 slot, port 8077) own **FP8+EP + CUDA-graphs + lossless parity gate + the novel
  route-aware/expert-union tree-shaping**; you own your **bf16 floor-bound "over-delivery"
  hypothesis + tree-shape (W×D) optimizer + kernel**. Different model (FP8 vs bf16), different
  slots, different ports — both data points useful, no redundant baseline re-runs. I'll post
  the real FP8 accept-length + tok/s here once measured so your analytical model gets ground truth.
- **LOOP-A (EAGLE3) → team/LOOP-B (2026-06-20 07:55 UTC):** Resuming on **EAGLE3 spec-decode**
  (sibling kv-fp8 loop stopped). **BLOCKER being resolved now (non-GPU prep, no lock/slot held):**
  box has system **vLLM 0.10.1 which REJECTS qwen3 EAGLE3** (needs ≥0.10.2). To avoid breaking
  teammates on the shared system vLLM, I'm **NOT upgrading system vLLM** — instead building an
  **isolated venv at `/alloc/data/eagle3-venv` (vLLM 0.11.0, its own torch 2.8/cu128)**. Shared
  HF cache reused (FP8 235B already cached, 221G). Also downloading the converted head
  `nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys` (~2GB) to shared cache. Driver
  560/CUDA12.6 runs torch2.8 via CUDA-12 minor-version compat. **No GPU touched** — disk/network
  only. Team: keep using `/usr/local/bin/vllm` (0.10.1) unaffected; my EAGLE3 runs use
  `/alloc/data/eagle3-venv/bin/vllm` and only during my :45–:00 slot under gpu.lock.
- **LOOP-A → LOOP-B:** Ack — 07:45 mine, 08:45 yours, lock arbitrates if timing slips.
  Like the KV-as-memory-win reframe (the HBM headroom stacks with my top-k — agreed
  orthogonal). Team status FYI: **Charles is now also proving adaptive-k** (k-sweep on his
  tuned K5 kernel) and found **TTFT is dominated by missing prefix caching (~50–100×)** +
  confirmed **spec-decode amortizes the comms floor**. So I'm **pivoting my creative
  research to the COMMS FLOOR** (the real dominant term, ~188 serial all-reduces): self-
  speculative **layer-skip / depth reduction** (fewer layers = fewer collectives + less
  weight), **attention-replication** to halve collectives/layer, and **NVLS / comms-overlap**.
  Launching research agents now; findings + any reusable harness posted here.
- **LOOP-B → LOOP-A:** Ack — adopted your atomic lock (`mkdir /alloc/data/gpu.lock` +
  `holder`, 20-min stale takeover, `rmdir` release); I was using a file-lock, now fixed in
  `kv_ab.sh`. **Yielding 07:45 to you** (your A/B is armed). I'll take **08:45** (kv=auto
  baseline sweep) and a later slot for kv=fp8; lock arbitrates if timing slips. Your
  comms-bound finding **matches my roofline**: this model is GQA-4 (4 KV heads, 94 layers),
  so KV is only ~6.7% of per-token bytes at 8k, ~22% at 32k → fp8-KV is a ~11% TPOT *ceiling*
  at 32k and **less after comms**. So I'm framing KV-fp8 as a **MEMORY win** (half KV
  footprint → longer ctx fits / HBM headroom for your top-k), quality-gated on long-ctx
  needle recall. Orthogonal + stackable, no path/port conflict. My harness is in `tools/kv_*`,
  predictor `tools/kv_roofline.py`.
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
- 2026-06-20 08:06 LOOP-A: **ARMED EAGLE3 slot runner** (`/alloc/data/slot_eagle3.sh`, pid 79926)
  waiting for the **08:45** slot. De-risked non-GPU: venv vLLM 0.11 imports OK, `speculative_config`
  is a valid arg, head config = Eagle3Speculator/algorithm=eagle3/verifier=FP8-target (verified).
  Slot plan: EAGLE3 eager (parity+accept-len+tok/s) → baseline FP8 graphs (denominator) → parity
  gate → push results to origin/djamoils-results + /alloc/data/eagle3/. LOOP-B stopped ⇒ I own 08:45.
- 2026-06-20 07:58 LOOP-A: **BLOCKER RESOLVED (no GPU used).** Isolated venv built clean →
  `vllm 0.11.0 / torch 2.8.0+cu128` (RC=0) at `/alloc/data/eagle3-venv`. EAGLE3 head fully
  cached. EAGLE3 is now turnkey for the next full slot. **Next GPU launch: 08:45 UTC slot**
  (current slot had only ~2min left). Will run eager-first parity gate → decode tok/s +
  accept-len → drop --enforce-eager for graph headline, vs FP8 baseline.
- 2026-06-20 07:53 LOOP-A: probed box (NO lock, all 8 GPUs free ~81GB). Only ~7min left in slot →
  did NOT launch (235B load > remaining time). Started non-GPU prep: isolated vLLM-0.11.0 venv
  build (pid 77967, log /alloc/data/eagle3_venv_build.log) + EAGLE3 head download
  (pid 77811, log /alloc/data/eagle3_head_dl.log). Next GPU launch target: a full :45–:00 slot
  once venv+head ready.
- (plan) LOOP-B: take 08:45 for kv=auto sweep (ctx 128/2k/8k/16k/32k + quality) →
  results/kv_fp8/auto/; later slot for kv=fp8. Yielding 07:45 to LOOP-A.
- (pending) LOOP-A: A/B armed for 07:45 (pid via /alloc/data/slot.pid).
