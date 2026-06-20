# GPU Experiment Queue — `charles-work` coordination channel

> **Async coordination.** The *planning agent* (Claude, no GPU) queues experiments here. The *GPU agent*
> (uncontested GPU window) runs them in priority order, writes raw results into the **Results Log**, and
> commits to `charles-work` with a clear message + notes any blockers. Planning agent reads results,
> updates the queue, drafts the next round. **Run in priority order; if you skip/reorder, say why.**

## Box facts (confirmed on-box)
- **8×H100 80GB HBM3 (~3.35 TB/s)**, CUDA 12.6, `nvcc` at `/usr/local/cuda-12.6/bin/nvcc`, 132 SMs.
- FP8 model cached: `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`. BF16 local: `/alloc/data/Qwen3-235B-A22B`.
- Harness: `bench/measure.py` (TTFT/TPOT/tok-s over OpenAI SSE), `bench/roofline.py` (dominant term), `bench/sweep.py`.
- Kernel microbench: `kernels/k5_microbench.cu` (correctness + realized HBM efficiency `e`).
- **GOTCHA (verified live):** vLLM `--tensor-parallel-size 8` on the FP8 block-128 checkpoint **crashes**
  (`gate/up output_size 192 not divisible by block_n 128`; 1536/8=192). Use **`--enable-expert-parallel`**
  (experts kept whole) or TP4×EP2 (384) or BF16. **Name the served model `qwen3-235b-a22b`** so
  `measure.py` (which defaults to that id) works unmodified.

## Experiment queue (priority order)

### E0 — Real collective latency (nccl-tests)  ⟵ DO FIRST: cheapest, decides the whole strategy
Goal: settle comms-bound vs weight-bound *without a model load*. The team's model uses
`collective_latency_s = 5µs` (a ballpark) → TP8 comms 0.94 ms (dominant). If the real small-message
all-reduce is ~1.5µs, comms is only 0.28 ms and **weight** dominates instead. One measurement decides
whether to prioritize comms/prefetch or int4/kernels (see `docs/team-coordination.md`).
```bash
cd /workspace/nccl-tests   # already built
./build/all_reduce_perf -b 8 -e 64K -f 2 -g 8     # 8B..64KB across 8 GPUs; read latency(us) at 8-16KB
./build/alltoall_perf    -b 8 -e 64K -f 2 -g 8     # EP dispatch/combine latency
```
Record: the **8–16 KB all-reduce latency (µs)** and all-to-all latency. Then set
`src/inferutil/hardware.py: collective_latency_s` to the measured value and re-run the bench model.
**Go signal:** if all-reduce ≤ ~2µs → weight-bound → prioritize E7 (int4) + E4 (kernel); if ≥ ~4µs →
comms-bound → prioritize route-prefetch (`engine/routing/scheduler.rs`) + one-shot all-reduce, then E6 spec.

### E-attr — Split the floor (Nsight Systems)  ⟵ DO BEFORE optimizing it
Two data rounds show the engine is **floor-bound** (vLLM 16% of roofline; the adaptive engine 2.5%), and the
real TPOT decomposes to ~60% **unmodeled overhead**, 26% comms, 14% weight (`overhead-attribution.md`,
`results-reaction-02.md`). Before tuning comms OR kernels, **split the floor** so the #1 lever is known:
```bash
# trace ~20 decode steps on the bf16-TP8 (or fp8-TP8) server:
nsys profile -t cuda,nvtx,nccl -d 20 -o /root/decode_trace \
  python3 bench/measure.py --base http://localhost:8001 --model q --ctx 512 --decode 64
nsys stats --report cuda_gpu_kern_sum,nccl_sum /root/decode_trace.nsys-rep | head -40
# also: relaunch with --enforce-eager, re-measure TPOT (delta = launch/host the graph hides)
```
Record: % of a step in MoE/attn kernels (+ their achieved DRAM BW) vs NCCL all-reduce vs idle gaps.
**Branch:** NCCL dominates → E0b is #1; MoE kernels at low BW → the K5 kernels are #1 (vLLM fused_moe ~0.16
vs K5 0.46); idle gaps → fast-path/sampling. **Weight levers (fp8/int4/adaptive-topk) stay LAST until the
floor is down** — `ab_adaptive` proved they're invisible while floor-bound.

### E0b — Comms tuning sweep  ⟵ #1 IF E-attr says comms (E0 showed 16µs all-reduce → 3.0ms)
Use the **bf16 TP8** baseline (current best 85.7 tok/s / 11.67ms; bf16 has no 192 constraint so pure TP8
launches). Each variant is env + relaunch; record TPOT. Goal: cut the 16µs all-reduce → shrink the ~3ms
comms term, and **diagnose comms-vs-overhead** (if TPOT barely moves, the 11.67ms is mostly kernel/overhead
→ K5/CUDA-graph work, E3/E4).
```bash
# pure TP8 bf16 (NO --enable-expert-parallel):
BASE="vllm serve /alloc/data/Qwen3-235B-A22B --served-model-name q --tensor-parallel-size 8 \
  --dtype bfloat16 --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.92 --trust-remote-code"
# run measure.py (512/128) after each; record TPOT:
$BASE                                                   # 1. baseline (confirm ~11.67ms)
NCCL_PROTO=LL $BASE                                     # 2. low-latency small-message protocol
NCCL_NVLS_ENABLE=1 $BASE                                # 3. NVLink-SHARP in-switch all-reduce (mesh confirmed)
NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=2 $BASE         # 4. fewer channels for tiny payloads
$BASE --disable-custom-all-reduce                       # 5. isolate vLLM's one-shot AR vs NCCL
NCCL_PROTO=LL NCCL_NVLS_ENABLE=1 NCCL_MAX_NCHANNELS=2 $BASE   # 6. stacked best-guess
```
Record TPOT per variant + the winner. Expect the model's 16→~4–8µs → bf16-TP8 floor 216→320–421. See
`results-reaction-01.md`. (15-min budget: one load + a couple of env relaunches; pick the 2–3 most promising.)

### E2b — fp8 + TP8 via dynamic quant (the PRIZE cell, one-flag unblock)  ⟵ run right after E0b
The best physics cell (TP8 no-EP-penalty + fp8 ½-weight) that the released block-128 ckpt can't reach.
Dynamic fp8 has no `block_size` → no 192 crash:
```bash
vllm serve /alloc/data/Qwen3-235B-A22B --served-model-name q --quantization fp8 \
  --tensor-parallel-size 8 --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.92 --trust-remote-code
# then measure.py 512/128; and re-run with the E0b comms env (NCCL_PROTO=LL NCCL_NVLS_ENABLE=1 ...)
```
Record: does it launch (vs the block-128 192-crash)? TPOT/tok-s vs bf16-TP8 (85.7) and fp8-EP8 (64.5).
**Predicted floor 262 (16µs) → 638 (tuned comms).** This + E0b comms tuning is the highest-ceiling combo
without a requant. If dynamic fp8 fails/regresses, fall back to a block-64 FP8 requant (192/64=3).

### E1 — End-to-end B=1 engine baseline (FP8 + expert-parallel)  ⟵ now lower priority than E0b/E2b
Goal: the headline real single-user tok/s + which term dominates.
```bash
# launch (served name = measure.py default)
vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 8192 --port 8001 --served-model-name qwen3-235b-a22b \
  --gpu-memory-utilization 0.88 > /root/vllm_e1.log 2>&1 &
# wait for /v1/models, then:
for ctx in 128 2048 8192; do
  echo "ctx=$ctx"; python3 bench/measure.py --base http://localhost:8001 --ctx $ctx --decode 128
done
python3 bench/roofline.py --ctx 2048 --weight-bytes 1 --tpot-ms <TPOT@ctx2048>
# capture GPU balance during a decode (EP hotspot check):
nvidia-smi dmon -s u -c 20
```
Record: TTFT, TPOT, decode tok/s, % of roofline, dominant term, per-GPU util spread. **Compare tok/s to
the roofline ceiling (~994 weight-only / ~500–547 realistic at ctx≤8K, H100 FP8).**

### E2 — Layout comparison: EP8 vs TP4×EP2  ⟵ validates the EP→TP thesis
Goal: does a TP-heavier layout beat EP8 at B=1 on the real engine (spec's central claim)?
- **EP8**: as E1.
- **TP4×EP2**: find the vLLM flags that put TP=4 with 2 expert-parallel groups across 8 GPUs
  (try `--tensor-parallel-size 4 --enable-expert-parallel --data-parallel-size 2`; if vLLM rejects the
  combo, document the closest working layout and its tok/s).
- **TP8 column-shard**: needs a block-64 FP8 requant (192/64=3) — skip unless a requant exists.
Record tok/s for each working layout. Expect EP8 ≤ TP-heavier at B=1.

### E3 — CUDA-graph win
EP8 default (graphs on) vs `--enforce-eager`. Record TPOT both ways → the launch-overhead delta at B=1.

### E4 — K5 kernel: confirm e + Nsight + test next variant
```bash
cd kernels && /usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math k5_microbench.cu -I. -o k5bench
./k5bench 264 1024 3350            # expect e≈0.46, ~100x vs scalar, max_rel~3e-5
# profile to find the next bound (esp. the down-proj kernel k5b, e≈0.405):
/usr/local/cuda-12.6/bin/ncu --set full -k regex:k5 -c 4 ./k5bench 264 1024 3350 2>&1 \
  | grep -iE "DRAM Throughput|Achieved Occupancy|Memory Throughput|Stall|Issue Slots"
```
Record: Nsight DRAM-throughput %, occupancy, top stalls for `k5a_gateup_warp` and `k5b_down_warp`.
Then test the **front-loaded down-proj fix** (`kernels/k5_experts_warp2.cu` = per-slot 6 KB smem vs the
winner's 48 KB all-`a`, to lift occupancy):
```bash
/usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math kernels/k5_downproj_bench.cu -I kernels -o dbench
CUDA_VISIBLE_DEVICES=0 ./dbench 3350   # compares winner k5b (e≈0.405) vs v2 across tile/block configs
```
Record the v2 best `e` + maxrel. **Interpretation:** if v2 wins → the down kernel was occupancy-bound (fold
v2 into `k5_experts_warp.cu`); if no improvement → it's DRAM- or reduce-bound, and the next fix is sub-warp
split-K (fewer lanes/row, more rows/warp) — report Nsight so the planning agent designs it. Target: push
blended K5 `e` from 0.46 toward ~0.50.

Also test the **int4 expert GEMV** (the biggest *byte* lever — halves the dominant expert term):
```bash
/usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math kernels/k5_int4_bench.cu -I kernels -o i4bench
CUDA_VISIBLE_DEVICES=0 ./i4bench 3350   # fp8 winner vs int4 throughput + int4-unpack correctness
```
Record: int4 speedup vs fp8 (**2.0× = bandwidth-bound ideal; <2.0× = the nibble unpack is issue-bound**) +
the int4 unpack `max_rel` (should be ~0). This is the key open question int4 raises — whether the unpack
eats the byte win. Result decides whether int4 experts are worth wiring into the engine (E7) / cudarc path.

### E5 — Speculative decode acceptance (if a draft is wired)
If EAGLE/MTP/n-gram drafting is available, measure `spec_accept_rate` (τ) + tok/s uplift via `x_summary`.

### E6 — n-gram / prompt-lookup spec-decode (cheapest real multiplier)  ⟵ do right after E1
Relaunch the E1 engine adding a speculative config (validate exact flag form vs vLLM 0.10.1):
```bash
vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 8192 --port 8001 --served-model-name qwen3-235b-a22b --gpu-memory-utilization 0.88 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3,"prompt_lookup_min":1}'
# then re-run E1's measure.py at ctx 128/2048 and read acceptance + tok/s
```
Record: decode tok/s vs E1 (no-spec), `spec_accept_rate`/τ. **Go/no-go is REALIZED tok/s, not acceptance.**
**UPDATE — promoted to a top-2 lever (run NOW): while floor-bound, spec amortizes the dominant FLOOR.** The
verify is one batched forward that pays the per-step floor (188 all-reduces + launch) **once**, amortized over
τ → **≈τ× (~2× at τ=2 on the current bf16-TP8)**. The MoE verify-tax adds only to the 14% weight term, so it
**barely bites now** → **try k=4–6** (not 2–3) and **gate on realized tok/s**. As the floor falls (comms
tuning + kernel work → weight-bound), the tax returns → **shrink k→2–3** (make it adaptive on `RoundStats`).
Full reasoning: **`docs/spec-decode-floor-bound.md`** (supersedes the weight-bound k≤3 in `spec-decode-moe-tax.md`
*for the floor-bound regime*). n-gram is free-draft (ideal); prose needs self-spec (E9) / MTP.

### E7 — INT4/AWQ expert weights (biggest byte win; gated on a checkpoint)
First resolve the blocker: does an AWQ/GPTQ-INT4 `Qwen3-235B-A22B` checkpoint exist on HF for vLLM?
```bash
# search HF cache / hub for an int4/awq variant; if present, serve it (W4A16):
vllm serve <int4-or-awq-qwen3-235b> --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 8192 --port 8001 --served-model-name qwen3-235b-a22b --gpu-memory-utilization 0.88
# bench vs the FP8 baseline (E1); then quality-gate vs FP8 (bitwise/PPL/task).
```
Record: decode tok/s vs FP8 (expect **~1.13–1.20×** e2e, not 2× — only the expert term halves) + the
accuracy delta. If no checkpoint exists, note it and defer (quantizing 235B is a separate task).
See `docs/next-levers-research.md` L2.

### E8 — Verify route-prediction (DirectProxy) on a real MoE  ⟵ validates `engine/routing/predictor.rs`
Goal: confirm the team's zero-training route predictor actually predicts next-layer experts (so the
prefetch/early-dispatch in `scheduler.rs` is worth wiring) — *before* deploying on the 235B. Pure
inference; no engine relaunch. The box has torch 2.7.1.
```bash
# quick sanity on a small MoE (downloads ~14GB once), or point straight at the 235B:
python tools/verify_route_prediction.py --model allenai/OLMoE-1B-7B-0924 --device cuda --dtype bfloat16
python tools/verify_route_prediction.py --model /alloc/data/Qwen3-235B-A22B --device cuda --dtype bfloat16
```
Record: (A) DirectProxy accuracy, (B) layer-to-layer + (C) token-to-token expert overlap, all vs the
random baseline (top_k/n_experts). **Go signal:** (A) ≫ random → prefetch is sound, and 1−(A) is the
misprediction (wasted-prefetch) rate to budget for. Feeds the markov/route-cache work. *(I couldn't run
this locally — no torch on the Mac, and the only cached small MoE is a hybrid-arch GGUF/MLX without clean
router hooks — so it's queued for the box where torch + Qwen3 + standard `output_router_logits` exist.)*

### E9 — Self-speculation viability (shallow-pass agreement curve)  ⟵ decides the n-gram fallback
Goal: does a shallow pass (first L_d layers + logit-lens) predict the next token well enough to draft
without a trained head / extra GPU? Decides whether self-spec is a viable general-text fallback to n-gram.
Pure inference, no engine relaunch.
```bash
python tools/verify_self_speculation.py --model /alloc/data/Qwen3-235B-A22B --device cuda --dtype bfloat16
# (or a small MoE first: --model allenai/OLMoE-1B-7B-0924)
```
Record: top-1 agreement vs depth. **Go signal:** τ≈agreement·k must beat break-even (~1.86 at L_d=12,k=2)
at a *small* L_d (cheap draft). See `docs/self-speculation-design.md`. If even deep L_d barely clears it,
drop self-spec → rely on E6 n-gram (repetitive) + a trained MTP head (general).

## Results Log  (GPU agent: append, newest first; format below)
<!-- ### YYYY-MM-DD  E<n> — <one-line result>
     launch/config: ...
     raw: TTFT=.. TPOT=.. tok/s=.. %roofline=.. dominant=..
     notes/anomalies: ... -->
_(empty — awaiting first GPU window)_

## Blockers / questions → planning agent
_(none yet)_
