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

### E1 — End-to-end B=1 engine baseline (FP8 + expert-parallel)  ⟵ HIGHEST
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
Then build/run any `kernels/k5_experts_warp2.cu` the planning agent pushes (down-proj block-reduce) and
record its `e`. Target: push blended `e` from 0.46 toward 0.55+.

### E5 — Speculative decode acceptance (if a draft is wired)
If EAGLE/MTP/n-gram drafting is available, measure `spec_accept_rate` (τ) + tok/s uplift via `x_summary`.

## Results Log  (GPU agent: append, newest first; format below)
<!-- ### YYYY-MM-DD  E<n> — <one-line result>
     launch/config: ...
     raw: TTFT=.. TPOT=.. tok/s=.. %roofline=.. dominant=..
     notes/anomalies: ... -->
_(empty — awaiting first GPU window)_

## Blockers / questions → planning agent
_(none yet)_
