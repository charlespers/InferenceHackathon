# Config sweep checklist — Qwen3-235B-A22B B=1 latency on 8×H100

Full parameter space we want to look into, organized by category, with which
metric each one targets (see `DESIGN.md` for the metric definitions: TPOT,
TTFT, bytes/token, comms latency, expert imbalance, spec accept rate).

Status key: ✅ = real measured number in hand · 🔲 = not yet run

```mermaid
graph TD
    ROOT["Qwen3-235B-A22B B=1 latency — config space"]

    ROOT --> PAR["Parallelism layout"]
    ROOT --> PREC["Precision"]
    ROOT --> COMM["NCCL / comms"]
    ROOT --> KERN["Kernels / runtime"]
    ROOT --> SPEC["Speculative decoding"]
    ROOT --> ENG["Engine choice"]

    PAR --> PAR1["TP degree: 1 / 2 / 4 ✅8(measured, no EP)"]
    PAR --> PAR2["EP degree: 1 / 2 / 4 / ✅8(measured, confounded w/ fp8)"]
    PAR --> PAR3["plan: ✅tp(bf16) / ✅ep(fp8, forced by quant block_size) / hybrid 🔲"]
    PAR -.affects.-> M_BYTES["bytes/token (per-GPU)"]
    PAR -.affects.-> M_IMBAL["expert imbalance"]

    PREC --> PREC1["weight dtype: bf16 ✅TP=8 / fp8 ✅TP=8+EP=8(confounded) / fp8-otf ✅TP=8 no-EP(clean) / int4 🔲(quality risk)"]
    PREC --> PREC2["KV dtype: bf16 ✅ / fp8-int8 🔲"]
    PREC -.affects.-> M_BYTES

    COMM --> COMM1["NCCL_ALGO: default ✅measured / RING 🔲 / TREE 🔲"]
    COMM --> COMM2["NCCL_PROTO: default ✅ / LL128 🔲 / LL 🔲"]
    COMM --> COMM3["NCCL_P2P_LEVEL: default ✅ / NVL 🔲"]
    COMM --> COMM4["NCCL_MAX_NCHANNELS: default ✅ / 32 🔲"]
    COMM -.affects.-> M_COMMS["comms latency (measured: 10.5µs a2a@8 / 16µs ar@8 / 6.5µs ar@2)"]

    KERN --> KERN1["CUDA graphs: on 🔲 / --enforce-eager 🔲(baseline-only so far)"]
    KERN --> KERN2["kernel fusion: gated-FFN, attn+norm 🔲"]
    KERN --> KERN3["attention kernel: default 🔲 / FlashInfer single-query 🔲"]
    KERN --> KERN4["KV layout: contiguous 🔲 / paged 🔲"]
    KERN -.affects.-> M_TPOT["TPOT (overhead component)"]

    SPEC --> SPEC1["algorithm: none ✅baseline / n-gram 🔲 / EAGLE-MTP 🔲"]
    SPEC --> SPEC2["draft length / num_steps: 🔲"]
    SPEC -.affects.-> M_ACCEPT["spec accept rate → effective TPOT"]

    ENG --> ENG1["SGLang 🔲(not yet launched)"]
    ENG --> ENG2["vLLM ✅measured (TP=8, bf16, 11.67ms/tok)"]
    ENG --> ENG3["plain transformers ✅measured (289ms/tok baseline)"]
    ENG -.affects.-> M_TTFT["TTFT, TPOT (engine overhead floor)"]
```

## Where we stand

| Category | Measured | Still open |
|---|---|---|
| Parallelism | plain TP=8 bf16 (vLLM) ✅ + TP=8+EP=8 fp8 (vLLM) ✅ + naive HF sharding ✅ | bf16+EP=8 (isolate EP-vs-TP cleanly — still not done; every EP run so far has been confounded with precision or eager mode) — TP=2×EP=8 hybrid plan untested |
| Precision | bf16 ✅ + FP8 pre-quantized+EP=8 (confounded) ✅ + **FP8 on-the-fly quant, TP=8, no EP (clean A/B) ✅** — see below | KV-cache FP8/INT8, int4 |
| NCCL | default-algo small-message latency (10.5/16/6.5µs, see below) | `NCCL_ALGO`/`NCCL_PROTO`/`NCCL_P2P_LEVEL` sweep |
| Kernels | vLLM default (CUDA graphs + torch.compile, on) ✅ | `--enforce-eager` comparison, manual fusion, FlashInfer |
| Spec decode | none (greedy-only baseline) | n-gram first, then EAGLE/MTP if Qwen3 supports it |
| Engine | vLLM TP=8 bf16 ✅ + vLLM TP=8+EP=8 fp8 ✅ + transformers (naive) ✅ | SGLang still unlaunched |

## Measured numbers so far (8×H100, this box)

- **Naive baseline** (plain `transformers.generate()`, device_map="auto", no
  TP/EP/graphs/fusion/FP8): **289.1 ms/token** (p50=285.7, p95=304.4) — see
  `routing_analysis.py` run, `/alloc/data/routing_stats.json` on the box.
- **Real expert-routing imbalance**: 5-8× on several layers (worse than the
  analytical model's uniform-routing estimate of ~2.6×). Hottest single
  expert (`L17·E78`) saw 770 activations vs. an expected-uniform average of
  ~62 over the same run.
- **NCCL small-message latency** (default algo/protocol, `nccl-tests`):
  - all-to-all, 8 GPUs: ~10.3–10.7 µs
  - all-reduce, 8 GPUs: ~16–17 µs
  - all-reduce, 2 GPUs (TP=2 group size): ~6.4–7.3 µs
  - vs. the model's flat `collective_latency_s = 5e-6` assumption — real
    comms cost is ~1.3–3.4× higher depending on collective/group size.
- **NVSwitch topology**: confirmed full mesh — every GPU pair shows `NV18`
  in `nvidia-smi topo -m`, validating the hybrid TP=2×EP=8 plan's precondition.
- **vLLM baseline** (TP=8, bf16, no EP, no FP8, no spec decode, CUDA graphs +
  torch.compile on — vLLM 0.10.1 defaults, `--max-model-len 8192`): single
  streamed request, 127 tokens, greedy.
  - TTFT: 776.7 ms
  - TPOT mean / p50 / p95: 11.67 / 11.58 / 12.35 ms
  - decode: **85.7 tok/s** — 15.9% of the analytical floor (540 tok/s), vs.
    35.6% for the hybrid-bf16 analytical estimate (192 tok/s) and 0.64% for
    the naive transformers baseline. ~25× faster than naive, ~2.2× short of
    the hybrid-plan estimate — expected, since this run is plain TP=8 with
    no expert parallelism, so it's really validating the `latency.py` "tp"
    row (3.03ms floor-only) against real CUDA-graph/kernel/comms overhead,
    not the hybrid plan.
- **vLLM FP8 run** (pre-quantized `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`,
  e4m3 dynamic activation scheme, weight `block_size=[128,128]`). Plain
  `--tensor-parallel-size 8` **fails to load**: the per-GPU expert FFN slice
  under TP=8 is `1536/8=192`, not divisible by the quant block size 128
  (`ValueError: output_size... not divisible by weight quantization
  block_n = 128`). Worked around with `--enable-expert-parallel`, which
  shards whole experts across GPUs instead of slicing each expert's FFN —
  avoids the block-size constraint entirely. Same single-request benchmark:
  - TTFT: 631.1 ms (faster than bf16's 776.7 ms)
  - TPOT mean / p50 / p95: 15.51 / 15.50 / 15.77 ms (**slower** than bf16's 11.67 ms)
  - decode: **64.5 tok/s** — *25% slower* than the bf16 TP=8 run (85.7 tok/s),
    contradicting the model's prediction that FP8 should roughly halve the
    weight-read term and speed things up.
  - **This result is confounded, not a clean precision comparison.** The FP8
    run is TP=8 **+ EP=8**; the bf16 run was TP=8 with no EP at all. The
    slowdown direction matches exactly what the analytical model predicts for
    naive EP at B=1 (`DESIGN.md`: EP=8 busiest-GPU imbalance ~2.6×, naive EP
    slower than plain TP) — so we may be looking at the imbalance penalty
    swamping whatever speedup FP8 alone provides, not evidence that FP8 is
    actually slower.
  - **Next run needed to isolate this**: bf16 + `--enable-expert-parallel`
    (same parallelism strategy as the FP8 run, precision held as the only
    variable) — queued, not yet run. Until then, neither "FP8 is slower" nor
    "FP8 is faster" is a supportable conclusion from this data.

- **FP8 on-the-fly quantization, TP=8, NO expert-parallel (the clean A/B)**.
  `vllm serve` on the **bf16** checkpoint with `--quantization fp8` (vLLM's
  own dynamic per-tensor/per-channel cast at load time) instead of the
  pre-quantized block-quantized checkpoint — this avoids the
  `block_size=128` / TP=8 divisibility error entirely, so no `--enable-
  expert-parallel` was needed. Same TP=8, same CUDA-graph mode as the bf16
  baseline on both sides — genuinely single-variable this time. Measured via
  `tools/measure_baseline.py` (5 repeats, median):
  - TPOT: 14.49 ms (vs. bf16's 11.67 ms)
  - decode: **69.0 tok/s** (vs. bf16's 85.7 tok/s) — **~19% slower**
  - % of roofline: 12.8% (`dominant_term_hint: DOMINATED by floor —
    launch/host/comms`)
  - TTFT (23.9 ms vs. bf16's 776.7 ms) is **not a reliable comparison point**
    here — almost certainly a prefix-cache-hit artifact from prior warmup
    requests on this server, not a real precision effect. TPOT/decode-tok-s
    is the trustworthy number since it reflects steady-state inter-token
    timing.
  - **Conclusion: FP8 has now underperformed bf16 in two independent,
    differently-confounded attempts** (pre-quantized+EP: 64.5 tok/s;
    on-the-fly+no-EP: 69.0 tok/s; both below bf16's 85.7 tok/s). This is no
    longer a confound artifact — it's a repeated result. The likely
    explanation is the same overhead-dominated regime djamoils's adaptive-
    top-k experiment found (only 1.8-12.8% of roofline across every EP/FP8
    run so far): when launch/host/comms overhead dominates, shrinking the
    weight-byte term doesn't help and may even add overhead (extra
    dequant/scale-handling kernels) that costs more than it saves.
  - Quality probe captured (`alyssa_fp8otf_quality.json`, 10 prompts) for a
    future `tools/quality_compare.py` diff once a comparable bf16-no-EP
    quality probe exists.

- **Opportunistic FP8 (pre-quantized) + EP=8 probe** against Jaymin's
  already-running server (`max-model-len=4096`, `gpu-mem-util=0.92`, graphs
  on, different launch from our own FP8+EP attempt but same core config).
  `tools/measure_baseline.py`, 5 repeats:
  - TPOT: 16.43 ms, decode: **60.86 tok/s** (5/5 runs within 0.03ms — very
    consistent), 11.3% of roofline.
  - Close to our own independent FP8+EP measurement (64.5 tok/s) — two
    separate launches landing in the same range adds confidence this is a
    real number, not a one-off artifact.
  - Notably *better* than the eager-mode EP numbers (1.8-2.5% of roofline)
    — this server has CUDA graphs on, reinforcing that graphs-vs-eager is a
    bigger lever than precision or EP choice.

- **djamoils: adaptive top-k (k=4) vs baseline (k=8), both bf16+EP=8+
  `--enforce-eager`** (`results/ab_baseline.json`, `results/ab_adaptive.json`):
  - baseline 13.23 tok/s -> adaptive 9.67 tok/s — a **~27% regression**.
  - Both runs: `dominant_term_hint: DOMINATED by floor (launch/host/comms)`,
    1.8-2.5% of roofline. Cutting expert-bytes doesn't help when bytes
    aren't the bottleneck, and the adaptive policy's own branching logic
    costs more than the bytes it saves.
  - Caveat: this baseline (13.23 tok/s) is eager-mode EP, not our clean
    graphs-on TP=8 baseline (85.7 tok/s) — the regression is real, but its
    *magnitude* is specific to the eager+EP regime, not necessarily
    representative of a graphs-on hybrid config.

- **NCCL sweep** (`tools/nccl_sweep.sh`, 1024B messages, one variable at a
  time): `NCCL_ALGO` {RING,TREE}, `NCCL_PROTO` {LL,LL128,SIMPLE},
  `NCCL_P2P_LEVEL=NVL`, `NCCL_MAX_NCHANNELS=32`, vs. defaults.
  - **Defaults already win or are within noise of winning in every group.**
    Explicit overrides mostly made things *worse* — `NCCL_PROTO=SIMPLE` on
    all-reduce@8 was 34.72µs vs. default's 15.98µs, more than 2x worse.
  - Only "win": `NCCL_P2P_LEVEL=NVL` on all-reduce@8, +1.5% (15.74 vs
    15.98µs) — likely within run-to-run noise, not a real effect.
  - **Conclusion: no free win available via NCCL env vars on this
    topology.** NCCL's auto-selection is already near-optimal. The gap
    between measured comms cost and the model's flat 5µs assumption is a
    **model-calibration problem, not a config problem** — confirms the
    `measured_max_experts`-style override pattern is the right fix, not a
    tuning knob.
  - Re-measured baseline numbers this run came in ~50% higher than the
    original measurement from earlier today (15.56-15.99µs vs. 10.72-15.99µs
    originally) on the *same* defaults — real run-to-run variance in the
    microbenchmark itself; treat absolute NCCL numbers as noisy to within
    that range, not precise to the µs.

- **Hybrid TP=2×EP=8 attempt (the playbook's actual recommended config) —
  FAILED, CUDA OOM.** Launched via `--tensor-parallel-size 2
  --data-parallel-size 4 --enable-expert-parallel` (4 DP groups × TP=2 each
  -> EP=8 spanning all 8 GPUs, per vLLM's "MoE shards by the TP×DP product"
  docs). Weights alone consumed ~64.3GB/GPU (vs. ~58GB/GPU in the plain
  TP=8+EP=8 case), leaving only ~622MB free — OOM'd during sampler warmup
  before ever reaching "ready."
  - Open question, not yet resolved: did this actually achieve EP=8 sharding
    across the full DP×TP pool, or did each DP=4 group fall back to local
    EP=2 (TP size), meaning each GPU held ~4x more expert weight than the
    EP=8 case? The memory numbers are consistent with the latter, but this
    needs confirming against vLLM's actual DP+EP source/logs before trusting
    either interpretation.
  - Next attempt should add `--gpu-memory-utilization` lower than default
    and/or `--max-num-seqs 1` (the flag missing here that other successful
    launches in this doc didn't need, but apparently this topology does) to
    leave warmup headroom, and verify true global EP=8 sharding is in effect
    before drawing any speed conclusion.

## Dependency note

Engine choice gates most of the rest of this list — CUDA graphs, fusion,
and FP8-in-practice all need a real serving engine (vLLM or SGLang) up and
running before they're testable. The naive-transformers and NCCL-microbench
rows are the only ones we could measure without one.
