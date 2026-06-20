# KV-cache FP8 quantization for B=1 decode (Qwen3-235B-A22B, 8×H100)

**Lever:** quantize the KV cache to FP8 (e4m3) so each decode step re-reads half
the KV bytes from HBM. The win lands on **TPOT** (per-token decode latency) and
**grows with context length**, because at B=1 decode is HBM-bandwidth bound and
the KV read scales with sequence length while the weight read is fixed per step.
Orthogonal to — and stacks with — the expert/weight/comms levers others own
(weights already FP8; this is the *cache*, untouched by them).

## Config (validated against vLLM FP8-KV blog, 2026-04-22)

- Flag: `--kv-cache-dtype fp8` (== `fp8_e4m3`). Baseline = `auto` (bf16 KV).
- Backend: FlashAttention-3 — **default** on Hopper H100, no `VLLM_ATTENTION_BACKEND`
  needed. FA3 is what makes fp8-KV *fast*, not just memory-saving; wrong backend ⇒
  memory win only. Verify FA3 in the launch log.
- Scales: default per-tensor, **uncalibrated** (scale=1.0). Blog: recovers 97–98%
  AUC@128k on 70B-class; calibration (llm-compressor) only if recall regresses.
- Model: global-attention MoE ⇒ **no** `--kv-cache-dtype-skip-layers` (that's for
  sliding-window/hybrid models like gpt-oss).
- Serve: FP8 weights + `--enable-expert-parallel` + CUDA graphs (no `--enforce-eager`),
  `--max-num-seqs 1`, `--max-model-len 36864`, TP=8, port **8088**.

## Hypothesis / crossover

Blog reports decode break-even ≈ **7k tokens** for dense Llama-3.1-8B (ITL slope
→ 54% of bf16). Qwen3-235B is **MoE with only ~22B active params**, so compute per
decode step is low relative to the KV read ⇒ the KV-read fraction of each step is
*larger* ⇒ we expect the crossover **at or below 7k**, and a *bigger* TPOT win at
32k than the dense-8B number. To test:

| ctx    | expectation |
|--------|-------------|
| 128    | neutral / slight fp8 loss (quant overhead, tiny KV) |
| 2 048  | ~neutral, near break-even floor |
| 8 192  | fp8 ahead on TPOT (past crossover) |
| 32 768 | fp8 clearly ahead; largest decode tok/s gain |

## Harness

- `tools/kv_measure.py` — B=1 streaming probe; greedy; reports TTFT, TPOT,
  decode tok/s, and **server-measured prompt_tokens** (compare at equal real ctx).
- `tools/kv_ab.sh {auto|fp8}` — one dtype's full ctx sweep + quality capture,
  sized to one 15-min slot. GPU-gated (skips if min free < 65 GB). Run twice across
  slots, then compare.
- `tools/kv_quality.py` — greedy capture + offline compare; short determinism probes
  + needle-in-haystack recall at 800/4k/16k words and depth 10/50/90% (the regime
  where KV quant is most likely to hurt). Gate = no recall regression vs baseline.

## Run order

1. Slot A (:45–:00 UTC): `bash tools/kv_ab.sh auto` → `results/kv_fp8/auto/`
2. Slot B: `bash tools/kv_ab.sh fp8` → `results/kv_fp8/fp8/`
3. Offline: `python3 tools/kv_quality.py compare results/kv_fp8/auto/quality.json results/kv_fp8/fp8/quality.json`
4. Pull results to laptop, write `FINDINGS.md`, commit (`git add -f results/`), push.
