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

## Hypothesis / crossover (refined by roofline — see tools/kv_roofline.py)

Initial guess was "crossover ≤7k because MoE has low active compute." The roofline
(real config: 94 layers, **GQA with only 4 KV heads**, head_dim 128, 22B active fp8
weights) **corrects this**: Qwen3-235B's KV cache is *small per token* (188 KB bf16
/ 94 KB fp8) relative to the 22 GB/token weight read, so the KV-read fraction is low
until long context. Bandwidth-bound prediction:

| ctx    | KV/(wt+KV) | roofline TPOT win | note |
|--------|-----------|-------------------|------|
| 128    | 0.1%  | 0.1%  | negligible |
| 2 048  | 1.8%  | 0.9%  | negligible |
| 8 192  | 6.7%  | 3.3%  | modest |
| 16 384 | 12.5% | 6.3%  | **crossover region** |
| 32 768 | 22.3% | 11.1% | clear (ceiling) |

So crossover is **~16k, later than the dense-8B 7k** — GQA-4 shrinks the KV term.
And these are bandwidth-bound *ceilings*; real wall TPOT carries TP+MoE comms, so the
measured latency win is SMALLER. The surer payoff for this model is **memory** (half
KV footprint ⇒ longer context fits / headroom for other levers). The A/B measures
which effect dominates in practice. Sweep ctx 128/2k/8k/16k/32k.

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
