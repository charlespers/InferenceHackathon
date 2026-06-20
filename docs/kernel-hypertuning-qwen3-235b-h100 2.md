# Kernel Hypertuning Runbook — Qwen3-235B-A22B @ B=1 on 8×H100

Latency-oriented, batch-size-1 decode. The objective is a single number — **bytes
moved per token** — because B=1 has no batch to amortize weight reads.

```
TPOT ≈ max(  per-GPU bytes_read / HBM_BW ,  expert all-to-all latency ,  kernel-launch overhead )
```

Every kernel and fusion below is judged by whether it removes an HBM round-trip,
shrinks the bytes read, or removes launch/sync overhead — nothing else.

> Architecture is for the public model **Qwen/Qwen3-235B-A22B** (config.json verified).
> All techniques here are standard CUDA/GPU practice. No proprietary engine code.

---

## 1. Confirmed architecture (drives every kernel shape)

- 94 layers, **all MoE** (`decoder_sparse_step=1`, `mlp_only_layers=[]`); hidden 4096; separate lm_head 4096→151936; `tie_word_embeddings=false`.
- **Attention:** GQA **16:1** (64 Q heads, 4 KV heads), `head_dim=128` (explicit; note 64×128=8192≠4096). `q_proj 4096→8192`, `k/v_proj 4096→512`, `o_proj 8192→4096`. **No biases.**
- **QK-norm:** per-head **RMSNorm over the 128-dim** on Q and K (128 weights each), applied **before RoPE**. RoPE θ=1e6. `rms_norm_eps=1e-6`.
- **MoE:** 128 experts, **top-8, no shared expert**. Router = `gate 4096→128` → **fp32 softmax over 128** → top-8 → **renormalize to sum 1** (`norm_topk_prob=true`). Expert = SwiGLU: `gate/up 4096→1536`, `down 1536→4096`, SiLU.

### Byte budget per decode token
- Active params ≈ **21.57B** → fp8 **~21.6 GB**, int4 **~10.8 GB**. Split: experts **~14.2B** (the bottleneck), attention+router+norms ~6.8B, lm_head 0.62B.
- KV cache: **96,256 elems / cached token** (94 layers × 4 KV heads × 128 × 2). At 32k ctx: **6.3 GB fp16 / 3.15 GB fp8**.
- Roofline (8×H100 = 26.8 TB/s aggregate, fp8 weights): **~1,240 tok/s** short-ctx → **~1,080 @ 32k** → **~780 @ 128k**. Idealized ceiling; real systems land at 40–70% of it.

---

## 2. Fusion map — one decode layer (B=1, M=1 GEMVs)

| Kernel | Fuses | Shapes | Primary tuning axis |
|---|---|---|---|
| **K1 attn-prologue** | input-RMSNorm → fused-QKV GEMV → per-head QK-norm → RoPE → write-KV | `W_qkv 4096×9216`; q→64×128, k/v→4×128; KV write 1024 elems | HBM-bound on W_qkv: 128-bit coalesced loads, K-major layout, dequant ILP. QK-norm(fp32)+RoPE are free epilogue ops — never round-trip |
| **K2 flash-decode** | online-softmax single-query attn, GQA broadcast, in-reg KV dequant | read KV (ctx×1024); q(8192)→attn(8192) | **split-KV / flash-decoding**: partition seq across CTAs + 2-pass combine, or 64 heads underfill the SMs. KV dtype = the bytes term |
| **K3 attn-epilogue** | O-proj GEMV **+ residual add (fuse this)** | `W_o 8192×4096`; +h → h′ | BW on W_o; fold residual into the epilogue (one less dispatch) |
| **K4 router** | post-RMSNorm → gate GEMV → fp32 softmax(128) → top-8 → renorm | `W_gate 4096×128` → ids[8], gates[8] | keep entirely on-device (no host sync); fp32 softmax stability |
| **K5 experts** ⟵ *bottleneck* | gate+up GEMV → silu⊙ → down GEMV ×gate_e → accumulate into residual | per expert `[W_gate|W_up] 4096×3072`→a(1536); `W_down 1536×4096`; ×8 | persistent/grouped GEMV over the 8; fold gate_e + Σ + residual into down epilogue; **fp8 = biggest byte win**; EP placement → 1 expert/GPU/token |
| **K6 step capture** | capture K1–K5 ×94 + final-norm + lm_head + sample → replay | lm_head 4096×151936; on-device sampling | CUDA-graph the whole step → kill launch latency; must survive speculative decode (see §5) |

**Cross-GPU (EP=8 × TP=2):** K5 = all-to-all dispatch of y(4096) to the GPUs holding
the 8 selected experts, gather out_e back; K1–K3 carry small TP all-reduces. The only
B=1 overlap available: dispatch layer L+1's router while layer L's expert-combine runs.

---

## 3. Open hypertuning levers (where effort pays off), prioritized

1. **EP=8 / TP=2 all-to-all for K5.** With 8 active experts and 8 GPUs, choose expert
   placement so the expected top-8 lands ≈1/GPU → balanced 8×8 dispatch, no hotspot.
   Tune: placement permutation, all-to-all chunk size, NCCL small-message settings,
   overlap of dispatch with the prior layer's combine.
2. **Graph capture that survives speculative decoding + on-device sampling.** Naive
   capture breaks when accept-length varies. Capture the fixed verify pass over a
   max-draft window and mask rejected positions; keep argmax/top-p sampling on-device so
   no per-token D2H sync. This unlocks the launch-overhead win *and* speculation together.
3. **fp8 dequant epilogues.** Retarget the K1/K5 fused GEMVs to fp8 e4m3 (H100 tensor
   cores) with per-channel/row scales; this both halves bytes vs bf16 and changes the
   epilogue math. Co-tune scale granularity vs quality.
4. **Persistent MoE expert kernel.** Keep the token activation resident and stream the 8
   experts' weights through one persistent kernel; amortize launch and reuse h_row.
5. **Fused attention residual (K3).** Fold `+residual` into the O-proj epilogue — one
   fewer dispatch per layer × 94 layers.

---

## 4. Precision degrees of freedom (co-tuned with a quality gate)
- Weight format: fp8 e4m3 (default) vs int4 (half the bytes, more dequant work) vs mixed per-tensor.
- KV dtype: fp8/int8 — directly halves the KV term (dominant past ~32k ctx).
- Activation precision: bf16 vs fp8 on the GEMV inputs.
- Scale granularity: per-tensor vs per-channel vs per-group (dequant cost vs accuracy).

---

## 5. Speculative decoding × graph capture (the hard, high-value piece)
B=1 is memory-bound, so drafting K tokens and **verifying them in one weight-read pass**
converts K single-token reads into one amortized pass — the single biggest lever.
- Draft source: prompt-lookup/n-gram (zero training) → MTP/EAGLE (learned, ~60–80% accept).
- The challenge: accept-length is data-dependent, which breaks static graph capture.
  Fix: capture a fixed max-draft verify graph; handle variable accept by masking, not
  re-capture. Keep sampling + accept-test on-device so the graph never needs a host sync.
- Tune: draft length K, accept threshold, draft model size vs accept rate.

---

## 6. The prompt as a tuning axis
- **Prefix-KV reuse** for system/few-shot prompts → skip re-prefill (TTFT win).
- **Speculative accept ∝ prompt structure** — quoting/agentic/structured prompts make
  n-gram/prompt-lookup drafting hit; format prompts to raise accept.
- **Routing balance** — monitor the per-expert hit histogram; a domain that concentrates
  the top-8 creates a GPU hotspot. Rebalance placement (or a small inference-time
  load-balance bias) to keep ≈1 expert/GPU/token.
- **Context length** is linear in the KV term — tighter prompts lower TPOT directly;
  **constrained/grammar decoding** shrinks lm_head sampling and raises speculative accept.

---

## 7. Method — roofline chase
Fixed harness: B=1, 512-prompt / 128-decode window, seed 0. Per change, measure which
term dominates — **weight-BW / KV-BW / comms / launch** — tune the DoF that attacks it,
re-measure (the dominant term *shifts* as you optimize). Watch: TPOT, bytes/token,
per-GPU HBM utilization %, speculative accept rate, per-expert/GPU load balance.

Suggested order: fp8 weights+KV → fuse K3 residual → CUDA graph (greedy) → EP=8 placement
→ persistent expert kernel → speculative decode + graph-survives-spec → TP=2 → NCCL tuning.
Stop when TPOT approaches `bytes_per_token / usable_HBM_BW` — then you're physics-limited
and only fewer bytes (more quant / more speculation) or more GPUs help.
