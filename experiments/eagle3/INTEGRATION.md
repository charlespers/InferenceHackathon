# EAGLE3 Speculative Decoding for Qwen3-235B-A22B in vLLM — Integration De-risk

Target: `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` on 8×H100, `--tensor-parallel-size 8 --enable-expert-parallel`.
Goal: ~1.8–2.4× lossless B=1 win via the published EAGLE3 draft head `lmsys/Qwen3-235B-A22B-EAGLE3`.
Stack assumed in prompt: vLLM ~0.10.x, torch 2.7.1+cu126.

---

## TL;DR / Verdict

- **EAGLE3 + Qwen3-MoE (235B-A22B) is supported by vLLM**, but only on **recent versions** (0.10.2+; safest is current 0.11.x / nightly). vLLM **0.10.1.1 does NOT support qwen3** EAGLE3 out of the box (model-type allowlist was `["llama","qwen"]`; the index-fix PR #24392 landed ~Sept 2025 / 0.10.2).
- **Do NOT point vLLM at the raw `lmsys/...` repo and expect it to "just work."** That card is **SGLang/SpecForge-documented only**, arch `LlamaForCausalLMEagle3`. The **safe, known-good path is the pre-converted speculators-format checkpoint** `nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys` (same weights, repackaged to vLLM's `Eagle3Speculator` format, verifier already pinned to `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`).
- **Memory: trivially fits.** Draft head is ~**1B params** (~2 GB BF16), single layer, vocab 151936 (+ draft vocab 32000). Negligible vs the FP8 235B target sharded over 8×H100 (80 GB each). Use **`draft_tensor_parallel_size: 1`** (run the small head un-sharded on one GPU; sharding a 1-layer head over 8 is pure overhead).
- **Biggest risk: CUDA-graph + EAGLE3 instability on the MoE+EP path.** Mitigation: launch with **`--enforce-eager` first** to validate correctness/accept-length, then re-enable graphs.

---

## 1. Exact vLLM speculative config

Verified key names and method string (vLLM EAGLE3 docs + Qwen3 EAGLE3 issue/PR):

- `"method": "eagle3"` — **distinct from `"eagle"`** (EAGLE-1). Must be exactly `eagle3` for an EAGLE3 head.
- `"model"`: draft-head repo/path.
- `"num_speculative_tokens"`: **required**. Start at **3** (the value baked into both the lmsys SGLang card and the nm-testing speculators config). Sweep 3→5→8; published accept length for this head is ~3–3.5, so 5 is usually the throughput sweet spot, with diminishing returns past that.
- `"draft_tensor_parallel_size"`: may only be **1** or equal to the target TP (8). Use **1**.

```json
{"method":"eagle3","model":"nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys","num_speculative_tokens":3,"draft_tensor_parallel_size":1}
```

Qwen3-MoE 235B-A22B EAGLE3 support is explicitly claimed by the vLLM/speculators project (seamless deployment "including Qwen3 MoE: 235B-A22B"), with reported accept length 1.8–3.5 and speedup up to ~1.9×.

## 2. Conversion / key remap

- **`lmsys/Qwen3-235B-A22B-EAGLE3` is NOT a drop-in for vLLM.** It is published in SGLang/SpecForge form (`LlamaForCausalLMEagle3`, BF16, 1B, ctx 8192, top-k 8, up to 32 draft tokens). The HF card documents **SGLang only**.
- **Two ways to get a vLLM-loadable head:**
  - **(A — RECOMMENDED) Use the already-converted checkpoint.** `nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys` is exactly lmsys's weights re-emitted in vLLM speculators format. Its `config.json`:
    - `architectures: ["Eagle3Speculator"]`, transformer_layer `model_type: llama`, 1 layer, hidden 4096, draft_vocab 32000, target vocab 151936, 64 heads / 4 KV heads / head_dim 128.
    - `speculators_config`: `algorithm: eagle3`, `num_speculative_tokens: 3`, `verifier: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`.
    - `eagle_aux_hidden_state_layer_ids: [1, 46, 90]` lifted to top level (EAGLE3 consumes 3 aux hidden states from the target).
    - Weights unchanged; only config restructured (`LlamaForCausalLMEagle3` → `Eagle3Speculator`, split into `transformer_layer_config` + `speculators_config`). **This is the load-as-is path.**
  - **(B — fallback if you must convert yourself)** Use `speculators` (`pip install speculators`) `EagleConverter`:
    ```python
    from speculators.convert.eagle.eagle_converter import EagleConverter
    EagleConverter().convert(
        input_path="lmsys/Qwen3-235B-A22B-EAGLE3",
        output_path="./qwen3-235b-eagle3-speculators",
        base_model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        layernorms=True,      # EAGLE3 heads carry extra norms; set True
        validate=True,
    )
    ```
    Then point `"model"` at `./qwen3-235b-eagle3-speculators`.
  - **(C — crude workaround, NOT recommended for prod)** On older vLLM, some users loaded a raw qwen3 eagle3 head by editing its `config.json` `model_type` `"qwen3"`→`"llama"` to pass the allowlist. Brittle; prefer (A).
- **Cross-check via the smaller sibling:** `Tengyunw/qwen3_8b_eagle3` / `AngelSlim/Qwen3-14B_eagle3` are the commonly-cited vLLM-working Qwen3 EAGLE3 heads and use the same `{"method":"eagle3", "num_speculative_tokens":3}` shape — confirms the config grammar before you commit GPU hours.

## 3. Memory feasibility

- Draft head ≈ **1B params, single decoder layer** → ~2 GB in BF16. Plus a draft-vocab LM head (32000×4096) and the target-vocab embedding sized 151936. Total well under a few GB.
- Verdict: **fits with comfortable headroom** alongside the FP8 235B target on 8×H100. The MoE target dominates; the head is rounding error.
- **`draft_tensor_parallel_size: 1`** (un-sharded, lives on one rank). Sharding a 1-layer head over TP=8 adds all-reduce latency per draft step with no memory benefit — counterproductive for a B=1 latency win.
- Keep `--gpu-memory-utilization` modest (e.g. **0.85–0.90**): EAGLE3 needs extra KV/activation headroom for the draft model's KV cache and the verify-step's widened batch (`num_spec_tokens+1` tokens per request).

## 4. Gotchas

1. **Min version.** Use **vLLM ≥ 0.10.2** for Qwen3 EAGLE3 (PR #24392, the qwen3 layer-index fix, merged ~Sept 2025). The stated 0.10.x is borderline — **0.10.1.1 will reject qwen3**. Strongly prefer **0.11.x or nightly** for MoE+EAGLE3 stability (later releases added per-draft-model MoE backend, eagle3 quant_config propagation, `norm_before_fc` propagation — all relevant to a MoE+FP8 target).
2. **FP8 block-128 / `1536/8=192 %128` issue.** This is the known MoE-FP8 error: `output_size ... = 192 is not divisible by weight quantization block_n = 128` when a per-expert gate/up projection is TP-sharded to a size not divisible by the 128 block. The documented fix is exactly what you're already doing — **`--enable-expert-parallel`** (experts replicated/distributed instead of column-sharded), so keep it. This is a *target-model* constraint and is orthogonal to the EAGLE3 head (the head is BF16, not block-FP8).
3. **CUDA-graph compatibility — the main one.** EAGLE3 + graph mode has a documented history of startup failures that disappear under **`--enforce-eager`**. Plan: **first launch with `--enforce-eager`** to confirm correctness and measure accept length; then drop it to capture graphs for the real perf number. If graphs crash, that's the known issue, not your config. If you run full-cudagraph/fullgraph EAGLE, `cudagraph_capture_sizes` must account for spec tokens (~`n*(K+1)` per batch size).
4. **EAGLE3 + EP interaction.** MoE + EAGLE3 is newer than dense + EAGLE3; the eagle3-quant-config / MoE-backend propagation fixes are recent. Validate **lossless parity** (your existing token-level parity gate) before trusting throughput — a silently-wrong draft still "works" but accept length collapses.
5. **Verifier pin.** The nm-testing head's `speculators_config.verifier` is pinned to `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` — matches your target exactly, so no mismatch. If you serve a different target revision, regenerate via path (B).
6. **transformers UPPER-BOUND (MEASURED 2026-06-20).** vLLM 0.11.0 requires only `transformers>=4.55.2` with **no upper bound**, so a fresh `pip install vllm==0.11.0` pulls **transformers 5.x** (5.12.1), which **removed `all_special_tokens_extended`** → vLLM's `get_cached_tokenizer` crashes at startup with `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended` (hits BOTH spec and plain baseline — it's pre-GPU, at tokenizer init, nothing to do with EAGLE3). **Fix: pin `transformers==4.57.1`** (contemporary with vLLM 0.11.0; tokenizers 0.22.2 OK). Reproduce/verify NON-GPU: `python -c "from vllm.transformers_utils.tokenizer import get_tokenizer; get_tokenizer('Qwen/Qwen3-235B-A22B-Instruct-2507-FP8')"`. The box venv `/alloc/data/eagle3-venv` + its build script are already pinned.

---

## RETURN

### Copy-pasteable launch command (most likely to work first try)

Validation pass (correctness first — eager to dodge the CUDA-graph gotcha):

```bash
vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --enforce-eager \
  --speculative-config '{"method":"eagle3","model":"nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys","num_speculative_tokens":3,"draft_tensor_parallel_size":1}'
```

Performance pass (after parity confirmed — drop `--enforce-eager`, bump spec tokens):

```bash
vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --speculative-config '{"method":"eagle3","model":"nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys","num_speculative_tokens":5,"draft_tensor_parallel_size":1}'
```

> Requires **vLLM ≥ 0.10.2** (prefer 0.11.x / nightly). If 0.10.1.x is pinned, upgrade — qwen3 EAGLE3 is not supported there.

### Conversion steps

None required if you use `nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys` (recommended). If you must convert the raw `lmsys/...` head yourself: `pip install speculators` and run the `EagleConverter().convert(..., layernorms=True, validate=True, base_model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")` snippet in §2(B).

### Memory feasibility verdict

**Fits easily.** ~1B-param / single-layer BF16 head (~2 GB) is negligible beside the FP8 235B MoE target on 8×H100. Run the head with `draft_tensor_parallel_size: 1`; keep `--gpu-memory-utilization 0.85–0.90` for KV + verify-batch headroom.

### Single biggest risk + mitigation

**Risk:** CUDA-graph capture failing (or silently degrading) on the EAGLE3 + MoE/EP path — the documented graph-mode startup failure for EAGLE3. **Mitigation:** bring it up with `--enforce-eager` first to lock in correctness and measure accept length (expect ~3–3.5, ~1.8–2.4× B=1); only then remove `--enforce-eager` to capture graphs for the headline number. Secondary guardrail: run your existing token-level parity gate to confirm the win is lossless before trusting throughput.

---

## Sources
- nm-testing converted speculators checkpoint: https://huggingface.co/nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys
- lmsys raw EAGLE3 head (SGLang-only): https://huggingface.co/lmsys/Qwen3-235B-A22B-EAGLE3
- vLLM EAGLE docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/
- vLLM issue #23464 (qwen3 eagle3 support / model_type allowlist): https://github.com/vllm-project/vllm/issues/23464
- vLLM PR #24392 (qwen3 eagle3 index fix): https://github.com/vllm-project/vllm/pull/24392
- speculators converter API: https://docs.vllm.ai/projects/speculators/en/latest/reference/speculators/convert/eagle/eagle_converter/
- speculators repo: https://github.com/vllm-project/speculators
- EAGLE 3.1 blog: https://vllm.ai/blog/2026-05-26-eagle-3-1
- Expert parallel deployment / FP8 block-128 fix: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- CUDA-graph/enforce-eager EAGLE3 issue: https://github.com/vllm-project/vllm-ascend/issues/2481
