# vLLM e2e bootstrap — real-token Qwen3-235B + our spec/kernels (branch `djamoils-vllm-e2e`)

**Why this branch.** The native engine (`kernels/decode_step_tp8.cu`) is a LATENCY PROXY — dummy fp8
weights (one layer reused ×94), no real weight loading, prefill exists only as standalone microbenches
(`prefill_attn/moe/wgmma.cu`), and `server/main.py` is a mock. So the native high-perf path does **not**
produce real tokens. vLLM (in `/alloc/data/eagle3-venv`, v0.11.2) DOES: real weight load + prefill +
decode + EAGLE3 spec. This branch borrows vLLM's plumbing to get a real-token e2e, then swaps our
optimizations in for perf. (vLLM is Apache-2.0; `vllm_ref/` files are copied for reference/attribution.)

## Reference copied (`vllm_ref/`, from the box's installed vLLM)
- `qwen3_moe.py` — the Qwen3-235B-A22B model def + `load_weights()` (the safetensors→tensor mapping,
  `stacked_params_mapping`, `FusedMoE.make_expert_params_mapping` for the 128-expert MoE, TP sharding).
  THIS is the weight-loading spec the native engine lacks.
- `default_loader.py` / `base_loader.py` — vLLM's weight-loading machinery (safetensors streaming).
- `eagle.py` (`v1/spec_decode/`) — vLLM's EAGLE3 verify/propose loop. **This is the thing to make flat**:
  vLLM's verify isn't flat in M (→ ~1× at B=1); our `engine/native/tc_verify_attn.cuh` IS flat (~3.3–3.4×).
- `qwen3_235b_config.json` — shapes (HIDDEN 4096, 94 layers, 64 Q/4 KV heads, 128 experts, top-8).

## Fastest path to real-token e2e + perf (recommended milestones)
1. **Baseline (DONE/runnable):** `vllm serve /alloc/data/Qwen3-235B-A22B --tensor-parallel-size 8` (the live
   demo runs this). Add `--speculative-config` w/ the RedHat EAGLE3 head for lossless spec.
2. **Measure real numbers:** baseline vs EAGLE3, B=1..N, to confirm vLLM spec ~1× at B=1 (verify not flat).
3. **PERF SWAP (the win):** replace vLLM's verify attention with our flat TC verify (`tc_verify_attn.cuh`)
   via a custom attention backend / torch custom-op in the EAGLE3 verify path (`eagle.py`). This is where
   the ~3.3–3.4× comes from — vLLM gives the plumbing, we give the flat verify.
4. **Parity gate:** spec output == greedy (lossless), cross-check vs unmodified vLLM EAGLE3 greedy.

## Coordination
Someone else may be attempting the e2e bootstrap — **this branch is `djamoils-vllm-e2e`**; sync via
`danielAgentScheduling.md` to avoid duplication. The GPU is demo-locked (USER-PRIORITY); weight-load +
prefill wiring is CPU/build work until a real run is needed. Owner: TBD (recommend a dedicated agent —
LOOP-A continues spec-decode perf tuning in parallel on `djamoils-work`).
