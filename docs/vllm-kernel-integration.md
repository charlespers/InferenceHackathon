# Wiring the two 1000-tok/s kernels into vLLM (the actual build, no new engine)

`path-to-1000.md`: 1000 = graphs(fp8-K5-at-e→1 + NVLS-AR) + scheduler-free loop + small-tree spec — **two custom
kernels, integrated into vLLM as ops**, not a new engine. This is *where* and *how* in the vLLM tree. Both ops
are CUDA kernels → graph-capturable → vLLM's existing CUDA-graph path folds them into one launch/step for free.

## Kernel 2 (NVLS all-reduce) — the EASIER integration point, and the make-or-break. Do first.
vLLM already has a **custom one-shot all-reduce** for small messages, graph-captured:
- `vllm/distributed/device_communicators/custom_all_reduce.py` (+ the CUDA in `csrc/custom_all_reduce.cu`).
  It picks the one-shot path under a size threshold (256 KB on 8×H100) — our 8 KB hits it.
- **The change:** add/replace the reduce body with the **multimem NVLS** path (`kernels/nvls_allreduce.cu`'s
  `multimem.ld_reduce`/`st`) when NVSwitch+multicast is available. This is a localized kernel swap inside an
  already-captured op — no scheduler or model changes. It flows through `tensor_model_parallel_all_reduce`
  (`vllm/distributed/parallel_state.py`) automatically, so **every layer's AR uses it with zero model edits**.
- **Validate the op in isolation first** (`measure_collective.sh` then the kernel's microbench) — C decides 1000.
- *Gotcha:* the multicast buffers must be registered once at init (the custom-AR already allocates a shared
  buffer — extend its setup with `cuMulticastCreate`/bind, guarded by a capability check; fall back to the
  existing one-shot if NVLS is unavailable so nothing regresses).

## Kernel 1 (fp8 K5 at e→1) — the MoE expert path
vLLM's MoE is `vllm/model_executor/layers/fused_moe/` (the Triton `fused_moe_kernel` + `fused_experts`).
- **The change:** register `kernels/k5_experts_pipelined.cu` as a custom op (`torch.library.custom_op` or the
  C++ `TORCH_LIBRARY` path vLLM uses in `csrc/`), and dispatch to it **for the B=1 decode path** (M=1) where the
  Triton grouped-GEMM is sub-roofline. Keep Triton for prefill/large-M (it's a real GEMM there).
- **Interface to match:** vLLM passes `(hidden_states[M,H], w1[E,2I,H], w2[E,H,I], topk_ids, topk_weights)`. K5
  consumes the selected experts' fp8 weights + the per-channel scales (`w1_scale`/`w2_scale` in the fp8 config).
  Match that layout (it's the `Fp8MoEMethod` path) so the routing/scales come through unchanged.
- **Graph-safe:** the kernel has no host sync / dynamic alloc → captures cleanly. Validate parity vs the Triton
  path (max_rel < 1e-3) before enabling.
- *Note:* this is the **decode (GEMV)** kernel; the spec **verify** (batched M=W×D) should keep the grouped-GEMM
  (Triton) path — it's a real GEMM there (`why-spec-wins.md`). So the dispatch is M-dependent: M=1 → K5, M>threshold → Triton.

## The runtime (overhead → 0), all config, no kernels
- **CUDA graphs ON** (not `--enforce-eager`) — captures both custom ops + the layers into one replay/step.
- **`VLLM_USE_V1=1` + `--max-num-seqs 1` + `--disable-log-requests`** — the scheduler-free-ish B=1 path
  (`vllm-b1-config.md`). The residual host work is what `E-attr`'s idle-gaps measure; V1 minimizes it.
- **spec** via `--speculative-config` (EAGLE3 small tree at the weight-bound limit, or n-gram).

## Order (each step independently shippable + measurable)
1. **NVLS into custom_all_reduce** → re-measure decode tok/s (comms 3.0 → ~0.4 ms is the biggest single jump:
   rung 4 of `ladder_to_1000.py`, 260 → ~800). **This is the make-or-break and the highest-leverage integration.**
2. **K5 into fused_moe (M=1 dispatch)** → weight at roofline (kills the kernel-inefficiency part of the overhead).
3. **graphs + V1 + max-num-seqs=1** → launch/host → 0.
4. **small-tree spec** → ride to ~1000–1170.
Each is a localized vLLM change behind a flag/capability-check, falls back cleanly, and is parity-gated.

## Why this beats "build a new engine"
The cudarc megakernel is a clean parallel path, but it re-implements attention, MoE routing, sampling, KV, and
the scheduler from scratch. **Integrating two kernels into vLLM reuses all of that** and gets the same ~1000
(the persistent-kernel's extra is 0.06 µs/token). Ship the NVLS op first — it's the one experiment that proves
the whole path, in the existing engine.
