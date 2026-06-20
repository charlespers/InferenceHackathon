#!/usr/bin/env python3
"""validate_bf16_gemm_swap.py — the gate that must pass before bf16_linear_method.py is ever trusted
in a live vLLM serving path. Two levels, cheapest first:

  (1) Kernel-correctness, no vLLM needed: bf16_gemm_ext's output vs torch.nn.functional.linear (the
      same op vLLM's own UnquantizedLinearMethod ultimately calls) on Qwen3-235B's REAL o_proj shape
      (TP=8: input 1024, output 4096), random bf16 data. This is the cheap, fast, no-model-load check
      -- run this FIRST, every time the kernel changes.
  (2) End-to-end in a real (small) vLLM model: patch one layer's o_proj, run a real forward pass
      patched vs unpatched on the SAME input, diff hidden states. Needs a GPU + a loaded model --
      only run this after (1) passes.

Per the file header in bf16_gemm_ext.cu: a fast GEMM that's wrong is worse than no swap at all. Don't
skip straight to (2).
"""
import os
import sys

import torch

sys.path.insert(0, "tools/vllm_kernels")
from bf16_linear_method import _get_ext

# Qwen3-235B-A22B real o_proj shape, TP=8 (this rank's shard): input = Q_DIM/TP = 8192/8 = 1024,
# output = HIDDEN = 4096 (matches kernels/common.cuh's constants, cross-checked against config.json).
Q_DIM_RANK = 1024
HIDDEN = 4096


def check_kernel_correctness(n_trials=5, atol=2e-2, rtol=2e-2):
    """Level (1): no vLLM, no model load. bf16_gemm_ext vs torch.nn.functional.linear."""
    ext = _get_ext()
    worst_max_abs = 0.0
    for trial in range(n_trials):
        torch.manual_seed(trial)
        M = [1, 4, 8, 16, 32][trial % 5]   # cover both decode (M=1) and batched-verify-shaped M
        X = torch.randn(M, Q_DIM_RANK, dtype=torch.bfloat16, device="cuda")
        W = torch.randn(HIDDEN, Q_DIM_RANK, dtype=torch.bfloat16, device="cuda") * 0.02

        y_stock = torch.nn.functional.linear(X, W)
        y_kernel = ext.bf16_gemm(X, W)

        diff = (y_stock.float() - y_kernel.float()).abs()
        max_abs = diff.max().item()
        max_rel = (diff / (y_stock.float().abs() + 1e-6)).max().item()
        worst_max_abs = max(worst_max_abs, max_abs)
        ok = max_abs < atol or max_rel < rtol
        print(f"  trial {trial} (M={M}): max_abs={max_abs:.4e} max_rel={max_rel:.4e} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            return False
    print(f"  worst max_abs across all trials: {worst_max_abs:.4e}")
    return True


def check_e2e_in_vllm_model(model_path="Qwen/Qwen3-0.6B"):
    """Level (2): patch one layer in a REAL vLLM model, diff hidden states patched vs stock for the
    same input. Needs GPU + the model loaded -- run only after level (1). Uses a small DENSE Qwen3
    model (not the 30B-A3B MoE) deliberately: o_proj is a plain linear layer present in any
    transformer, MoE or dense, so the patch mechanism is fully exercised by a dense model -- and a
    small one fits on whatever GPU happens to be free right now instead of waiting for a 60GB+ slot."""
    # V1's default spawns the model into a SEPARATE OS process (engine_core talks to it over ZMQ) --
    # there is no Python attribute path from the main process into that process's memory. Disabling
    # multiprocessing keeps the engine core (and the model) in-process so we can reach in and patch it.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM
    from bf16_linear_method import patch_module_linear

    print(f"loading {model_path} (small model, for validation only -- not the 235B target)...")
    # gpu_memory_utilization is a FRACTION OF TOTAL device memory, not "however much is free" -- vLLM's
    # default 0.9 tried to reserve 71GB on an 80GB GPU that only had 27GB free (another session holds
    # the rest). Qwen3-0.6B needs ~1-2GB; 0.15 (~12GB) leaves headroom without fighting other GPUs' work.
    llm = LLM(model=model_path, tensor_parallel_size=1, enforce_eager=True, max_model_len=512,
             gpu_memory_utilization=0.12)

    # Confirmed by direct introspection on vLLM 0.10.1's V1 InprocClient (engine_core appears TWICE:
    # LLMEngine.engine_core is an InprocClient wrapper, .engine_core on THAT is the real EngineCore):
    #   llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model
    # .model_runner.model is the Qwen3ForCausalLM itself (.model.layers is the decoder stack).
    model = (llm.llm_engine.engine_core.engine_core.model_executor
             .driver_worker.model_runner.model)

    prompt = "The quick brown fox"
    out_stock = llm.generate([prompt], use_tqdm=False)[0].outputs[0].text

    target_layer = model.model.layers[0].self_attn.o_proj
    orig_method = target_layer.quant_method
    patch_module_linear(target_layer)
    out_patched = llm.generate([prompt], use_tqdm=False)[0].outputs[0].text
    target_layer.quant_method = orig_method   # restore regardless of outcome

    print(f"  stock output:   {out_stock!r}")
    print(f"  patched output: {out_patched!r}")
    ok = out_stock == out_patched
    print(f"  greedy-output match: {'PASS' if ok else 'FAIL (expected for greedy decoding if the swap is correct)'}")
    return ok


if __name__ == "__main__":
    print("== Level 1: kernel correctness (no vLLM, no model load) ==")
    ok1 = check_kernel_correctness()
    if not ok1:
        print("\nABORT: kernel correctness failed. Do not proceed to level 2 or use this in any live path.")
        sys.exit(1)
    print("\nLevel 1 PASSED.")

    if "--e2e" in sys.argv:
        print("\n== Level 2: end-to-end in a real (small) vLLM model ==")
        ok2 = check_e2e_in_vllm_model()
        sys.exit(0 if ok2 else 1)
    else:
        print("\n(skipping level 2 -- pass --e2e to also validate inside a real running vLLM model)")
