"""bf16_linear_method.py — the vLLM-side half of the midpoint compromise: a custom LinearMethodBase
that vLLM's RowParallelLinear/QKVParallelLinear will dispatch to via `self.quant_method.apply(...)`
(confirmed directly from vllm/model_executor/layers/linear.py's forward()), instead of vLLM's own
UnquantizedLinearMethod. Keeps everything else -- tokenizer, scheduler, PagedAttention, sampling,
weight loading -- exactly as vLLM already does it. Only the GEMM itself is swapped.

This does NOT touch weight loading: vLLM loads `layer.weight` as it always does (bf16, TP-sharded by
vLLM's own mechanism); this method just computes the matmul differently when called.
"""
import os

import torch
from torch.utils.cpp_extension import load
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        src = os.path.join(os.path.dirname(__file__), "bf16_gemm_ext.cu")
        _ext = load(name="bf16_gemm_ext", sources=[src], extra_cuda_cflags=["-O3"], verbose=True)
    return _ext


class NativeBF16LinearMethod(LinearMethodBase):
    """Drop-in replacement for vLLM's UnquantizedLinearMethod, routing the matmul through the team's
    cuBLASLt bf16 GEMM (tools/vllm_kernels/bf16_gemm_ext.cu) instead of vLLM's own torch.nn.functional
    .linear call. create_weights is intentionally IDENTICAL to vLLM's default -- this only changes
    HOW the matmul runs, not how/what gets loaded."""

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list, input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        weight = torch.nn.Parameter(
            torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype),
            requires_grad=False)
        from vllm.model_executor.utils import set_weight_attrs
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
        ext = _get_ext()
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        y = ext.bf16_gemm(x2d, layer.weight)
        if bias is not None:
            y = y + bias
        return y.reshape(*orig_shape[:-1], -1)


def patch_module_linear(linear_module: LinearBase):
    """Swap ONE already-constructed linear layer's quant_method in place. Reuses the EXISTING weight
    tensor vLLM already loaded -- no reload, no re-shard, just a different apply() going forward."""
    linear_module.quant_method = NativeBF16LinearMethod()


def patch_qwen3_o_proj(model: torch.nn.Module, layer_indices=None):
    """Patch o_proj on the given decoder layers (default: all). The lowest-risk first target per the
    scoping discussion: O-proj is a single stateless linear, no KV-cache/paging/routing entanglement
    (unlike attention or MoE), called once per layer -> real, recurring, easy-to-isolate speed impact."""
    layers = model.model.layers
    indices = layer_indices if layer_indices is not None else range(len(layers))
    patched = []
    for i in indices:
        layer = layers[i]
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            patch_module_linear(layer.self_attn.o_proj)
            patched.append(i)
    print(f"patched o_proj on {len(patched)} layers: {patched[:5]}{'...' if len(patched) > 5 else ''}")
    return patched
