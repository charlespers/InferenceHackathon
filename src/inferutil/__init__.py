"""inferutil — latency-oriented multi-GPU MoE inference utilities.

Phase 0 (pre-hackathon, no GPU / no conifer yet): a pure-stdlib analytical
model that answers "where does B=1 decode latency go for Qwen3-235B-A22B on
8xH100, and what is worth optimizing?". When conifer + hardware land, the
measured numbers slot in beside these predictions.
"""

from .hardware import GPUS, H100_SXM, H200_SXM, Cluster, GPU
from .model import MoEConfig, QWEN3_235B
from .latency import decode_latency, DecodeBreakdown, expected_max_experts_per_gpu

__all__ = [
    "GPUS", "H100_SXM", "H200_SXM", "Cluster", "GPU",
    "MoEConfig", "QWEN3_235B",
    "decode_latency", "DecodeBreakdown", "expected_max_experts_per_gpu",
]
