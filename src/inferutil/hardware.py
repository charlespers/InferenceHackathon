"""GPU and interconnect specs for the latency model.

All numbers are vendor dense (non-sparsity) figures. Bandwidths in bytes/sec,
compute in FLOP/sec, memory in bytes. Keep everything in SI so the latency
model never has to guess units.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1_000_000_000  # we use decimal GB/TB to match vendor spec sheets
TB = 1_000_000_000_000
TFLOP = 1_000_000_000_000


@dataclass(frozen=True)
class GPU:
    name: str
    hbm_bytes: int          # capacity per GPU
    hbm_bw: float           # bytes/sec, HBM read+write bandwidth
    bf16_flops: float       # dense bf16 tensor-core FLOP/sec
    fp8_flops: float        # dense fp8 tensor-core FLOP/sec
    nvlink_bw: float        # bytes/sec, *unidirectional* per-GPU NVLink BW
    # Rough fixed cost to launch + sync one collective on NVSwitch fabric.
    # At B=1 the payloads are tiny, so collective time is latency- not
    # bandwidth-bound and this dominates. Empirical ballpark; override per run.
    collective_latency_s: float = 5e-6


# H100 SXM5 80GB (HBM3). NVLink4 = 900 GB/s bidirectional => 450 GB/s each way.
H100_SXM = GPU(
    name="H100-SXM-80GB",
    hbm_bytes=80 * GB,
    hbm_bw=3.35 * TB,
    bf16_flops=989.4 * TFLOP,
    fp8_flops=1978.9 * TFLOP,
    nvlink_bw=450 * GB,
)

# H200 SXM 141GB (HBM3e). Same compute & NVLink as H100, much more BW+capacity.
H200_SXM = GPU(
    name="H200-SXM-141GB",
    hbm_bytes=141 * GB,
    hbm_bw=4.8 * TB,
    bf16_flops=989.4 * TFLOP,
    fp8_flops=1978.9 * TFLOP,
    nvlink_bw=450 * GB,
)


@dataclass(frozen=True)
class Cluster:
    gpu: GPU
    n_gpus: int = 8

    @property
    def total_hbm(self) -> int:
        return self.gpu.hbm_bytes * self.n_gpus

    @property
    def aggregate_hbm_bw(self) -> float:
        return self.gpu.hbm_bw * self.n_gpus


GPUS = {g.name: g for g in (H100_SXM, H200_SXM)}
