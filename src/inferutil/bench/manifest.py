"""Reproducibility manifest — everything needed to re-run and trust a bench.

Captures the host, the exact code revision, the model identity (a hash over its
architecture config), the hardware peaks, and the full bench config + seed. A
run is only as credible as its manifest; without it, numbers can't be compared
across machines or commits.
"""

from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
import sys
from dataclasses import astuple, asdict

from ..model import MoEConfig
from ..hardware import Cluster
from .config import BenchConfig, config_id

SCHEMA = "inferutil/bench/manifest/1"


def model_hash(cfg: MoEConfig) -> str:
    """Stable 12-hex id over the model architecture fields."""
    raw = "|".join(str(x) for x in astuple(cfg))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=2)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                             text=True, timeout=2)
        return bool(out.stdout.strip())
    except Exception:
        return False


def build_manifest(cfg: MoEConfig, cluster: Cluster, config: BenchConfig, *,
                   peak_bw_measured: float | None = None, cli: str | None = None,
                   runid: str | None = None, driver_version: str | None = None) -> dict:
    """Assemble the reproducibility manifest for one bench run."""
    gpu = cluster.gpu
    return {
        "schema": SCHEMA,
        "runid": runid,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "driver_version": driver_version or "unknown",
        },
        "code": {"git_commit": _git_commit(), "git_dirty": _git_dirty()},
        "model": {
            "name": cfg.name,
            "hash": model_hash(cfg),
            "total_params": cfg.total_params,
            "active_params": cfg.active_params,
        },
        "hardware": {
            "gpu": gpu.name,
            "n_gpus": cluster.n_gpus,
            "hbm_bw_per_gpu": gpu.hbm_bw,
            "aggregate_hbm_bw": cluster.aggregate_hbm_bw,
            "peak_bw_measured": peak_bw_measured,
            "bf16_flops": gpu.bf16_flops,
            "fp8_flops": gpu.fp8_flops,
        },
        "bench": {
            "config": asdict(config),
            "config_id": config_id(config),
            "seed": config.seed,
            "repeats": config.repeats,
            "warmup_steps": config.warmup_steps,
        },
        "cli": cli,
    }
