from __future__ import annotations

from dataclasses import dataclass

from ..model import MoEConfig
from ..hardware import Cluster
from ..latency import decode_latency
from .config import BenchConfig


@dataclass(frozen=True)
class TelemetrySummary:
    available: bool
    n_gpus: int
    temp_c_max: float
    sm_util_pct_mean: float
    mem_util_pct_mean: float
    power_w_mean: float            # mean TOTAL power across all GPUs (W)
    energy_j_per_token: float
    util_imbalance: float          # busiest-GPU mean util / mean-across-GPUs
    per_gpu_mean_util: tuple       # tuple[float, ...]


@dataclass(frozen=True)
class BenchResult:
    # latency
    ttft_s: float
    prefill_tok_per_s: float
    decode_tok_per_s: float
    tpot_p50_s: float
    tpot_p95_s: float
    total_s: float
    n_decode_tokens: int
    # bandwidth / efficiency (derived)
    bytes_per_token: int
    achieved_hbm_bw: float
    pct_of_peak_bw: float
    analytical_floor_tok_per_s: float
    pct_of_floor: float
    # device
    telemetry: TelemetrySummary


def bytes_per_token(cfg: MoEConfig, seq_len: int, dtype_bytes: int,
                    kv_dtype_bytes: int) -> int:
    """Active weight bytes + whole-KV read for this context (per latency.py terms)."""
    return cfg.active_params * dtype_bytes + seq_len * cfg.kv_bytes_per_token(kv_dtype_bytes)


def percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize_telemetry(samples, n_decode_tokens: int,
                        decode_window_s: float) -> TelemetrySummary:
    if not samples:
        return TelemetrySummary(False, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ())
    gpu_ids = sorted({s.gpu_index for s in samples})
    by_gpu = {g: [s for s in samples if s.gpu_index == g] for g in gpu_ids}
    per_gpu_mean_util = tuple(
        sum(s.sm_util_pct for s in by_gpu[g]) / len(by_gpu[g]) for g in gpu_ids)
    mean_util = sum(per_gpu_mean_util) / len(per_gpu_mean_util)
    imbalance = (max(per_gpu_mean_util) / mean_util) if mean_util else 0.0
    samples_per_gpu = len(samples) / len(gpu_ids)   # assumes equal sampling cadence
    power_total_mean = sum(s.power_w for s in samples) / samples_per_gpu
    energy = (power_total_mean * decode_window_s / n_decode_tokens) if n_decode_tokens else 0.0
    return TelemetrySummary(
        available=True, n_gpus=len(gpu_ids),
        temp_c_max=max(s.temp_c for s in samples),
        sm_util_pct_mean=sum(s.sm_util_pct for s in samples) / len(samples),
        mem_util_pct_mean=sum(s.mem_util_pct for s in samples) / len(samples),
        power_w_mean=power_total_mean, energy_j_per_token=energy,
        util_imbalance=imbalance, per_gpu_mean_util=per_gpu_mean_util)


def build_result(*, cfg: MoEConfig, cluster: Cluster, config: BenchConfig,
                 ttft_s: float, prefill_tok_per_s: float, decode_step_seconds: list,
                 telemetry_summary: TelemetrySummary) -> BenchResult:
    steps_sorted = sorted(decode_step_seconds)
    n = len(decode_step_seconds)
    total_decode = sum(decode_step_seconds)
    tpot_mean = (total_decode / n) if n else float("inf")
    decode_tok_per_s = (1.0 / tpot_mean) if tpot_mean else float("inf")
    bpt = bytes_per_token(cfg, config.seq_len, config.dtype_bytes, config.kv_dtype_bytes)
    achieved = (bpt / tpot_mean) if tpot_mean else 0.0
    floor_tok_s = decode_latency(
        cfg, cluster, plan=config.plan, dtype_bytes=config.dtype_bytes,
        kv_dtype_bytes=config.kv_dtype_bytes, seq_len=config.seq_len,
        tp=config.tp, ep=config.ep).tokens_per_s
    return BenchResult(
        ttft_s=ttft_s, prefill_tok_per_s=prefill_tok_per_s,
        decode_tok_per_s=decode_tok_per_s,
        tpot_p50_s=percentile(steps_sorted, 0.5),
        tpot_p95_s=percentile(steps_sorted, 0.95),
        total_s=ttft_s + total_decode, n_decode_tokens=config.decode_tokens,
        bytes_per_token=bpt, achieved_hbm_bw=achieved,
        pct_of_peak_bw=achieved / cluster.aggregate_hbm_bw,
        analytical_floor_tok_per_s=floor_tok_s,
        pct_of_floor=(decode_tok_per_s / floor_tok_s) if floor_tok_s else 0.0,
        telemetry=telemetry_summary)
