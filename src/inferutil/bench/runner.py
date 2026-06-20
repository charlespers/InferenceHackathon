from __future__ import annotations

from ..model import MoEConfig
from ..hardware import Cluster
from .config import BenchConfig
from .metrics import build_result, summarize_telemetry, BenchResult
from .telemetry import NullTelemetry


def run_benchmark(engine, config: BenchConfig, cfg: MoEConfig, cluster: Cluster,
                  telemetry=None) -> BenchResult:
    """Drive one fixed-window B=1 benchmark and return a BenchResult.

    Telemetry brackets ONLY the timed decode window (warmup discarded first) so
    sampling never perturbs the latency path.
    """
    if config.decode_tokens < 2:
        raise ValueError("decode_tokens must be >= 2 (need inter-token samples)")
    telemetry = telemetry or NullTelemetry()

    engine.reset(plan=config.plan, dtype_bytes=config.dtype_bytes,
                 kv_dtype_bytes=config.kv_dtype_bytes, tp=config.tp, ep=config.ep,
                 seq_len=config.seq_len)

    pre = engine.prefill(list(range(config.prompt_tokens)))
    ttft_s = pre.seconds + pre.first_token_seconds
    prefill_tok_per_s = (pre.n_prompt_tokens / pre.seconds) if pre.seconds else float("inf")

    for _ in range(config.warmup_steps):
        engine.decode_step()

    telemetry.start()
    step_seconds = [engine.decode_step().seconds for _ in range(config.decode_tokens - 1)]
    gpu_samples = telemetry.stop()

    summary = summarize_telemetry(gpu_samples, config.decode_tokens, sum(step_seconds))
    return build_result(cfg=cfg, cluster=cluster, config=config, ttft_s=ttft_s,
                        prefill_tok_per_s=prefill_tok_per_s,
                        decode_step_seconds=step_seconds, telemetry_summary=summary)
