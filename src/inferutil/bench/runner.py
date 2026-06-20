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
    samples_tok_s = []
    step_seconds = []
    gpu_samples = []
    last_breakdowns = []
    ttft_s = 0.0
    prefill_tok_per_s = 0.0
    for rep in range(config.repeats):
        pre = engine.prefill(list(range(config.prompt_tokens)))
        ttft_s = pre.seconds + pre.first_token_seconds
        prefill_tok_per_s = (pre.n_prompt_tokens / pre.seconds) if pre.seconds else float("inf")
        for _ in range(config.warmup_steps):
            engine.decode_step()
        last_rep = rep == config.repeats - 1
        if last_rep:
            telemetry.start()
        if last_rep:
            steps = [engine.decode_step() for _ in range(config.decode_tokens - 1)]
            step_seconds = [s.seconds for s in steps]
            last_breakdowns = [s.breakdown for s in steps]
        else:
            step_seconds = [engine.decode_step().seconds for _ in range(config.decode_tokens - 1)]
        if last_rep:
            gpu_samples = telemetry.stop()
        total = sum(step_seconds)
        samples_tok_s.append((len(step_seconds) / total) if total else float("inf"))

    summary = summarize_telemetry(gpu_samples, config.decode_tokens, sum(step_seconds))
    return build_result(cfg=cfg, cluster=cluster, config=config, ttft_s=ttft_s,
                        prefill_tok_per_s=prefill_tok_per_s,
                        decode_step_seconds=step_seconds, telemetry_summary=summary,
                        decode_tok_per_s_samples=samples_tok_s,
                        step_breakdowns=last_breakdowns)
