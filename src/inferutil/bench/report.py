# src/inferutil/bench/report.py
from __future__ import annotations

from .store import RunRecord
from .attribution import diagnose


def _pct(x) -> str:
    return f"{x*100:.1f}%" if x is not None else "—"


def is_significant(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """True when |mean_b - mean_a| exceeds combined run-to-run noise (~95%)."""
    noise = 2.0 * ((std_a ** 2 + std_b ** 2) ** 0.5)
    return abs(mean_b - mean_a) > noise


def format_result(record: RunRecord) -> str:
    r = record.result
    c = record.config
    ms = 1e3
    t = r.telemetry
    lines = [
        f"== BENCH: {c.name}  [{record.runid}]  plan={c.plan} "
        f"dtype={c.dtype_bytes*8}b tp={c.tp} ep={c.ep} "
        f"ctx={c.seq_len} ({c.prompt_tokens}+{c.decode_tokens}) ==",
        f"  env          : {record.env.get('gpu','?')} x{record.env.get('n_gpus','?')}",
        "  -- latency --",
        f"  TTFT         : {r.ttft_s*ms:8.2f} ms",
        f"  prefill      : {r.prefill_tok_per_s:8.1f} tok/s",
        f"  decode       : {r.decode_tok_per_s:8.1f} tok/s"
        + (f" ±{r.decode_tok_per_s_std:.1f} (n={r.n_repeats})" if r.n_repeats > 1 else "")
        + f"   (TPOT p50 {r.tpot_p50_s*1e3:.2f} / p95 {r.tpot_p95_s*1e3:.2f} ms)",
        f"  E2E (total)  : {r.total_s*ms:8.2f} ms",
        "  -- bandwidth / roofline --",
        f"  bytes/token  : {r.bytes_per_token/1e9:8.2f} GB",
        f"  achieved BW  : {r.achieved_hbm_bw/1e12:8.2f} TB/s "
        f"({r.pct_of_peak_bw*100:.1f}% of peak)",
        f"  vs floor     : {r.decode_tok_per_s:8.1f} / "
        f"{r.analytical_floor_tok_per_s:.1f} tok/s = "
        f"{r.pct_of_floor*100:.1f}% of floor",
    ]
    eff = r.efficiency
    if eff is not None:
        ai = f"{eff.ai_decode:.2f}" if eff.ai_decode is not None else "—"
        ridge = f"{eff.roofline_ridge:.0f}" if eff.roofline_ridge is not None else "—"
        lines += [
            "  -- efficiency (roofline) --",
            f"  MFU          : prefill {_pct(eff.mfu_prefill)} / decode {_pct(eff.mfu_decode)}",
            f"  MBU (decode) : {_pct(eff.mbu_decode)}   (KV byte share {_pct(eff.kv_byte_share)})",
            f"  AI decode    : {ai} FLOP/B   ridge {ridge}  -> {eff.regime_decode}",
        ]
    b = diagnose(r)
    lines += [
        "  -- bottleneck --",
        f"  dominant     : {b.dominant_term} ({b.share*100:.0f}% of time, "
        f"conf {b.confidence*100:.0f}%)",
        f"  -> {b.note}",
    ]
    mb = r.measured_breakdown
    if mb is not None:
        ms = 1e3
        lines += [
            "  -- measured decode breakdown (mean ms/token) --",
            f"  weight {mb.weight_s*ms:.3f}  kv {mb.kv_s*ms:.3f}  "
            f"comms {mb.comms_s*ms:.3f}  compute {mb.compute_s*ms:.4f}",
        ]
    if t.available:
        lines += [
            "  -- device telemetry --",
            f"  temp (max)   : {t.temp_c_max:8.1f} C",
            f"  SM util mean : {t.sm_util_pct_mean:8.1f} %   "
            f"mem util {t.mem_util_pct_mean:.1f} %",
            f"  power (total): {t.power_w_mean:8.0f} W   "
            f"energy/token {t.energy_j_per_token:.2f} J",
            f"  GPU imbalance: {t.util_imbalance:8.2f}x  "
            f"(busiest vs mean util across {t.n_gpus} GPUs)",
        ]
    else:
        lines.append("  -- device telemetry unavailable --")
    return "\n".join(lines)


def format_diagnosis(record: RunRecord, levers=None) -> str:
    """Bottleneck diagnosis + ranked next-lever recommendations for one run."""
    b = diagnose(record.result)
    ai = f"{b.ai_decode:.2f}" if b.ai_decode is not None else "—"
    ridge = f"{b.ridge:.0f}" if b.ridge is not None else "—"
    lines = [
        f"== DIAGNOSIS: {record.config.name}  [{record.runid}] ==",
        f"  regime       : {b.regime}  (AI {ai} vs ridge {ridge} FLOP/B)",
        f"  dominant     : {b.dominant_term}  ({b.share*100:.0f}% of per-token time)",
        f"  runner-up    : {b.second_term}  ({b.second_share*100:.0f}%)  "
        f"confidence {b.confidence*100:.0f}%",
        f"  headroom     : {b.headroom_to_floor*100:.0f}% below analytical floor",
        f"  -> {b.note}",
    ]
    if levers:
        lines.append("  -- ranked next levers (predicted) --")
        lines.append(f"  {'lever':<22}{'tok/s':>10}{'speedup':>9}{'effort':>8}")
        for lv in levers:
            lines.append(f"  {lv.name:<22}{lv.predicted_tok_s:>10.1f}"
                         f"{lv.speedup:>8.2f}x{lv.effort:>8}")
            lines.append(f"      {lv.rationale}")
    return "\n".join(lines)


def _delta(a: float, b: float) -> str:
    d = b - a
    return f"{d:+.1f}"


def format_compare(a: RunRecord, b: RunRecord) -> str:
    ra, rb = a.result, b.result
    ms = 1e3
    if ra.n_repeats < 2 or rb.n_repeats < 2:
        sig = "n/a (need repeats>=2)"
    elif is_significant(ra.decode_tok_per_s, ra.decode_tok_per_s_std,
                        rb.decode_tok_per_s, rb.decode_tok_per_s_std):
        sig = "SIGNIFICANT"
    else:
        sig = "within-noise"
    return "\n".join([
        f"== COMPARE: {a.runid} (A) vs {b.runid} (B) ==",
        f"  {'metric':<16}{'A':>12}{'B':>12}{'delta':>12}",
        f"  {'decode tok/s':<16}{ra.decode_tok_per_s:>12.1f}"
        f"{rb.decode_tok_per_s:>12.1f}{_delta(ra.decode_tok_per_s, rb.decode_tok_per_s):>12}",
        f"  {'TTFT ms':<16}{ra.ttft_s*ms:>12.2f}{rb.ttft_s*ms:>12.2f}"
        f"{_delta(ra.ttft_s*ms, rb.ttft_s*ms):>12}",
        f"  {'% of floor':<16}{ra.pct_of_floor*100:>12.1f}{rb.pct_of_floor*100:>12.1f}"
        f"{_delta(ra.pct_of_floor*100, rb.pct_of_floor*100):>12}",
        f"  {'% of peak BW':<16}{ra.pct_of_peak_bw*100:>12.1f}"
        f"{rb.pct_of_peak_bw*100:>12.1f}"
        f"{_delta(ra.pct_of_peak_bw*100, rb.pct_of_peak_bw*100):>12}",
        f"  {'temp max C':<16}{ra.telemetry.temp_c_max:>12.1f}"
        f"{rb.telemetry.temp_c_max:>12.1f}"
        f"{_delta(ra.telemetry.temp_c_max, rb.telemetry.temp_c_max):>12}",
        f"  {'decode sig?':<16}{sig:>36}",
    ])
