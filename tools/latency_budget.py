#!/usr/bin/env python3
"""Single-user latency budget calculator — projects TTFT + N·TPOT (and multi-turn chat) under a chosen set
of levers, using the team's latency model (src/inferutil/latency.py) + the measured constants and the
floor-amortization / prefix-cache findings (docs/single-user-latency-budget.md, ttft-analysis.md,
spec-decode-floor-bound.md). Pure-stdlib, no GPU.

  PYTHONPATH=src python3 tools/latency_budget.py                      # current measured baseline
  PYTHONPATH=src python3 tools/latency_budget.py --proven             # cheap proven levers stacked
  PYTHONPATH=src python3 tools/latency_budget.py --collective-us 4 --spec-tau 2 --prefix-cache --eff 0.30
"""
import argparse, dataclasses
from inferutil.model import QWEN3_235B
from inferutil.hardware import GPUS, Cluster
from inferutil.latency import decode_latency


def tpot_ms(plan, tp, ep, dtype, collective_us, eff, host_ms):
    gpu = dataclasses.replace(GPUS["H100-SXM-80GB"], collective_latency_s=collective_us * 1e-6)
    b = decode_latency(QWEN3_235B, Cluster(gpu=gpu, n_gpus=8), plan=plan, dtype_bytes=dtype,
                       kv_dtype_bytes=2, seq_len=512, tp=tp, ep=ep)
    # model's bandwidth terms scaled by realized efficiency (global factor, per the bench convention),
    # comms is latency-bound (not eff-scaled), plus an explicit host/launch/sampling residual.
    bw_ms = (b.weight_read_s + b.kv_read_s) * 1e3 / eff
    return bw_ms + b.comms_s * 1e3 + host_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="tp"); ap.add_argument("--tp", type=int, default=8); ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--dtype", type=int, default=2, help="2=bf16,1=fp8")
    ap.add_argument("--collective-us", type=float, default=16.0, help="measured 16µs default-NCCL; ~4 tuned")
    ap.add_argument("--eff", type=float, default=0.16, help="whole-model realized efficiency (measured ~0.16; K5 kernel 0.46)")
    ap.add_argument("--host-ms", type=float, default=0.0, help="extra host/launch/sampling residual per step")
    ap.add_argument("--spec-tau", type=float, default=1.0, help="accepted tokens/round (floor-amortization); 1=off")
    ap.add_argument("--ttft-ms", type=float, default=777.0, help="measured cold TTFT; overridden by --prefix-cache")
    ap.add_argument("--prefix-cache", action="store_true", help="cache hit -> TTFT ≈ first decode step")
    ap.add_argument("--decode", type=int, default=128); ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--proven", action="store_true", help="preset: prefix-cache + n-gram τ=2 + comms 8µs + eff 0.30")
    a = ap.parse_args()
    if a.proven:
        a.prefix_cache = True; a.spec_tau = 2.0; a.collective_us = 8.0; a.eff = 0.30

    base_tpot = tpot_ms(a.plan, a.tp, a.ep, a.dtype, a.collective_us, a.eff, a.host_ms)
    # spec amortizes the per-step floor over τ accepted tokens (valid while floor-dominated):
    eff_tpot = base_tpot / max(a.spec_tau, 1.0)
    ttft = (base_tpot if a.prefix_cache else a.ttft_ms)   # cache hit ≈ one decode step
    perceived = ttft + a.decode * eff_tpot
    tok_s = 1000.0 / eff_tpot

    print(f"== single-user budget ==  plan={a.plan} tp{a.tp}×ep{a.ep} {'bf16' if a.dtype==2 else 'fp8'}"
          f"  comms={a.collective_us}µs  eff={a.eff}  spec_tau={a.spec_tau}  prefix_cache={a.prefix_cache}")
    print(f"  TPOT (per token)     : {eff_tpot:6.2f} ms   ({tok_s:6.1f} tok/s)")
    print(f"  TTFT                 : {ttft:6.1f} ms   ({'cache-hit≈1 step' if a.prefix_cache else 'cold/measured'})")
    print(f"  perceived ({a.decode} tok)   : {perceived:7.1f} ms")
    if a.turns > 1:
        # multi-turn chat: every turn after #1 is a prefix-cache hit on the history
        total = a.ttft_ms + a.decode * eff_tpot                          # turn 1 (cold prefill)
        total += (a.turns - 1) * (base_tpot + a.decode * eff_tpot)       # turns 2..T (cached history)
        print(f"  {a.turns}-turn chat (turn1 cold, rest cached): {total/1000:5.2f} s total, "
              f"{total/a.turns:6.0f} ms/turn avg")
    print("  (baseline for reference: TPOT 11.67ms / 85.7 tok/s, TTFT 777ms, perceived 2271ms @128tok)")


if __name__ == "__main__":
    main()
