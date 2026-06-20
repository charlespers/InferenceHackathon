#!/usr/bin/env python3
"""Predicted B=1 decode tok/s across layout × precision × context, using the team's latency model
(src/inferutil/latency.py) calibrated by the MEASURED kernel efficiency e=0.46 (K5 on H100).

Pure-stdlib, no GPU. Produces the single reference table the experiment plan keys off.
  PYTHONPATH=src python3 tools/predict_matrix.py [efficiency=0.46] [gpu=H100-SXM-80GB]
"""
import sys
from inferutil.model import QWEN3_235B
from inferutil.hardware import GPUS, Cluster
from inferutil.latency import decode_latency

E = float(sys.argv[1]) if len(sys.argv) > 1 else 0.46
GPU = sys.argv[2] if len(sys.argv) > 2 else "H100-SXM-80GB"
cl = Cluster(gpu=GPUS[GPU], n_gpus=8)
LAYOUTS = [("tp", 8, 1), ("hybrid", 4, 2), ("hybrid", 2, 4), ("ep", 1, 8)]

print(f"# Predicted B=1 decode tok/s — {GPU}, measured efficiency e={E}\n")
print("floor = analytical roofline (e=1.0); real = floor × e (model applies e as a global TPOT factor).\n")
for dtype, dname in ((1, "fp8"), (2, "bf16")):
    for seq in (2048, 8192, 32768):
        print(f"## weights={dname}  ctx={seq}")
        print(f"| layout (tp×ep) | floor tok/s | real @e={E} | weight ms | kv ms | comms ms | E[busiest] |")
        print("|---|---|---|---|---|---|---|")
        for plan, tp, ep in LAYOUTS:
            b = decode_latency(QWEN3_235B, cl, plan=plan, dtype_bytes=dtype,
                               kv_dtype_bytes=dtype, seq_len=seq, tp=tp, ep=ep)
            r = b.as_row()
            print(f"| {plan} ({tp}×{ep}) | {r['tok_per_s']:.0f} | {b.tokens_per_s*E:.0f} "
                  f"| {r['weight_ms']:.2f} | {r['kv_ms']:.2f} | {r['comms_ms']:.2f} | {r['expert_imbalance']:.2f} |")
        print()
