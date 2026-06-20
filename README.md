# InferenceHackathon

Latency-oriented multi-GPU inference utilities for **Qwen3-235B-A22B on 8×H100**.
Goal: minimize **per-token decode latency at batch size 1** (not throughput).

## Status

Pre-hackathon. Conifer (the foundational engine) and the GPU node aren't here
yet, so phase 0 is a **pure-stdlib analytical latency model** — no torch, no GPU
required — that answers *where does B=1 decode latency go, and what's worth
optimizing?* When conifer + hardware arrive, measured numbers slot in beside it.

## Quick start

```bash
python -m pytest tests/                 # or: python tests/test_model.py
PYTHONPATH=src python3 -m inferutil      # the roofline report
PYTHONPATH=src python3 -m inferutil --gpu H200-SXM-141GB
```

## What's here

```
src/inferutil/
  hardware.py   # H100/H200 specs (HBM BW, NVLink, compute)
  model.py      # Qwen3-235B-A22B arch + param/memory accounting (validated)
  latency.py    # B=1 decode latency model: weight/KV/comms/compute breakdown
  cli.py        # the report
docs/DESIGN.md  # findings + optimization priorities + workstream split
tests/          # reproduces the 235B/22B headline numbers
```

## Headline findings

See `docs/DESIGN.md`. In short:

- B=1 decode is **memory-bandwidth bound** — compute is ~0.1% of the budget.
- Roofline floor ≈ **1.85 ms/token (~540 tok/s)** on 8×H100 bf16.
- **FP8 weights** are the highest-ROI lever (~halves the dominant term).
- Naive expert-parallelism is *slower* than tensor-parallel at B=1 (routing
  imbalance) — which is exactly what **expert prediction/prefetch** can fix.
