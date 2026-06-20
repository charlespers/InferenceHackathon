# Multi-GPU Inference Console

Latency-oriented **B=1** MoE inference on **8×H100**, targeting
**Qwen3-235B-A22B** (235B total / 22B active, 128 experts/layer, top-8 routing).
Streams chat over an **OpenAI-compatible API** and visualizes per-token **expert→GPU
routing** and **latency** (TTFT, tok/s, inter-token, speculative-decode acceptance).

No proprietary engine code included. The UI talks to any OpenAI-compatible backend
over SSE (non-standard fields are `x_`-namespaced and optional). A Python mock ships
so the whole thing runs and demos without a real engine.

## Architecture

```
ui/           Vite + React + TS + Tailwind SPA       (the deliverable)
server/       FastAPI mock backend + adapter stub     (contract authority + demo data)
src/inferutil/ Pure-stdlib roofline / latency model   (runs on any machine, no GPU)
docs/         design spec, plan, DESIGN.md, H100 tuning playbook
```

The server seam is three endpoints: `GET /v1/models`, `GET /v1/topology`, and
`POST /v1/chat/completions` (SSE). Per-token telemetry rides the stream as optional
`x_telemetry`; a final `x_summary` carries the turn's latency totals.

## Run (two processes)

```bash
# 1. backend (mock)
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
STREAM_DELAY=0.03 uvicorn server.main:app --host 0.0.0.0 --port 8000

# 2. UI
cd ui && npm install && npm run dev   # http://localhost:5173
```

Point the backend-url field (top-right of the UI) at your server. For the H100 box:
`ssh -L 8000:localhost:8000 <box>`, run the server there, keep the UI local at `http://localhost:8000`.

## Swap in the real engine

Run an OpenAI-compatible server (SGLang / vLLM / conifer) on the 8×H100s and implement
`server/backend.py:RealEngineBackend` to forward requests and map routing hooks into
`x_telemetry`. Set `BACKEND=real`. See `docs/h100-tuning-playbook.md` for deployment + tuning.

## Latency model

A pure-stdlib roofline analyzer is in `src/inferutil/` — no GPU or torch required:

```bash
PYTHONPATH=src python3 -m inferutil       # B=1 decode latency breakdown on 8×H100
PYTHONPATH=src python3 -m inferutil --gpu H200-SXM-141GB
```

See `docs/DESIGN.md` for findings: roofline floor ~1.85 ms/token, FP8 is highest-ROI,
and naive expert-parallelism is slower than tensor-parallel at B=1 (routing imbalance).

## Tests

```bash
python -m pytest server/        # backend: contract, topology, mock engine, adapter
python tests/test_model.py      # latency model sanity checks
cd ui && npm run test           # frontend: SSE parse, chat hook, components, stats
cd ui && npm run e2e            # end-to-end smoke (needs the mock running on :8000)
```

## Docs
- `docs/DESIGN.md` — latency model findings + optimization priorities
- `docs/h100-tuning-playbook.md` — 8×H100 latency tuning playbook
- `docs/superpowers/specs/2026-06-19-multigpu-inference-ui-design.md` — UI design spec
- `docs/superpowers/plans/2026-06-19-multigpu-inference-ui.md` — implementation plan

## Benchmark harness (`inferutil.bench`)

Offline B=1 decode benchmarks measured against the analytical roofline. Runs
today on a `MockEngine` (no GPU); a `ConiferEngine` slots in behind the same
`Engine` seam when the engine lands.

```bash
PYTHONPATH=src python -m inferutil.bench run --name fp8-hybrid --dtype 1 --plan hybrid --tp 2 --ep 8
PYTHONPATH=src python -m inferutil.bench report --name fp8-hybrid          # latest run
PYTHONPATH=src python -m inferutil.bench compare <runidA> <runidB> --name fp8-hybrid
```

Each run captures TTFT, decode/prefill tok/s, TPOT p50/p95, derived achieved
bandwidth (% of peak and % of analytical floor), and NVML device telemetry
(temps, util, power, energy/token, per-GPU imbalance). Results are JSON under
`results/<name>/` and diffable run-to-run.

### Agent-loop notes & limitations

- All numbers today run on `MockEngine`; timing-derived metrics are analytical
  until `ConiferEngine` lands on real hardware.
- Use `repeats>=5` for the `compare` significance verdict to be meaningful;
  with `repeats=1` the output reports `n/a (need repeats>=2)`.
- The quality parity gate (`--min-quality`) is inert on the mock — token ids
  are synthetic and independent of timing knobs (see `quality.py` docstring).
