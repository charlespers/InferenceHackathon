# Conifer · Inference, measured

A latency-oriented inference console for **B=1** MoE serving on **8×H100**, targeting
**Qwen3-235B-A22B** (235B total / 22B active, 128 experts/layer, top-8 routing). Two views:

- **Race** — fires one prompt at two engines (**Conifer** vs a **vLLM** baseline) at the
  same instant and times them on the wall clock. The speedup is *measured*, not asserted.
  A test-time-compute panel then translates that speed into task quality: in vLLM's
  time-to-answer, Conifer fits *k* reasoning passes → self-consistency lifts accuracy.
- **Console** — single-stream chat with live **latency** (TTFT, tok/s, inter-token,
  speculative-decode acceptance) and per-token **expert→GPU routing** across the 8 H100s.

Built fresh for the hackathon — **no proprietary engine code is included**. The UI talks
to any OpenAI-compatible backend over standard SSE (non-standard fields are `x_`-namespaced
and optional). A Python mock ships two serving profiles so the whole thing runs and demos
with no real engine; the accuracy curve is an illustrative self-consistency model, clearly
labeled as such.

## Architecture

```
ui/       Vite + React + TS + Tailwind SPA       (the deliverable)
server/   FastAPI mock backend + adapter stub     (contract authority + demo data)
docs/     design spec, plan, H100 tuning playbook
```

The seam is three endpoints: `GET /v1/models`, `GET /v1/topology`, and
`POST /v1/chat/completions` (SSE). Per-token telemetry rides the stream as optional
`x_telemetry`; a final `x_summary` carries the turn's latency totals.

## Run (two processes)

```bash
# 1. backend (mock) — paces each token at the engine profile's real latency
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000

# 2. UI
cd ui && npm install && npm run dev   # http://localhost:5173
```

Point the backend-url field (top-right of the UI) at your server. For the H100 box:
`ssh -L 8000:localhost:8000 <box>`, run the server there, keep the UI pointed at
`http://localhost:8000`.

**Demo pacing.** The mock sleeps each token at its profile's real per-token latency so the
race speedup is measured by the UI's wall clock. `STREAM_SCALE` warps time (`1.0` = real
time; `0.5` = twice as fast for a snappier demo); `STREAM_DELAY` adds a flat per-token delay.

```bash
STREAM_SCALE=0.6 uvicorn server.main:app --host 0.0.0.0 --port 8000
```

**Race two real engines.** By default both lanes share the backend-url and differ only by an
`engine` profile (`conifer` / `vllm`). To race real OpenAI-compatible servers head-to-head,
point each lane at its own URL:

```bash
cd ui && VITE_CONIFER_BASE=http://localhost:8001 VITE_VLLM_BASE=http://localhost:8002 npm run dev
```

## Swap in the real engine

Run an OpenAI-compatible server (SGLang / vLLM / custom) on the 8×H100s and implement
`server/backend.py:RealEngineBackend` to forward requests and map the engine's routing
hooks into `x_telemetry`. Set `BACKEND=real`. Until then, chat + latency work against any
OpenAI server; the routing viz runs on the mock. See `docs/h100-tuning-playbook.md` for
deployment + latency tuning.

## Tests

```bash
python -m pytest server/        # backend: contract, topology, mock engine, adapter
cd ui && npm run test           # frontend: SSE parse, chat hook, components, stats
cd ui && npm run e2e            # end-to-end smoke (needs the mock running on :8000)
```

## Docs
- `docs/superpowers/specs/2026-06-19-multigpu-inference-ui-design.md` — design spec
- `docs/superpowers/plans/2026-06-19-multigpu-inference-ui.md` — implementation plan
- `docs/h100-tuning-playbook.md` — 8×H100 latency tuning playbook
