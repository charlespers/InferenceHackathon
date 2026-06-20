# server/ — mock inference backend + adapter stub

Implements the API contract the UI consumes. In mock mode it synthesizes streamed
tokens with fabricated 8×H100 expert routing (Qwen3-235B-A22B-shaped: 128 experts,
top-8 per token) and speculative-decode stats, so the UI and visualization run with
no real engine.

## Run

```bash
cd /Users/charles/Desktop/InferenceHackathon
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
# demo pacing: STREAM_DELAY=0.03 uvicorn server.main:app --port 8000
```

## Endpoints
- `GET /health`
- `GET /v1/models`
- `GET /v1/topology`
- `POST /v1/chat/completions` (OpenAI-compatible; SSE when `stream:true`)

## On the H100 box
Run there, then from your laptop: `ssh -L 8000:localhost:8000 <box>` and point the
UI at `http://localhost:8000`. Swap mock for the real engine via the adapter
(`server/backend.py`, `BACKEND=real`).
