# Inference Server

Two server options: Python (legacy, mock-capable) and Rust (production, real predictor).

---

## Rust Server (preferred)

Replaces the Python server. Proxies vLLM, runs the Markov predictor per token, and exposes `/api/tasks`.

### Run

```bash
# On the H100 box:
source $HOME/.cargo/env
cd /alloc/data/InferenceHackathon

# Port 9000 is dedicated to this server — don't use 8000 (shared / Python)
PORT=9000 cargo run --release --bin server -p engine
```

Runs in tmux so it survives disconnects:
```bash
tmux new-session -d -s rustserver 'source $HOME/.cargo/env && cd /alloc/data/InferenceHackathon && PORT=9000 cargo run --release --bin server -p engine 2>&1 | tee /alloc/data/server.log'
```

### Tunnel + UI

```bash
# Laptop:
ssh -L 9000:localhost:9000 root@147.185.41.162 -p 31025 -N
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"status":"ok"}` |
| GET | `/v1/models` | Model list |
| GET | `/v1/topology` | GPU stats + expert placement |
| POST | `/v1/chat/completions` | OpenAI-compatible SSE proxy to vLLM |
| GET | `/api/tasks` | **Live request monitor** (see below) |

### /api/tasks — Live Request Monitor

Shows who is running inference right now. Auto-refreshes every 2 seconds.

**Browser** (returns dark-themed HTML table):
```
http://localhost:9000/api/tasks
```

**JSON** (curl or scripts):
```bash
curl localhost:9000/api/tasks
```

```json
{
  "active": [
    {
      "user": "jaymin",
      "prompt": "Explain the transformer attention mechanism…",
      "elapsed_s": 12,
      "tokens": 45,
      "tok_per_s": 3.7
    }
  ],
  "total_served": 142,
  "uptime_s": 3600
}
```

**Tag your requests** so you appear by name:

```bash
# X-User header:
curl -H "X-User: jaymin" -X POST localhost:9000/v1/chat/completions ...

# Or "user" field in the body:
{"user": "jaymin", "messages": [...], "stream": true}

# benchmark.py --user flag does this automatically:
python3 tools/benchmark.py --user jaymin
```

Untagged requests show as `unknown`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Listening port (use `9000` on the box) |
| `VLLM_URL` | `http://localhost:8001` | vLLM backend |

---

## Python Server (legacy)

Still usable as a fallback. Supports mock mode (no vLLM needed). Does **not** have `/api/tasks` or the Rust predictor.

```bash
cd /alloc/data/InferenceHackathon
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000

# Mock mode (no vLLM):
BACKEND=mock uvicorn server.main:app --port 8000
```

---

## vLLM

Both servers proxy to vLLM at `localhost:8001`. Start it with:

```bash
tmux new-session -d -s vllm 'vllm serve /alloc/data/Qwen3-235B-A22B --tensor-parallel-size 8 --port 8001 --disable-log-requests 2>&1 | tee /alloc/data/vllm.log'

# Watch startup (~5 min):
tail -f /alloc/data/vllm.log
# Ready when you see: "Application startup complete"
```
