# Inference Server

---

## Rust Server

Proxies vLLM, runs the Markov predictor per token. Listens on two ports:

| Port | What's there |
|------|-------------|
| **8000** | Main proxy: `/health`, `/v1/models`, `/v1/topology`, `/v1/chat/completions` |
| **9000** | Monitor only: `/api/tasks` (live GPU + request dashboard) |

### Run

```bash
# On the H100 box:
source $HOME/.cargo/env && cd /alloc/data/InferenceHackathon
cargo run --release --bin server -p engine
```

In tmux:
```bash
tmux new-session -d -s rustserver 'source $HOME/.cargo/env && cd /alloc/data/InferenceHackathon && cargo run --release --bin server -p engine 2>&1 | tee /alloc/data/server.log'
```

### Tunnel

```bash
# Laptop — tunnel both ports:
ssh -L 8000:localhost:8000 -L 9000:localhost:9000 root@147.185.41.162 -p 31025 -N
```

### /api/tasks — Live GPU Monitor (port 9000)

Shows GPU utilisation, vLLM queue depth, and active requests — including requests
that bypass this server and go directly to vLLM.

```
http://localhost:9000/api/tasks        # browser (dark HTML, auto-refresh 2s)
curl localhost:9000/api/tasks          # JSON
```

**Tag your requests** so you appear by name:

```bash
curl -H "X-User: jaymin" -X POST localhost:8000/v1/chat/completions ...
# Or "user" field in the body. benchmark.py --user does this automatically.
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Main proxy port |
| `TASKS_PORT` | `9000` | /api/tasks port |
| `VLLM_URL` | `http://localhost:8001` | vLLM backend |

---

## vLLM

### Option A — with routing hook (recommended for benchmarking)

Captures real per-token expert selections and streams them to the Rust server.
The Rust server uses real routing data instead of simulation when connected.

```bash
tmux new-session -d -s vllm 'cd /alloc/data/InferenceHackathon && python3 tools/start_vllm.py 2>&1 | tee /alloc/data/vllm.log'

# Watch startup (~5 min):
tail -f /alloc/data/vllm.log
# Ready when you see: "Application startup complete"
```

### Option B — plain vLLM

```bash
tmux new-session -d -s vllm 'vllm serve /alloc/data/Qwen3-235B-A22B --tensor-parallel-size 8 --port 8001 --disable-log-requests 2>&1 | tee /alloc/data/vllm.log'
```
