# QWEN3-30B live chat — runbook

How to bring up **real, fast chat** in the Inference Console backed by **Qwen3-30B-A3B** on the 8×H100 box.
Measured live: **155.9 tok/s** decode, real generation (ttft ~760 ms incl. prompt processing).

> Why the 30B: the custom TP8 engine proves **116 tok/s** on the 235B but runs on dummy weights (a latency
> proxy — it can't emit real text yet). vLLM is the only chattable path; on the 235B it's ~75–85 tok/s
> (overhead/kernel-bound at B=1, and fp8 is *slower* there, spec ≈1×). The **30B-A3B is a much smaller model
> (3B active vs 22B), so plain vLLM already does ~156 tok/s** — real chat that clears the 116 target. The
> optimization *architecture* (graphs + NVLS comms + kernel tuning + flat-K2 spec verify) is transferable to the
> 30B; only the tuned 235B kernels are model-specific.

## Quickstart (TL;DR)
Box has the model cached; you have this repo + the venv. Five steps to a live chat at ~156 tok/s:
```bash
# 1. BOX: start the 30B (launch script contents in "Box" §1 below)
ssh … root@HOST 'nohup bash /root/launch_30b.sh </dev/null >/root/start_30b.out 2>&1 &'

# 2. LOCAL: tunnel the box's vLLM to localhost:8001
ssh -N -L 8001:localhost:8001 -p 31025 -i ~/.ssh/id_github root@HOST &

# 3. LOCAL: wait until `curl localhost:8001/v1/models` shows qwen3-30b, THEN start the backend on the 30B
VLLM_MODEL=qwen3-30b .venv/bin/uvicorn server.main:app --port 8000 &     # NOT BACKEND=mock

# 4. LOCAL: UI
cd ui && npm run dev

# 5. open http://localhost:5173 and chat -> real generation, ~156 tok/s
```

## How it beats vLLM — the measured comparison (keep two claims straight)

| setup | model | tok/s (B=1, 8×H100) | what it shows |
|---|---|---|---|
| **vLLM baseline** | Qwen3-235B bf16 | **75–85** | the reference OpenAI-compatible server (already CUDA-graphed) |
| **Conifer custom engine** | Qwen3-235B fp8 | **116** | **1.5× vLLM on the *same* model** — the real "beats vLLM" result |
| **this runbook (30B chat)** | Qwen3-30B bf16 | **156** | fast *real* chat — a smaller model on vLLM (NOT a Conifer-vs-vLLM win) |

**What beats vLLM is the Conifer custom TP8 engine, apples-to-apples on the 235B: 116 vs 75–85 ≈ 1.5×.** It wins
by removing exactly what makes vLLM overhead/kernel-bound at B=1 — *not* by changing the model:
- **CUDA-graph capture + scheduler-free B=1 loop** — kills per-kernel launch/host overhead.
- **NVLS in-switch all-reduce** — the tensor-parallel comms barrier, 17 µs → 9 µs/collective.
- **Per-kernel occupancy/MBU tuning** — router split-K (106 → 16 µs), K2 attention, fp8 GEMM.
- *(roadmap)* **flat-K2 tensor-core spec verify + a trained EAGLE3 head** — the path toward ~1000 tok/s.

> ⚠️ **The 30B chat in this runbook is plain vLLM** (a smaller model). It's fast because 30B-A3B activates only
> 3B params vs the 235B's 22B — it does **not** itself beat vLLM; it's the snappy real-chat demo. The Conifer
> architecture is **portable to the 30B**: porting the tuned kernels would stack the same ~1.5× on top of the 156
> (the *techniques* are model-agnostic; only the 235B kernel *shapes* are baked in today). So the console's
> honest story = **"vLLM 75–85 → our engine 116 on the hard 235B case (1.5×), and here's a 156 tok/s live chat."**

## Topology
```
browser  ──HTTP──>  UI (Vite :5173)  ──fetch──>  console backend (FastAPI :8000)
                                                      │  VLLMBackend, VLLM_MODEL=qwen3-30b
                                                      ▼
                                          localhost:8001  ──SSH -L tunnel──>  box:8001 (vLLM, 30B, TP8)
```

## Box (the GPU host)
Session box: `root@147.185.41.162 -p 31025 -i ~/.ssh/id_github` (substitute your host).

### 1. Launch script — write it to the box (DO NOT inline the vllm command over SSH; see Gotchas)
`/root/launch_30b.sh`:
```bash
#!/bin/bash
fuser -k 8001/tcp 2>/dev/null              # free the port (NOT `pkill -f "vllm serve"` — see Gotchas)
pkill -9 -f "VLLM::EngineCore" 2>/dev/null # drop any prior model
sleep 5
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1                     # use the local HF cache, no download
vllm serve Qwen/Qwen3-30B-A3B \
  --served-model-name qwen3-30b \
  --tensor-parallel-size 8 --dtype bfloat16 \
  --max-num-seqs 1 --port 8001 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --trust-remote-code
```

### 2. Run it detached (survives the SSH channel closing)
```bash
ssh … root@HOST 'nohup bash /root/launch_30b.sh </dev/null >/root/start_30b.out 2>&1 & echo started $!'
```
Loads in ~1–2 min (small model). Ready when `curl -s localhost:8001/v1/models` (on the box) returns `qwen3-30b`.

## Local machine (the console)
### 3. SSH tunnel — forward the box's vLLM to localhost:8001
```bash
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ExitOnForwardFailure=no \
    -N -L 8001:localhost:8001 -p 31025 -i ~/.ssh/id_github root@HOST &
```

### 4. Console backend (FastAPI) — point it at the 30B
The backend auto-detects vLLM via `_vllm_healthy()` (probes `VLLM_URL`, default `localhost:8001`) and is selected
ONCE at startup (`server/main.py: BACKEND = get_backend()`), so start it **after** the tunnel + vLLM are up.
`VLLMBackend` sends `model = $VLLM_MODEL` (default `qwen3-235b-fp8`), so override it to the served name:
```bash
cd <repo root>
VLLM_MODEL=qwen3-30b .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000
# (do NOT set BACKEND=mock — that forces the hardcoded demo profile)
```

### 5. UI (Vite)
```bash
cd ui && npm run dev        # serves http://localhost:5173
```
The UI's default API base is `http://localhost:8000` (`ui/src/config.ts`); no `.env` needed. Both lanes
(`conifer`, `vllm`) share the backend and now stream the **real 30B**.

### 6. Verify
```bash
curl -s -m60 -X POST localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-30b","engine":"conifer","messages":[{"role":"user","content":"hi"}],"max_tokens":40}' \
  | tr ',' '\n' | grep -iE '"content"|decode_tok_per_s'
```
Expect real text + `decode_tok_per_s` ≈ 150–160. Then open **http://localhost:5173** and chat.

## Gotchas (each cost a failed launch)
1. **Never `pkill -f "vllm serve"` from inside an SSH command whose own args contain `vllm serve`** — `pkill -f`
   matches the *full command line*, so it kills the SSH login shell running your script before the launch line
   executes (symptom: command returns with **no output**, model never starts). Kill by **port** (`fuser -k
   8001/tcp`) and by the worker name (`pkill -f "VLLM::EngineCore"`) instead.
2. **Launch via a script + `nohup bash script </dev/null >log 2>&1 &`**, not an inline `nohup vllm … &` over SSH.
   Inline, the backgrounded process is still tied to the SSH channel's stdin and dies (or eats the foreground
   echoes) when the channel closes. A `</dev/null`-detached script survives.
3. **Restart the backend after vLLM is ready** — `get_backend()` is evaluated once at import; a backend started
   before vLLM (or with `BACKEND=mock`) stays on the mock profile and streams hardcoded text.
4. **`VLLM_MODEL` must equal `--served-model-name`** or vLLM 404s the request.

## To push the 30B even faster
TP=8 on a 3B-active model is comms-heavy (the small per-GPU weight is dwarfed by the per-layer all-reduces).
**TP=4** (or 2) cuts the collective count and usually lands ~200+ tok/s at B=1 — set `--tensor-parallel-size 4`
in the launch script. Trade-off: fewer GPUs, off the "8×H100" headline.

## Switching back to the 235B
Same flow with `start_vllm_live.sh` (serves `/alloc/data/Qwen3-235B-A22B` bf16 as `qwen3-235b-fp8` on :8001),
and restart the backend without the `VLLM_MODEL` override (its default already matches). ~4–5 min load,
~75–85 tok/s real chat.
