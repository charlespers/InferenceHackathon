# Initial Trajectory — Qwen3-235B-A22B @ B=1 on 8×H100

The first decisions here set the autoregressive search that follows. Goal: minimize
**TPOT** (time per output token) for a single request. Everything is judged against the
**bytes-moved-per-token** roofline; comms *count* is a co-equal term at B=1.

Public-model + standard-technique only; no proprietary engine internals.

---

## 0. The 60-second checklist the moment SSH lands

```bash
# 1. Confirm the mesh is full NVLink/NVSwitch (expect NV# between every pair, not SYS/PHB)
nvidia-smi topo -m
nvidia-smi --query-gpu=name,memory.total,power.max_limit --format=csv

# 2. Pull an fp8 checkpoint (avoid runtime quant if a prebuilt fp8 exists)
#    Qwen3-235B-A22B-FP8 if available; else bf16 + engine-side fp8.
huggingface-cli download Qwen/Qwen3-235B-A22B-FP8 --local-dir /models/qwen3-fp8

# 3. Launch the bootstrap engine (SGLang) — get a NUMBER before touching kernels
python -m sglang.launch_server --model-path /models/qwen3-fp8 \
  --tp 4 --ep 2 --kv-cache-dtype fp8_e5m2 --enable-torch-compile \
  --cuda-graph-max-bs 1 --host 0.0.0.0 --port 8000      # flags: validate vs installed version

# 4. First baseline: end-to-end TPOT over the same OpenAI SSE contract the UI uses
python bench/measure.py --base http://localhost:8000 --ctx 2048 --decode 128
#    Record TTFT, TPOT, decode tok/s, and % of the fp8 roofline (bench/roofline.py).
```
If anything blocks (weights, flags), fall back to bf16 + `--quantization fp8` and a smaller
context to get a baseline; optimize from a real number, never from theory.

---

## 1. Target envelope (what "good" looks like)

fp8 roofline (26.8 TB/s aggregate): **~1,240 tok/s** short-ctx → **~1,080 @ 32k** → **~780 @ 128k**.
Real systems land at 40–70%. **First milestone: ≥40% of roofline at 32k (≈430 tok/s, ≈2.3 ms TPOT).**
Stretch with speculation: 1.5–2.5× on accepted tokens.

Per-GPU byte budget when balanced across 8: ~2.7 GB/token (21.6 GB / 8) + KV share. That's
~0.8 ms of pure HBM time — so **anything that adds >~0.1 ms/token of comms or launch overhead
is first-order.** This is why the parallelism choice (next) is the highest-leverage decision.

---

## 2. The parallelism decision (do this first, it dominates everything)

At B=1 there is **one token in flight**, so:
- **GQA caps clean tensor-parallel at TP≤4** (only 4 KV heads; TP=8 forces 2× KV replication).
- **Collective COUNT matters more than volume.** Per token: TP→ ~2 all-reduces/layer; pure
  EP→ dispatch+combine all-to-all/layer (~188 tiny collectives/token across 94 layers). At
  B=1 each expert sees exactly one token, so EP has near-zero arithmetic intensity and the
  all-to-all latency (not bandwidth) is what you pay.

**Start: TP=4 × EP=2.** Clean GQA sharding (1 KV head/TP rank), experts split into 2 groups
(64 experts/group, ~8 active land in each), moderate all-to-all. **Bench against:**
1. **TP=8 (KV replicated ×2)** — fewest collectives, perfectly balanced; pays 2× KV bytes.
2. **EP=8 + attention TP/replicated** — only wins if a low-latency all-to-all (DeepEP/NVSHMEM)
   makes the 188 collectives cheap (~µs) *and* expert placement stays balanced.

Decision rule: pick the layout with the lowest measured TPOT at 32k; expect **TP=8 or TP=4×EP=2
to win at B=1**, with pure-EP only competitive once DeepEP + CUDA-graph capture are in.
Details + placement math: `ep-parallel-schedule.md`.

---

## 3. Tuning order (one change at a time; re-measure; the dominant term shifts)

| # | Lever | Expected win | Confirm with |
|---|---|---|---|
| 1 | **fp8 weights + fp8/int8 KV** | 1.5–2× vs bf16 | bytes/token drops; tok/s up |
| 2 | **Parallelism layout** (§2) | balanced + min comms | TPOT across TP4×EP2 / TP8 / EP8 |
| 3 | **CUDA graph, greedy** (capture whole step) | remove launch latency | TPOT step-down, low CPU |
| 4 | **Speculative decode** (EAGLE-3/MTP) | 1.5–2.5× on accept | accept rate × tok/s |
| 5 | **Graph survives spec-decode** (the crux) | keep #3 win *with* #4 | no per-token D2H sync |
| 6 | **Kernel fusion** (K1/K3/K5 from `kernels/`) | 1.1–1.4× | per-kernel µs, fewer dispatches |
| 7 | **Expert placement / NCCL tuning** | last 5–10% | per-GPU balance, all-to-all µs |

Stop when TPOT ≈ `bytes_per_token / usable_HBM_BW` — then you're physics-limited; only fewer
bytes (more quant / more speculation) or more GPUs help.

---

## 4. Decision tree — "if <dominant term> then tune <lever>"

`bench/sweep.py` measures the four terms each run; act on whichever dominates:
- **Weight bandwidth dominates** → quantize harder (fp8→int4 on experts), confirm dequant isn't compute-bound.
- **KV bandwidth dominates** (long ctx) → fp8/int8 KV, then KV-cache compression / shorter ctx / prefix reuse.
- **Comms dominates** (lots of small collectives) → switch layout toward TP (fewer collectives), adopt DeepEP/NVSHMEM low-latency all-to-all, overlap dispatch with prior-layer combine, capture in graph.
- **Launch/host overhead dominates** (low GPU util, gaps in Nsight) → CUDA graph capture; make sampling + accept on-device.
- **Speculative accept is low** → better draft (EAGLE-3/MTP > n-gram), tune draft length K and threshold; check prompt structure (§6 of the runbook).

---

## 5. Top risks (mitigations)

1. **fp8 weights unavailable / quality regression** → keep a bf16 fallback lane; validate task accuracy (the UI's task presets) alongside speed.
2. **B=1 EP all-to-all latency > weight read** → do NOT default to pure EP; start TP-heavy (§2); only adopt EP with DeepEP + graph capture.
3. **CUDA graph + spec-decode incompatibility** → if variable accept breaks capture, run graph on the greedy verify path with fixed max-draft + masking (`spec-decode-cuda-graph.md`); keep a non-graph fallback so you always have a number.
4. **Engine flag drift** → validate every launch flag against the installed SGLang/vLLM version; the commands here are first-guess.
5. **Hand-written kernels won't beat the engine quickly** → kernels in `kernels/` are the *second-half* play (lever #6); ship wins from quant + parallelism + graph + spec first.

---

## 6. Pointers
- Parallelism + placement + comms: `docs/kernel-design/ep-parallel-schedule.md`
- Speculative decode × CUDA graph (the crux): `docs/kernel-design/spec-decode-cuda-graph.md`
- Fused CUDA kernel skeletons (K1–K6): `kernels/README.md`
- Autoresearch + benchmark harness: `bench/README.md`
- Byte budget / fusion map / DoF: `docs/kernel-hypertuning-qwen3-235b-h100.md`
