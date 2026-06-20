# Team coordination — how the kernel/research work fits `origin/main`

Read after surveying `origin/main` (the Rust `engine/` crate + Python `src/inferutil/` bench) and
`SCHEDULE.md` (Charles tests **:30–:45**, 15 min/hour). This reconciles my work (on `charles-work`) with
what the team already built, corrects one of my earlier calls, and revises the GPU plan for the 15-min budget.

## What already exists on `origin/main` (don't duplicate)
- **`src/inferutil/latency.py`** — a B=1 decode latency model that *independently implements the same
  busiest-rank physics I derived*: `expected_max_experts_per_gpu(top_k, ep)` = balls-in-bins E[max], plus
  weight/kv/comms/compute breakdown per parallelism plan. **My research/spec converges with this.**
- **`src/inferutil/bench/`** — a calibrated bench keyed on an **`efficiency`** parameter (BW efficiency,
  1.0 = analytical floor) + NVML telemetry. **`efficiency` is exactly the realized `e` my kernel measures.**
- **Rust `engine/routing/`** — `DirectProxy` route predictor (predicts next layer's experts from the
  residual stream `h_{L+1}≈h_L` via the next router weights, zero-training) + `scheduler.rs` prefetch
  (issues the next-layer all-to-all dispatch early on a `prefetch_stream` to hide the ~5 µs collective).
- **Rust `engine/spec/`** — speculative decoding (accept/engine/model). **My L1/E6 overlaps this.**
- **`server/` VLLMBackend** — proxies the OpenAI SSE contract to real vLLM + injects `x_telemetry`.

## My complementary contributions (the gaps the team work doesn't fill)
1. **Measured kernel efficiency `e`** — the team's bench *assumes* `efficiency`; I *measured* it on-box
   (K5 MoE GEMV: **e=0.46**, gate/up 0.49 / down 0.405). This calibrates their model (below).
2. **Engine-launch validation** — reproduced the vLLM `192%128` TP8-FP8 crash my spec predicted; the fix
   (`--enable-expert-parallel` / TP4×EP2) is the engine config the team should use.
3. **The actual CUDA kernels** (`kernels/k5_experts_warp.cu`) — a reference the cudarc engine could call.
4. **The 15-min experiment plan** (`gpu-agent-experiments.md`) + the next-levers dossier.

## Cross-validation (their model × my measured e) — the EP→TP inversion, triple-confirmed
`python -m inferutil.bench run --gpu H100-SXM-80GB --n-gpus 8 --dtype 1 --efficiency 0.46 --plan <p> ...`
(ctx 2048, FP8):

| Layout | decode tok/s @ e=0.46 | floor @ e=1.0 |
|---|---|---|
| **TP8** (plan=tp tp=8) | **260.9** | 567.2 |
| EP8 (plan=ep ep=8) | 94.1 | 204.5 |
| hybrid TP4×EP2 | 98.6 | 214.4 |
| hybrid TP2×EP4 | 105.6 | 229.6 |

**TP8 is 2.8× EP8 in the team's own model** — independent agreement with my spec and my measured kernel.
**Pure TP8 is the layout to serve.** (Reminder: vLLM can't pure-TP8 the FP8 ckpt — 192%128 — so the
*engine* path is `--enable-expert-parallel`; the *model* says a true TP8 column-shard, e.g. via a block-64
requant or the cudarc engine, is worth ~2.8× over EP8.)

## New finding + open discrepancy (what E1 resolves)
At the TP8 **floor** (e=1.0), the model's breakdown is **weight 0.81 ms, comms 0.94 ms** — i.e. **comms
DOMINATES**. My earlier roofline put comms at ~0.14–0.28 ms. These disagree by ~3–6×. Consequences:
- If the team's comms model is right → **comms is the #1 lever**, and the routing `scheduler.rs`
  (early-dispatch prefetch) + one-shot/NVSHMEM all-reduce matter *more* than any byte lever. The
  prefetch hides ~5 µs/layer × 94 ≈ **~0.47 ms/token** — most of that 0.94 ms comms term.
- If my lower estimate is right → weight dominates and int4/quant pays first.
- **E1 (measured real TPOT) decides this.** Back out the true comms term: real TPOT − (weight/e) − kv.

## Correction to my dossier (L4 route-prefetch was under-rated)
My `next-levers-research.md` L4 called route-prediction prefetch "moot at B=1 (weights HBM-resident)."
**That was wrong.** The team's `scheduler.rs` is correct: the value at B=1 is **not** PCIe fetch — it's
(a) issuing the next all-to-all dispatch early to overlap the ~5 µs collective, and (b) L2-warming the
predicted experts. Given comms may be the dominant term (above), **this is a do-soon lever, not a
research-bet.** The `DirectProxy` predictor is zero-training and immediately usable.

## Revised plan for the 15-min window (Charles :30–:45)
A 235B vLLM load + graph-capture is **~3–4 min**, so the window affords **~1 engine launch**, not a sweep.
Spend it to extract maximum signal from a single load; do all model/kernel work that needs **no GPU load**
outside the window or in parallel.

1. **Before the slot (no GPU):** run the bench *model* for every layout/precision (done above — free,
   pure-stdlib). Pre-stage the launch command + the `measure` command.
2. **In the slot (one engine load):** launch FP8 `--enable-expert-parallel` once (E1). While it serves,
   run: ctx-128/2048/8192 `measure`, `nvidia-smi dmon` for EP balance, and — same load — toggle
   `--enforce-eager` only if time permits (a relaunch). **Capture the real TPOT breakdown → resolves the
   comms discrepancy.**
3. **Anytime (no engine load, seconds):** the kernel microbenches (`k5_microbench`, `k5_downproj_bench`)
   + Nsight — these don't need the model loaded, so run them in any leftover GPU seconds or a separate slot.
4. **Calibration loop:** feed the measured TPOT back as `--efficiency <backed-out>` to reconcile the bench
   model with reality, and compare to my kernel `e=0.46`.

**Priority for the first real slot:** E1 (FP8+EP baseline + breakdown) > E4 kernel microbench/Nsight (no
load) > E6 n-gram spec (one relaunch). The big unknown is whether reality is comms-bound (→ prefetch/
spec) or weight-bound (→ int4) — E1 answers it in one load.
