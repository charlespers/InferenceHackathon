# Hackathon Schedule — 4-Person Team

3 rounds of 45 min work + 15 min group testing. Ownership is split by codebase area to avoid merge conflicts.

**Branch:** `jminding/graph+draft` — coordinate before pushing, or cut personal branches and merge at each testing checkpoint.

**Box:** `ssh root@147.185.41.162 -p 31025`
**Tunnel:** `ssh -L 8000:localhost:8000 root@147.185.41.162 -p 31025 -N`
**UI:** `http://localhost:5173` (run `npm run dev` in `ui/`)

---

## Round 1 — 45 min work

| Person | Task | Files |
|--------|------|-------|
| **Jaymin** | Seed `MarkovTransition` predictor with real routing_stats.json matrices; wire `EnsemblePredictor` so it runs on every token | `engine/src/routing/predictor.rs`, `engine/src/routing/optimizer.rs` |
| **Alyssa** | Replace FastAPI mock with Axum server — same SSE API contract (`/v1/chat/completions`, `/v1/topology`, `/v1/models`) | `engine/src/bin/server.rs` (new), `engine/Cargo.toml` |
| **Charles** | UI: highlight replicated experts in a different color, add `placement_source` badge ("optimized" vs "round-robin"), add imbalance metric to GPU cards | `ui/src/components/GpuExpertViz.tsx`, `ui/src/types.ts` |
| **djamoils** | Stub `ModelRunner` + `DrafterPool` trait impls that shell out to vLLM over local socket — lets speculative engine run before conifer lands | `engine/src/spec/model.rs`, `tools/vllm_runner.py` |

### Testing checkpoint — 15 min
- `git pull` on box + restart server (`uvicorn server.main:app --host 127.0.0.1 --port 8000`)
- Verify `/v1/topology` returns `"placement_source": "optimized"`
- Alyssa's Axum server starts and responds at `/health`
- Charles's UI shows hot expert badges and placement source badge
- djamoils's stub ModelRunner passes the 3 spec engine tests (`cargo test --package engine`)

---

## Round 2 — 45 min work

| Person | Task | Files |
|--------|------|-------|
| **Jaymin** | Wire `PrefetchScheduler` into the token loop: after attention, call predictor → scheduler → emit `EarlyDispatch` actions; log hit rate to telemetry | `engine/src/spec/engine.rs`, `engine/src/routing/scheduler.rs` |
| **Alyssa** | Load Qwen3-235B in vLLM (`--tensor-parallel-size 8 --dtype fp8`) and serve real completions through the Axum server | box SSH session + `engine/src/bin/server.rs` |
| **Charles** | UI: real latency chart (ms/tok over time), spec accept rate bar, add prefetch hit rate % to GPU panel | `ui/src/components/LatencyPanel.tsx`, `ui/src/components/GpuExpertViz.tsx` |
| **djamoils** | Stream live routing telemetry from vLLM hooks (same hooks as `tools/routing_analysis.py` but streaming); push events to Axum SSE | `tools/vllm_runner.py`, `engine/src/bin/server.rs` |

### Testing checkpoint — 15 min
- Send a real chat message through the UI
- Verify telemetry tokens light up the GPU expert cards live
- Check real ms/tok numbers appear in the latency panel
- Prefetch hit rate shows up in server logs

---

## Round 3 — 45 min work

| Person | Task | Files |
|--------|------|-------|
| **Jaymin** | Re-run `tools/routing_analysis.py` after 100+ real inference tokens; rebuild optimized placement with `./target/release/optimize_placement`; measure before/after imbalance delta | box SSH session |
| **Alyssa** | Enable speculative decoding in vLLM (`--speculative-model Qwen/Qwen3-1.7B --num-speculative-tokens 8`) or wire our Rust spec engine loop | box SSH session |
| **Charles** | Demo polish: clean layout, add "Expert Placement" toggle (round-robin vs optimized) to show the routing difference visually | `ui/src/App.tsx`, `ui/src/components/` |
| **djamoils** | Write results summary: measured imbalance reduction %, ms/tok baseline vs speculative, prefetch hit rate | `RESULTS.md` |

### Final test — 15 min
- Full demo: send 5 prompts, watch GPU cards light up with real routing, latency numbers live
- Toggle placement (round-robin vs optimized) to show the improvement
- Screenshot everything for the writeup

---

## Key context

- **Model:** Qwen3-235B-A22B at `/alloc/data/Qwen3-235B-A22B` (470 GB bf16, all 118 shards downloaded)
- **Routing stats:** `/alloc/data/routing_stats.json` — real activation counts from 20 prompts × 50 tokens
- **Optimized placement:** `/alloc/data/optimized_placement.json` — 1.45x → 1.07x imbalance (26.6% reduction), 1591 expert-layer pairs replicated
- **Rust engine:** `engine/` crate — spec decoding, route prediction, placement optimizer all implemented
- **Python tools:** `tools/routing_analysis.py` for re-collecting routing stats after real inference
- **UI:** React/TS/Tailwind, polls `/v1/topology` every 2s for live GPU stats
