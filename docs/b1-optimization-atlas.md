# B=1 optimization atlas — every lever, status, and the one execution sequence

The complete map of the converged team effort (my docs + `research/` + `tools/`). Status: ✅measured ·
📊projected · 🧪experiment-ready · 🔬research. Current: **bf16-TP8 85.7 tok/s / 777ms TTFT / 2271ms@128tok**.

## 🎯 NORTH STAR: 1000 tok/s — UPDATED by measured data (`results-reaction-04.md`, `1000-experiments.md`)
Squeeze round changed the picture: **comms is BARRIER-bound (~16µs/collective, can't make it faster — in-kernel
recursive-doubling is worse), int4 RULED OUT at B=1 (0.58×), spec is the lever.** So:
- **Engine: graphs + scheduler-free loop + fp8-K5 (e→1) + big-tree spec amortizes the barrier-bound comms → ~745–870 LOSSLESS** (`ladder_to_1000.py --C 16 --ncoll 188`; route-aware big-tree verify adds ~15%, now first-order).
- **The 300→1000 leap = ONE of:** (a) **proxy/stale-TP** hides the comms (LOOP-C; quality-gated — use **DirectProxy** as the predictor to preserve routing) → ~1218; (b) **EAGLE3 realized ×3.8** (trained draft) → ~1003; (c) a *lossless* EP collective-count cut (flagged uncertain — 2 TP ARs are intrinsic, DP-attn is a net loss).
- **int4 / count-via-DP-attn / making-the-AR-faster are dead ends.** **Make-or-break experiments (`1000-experiments.md`):** #1 comms C (`measure_collective.sh`, no model load), #2 EAGLE3 realized S (09:45), #3 proxy/stale-TP quality (LOOP-C).
- **Cheap first ship (~300, lossless today):** spec + prefix-cache on bf16-TP8.

## The decode-step decomposition (the spine — `overhead-attribution.md`)
TPOT 11.67ms = **overhead ~7.0ms (60%) · comms ~3.0ms (26%) · weight ~1.6ms (14%)**. Floor-bound: weight
levers are invisible until the floor is down (`results-reaction-02.md`, proven by `ab_adaptive`).

## Levers, ranked by current gain-per-effort
| # | lever | gain | status | where |
|---|---|---|---|---|
| 1 | **Prefix caching** (TTFT) | ~50–100× TTFT (cached) | 🧪 high-conf | `ttft-analysis.md` · `run_bench6.sh` |
| 2 | **Spec decode** (floor-amortization) | **~2–3×** (k=8/EAGLE3 now) | 📊→🧪 | `spec-decode-floor-bound.md` · `spec_floor_model.py` · `run_bench5.sh` |
| 3 | **E-attr** (split the 7ms floor) | diagnostic → routes 4/5 | 🧪 | `overhead-attribution.md` |
| 4 | **Kernel efficiency** (K5: vLLM 0.16→0.46) | shrinks ~½ the 7ms | ✅ kernel / 🧪 e2e | `k5_experts_warp.cu` · `k5-kernel-results-h100.md` |
| 5 | **Comms latency** (NVLS one-shot, 16→7µs) | comms 3.0→1.3ms | 📊 | `research/comms_floor.md` · `seriality-breaking-notes.md` · E0b |
| 6 | **Layout = TP8** | avoid EP 2.53× busiest | ✅ measured | `b1-tp8-moe-rearchitecture-h200.md` · `predicted-tok-s-matrix.md` |
| 7 | **fp8 weights** (E2b dynamic-quant unblock) | ~7% now (weight=14%) | 🧪 | `results-reaction-01.md` · `run_bench4.sh` |
| 8 | **Depth reduction / EAGLE3** | ~3× (spec-flavored) | 🔬 | `research/depth_reduction.md` |
| 9 | **B=1 fast-path** (per-step scheduler tax) | ~1.3–2× on the residual | 🔬 (last) | `b1-fast-path-design.md` |
| 10 | **Long-context KV (fp8 KV)** | ~2× at 128K | 📊 (post-floor) | `long-context-chat.md` |
| 11 | **EP placement** (per-step busiest, IF EP) | EP-path mitigation | ✅ demo | `ep-placement-for-b1.md` · `placement_b1.py` |
| 12 | **Route prediction** (prefetch deep layers) | comms-overlap | ✅ persistence 0.45 | `routing_predict_early.json` · `verify_route_prediction.py` |
| — | adaptive top-k / int4 | **LAST** (invisible floor-bound) | ✅ regressed | `router_mass.py` · `k5_int4_bench.cu` |

## The one execution sequence (cheap-first, data-gated)
1. **Ship now (config flags, free):** prefix caching (#1) + n-gram/EAGLE3 spec (#2) on bf16 **pure-TP8** (#6).
   → `run_bench_best.sh` validates the stacked ~5× in one slot.
2. **`E-attr`** (#3): split the 7ms → does the next decode win come from **comms** (#5) or **kernels** (#4)?
3. Land that one, then **fp8** (#7) for the headroom (small until the floor is down).
4. **Research bets** as the floor falls: depth/EAGLE3 (#8), the cudarc fast-path (#9), route-prefetch (#12).
5. **When chats run long:** fp8 KV (#10). **If ever forced onto EP:** co-activation placement (#11).

## What's proven vs projected (honesty)
- **Measured on-box:** the floor decomposition; EP→TP (64.5<85.7); collective latency 16µs; K5 e=0.46;
  route persistence 0.45 / affinity placement; adaptive-top-k regression.
- **Projected (model + measured constants):** the ~5× endpoint, the spec ~2–3×, the NVLS comms win. The
  slot benches (`run_bench_best/4/5/6`) convert these to measured the moment they run.
- **The recurring caveat:** the *model* floor (e.g. 216 tok/s bf16-TP8 @16µs) overstates wins vs the *real*
  overhead-bound 85.7 — because the 7ms overhead isn't in the model. `E-attr` closes that gap; until then,
  every model projection is an upper bound the overhead pulls down.

## Files
Mine (`charles-work`): the docs above + `tools/{latency_budget,predict_matrix,spec_floor_model,placement_b1,
verify_route_prediction,verify_self_speculation}.py` + `kernels/k5_*` + `bench/run_bench{4,5,6,_best}.sh`.
Team (`research/`, `tools/`, `engine/`): `seriality_breaking · depth_reduction · comms_floor · spec_moe_model`
· `router_mass · routing_predict · slot_runner` · the Rust `engine/{routing,spec}` + the fused `kernels/`.
