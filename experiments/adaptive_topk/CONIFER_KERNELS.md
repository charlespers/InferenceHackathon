# Conifer locate + Charles's kernels vs adaptive-top-k synergy

Investigation date: 2026-06-20. Author context: building a confidence-adaptive top-k
expert-reduction optimization for B=1 decode (route to fewer experts when the router
softmax is concentrated) for Qwen3-235B-A22B on 8×H100.

---

## TASK 1 — Does "conifer" exist?

**No — there is no real conifer inference engine.** There is a placeholder repo and nothing else.

| URL probed | Result |
|---|---|
| `github.com/charlespers/conifer` | **Exists but EMPTY** — GitHub API: `size: 0`, `language: null`, `description: null`, `default_branch: main`. `created_at == pushed_at == 2026-04-11T22:45:23Z` (created once, never committed to since). Web view renders "This repository is empty." |
| `github.com/jminding/conifer` | 404 (does not exist) |
| `github.com/AlyssaC576/conifer` | 404 (does not exist) |
| `github.com/djamoils/conifer` | 404 (does not exist) |
| `charlespers?tab=repositories` | Lists `Conifer` among 21 repos, but it is the empty placeholder above. No other conifer-named repo on any of the four users. |
| WebSearch "conifer MoE inference engine CUDA kernels" | No project called "conifer" in the MoE-inference space. Only generic MoE-kernel material (DeepEP/DeepGEMM, vLLM fused MoE, PyTorch Triton fused-MoE, etc.). |

**Conclusion:** `charlespers/Conifer` is a reserved-name stub created 2026-04-11 with zero
bytes of content — no kernels, no weight loading, no decode loop, no routing, no FP8 GEMV.
There is nothing to pull over. This confirms the repo's own README/DESIGN.md framing:
conifer is a *planned* engine that never landed. The real custom-inference work is Charles's
CUDA kernels on `origin/charles-work` (the "conifer kernels" in spirit), and serving is vLLM.

---

## TASK 2 — Charles's kernels (origin/charles-work) and adaptive-k synergy

Source read directly from git: `kernels/k4_router.cu`, `k5_experts_warp.cu`,
`k5_experts_tuned.cu`, `k5_experts_int4.cu`, `k5_microbench.cu`, `kernels/README.md`,
`k6_graph_capture.cu`; docs `k5-kernel-results-h100.md`, `kernel-design/ep-parallel-schedule.md`,
`charles-work.md`.

### (a) Design + measured state of K5 (expert kernel) and K4 (router)

**K5 — the MoE expert GEMV (the B=1 bottleneck, ~14.2B of ~21.6B active params/token).**
- Real, MEASURED on a physical H100 80GB (sm_90a, CUDA 12.6). Numerically validated vs a
  scalar reference (max relative error 3.2e-5, fp32 accumulation).
- Optimization journey (measured, 8 active experts, fp8, 151 MB moved/call, 3.35 TB/s peak):
  scalar 9.82 ms (e=0.005) → 128-bit loads + smem-y + hoisted scale 1.15 ms (e=0.039) →
  tile experts across CTAs 0.175 ms (e=0.257) → warp-per-row + split-K 0.105 ms (e=0.431) →
  fp8x2→half2 hardware dequant **0.098 ms, 1538 GB/s, e=0.459, ~100× vs scalar** (winner =
  `k5_experts_warp.cu`).
- Architecture that carries the win at B=1: **warp-per-row + split-K across the 32 lanes**
  (consecutive lanes read consecutive 16-byte chunks of the SAME weight row → fully coalesced
  HBM, then a shuffle-reduce), and **fill the machine** (grid-stride warps over (slot, channel),
  best launch 264 CTAs × 1024 threads = 8448 warps, instead of 1 CTA/expert which idles 124/132 SMs).
- Split into two kernels around a global `a` buffer: `k5a_gateup_warp` (gate+up+silu, 101 MB,
  e=0.490) and `k5b_down_warp` (down-proj, 50 MB, e=0.405 — the weaker one; short 1536 contraction
  + 48 KB all-`a` smem occupancy cap). Down-proj uses `atomicAdd` into the residual.
- An **int4 variant** (`k5_experts_int4.cu`, W4A16) exists as the "next ~2× byte win" — same
  warp-per-row/split-K structure, measures whether the in-register nibble unpack becomes issue-bound.
  Written, not yet benched on-box.

**K4 — the router (`k4_router.cu`).**
- Fully on-device, no host sync: post-RMSNorm → gate GEMV (4096→128) → fp32 softmax over 128
  → top-8 → renormalize selected weights to sum 1 (`norm_topk_prob=true`, no shared expert).
- **It is a skeleton.** The top-8 selection is a single-threaded `threadIdx.x==0` O(128×8) argmax
  loop with explicit `TODO(on-box)` to parallelize (block-wide max/sum reduce, lane-parallel top-k).
  It emits `sel_idx[TOP_K]` and `sel_w[TOP_K]` into device memory for K5/EP dispatch to read with no sync.
- **This is the integration point for adaptive-k** (see below): K4 already computes the full fp32
  softmax over all 128 experts, so the routing-concentration signal you need (top-1 prob, entropy,
  cumulative-mass threshold) is *already in registers/smem at the moment of selection*.

### (b) Does adaptive-k integrate MORE cleanly into Charles's kernel than vLLM's fused path?

**Yes — decisively, and for a concrete structural reason.**

- **vLLM fused path (your current limitation, confirmed):** `moe_align_block_size` pads the
  expert→block grid to a fixed capacity. Dropping experts at B=1 skips the HBM weight *load* but
  NOT the launch/align overhead, so the byte saving only shows up when weight-reads already dominate.
  There is a hard padding floor you can't get under.
- **Charles's K5 has NO padding floor.** The number of expert GEMVs is driven directly by the
  routing output — `k5_experts_tuned.cu` launches `grid.x = TOP_K_local` (one CTA per active
  expert slot), and the warp kernels grid-stride over `nslot * MOE_INTER` / `nslot * HIDDEN` where
  `nslot` is literally the count of selected experts. **Make K4 emit a variable `nslot` (e.g. 4
  instead of 8 when softmax is concentrated) and K5 does proportionally less work — fewer warp items,
  fewer HBM weight rows read — with no fixed-capacity grid to pad against.** The byte saving becomes
  real wall-clock: at B=1 decode is ~1 FLOP/byte (pure bandwidth), so dropping 8→4 experts halves
  the dominant 14.2B-param/token weight read → ~½ the K5 time, the single largest decode term.
  Both the warp kernels (via `nslot`) and the tuned single-CTA kernel (via `grid.x = nslot`) skip
  **launch + compute + load** for dropped experts. This is exactly the "no moe_align padding floor"
  property you want.
- Bonus: K4 already has the concentration signal for free (full fp32 softmax over 128), so adaptive-k
  costs ~nothing to *decide* — just change the stopping rule in the top-k loop from "pick 8" to
  "pick until cumulative prob ≥ τ, capped at 8, floored at e.g. 2."

### (c) Are the kernels wired into a runnable engine, or standalone microbenchmarks?

**Standalone microbenchmarks — NOT a runnable engine yet.** This is the key near-term caveat.

- K5 is exercised only by `k5_microbench.cu` (and `k5_int4_bench.cu`, `k5_downproj_bench.cu`):
  synthetic random fp8 weights, correctness = kernel-vs-reference equivalence, NOT model accuracy.
- `k6_graph_capture.cu` is the *intended* glue (capture K1→K5 ×94 + final norm + lm_head + sampling
  as one CUDA graph) but it is a host-side skeleton full of `TODO(on-box)`; the kernels are explicitly
  "not yet wired into k6," reductions are sketches, and nothing loads real Qwen3 weights or runs a
  full forward pass.
- `kernels/README.md` calls these "best-first-guess skeletons … the *second-half* play" — the team's
  shipped path is vLLM (`charles-work.md`: current best **bf16+TP8 = 85.7 tok/s**, decode is
  **comms-bound** at ~16 µs all-reduce). The kernels are a research track, not the serving engine.
- Therefore: a K5-based adaptive-k is **not** a drop-in win this week. It requires building the
  missing engine scaffolding (weight loading, the k6 graph, real-model validation) before any
  end-to-end tok/s number exists. K5 itself is real and fast; the *engine around it* is not built.

---

## DELIVERABLES — direct answers

**(1) Does conifer exist?**
No. `charlespers/Conifer` is an empty 0-byte placeholder repo (created and untouched since
2026-04-11); the other three users 404; no public "conifer" MoE engine exists. Nothing to reuse.
The real custom-kernel work is `origin/charles-work` `kernels/k1–k6`.

**(2) Single most reusable thing from Charles's kernels for adaptive-k:**
**K4's on-device full-softmax + top-k selection (`k4_router.cu`), paired with K5's `nslot`-driven
launch.** K4 already computes the fp32 softmax over all 128 experts and writes `sel_idx`/`sel_w` to
device memory, so the concentration signal for adaptive-k is free and the change is a ~5-line edit to
the selection loop (stop at cumulative mass τ instead of a fixed 8). K5 then consumes the variable
expert count with zero padding floor — the byte saving turns into wall-clock directly. (Concretely:
make K4 output a variable `nslot`; K5's `grid.x = nslot` / grid-stride over `nslot*…` does the rest.)

**(3) Recommendation — target vLLM, Charles's kernel, or both? Fastest path to a real wall-clock win:**

Target **vLLM now for the measurable win; design for K5 as the second-half ceiling.**

- **vLLM (now):** It is the only thing that actually runs the full model end-to-end today, so it is
  the only place you can produce a real, validated tok/s and quality number this week. Accept the
  `moe_align` padding floor as a known limitation and measure where weight-reads dominate (large
  context, fp8 weights) — that's the regime your adaptive-k shows a win in vLLM. Pair it with a
  quality gate (entropy/τ threshold tuned so dropped experts carry negligible probability mass).
  This gives a defensible result without building an engine.
- **Charles's K5 (the real ceiling, not near-term):** It is where adaptive-k becomes *clean* (no
  padding floor, byte saving = wall-clock saving) — but it is microbench-only, not wired into a
  runnable engine. Treat it as the design target: prototype the K4 cumulative-τ stop + variable
  `nslot` against `k5_microbench.cu` (synthetic weights, no model load) to **measure** the 8→k
  bandwidth/latency curve on the H100. That microbench result is cheap (no GPU model load, no engine)
  and proves the ceiling, even before an engine exists to ship it in.
- **Fastest path to a real wall-clock win:** (i) ship adaptive-k in vLLM in the weight-read-dominated
  regime + quality gate (real end-to-end number, this week); in parallel (ii) extend `k5_microbench.cu`
  to sweep nslot=2..8 with a concentration-driven stop, demonstrating the no-padding-floor speedup as
  the K5/k6 engine path matures. Do **not** block on building the K5 engine — it isn't there yet.

---

## Files referenced (origin/charles-work unless noted)
- `kernels/k4_router.cu` — on-device softmax + top-8 (skeleton; adaptive-k hook)
- `kernels/k5_experts_warp.cu` — measured winner, e=0.459 / 1538 GB/s, nslot-driven grid-stride
- `kernels/k5_experts_tuned.cu` — single-CTA-per-slot variant (`grid.x = TOP_K_local`)
- `kernels/k5_experts_int4.cu` — int4 W4A16 next-byte-lever (written, unbenched)
- `kernels/k5_microbench.cu` — correctness + HBM-bandwidth harness (the cheap prototype target)
- `kernels/k6_graph_capture.cu` — whole-step CUDA-graph glue (skeleton, TODO-heavy, not wired)
- `kernels/README.md` — "skeletons … second-half play"; engine not built
- `docs/k5-kernel-results-h100.md` — the measured journey
- `docs/kernel-design/ep-parallel-schedule.md` — B=1 EP/TP schedule, nslot framing
- `docs/charles-work.md` — vLLM is the shipped path (bf16+TP8 = 85.7 tok/s, comms-bound)
