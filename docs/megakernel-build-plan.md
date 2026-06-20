# Megakernel build plan — the concrete, staged path to 1000 tok/s

`path-to-1000.md`: 1000 tok/s needs fp8 + in-kernel NVLS + ~zero overhead, simultaneously — the **megakernel**.
This is how to build it incrementally, so each stage is **measurable, banked, and has a fallback**, and so we
know the moment vLLM tops out and the custom engine must take over. Stages 0–2 are vLLM; 3–5 are the cudarc
engine. Every stage validates against `bench/measure.py` (tok/s) + a parity gate (lossless) + `E-attr` (the
floor delta it claims).

## Stage 0 — the vLLM ceiling (measure where flags top out) · ~250–400 tok/s
**Do:** bf16-TP8 + EAGLE3/n-gram spec + prefix-cache + `max-num-seqs=1` + V1 + CUDA graphs (`vllm-b1-config.md`).
**Measures:** the cheap 3.5× (spec) on the *current* floor → ~300. **This is the floor-amortized rung, not the
floor-removed one.** Banks the lossless ~3.5×.
**Gate:** does graphs-on beat eager by ~5×? (Confirms graphs capture the kernels.) If not, graph capture is broken
on the MoE+spec path — fix before proceeding.

## Stage 1 — `E-attr`: split the 7 ms so we know the megakernel's exact job · diagnostic
**Do:** Nsight on bf16-TP8 graphs; attribute the step to **launch / host-gap / NCCL-comms / kernel-sub-roofline**.
**Decides the whole plan:**
- launch is big → graphs help (Stage 0 already banks it).
- **NCCL-comms is ~3 ms and graph-captured-but-not-faster → Stage 3 (the NVLS kernel) is THE lever.** (Expected:
  comms is the wall; graphs remove launch but the 188×16 µs all-reduce stays.)
- host-gap remains under graphs → Stage 4 (scheduler-free loop).
- kernel sub-roofline → Stage 2 (fp8 K5, e→1).

## Stage 2 — fp8 weights + efficient MoE kernel (make the weight fit) · the weight term → 0.78 ms
**Do:** `--quantization fp8` **with fused dequant** in the expert kernel (the K5 path, `k5-tuning-roadmap.md`
cp.async double-buffering so dequant overlaps the load → free). **Critical:** fp8 is *negative* today (Alyssa,
floor-bound) — it only pays once the floor is being removed (Stages 3–5) AND the dequant is fused. So land fp8
**together with** the comms/overhead work, not before. Validate the weight read hits ~0.78 ms (roofline) and
parity holds.
**Gate:** fp8 e2e ≥ bf16 once graphs+K5 are on (if still slower, the dequant isn't fused — fix the kernel).

## Stage 3 — the device-side NVLS/multimem all-reduce kernel (THE CRUX) · comms 3.0 → ~0.2 ms
This is the single highest-leverage piece and the reason a custom kernel is unavoidable: NCCL defaults won't
NVLS at 8 KB (Alyssa), and graphs don't make the all-reduce *faster*, only launch-free.
**Do:** an 8 KB all-reduce kernel using **`multimem.ld_reduce` / `multimem.st`** over the NVLink-SHARP in-switch
reduction (NVLS), on 2–8 SMs, **captured in the CUDA graph first** (standalone, replaces vLLM's AR), then moved
**in-kernel** at Stage 5. Target **~1–3 µs/collective → 0.19–0.56 ms** for the 188.
**Validate:** microbench the single 8 KB all-reduce latency (vs the 16 µs baseline) BEFORE wiring it — this is a
standalone kernel, testable in isolation (like K5 was). Parity: bit-exact reduction.
**Fallback:** if multimem/NVLS underperforms on this box, the comms floor is ~0.5–1.0 ms (3–6 µs one-shot) →
1000 is then only reachable with a lossy comms-count cut (depth reduction) — flag it early.

## Stage 4 — scheduler-free B=1 loop (host overhead → 0) · removes the residual ~0.5–3 ms host
**Do:** the `b1-fast-path-design.md` loop — one request, no continuous-batch scheduler, on-device sampled-token
self-feedback (no D→H→D per token), async detok, contiguous KV. In cudarc this is the native decode loop.
**Validate:** inter-kernel idle gaps (Nsight) → ~0. `--decode 1` vs `--decode 128` isolates per-step host cost.

## Stage 5 — fuse into the persistent megakernel (the last overhead + in-kernel comms) · → ~1.0 ms, 1000 tok/s
**Do:** collapse the per-layer kernels into **one persistent grid** looping over all 94 layers; the Stage-3 NVLS
reduce runs **inside** the kernel (grid-wide sync, no graph break); fp8 weights streamed with cp.async across
layer boundaries; the 8 KB activation stays in registers/smem between layers (`megakernel-b1.md`).
**Result:** `0.78 (fp8) + 0.19 (in-kernel NVLS) + ~0 (no launches/host) ≈ 0.97 ms → ~1033 tok/s, lossless.`
**Validate:** end-to-end tok/s + parity vs the bf16 reference.

## Past 1000 (margin + the lossy cushion)
- **Small-tree route-aware spec on the floor-removed engine** → ~1.5× → ~1500 (lossless; the F→0 column of
  `tree_spec_optimizer.py` — note the tree must *shrink* here, not grow).
- **int4 experts** (0.39 ms weight) → ~2000; **adaptive-top-k** (4 experts) → more margin — both quality-gated.
- **depth reduction** — the most leveraged lossy knob (fewer layers cut weight AND the 188 count together).

## The honest risk assessment (and why Stage 5 is NOT the gate)
- **Stage 3 (NVLS) is the make-or-break.** If the box's multimem in-switch reduce hits ~2–3 µs, 1000 is lossless
  (with small-tree spec). If it floors at ~4 µs, 1000 needs stale-TP (LOOP-C) to hide it or a lossy lever
  (depth/int4). Test this kernel *first and in isolation* — it gates everything.
- **The persistent megakernel (Stage 5) is OPTIONAL.** Its only gain over Stages 2–4 is keeping the activation
  on-chip between layers — **0.06 µs/token at fp8, negligible.** CUDA graphs already fold the per-layer kernels
  (incl. the Stage-3 NVLS kernel) into one launch/step, and the Stage-4 scheduler-free loop drives host→0. So
  **1000 is reachable as: graphs(fp8-K5-at-e→1 + NVLS-AR) + scheduler-free loop + small-tree spec** — two custom
  kernels integrated as vLLM custom ops, **not a whole new engine.** Stage 5 is the clean ~5–10% finish.
- **So the real ask is TWO isolation-testable kernels**, not a megakernel rewrite: **fp8-K5 at e→1** (Stage 2 —
  e=0.46→~1.0, the weight at roofline) and the **NVLS all-reduce** (Stage 3 — C≤~4 µs). Both graph-captured.
  vLLM gets ~300 on flags (Stage 0); the 300→1000 leap is these two kernels + the fast-path, integratable into
  vLLM (custom ops) — the cudarc engine is a *parallel* clean path, not a prerequisite.
- **Order discipline:** spec + prefix-cache now (bank 3.5×); fp8 only *with* the floor work; NVLS kernel is the
  pivot; the persistent megakernel is an optional polish. Don't chase fp8/int4/adaptive-k while the floor is up
  — invisible or negative there (proven twice: `ab_adaptive`, Alyssa's fp8).

## One line
**1000 = two isolation-testable, graph-captured kernels (fp8-K5 at e→1, and the NVLS all-reduce ≤4 µs) +
CUDA graphs + a scheduler-free loop + small-tree spec.** Bank spec ~300 now; the NVLS kernel is the make-or-break
pivot; the persistent megakernel is a negligible (0.06 µs) polish, not the gate. No new engine required.
