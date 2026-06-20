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

## The honest risk assessment
- **Stage 3 (NVLS) is the make-or-break.** If the box's multimem in-switch reduce hits ~1 µs, 1000 is lossless
  and tight (3% margin). If it floors at ~4 µs, comms is 0.75 ms and **1000 needs a lossy lever** (depth/int4).
  Test this kernel *first and in isolation* — it gates everything.
- **vLLM tops out ~400–600** (Stages 0–2 + a graph-captured NVLS): graphs leave a host residual and can't host a
  whole-model persistent kernel. **The 600→1000 leap is Stages 4–5 in the cudarc engine — that's the real ask.**
- **Order discipline:** spec + prefix-cache now (bank 3.5×); fp8 only *with* the floor work; NVLS kernel is the
  pivot; the megakernel is the finish. Don't chase fp8/int4/adaptive-k while the floor is up — they're invisible
  or negative there (proven twice: `ab_adaptive`, Alyssa's fp8).

## One line
Five stages: bank spec (~300) → `E-attr` → fp8+fused-dequant → **the NVLS all-reduce kernel (the crux, test it
in isolation first)** → scheduler-free loop → persistent megakernel = ~1.0 ms. vLLM gets ~half; the cudarc
megakernel is the 600→1000 finish.
