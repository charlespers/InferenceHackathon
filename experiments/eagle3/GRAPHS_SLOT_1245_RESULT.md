# EAGLE3 CUDA-graphs slot (12:45 UTC, 2026-06-20) — result + honest postmortem

Run: `slot_eagle3_graphs.sh`, MODE=graphs (`enforce_eager=False`), EAGLE3 `num_speculative_tokens=3`,
`draft_tensor_parallel_size=1`, RedHat head, FP8 target (Qwen3-235B-A22B-Instruct-2507-FP8), TP8 + EP,
gpu-mem-util 0.85, `measure_baseline.py --decode 256 --repeats 3`. Lock `LOOP-A(eagle3)` 12:45:03.

## Banked clean wins (real, robust)
1. **EAGLE3 + CUDA-graphs CAPTURE WORKS at k=3** — the documented graphs crash history (INTEGRATION
   §3) did **not** recur. `cudagraph_mode=FULL_AND_PIECEWISE`, capture_sizes [1,2], compiled and served
   cleanly. This unblocks the graphs path as a deployment option.
2. **Acceptance is preserved under graphs**: live SpecDecoding metrics gave **τ_graphs ≈ 2.5–2.67**,
   per-position accept ≈ 0.75 / 0.5 / 0.33 (first-pos ~0.7–0.75) — **matching the eager measurement**
   (τ≈2.7, first-pos~0.75). As expected, acceptance is a model property, mode-independent. Lossless-
   consistent. (Source: Prometheus `/metrics` — the clean signal.)

## KEY NEGATIVE FINDING (surprising, needs the diagnostic)
The eagle3_graphs **decode ran at ~1–2 tok/s** (Prometheus active-window drafted ~0.9–1.2 tok/s;
measure_baseline didn't finish even 3×256 tokens in ~10 min). That is **~5× SLOWER than the same
harness measured eagle3_EAGER at (m_eagle3_eager.json: 10.05 tok/s, tpot 99 ms)**. So in THIS config,
**CUDA-graphs did not deliver the expected speedup for spec decode — it ran degenerately slow**,
contradicting the "graphs is where S shows / graphs ~5× eager" expectation.

Candidate causes (unresolved): graphs⊗spec interaction (graph replay overhead / per-step recompile
with the draft), `draft_tensor_parallel_size=1` serializing the head against the TP8 target, or the
FP8-MoE Triton path (DeepGEMM unavailable). Root cause is the next experiment, not a conclusion yet.

## Measurement-reliability caveat (the real blocker)
`measure_baseline.py` single-stream decode tok/s is **not trustworthy on this shared box**: prior
`m_baseline_eager.json` = **1.95 tok/s** (tpot 513 ms) where analyze.py's own reference says FP8+EP
should be ~64 tok/s — a ~30× miss, and a 5× spread vs eagle3_eager 10.05 that can't both be physical.
So **absolute-tok/s-derived S from this harness is unreliable**, independent of the graphs finding.

## Why no S this slot
The ~10-min slow eagle3 measure ate the :45–:00 window before `baseline_graphs` could run, so there is
**no matched graphs baseline** → no S, no parity gate. To respect the slot boundary I **killed my PIDs
and released the lock at 13:04** rather than let baseline_graphs' compile hog the next owner's :00–:45.
No eagle3_graphs tok/s json was banked (killed mid-measure; the number would have been untrustworthy
anyway). `results/eagle3_redhat/` eager numbers + this slot's τ/capture are the durable record.

## Next: 13:45 DIAGNOSTIC (separate "graphs broken" from "spec+graphs broken")
1. **baseline_graphs (NO spec), FAST** (`--decode 64`) FIRST. If ~30–60 tok/s → graphs path is fine and
   the slowdown is the **spec+graphs** interaction. If also ~2 tok/s → the **graphs path itself** is
   broken in this venv/config (then compare to a fresh eager baseline to confirm).
2. Then eagle3_graphs `--decode 64` for the matched ratio (warm torch.compile cache → faster startup).
3. If spec+graphs is confirmed slow, test `draft_tensor_parallel_size=8` and `k=2` as the likely fixes.
Robust metric: prefer Prometheus accepted/drafted + a short controlled decode over the noisy single
long stream.
