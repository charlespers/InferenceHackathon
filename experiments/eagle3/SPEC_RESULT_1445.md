# EAGLE3 spec measurement (14:45 slot, run early 14:41) — vLLM spec ≈ 1× at B=1; native M=k verify is the path

Run: `slot_spec_diag.sh` (no-wait), `--decode 64`, isolated `eagle3-venv` (vLLM 0.11.2), FP8 target,
TP8 + EP, **capture-size FIX** applied (`cudagraph_capture_sizes:[4,8,16]` = multiples of k+1).

## The one clear WIN: the capture-size fix is confirmed
- `eagle3_graphs` went from **12:45's un-graphed ~2 tok/s → 66.92 tok/s** — a ~33× jump from one config fix.
- The vLLM log proves it: `Capturing CUDA graphs (decode, FULL): 100% 1/1` (**1 decode graph captured**) vs
  12:45's `0it` (zero), and the "No valid cudagraph sizes" warning is GONE. Root cause + fix validated.
- **Banked, real deployment fix:** EAGLE3 graphs needs `cudagraph_capture_sizes` = multiples of
  `(num_speculative_tokens+1)`, else the spec decode runs un-graphed.

## The sobering finding: vLLM EAGLE3 spec ≈ 1× at B=1 (no meaningful speedup)
Matched, same-warm-run decode tok/s (the absolutes are noisy ±5–10%, so use the RATIOS):

| config | tok/s | matched S_spec |
|---|---|---|
| baseline_graphs (no spec)      | 62.11 / 67.48 | — |
| eagle3 k=3 **draft_tp=1** (spec) | 66.92 | 66.92/62.11 = **1.08×** |
| eagle3 k=3 **draft_tp=8** (spec) | 63.41 | 63.41/67.48 = **0.94×** |

**S_spec ≈ 1.0 ± 0.07 — vLLM spec barely beats no-spec at B=1, and `draft_tp=8` does NOT help.** τ = 2.52
(first-pos 0.72), lossless-consistent — so acceptance is fine; the *throughput* gain is the problem.

### Why (the crux)
From wall-clock: a vLLM spec round emits τ=2.52 tokens but costs **~2.34–2.5× a single decode step**
(V = τ/S ≈ 2.34; cross-check: eagle3 63 tok/s ÷ τ2.52 = 25/s → 40 ms/round vs baseline 16 ms/step =
2.5×). **vLLM's verify forward over the k+1 positions is NOT flat** — it costs ~(k+1)× a decode step,
so the τ multiplier is cancelled by the per-round cost. This is the OPPOSITE of the **flat M=k GEMM
verify** Charles measured natively (`spec_verify_forward_gemm.cu`, T16/T1≈1.003). vLLM does not realize
the tensor-core "double win."

(Caveat: `diag_analyze.py` back-solved a wide "verify union → route-aware GO" from V — that's a
MIS-ATTRIBUTION; it models V as all verify-union cost and ignores the draft + the non-flat verify. The
high V here is the non-flat verify + draft, not a wide expert union. Route-aware union-shrinking NO-GO
still stands — shrinking the union won't lower a non-flat-verify / draft-bound V. Fixing the analyzer.)

## Implication — this VALIDATES the native-engine strategy
- The projected **2–2.8× spec speedup is real but only reachable via the NATIVE M=k-GEMM-verify-core**
  (assembly-plan **M1**), because only there is the verify flat. **vLLM at B=1 won't deliver it.**
- **vLLM EAGLE3 is the lossless REFERENCE (parity oracle), not the speedup vehicle at B=1.**
- So: build M1 (decode_step_tp8 GEMV→GEMM + M=k causal attention) — that's where the spec win lives. My
  `engine/native/spec_accept.h` (M3) + the wiring plan are ready to plug into it.

## Numbers banked
`results/eagle3_diag2_dtp1/` (eagle3 66.92 + baseline 62.11), `results/eagle3_diag2_dtp8/` (eagle3 63.41).
