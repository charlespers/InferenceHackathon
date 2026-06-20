# 13:45 graphs DIAGNOSTIC — result: CUDA-graphs gives 6.8× on plain decode (the 12:45 bug confirmed)

Run: `tools/slot_graphs_diag.sh`, `--decode 64`, isolated `eagle3-venv` (vLLM 0.11.2), FP8 target,
TP8 + EP. Two of three arms completed; `eagle3_graphs` hit the `:53` time-guard and was deferred.

## Numbers (results/eagle3_diag/)
| arm | tok/s | tpot | note |
|---|---|---|---|
| baseline_eager  (no spec, `--enforce-eager`) | **4.4**   | 227 ms | floor-dominated |
| baseline_graphs (no spec, graphs)            | **29.92** | 33 ms  | floor-dominated (5.5% roofline) |
| eagle3_graphs   (spec, graphs, capture fix)  | —         | —      | skipped (`:53` guard); next slot |

**S_graph_plain = 29.92 / 4.4 = 6.8×.**

## What it establishes
1. **CUDA-graphs DOES speed up plain B=1 decode — 6.8×** (even beats Alyssa's ~5× estimate). The eager
   regime is floor-bound (tpot 227 ms = launch/host/comms); graphs collapse that floor → 33 ms.
2. **The 12:45 "graphs is slow" (~2 tok/s) is now DEFINITIVELY the cudagraph mis-capture bug** — plain
   decode (capture sizes [1,2], valid for no-spec) graphs fine and is fast; the 12:45 spec run had
   capture sizes that weren't multiples of (k+1) → zero decode graphs → un-graphed → slow. Confirmed by
   contradiction: same venv, same target, graphs ON → 6.8× when capture is valid.

## Caveats (honest)
- **Absolutes are not representative.** This isolated `eagle3-venv` runs ~10–15× slower than the team's
  production vLLM (baseline_graphs 29.9 vs the team's ~64 FP8 / 85.7 bf16) — untuned FP8-MoE Triton
  config (`fused_moe.py:886` "Using default MoE config… sub-optimal") + EP + 0.11.2. **Use the within-venv
  RATIO (6.8×), not the absolute 29.9.** Even graphs-on is still floor-dominated here (5.5% roofline), so
  there's more floor to remove with a tuned config.
- **No spec arm yet** → no S_spec, no V=τ/S, no parity this slot. Deferred to the next :45.

## Next (14:45 slot)
Run `eagle3_graphs` (spec, graphs, capture sizes = multiples of k+1) vs the banked `baseline_graphs` →
S_spec + V=τ/S. With Charles's measured **flat-in-k GEMM verify** (`54d276a`: verify cost flat to the
16-wide fp8 tile, "pad to 16, maximize E[accepted]"), the expectation is S_spec ≈ τ (the verify doesn't
grow with the tree) and **V ≈ 1 → route-aware union-shrinking NO-GO at B=1** (the verify isn't
union-bound). The 14:45 V is my own end-to-end confirmation of that go/no-go.
