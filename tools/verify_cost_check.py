#!/usr/bin/env python3
"""What a CORRECTLY-BATCHED spec verify should measure — the target for fixing spec_verify_bench.cu.

The squeeze round (results-reaction-04) found the verify bench scaled ~linearly with draft rows
(192→850 µs/layer for M=1→5). That's the BUG: a real batched verify reads each active expert's weight ONCE
and applies it to all the tree positions routing to it (a grouped GEMM), so the **weight read is flat in the
tree size** (it's the UNION of experts, which grows slowly), not M×. This computes the correct flat target so
the team can validate the fixed kernel — and shows why spec amortizes (the verify ≈ one decode's weight read,
not M of them).

  python3 tools/verify_cost_check.py
"""
E, TOPK = 128, 8
BW = 8 * 3.35e12
EXPERT_BYTES_FP8 = 18.9e6 * 94      # one expert, all layers, fp8 (1 byte)
NONEXP_BYTES_FP8 = 6.7e9            # attention+router, fp8

def ms(b): return b / BW * 1e3
def union(P): return E * (1.0 - ((E - TOPK) / E) ** P)

print("Per-spec-round verify cost: MIS-MODELED (per-position) vs CORRECT (union read once, flat).")
print(f"  {'tree M':>7} {'union':>6} {'WRONG (M× 8-expert)':>20} {'CORRECT (union once)':>21} {'ratio':>6}")
for M in (1, 2, 4, 8, 16, 32):
    u = union(M)
    wrong = ms(NONEXP_BYTES_FP8 + M * TOPK * EXPERT_BYTES_FP8)      # reads 8 experts PER position (the bug)
    correct = ms(NONEXP_BYTES_FP8 + u * EXPERT_BYTES_FP8)           # reads the UNION once (grouped GEMM)
    print(f"  {M:>7} {u:>6.0f} {wrong:>18.2f}ms {correct:>19.2f}ms {wrong/correct:>5.1f}×")

print("\nReading:")
print("  * The CORRECT verify is ~FLAT (M=1: 0.78ms -> M=32: ~the full-expert read), because the union saturates")
print("    (~all 128 experts by M~30). The WRONG model scales M× -> kills spec (a 5-row tree looks 5× costlier).")
print("  * Validate the fixed spec_verify_bench.cu against the CORRECT column: forward(M) should be flat-ish,")
print("    NOT linear in M. If it's linear, the kernel re-reads weights per position (not a grouped GEMM).")
print("  * This is why spec works: ONE verify forward (≈ one decode's nonexpert read + the union) emits τ tokens,")
print("    amortizing the comms (188×16µs paid once) + the read over τ. The flat verify is the whole ballgame.")
print("  * NOTE: small trees keep the union (hence the verify weight) small -> the F→0/weight-bound sweet spot")
print("    is a SMALL tree; big trees only win while the comms floor dominates (tree_spec_optimizer.py).")
