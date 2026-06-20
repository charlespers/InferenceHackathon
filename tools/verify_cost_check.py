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

print("\nReading (precise — corrects both 'per-position' AND 'flat'):")
print("  * The CORRECT verify is NOT flat — it GROWS with the UNION (0.78→7.66ms, M=1→32). For SMALL trees")
print("    (M≤4) it's ~the same as the per-position model (union≈8M before saturation) — so the squeeze round's")
print("    'linear' M=1→5 wasn't purely a bug; the union really does grow. The two models only DIVERGE at big")
print("    trees (M=32: union saturates at 112, so union-once is 2.2× cheaper than per-position).")
print("  * What's FLAT (amortized by spec) is the COMMS (188×16µs, paid once per verify) + the nonexpert read.")
print("    What GROWS is the union expert weight — the spec verify-tax (spec_floor_model.py / spec-decode-moe-tax).")
print("  * So validate the fixed kernel against the UNION-once column (grows with union, saturating), NOT a flat")
print("    target and NOT M×8 per-position. If it's M×8 linear past M~8, it's re-reading per position (bug).")
print("  * Spec works because the COMMS amortizes (the dominant barrier-bound floor /τ); the union weight is the")
print("    tax that caps it -> the optimal tree balances τ vs union (small tree as the comms floor falls).")
print("  * The team's realized EAGLE3 speedup is the ground truth — it nets this union tax automatically.")
