#!/usr/bin/env python3
"""The whole-engine MBU ceiling at B=1 — why "compute at 58-80% MBU -> 700-1024 tok/s" is unreachable
WITHOUT the megakernel. (LOOP-C, 2026-06-20. Adversarial validation of the load-bearing 1000-projection.)

Every 1000-path projection (ceiling-and-1000.md, squeeze-to-700.md, ladder) applies a PER-KERNEL MBU
(K5 experts: 45.7% measured -> 80% projected) to the WHOLE byte budget (2.74 GB/GPU -> 0.78ms at e=1).
But the MEASURED whole-engine kernels-floor is 10.3 ms (commit 5f1150f), i.e. whole-engine effective MBU
= 0.78/10.3 = ~7.6% -- not 58%. The gap is NOT un-tuned kernels: it's that most of the compute is
LATENCY-bound (fixed us/op x 94 layers), which MBU tuning cannot touch. This tool decomposes it and shows
the un-fused ceiling, the role of the megakernel, and what spec adds.

MEASURED anchors (cited):
- byte floor (fp8, TP8 /8): 2.74 GB / 3.35 TB/s = 0.818 ms  -> 1223 tok/s @ e=1   (squeeze-to-700 §1a)
- whole-engine kernels-floor: 10.3 ms                        (5f1150f, post router-fix)
- router (K4): 24 us/call x 94 = 2.26 ms  -- a 0.52 MB GEMV; at 3.35 TB/s its BYTES need 0.16 us, so
  ~2.26 ms is pure LATENCY (occupancy/dispatch), ~0.7% MBU. MBU tuning CANNOT reduce it.  (5f1150f)
- K1 (attn prologue, Qwen3 per-head q/k-norm x68 + RoPE + small projs): "44% of kernel time" (5f1150f)
  -- small WEIGHT bytes (8.9 MB/layer /8) but many small ops -> latency-bound. Treated as ESTIMATED.
- K5 experts measured e: 0.281 default -> 0.58 tuned (v3).                       (k5 microbench / squeeze)
"""

BYTE_FLOOR_MS = 0.818     # e=1, the only genuinely BW-bound part (experts + attn weights + lm_head)
KERNELS_FLOOR_MS = 10.3   # measured whole-engine
ROUTER_MS = 2.26          # MEASURED latency-bound floor (router, 0.7% MBU, MBU-immune)
COMMS_NVLS_MS = 0.72      # NVLS full out-of-place (3.84us x 188); ~0 if fully overlapped

def tput(ms): return 1000.0 / ms

print("=== The conflation: per-kernel MBU vs whole-engine MBU ===")
eng_eff_mbu = BYTE_FLOOR_MS / KERNELS_FLOOR_MS
print(f"  byte floor (e=1) {BYTE_FLOOR_MS:.2f} ms ; measured kernels-floor {KERNELS_FLOOR_MS:.1f} ms")
print(f"  => whole-engine EFFECTIVE MBU = {eng_eff_mbu*100:.1f}%  (NOT the K5-isolated 58%)")
print(f"  squeeze-to-700 projects 'compute @58% MBU = 1.41ms = 709 tok/s' by applying K5's MBU to the")
print(f"  WHOLE budget -- but {KERNELS_FLOOR_MS-BYTE_FLOOR_MS:.1f} ms of the measured floor is NOT BW-bound.\n")

# Decompose the 10.3ms into BW-bound (scales with MBU) vs latency-bound (fixed, MBU-immune).
# BW-bound part at the measured tuned e=0.58: byte_floor/0.58
bw_bound_at_58 = BYTE_FLOOR_MS / 0.58
latency_floor = KERNELS_FLOOR_MS - bw_bound_at_58
print(f"=== Decomposition of the {KERNELS_FLOOR_MS} ms kernels-floor ===")
print(f"  BW-bound (experts+attn-wt+lm_head) @ e=0.58 : {bw_bound_at_58:.2f} ms  (MBU tuning shrinks this)")
print(f"  LATENCY-bound (router 2.26 + K1 attn + norms + small-ops x94 + gaps): {latency_floor:.2f} ms")
print(f"    of which MEASURED router = {ROUTER_MS:.2f} ms; rest (K1/norms/gaps) ~ {latency_floor-ROUTER_MS:.2f} ms (estimated)\n")

print("=== Ceilings (compute-only; add comms separately) ===")
def ceiling(label, bw_e, lat_ms):
    bw = BYTE_FLOOR_MS / bw_e
    comp = bw + lat_ms
    print(f"  {label:54} {comp:6.2f} ms -> {tput(comp):5.0f} tok/s compute-only")
    return comp
ceiling("today (e=0.58, full latency floor)", 0.58, latency_floor)
ceiling("MBU tuning ONLY: experts e->1.0, latency floor stays", 1.0, latency_floor)
print(f"    ^ MBU tuning alone is capped here -- the latency floor dominates; ~58-80% whole-engine is impossible.")
# Conservative measured-only bound: even if EVERYTHING but the measured router were free:
cons = BYTE_FLOOR_MS + ROUTER_MS
print(f"  CONSERVATIVE measured-only (byte floor + ONLY the measured router latency): "
      f"{cons:.2f} ms -> {tput(cons):.0f} tok/s")
print(f"    ^ the router ALONE (measured) caps un-fused compute at <= {tput(cons):.0f} tok/s, vs the projected 709.\n")
ceiling("MEGAKERNEL fuses away the latency floor (e->1)", 1.0, 0.0)
print(f"    ^ ONLY fusion (router/norms/small-ops folded into the persistent kernel) unlocks the BW regime.\n")

print("=== Sensitivity: un-fused compute ceiling vs the (estimated) latency floor ===")
print(f"  {'latency floor ms':>16} {'compute@e=1':>12} {'tok/s':>7}")
for lat in (2.26, 4.0, 6.0, latency_floor, 8.0):
    comp = BYTE_FLOOR_MS + lat
    print(f"  {lat:16.2f} {comp:12.2f} {tput(comp):7.0f}")
print(f"  -> robust: even at the most generous (only the measured router, 2.26ms), un-fused caps ~{tput(BYTE_FLOOR_MS+2.26):.0f};")
print(f"     at the realistic full floor, ~{tput(KERNELS_FLOOR_MS):.0f}-{tput(BYTE_FLOOR_MS+4):.0f}. NONE approach 709/1024.\n")

print("=== What it means for 1000 (compute + comms, x spec) ===")
def path(label, comp_ms, comms_ms, spec):
    t = (comp_ms + comms_ms) / spec
    print(f"  {label:50} {comp_ms:5.2f}+{comms_ms:.2f} /{spec} = {t:5.2f} ms -> {tput(t):5.0f} tok/s")
path("un-fused (lat floor) + NVLS, NO spec",       BYTE_FLOOR_MS/0.58 + latency_floor, COMMS_NVLS_MS, 1.0)
path("un-fused + NVLS + spec x3",                  BYTE_FLOOR_MS/0.58 + latency_floor, COMMS_NVLS_MS, 3.0)
path("MEGAKERNEL (lat->0, e=1) + NVLS, NO spec",   BYTE_FLOOR_MS, COMMS_NVLS_MS, 1.0)
path("MEGAKERNEL + NVLS + spec x2",                BYTE_FLOOR_MS, COMMS_NVLS_MS, 2.0)
print("""
VERDICT:
- The whole-engine MBU ceiling at B=1 in the UN-FUSED engine is ~8-15%, NOT 58-80% -- because ~7-9 ms of
  the measured 10.3 ms compute is LATENCY-bound (router 2.26 ms MEASURED + per-head attn norms + ~6 small
  ops x 94 layers), which MBU tuning cannot touch. The "compute @58-80% MBU -> 709-1024" projection applies
  a single big-kernel MBU to the whole budget and silently assumes the latency floor away.
- Conservative MEASURED-only bound: the router alone caps un-fused compute at ~329 tok/s; the full latency
  floor caps it at ~100-150. So "kernels+comms alone -> 650-1024" is UNREACHABLE by MBU+comms tuning.
- The MEGAKERNEL is therefore the GATING PRECONDITION, not one lever among many: its real value is
  collapsing the ~7-9 ms per-op LATENCY floor (router/norms/small-ops fused into one persistent kernel),
  which is ~3x larger than the comms term it also helps. Only after fusion does the BW regime (and 80% MBU,
  and NVLS, and spec) become the story.
- For 1000: megakernel (collapse latency floor) is REQUIRED; then NVLS + spec stack on the ~1280 BW base.
  Without the megakernel, even perfect MBU + NVLS + spec x3 stays ~300-450. This sharpens "1000 needs spec"
  -> "1000 needs the MEGAKERNEL first; spec/NVLS stack on top."
""")
