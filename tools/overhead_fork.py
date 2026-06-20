#!/usr/bin/env python3
"""The 7ms 'overhead' fork: is it LAUNCH/HOST-bound or KERNEL-low-e-bound? (they imply opposite #1 levers)

LOOP-C, 2026-06-20. SUPPORTING Charles's overhead attribution (he owns docs/overhead-attribution.md +
tools/backout_floor.py). This is the adversarial-validation companion: the 7ms 'overhead' term has TWO
decompositions that BOTH fit the measured TPOT but name OPPOSITE top levers — and the team's two artifacts
silently assume different ones. This tool makes the fork explicit, applies one GPU-free correction (the
baseline is graphs-ON, so the ladder's '+CUDA graphs recovers ~3.5ms launch' rung is illusory), and KILLS
the sampling/LM-head candidate from physics.

  measured: TPOT 11.67ms @ 85.7 tok/s, bf16-TP8, B=1, greedy, ctx 512, CUDA graphs ON (vLLM default;
            config-sweep.md:59, adaptive_topk/PLAN.md:369 — you must pass --enforce-eager to DISABLE).

  python3 tools/overhead_fork.py
"""

TPOT_MS   = 11.67
WEIGHT_FLOOR_MS = 20.9e9 * 2 / 8 / 3.35e12 * 1e3   # ~1.56 ms, active 20.9B bf16 TP8 at roofline (e=1)
COMMS_MS  = 2.5     # in-engine, from comms_floor_reconcile_e0.md (C~10-18us band -> ~2-3ms; midpoint)
KV_MS     = 0.07

# --- candidate KILL: sampling + LM head over the 151936 vocab (HBM BW = 3.35 GB/ms on H100) ---
VOCAB, HIDDEN, BW = 151936, 4096, 3.35       # BW in GB/ms (= 3.35 TB/s)
lm_head_gb        = VOCAB * HIDDEN * 2 / 1e9          # 1.245 GB total (bf16, tie=False, separate head)
lm_head_ms        = (lm_head_gb / 8) / BW             # TP column-parallel /8 -> per-GPU read time
logits_read_ms    = (VOCAB * 4 / 1e9) / BW            # fp32 logits read for argmax/softmax (152k)
sampling_total_ms = lm_head_ms + logits_read_ms
print("=== KILL candidate: sampling / LM head (151936 vocab) ===")
print(f"  LM head weight read (TP8-sharded): {lm_head_ms:.3f} ms")
print(f"  logits argmax read (152k fp32)    : {logits_read_ms:.4f} ms")
print(f"  TOTAL sampling+LMhead             : {sampling_total_ms:.3f} ms  = {sampling_total_ms/TPOT_MS*100:.1f}% of TPOT")
print(f"  -> NEGLIGIBLE vs the 7ms overhead (even an un-sharded LM head = {lm_head_gb/BW:.2f}ms). "
      f"Sampling is NOT the elephant. KILLED as a major term.\n")

# --- the FORK: both stories fit TPOT, opposite levers ---
# non-comms, non-kv budget that 'compute + launch + host' must fill:
R = TPOT_MS - COMMS_MS - KV_MS
print(f"=== The fork: non-comms budget R = {R:.2f} ms must split into compute(=weight/e) + launch + host ===")
print(f"{'story':46} {'impl. e':>8} {'launch+host ms':>14}  lever")
print("-"*92)
# Story A (ladder): kernels near-OK (e~0.44 -> compute 3.6ms), the rest is launch+host
eA = 0.44; compA = WEIGHT_FLOOR_MS/eA; lhA = R - compA
print(f"{'A (ladder): launch+host-bound (e=0.44)':46} {eA:8.2f} {lhA:14.2f}  fast-path/graphs/fused-sampling")
# Story B (E0 reconcile + measured e): kernels slow (e~0.18 -> compute 8.7ms), launch+host ~0
eB = 0.18; compB = WEIGHT_FLOOR_MS/eB; lhB = R - compB
print(f"{'B (reconcile): kernel-low-e-bound (e=0.18)':46} {eB:8.2f} {lhB:14.2f}  K5 kernels (recover ~3x)")
print(f"\n  measured whole-model e ~ 0.16-0.19 (overhead-attribution candidate-2 / K5) -> FAVORS story B.")

# --- the GPU-free correction ---
print(f"""
=== GPU-FREE CORRECTION (the discrete catch) ===
The 85.7 baseline is CUDA-graphs-ON (vLLM default). Under graph replay there is ONE launch per decode STEP,
not per kernel -> pure kernel-LAUNCH overhead is ~0 in the baseline. So the ladder's
  'rung 1: + CUDA graphs (launch ~3.5ms -> 0)'  is ILLUSORY for this baseline: there is no 3.5ms of
kernel-launch left to recover (it was already banked at vLLM's default). Story A's 'launch+host = {lhA:.1f}ms'
must therefore be HOST work uncaptured by the graph (python scheduler, detok, torch.compile guard syncs),
NOT launch. That is testable and is what splits A vs B.

CONSEQUENCE: the 7ms is kernel-low-e (story B, K5) OR uncaptured-host (story A minus launch) -> NOT launch.
The eager-vs-graphs delta will be SMALL if graphs are already capturing well (-> kernel-bound), LARGE if a
lot of host work escapes the graph (-> host-bound). Either way 'just turn on graphs' is not the free win the
ladder implies, because they are already on.

=== RESOLVERS (both Charles's; this only sharpens the question) ===
1. backout_floor.py (Charles) — F (floor fraction) from LOOP-A's V(k)=tau/S spec run, NO Nsight. F high =>
   floor-dominated; the launch-vs-kernel-vs-host split still needs (2).
2. E-attr (Nsight nccl_sum + cuda_gpu_kern_sum + idle-gap) over ~20 decode steps + an eager-vs-graphs A/B:
   idle-gaps => host (story A); kernel-busy-at-low-achieved-BW => kernel (story B). ONE slot resolves it.
""")
