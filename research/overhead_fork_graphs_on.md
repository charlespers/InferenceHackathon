# The 7ms "overhead" is kernel-bound, not launch-bound — the baseline is already graphs-ON

**LOOP-C, 2026-06-20.** Adversarial-validation companion to Charles's overhead attribution (he owns
`docs/overhead-attribution.md` + `tools/backout_floor.py`; this supports, doesn't replace). Tool:
`tools/overhead_fork.py` (re-runnable). **Net: the ~7ms overhead is kernel-low-e (→ K5), the ladder's
"+CUDA graphs recovers 3.5ms launch" rung is illusory, and sampling/LM-head is killed.**

## The fork (both fit TPOT 11.67ms, opposite #1 levers)
The non-comms budget `R = TPOT − comms(~2.5ms) − kv = 9.10ms` must split into `compute(=weight_floor/e)
+ launch + host`. Two stories fit:

| story | implied e | launch+host | #1 lever |
|---|---|---|---|
| **A** (`ladder_to_1000.py`: launch 3.5 + host 1.5 + kernel-ineff 2.0) | 0.44 | **5.56 ms** | fast-path / graphs / fused sampling |
| **B** (E0 reconcile + measured e) | 0.18 | **0.43 ms** | **K5 kernels (recover ~3×)** |

## Why B wins (two GPU-free facts)
1. **The baseline is CUDA-graphs-ON** (vLLM default — you must pass `--enforce-eager` to *disable*;
   `config-sweep.md:59`, `adaptive_topk/PLAN.md:369`). LOOP-A's eager runs are an EAGLE3-specific choice
   (graph instability on MoE+EP), *not* the 85.7 baseline. Under graph replay there is **one launch per
   decode step, not per kernel** → pure kernel-launch overhead ≈ 0 in the baseline.
   - ⇒ The ladder's **rung 1 "+ CUDA graphs (launch 3.5ms → 0)" is ILLUSORY** — that recovery was already
     banked at vLLM's default. You can't turn on graphs that are already on.
   - ⇒ Story A's 5.56ms would have to be **uncaptured HOST** work (Python scheduler, detok, torch.compile
     guard syncs), not launch. **5.56ms of host escaping the graph is not credible** — CUDA graphs exist
     precisely to collapse that. So Story A is implausible as stated.
2. **Measured whole-model e ≈ 0.16–0.19** (K5 microbench / `overhead-attribution.md` candidate-2) lands
   right on Story B's 0.18. The kernels really do run at ~3× below roofline at B=1 (skinny MoE expert GEMVs,
   ~3.5% occupancy per the tp_degree thread).

**Conclusion:** the 7ms is **kernel sub-roofline**, confirming K5 (e→1) is the correct top lever for it.
This *supports* Charles's K5 emphasis; it corrects the ladder's framing (it's not a free graphs win) and
removes the launch candidate.

## Candidate KILL — sampling / LM head (151936 vocab)
- LM head weight read, TP8-sharded: 151936×4096×2B /8 / 3.35 GB/ms = **0.046 ms**.
- Logits argmax/softmax read (152k × fp32): **0.0002 ms**.
- **Total ≈ 0.05 ms = 0.4% of TPOT.** Even an *un-sharded* LM head is 0.37ms. **Sampling is NOT the
  elephant** — drop it from the 7ms candidate list (it only matters if fused-sampling is free anyway).

## What this leaves for the resolver (Charles's, GPU-gated)
The only surviving ambiguity is *kernel-low-e (B)* vs *a smaller-but-real uncaptured-host residual*. Both
of Charles's methods address it: `backout_floor.py` (F from LOOP-A's spec run, no Nsight) bounds the floor
fraction; **E-attr** (Nsight `cuda_gpu_kern_sum` achieved-BW + idle-gap, plus an eager-vs-graphs A/B) splits
kernel-busy-at-low-BW (B) from idle-gaps (host) definitively in one slot. Prediction to falsify: the
eager-vs-graphs delta is *small* (graphs already on) and the kernels show *low achieved BW* → kernel-bound.

## CONFIRMED + SELF-CORRECTION (2026-06-20, LOOP-A's 13:45 diag, commit 967ccdf)
Measured eager-vs-graphs on plain B=1 decode (isolated venv): **eager 4.4 tok/s (227ms) → graphs 29.92
(33ms) = 6.8×**; production matches (eager ~10 → graphs 85.7 ≈ 8.5×, commit 27be031).
- **CONFIRMED — the load-bearing claim holds:** graphs deliver a *huge* win and it is **already banked in
  the 85.7 baseline** ⇒ the ladder's "rung 1: +CUDA graphs" is **illusory** (no second graphs win to take).
- **SELF-CORRECTION:** my "prediction to falsify" was wrong as worded — the eager→graphs delta is **large**
  (6.8×), not small. The reason refines the fork: the eager floor is **HOST-dominated** (per-step Python
  scheduler/sample/detok, ~190ms in the slow venv / ~88ms implied production), not kernel-launch (~1.13ms,
  k6). So the "overhead" the ladder split as "launch 3.5 + host 1.5" is really **mostly HOST, removed by
  graphs**. This does NOT change the conclusion: that host floor is *gone* post-graphs, so the residual 7ms
  in the 85.7 (graphs-on) baseline is **kernel + comms (Story B)** — what graphs can't touch. Story A's
  "launch/host" is real but it lives in the eager regime the baseline already escaped.
- **Caveat:** this diag is a host-dominated *eager-vs-graphs* split; it does NOT itself prove the post-graphs
  residual is kernel-vs-(small)host. That still needs the production eager-vs-graphs A/B at the achieved-BW
  level (E-attr) — the diag's 5.5%-roofline graphs-on number is *consistent* with kernel-bound but is untuned
  Triton, not the production engine. Net: Story B stands on the e≈0.18 evidence; the illusory-rung point is now
  empirically nailed.
