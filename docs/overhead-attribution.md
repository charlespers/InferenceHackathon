# The real TPOT is overhead-dominated — attribute the 7 ms before optimizing

A correction to `results-reaction-01.md`'s "comms is #1." Decomposing the **measured** bf16-TP8 TPOT
(11.67 ms) against the physics model (at the measured 16 µs comms) shows comms is only the *second* term:

| term | ms | share | source |
|---|---|---|---|
| **unmodeled overhead** | **~7.04** | **60%** | NOT in the model — see below |
| comms (2 all-reduce/layer × 94 × 16 µs) | 3.01 | 26% | `nccl-tests` microbench (a *lower bound*) |
| weight read (bf16, TP8, /8) | 1.61 | 14% | physics |
| kv read (ctx 512) | 0.01 | ~0 | physics |
| **measured total** | **11.67** | | vLLM bf16 TP8, greedy, B=1 |

**The elephant is the 7 ms, not the 3 ms comms — and the weight term (where int4/fp8 live) is the *smallest*
at 14%.** Halving it with fp8 saves ~0.8 ms = ~7% of TPOT. So the lever order from the *data* is:
**layout (TP8) ≫ the 7 ms overhead ≈ comms ≫ weight.** This recolors the whole plan: quantization is a minor
lever right now; the prize is the overhead + comms + the TP8 layout.

## What is the 7 ms? Four candidates (graphs are already ON, so launch is ~0)
1. **In-engine comms > microbench.** The 16 µs is `nccl-tests` best-case; real in-forward all-reduce isn't
   perfectly overlapped/pipelined, so true comms could be 2–3× → eats part of the 7 ms.
2. **MoE kernel inefficiency.** vLLM's `fused_moe` at B=1 runs the expert GEMV at low utilization — the whole
   premise of the K5 work, which measured **e=0.46 tuned vs ~0.16 whole-model** (≈3× headroom). Plausibly
   3–4 ms of the 7 ms is the experts + attention kernels running well below roofline.
3. **Sampling / detokenize / Python per-token.** Logit→token sampling over a 152k vocab, detok, and the
   per-step host work — usually small with graphs, but non-zero.
4. **torch.compile guards / residual host syncs** not captured by the graph.

## Attribute it (one slot, ~3 runs) — `E-attr`
Don't optimize the 7 ms blind; split it first.
```bash
# 1) Nsight Systems timeline of ~20 decode steps — the single best diagnostic (kernel vs NCCL vs gaps):
nsys profile -t cuda,nvtx,nccl -d 20 -o /root/decode_trace \
  python3 bench/measure.py --base http://localhost:8001 --model q --ctx 512 --decode 64
nsys stats --report cuda_gpu_kern_sum,nccl_sum /root/decode_trace.nsys-rep | head -40
#    -> % of a step in MoE/attn GEMM kernels  vs  NCCL all-reduce  vs  idle gaps (launch/host)
# 2) graphs vs eager (isolates launch/host overhead the graph hides):
#    relaunch with --enforce-eager, re-measure TPOT; delta = what graphs are saving (should be large if launch-bound)
# 3) comms share: re-measure with E0b NCCL tuning; TPOT drop = the comms portion of the 7 ms
```
**Decision from the trace:**
- NCCL dominates the timeline → the 7 ms is mostly comms → **E0b (NCCL LL/NVLS) is #1**, and reducing the
  collective *count* (fold expert-reduce, fewer all-reduces) matters.
- MoE/attn kernels dominate at low achieved-BW → the 7 ms is **kernel inefficiency** → the **K5 kernels**
  (`k5_experts_warp.cu`, e=0.46 vs vLLM's ~0.16) and a custom-kernel path (the cudarc engine) are #1.
- Big idle gaps → host/launch/sampling → fast-path + fused sampling + tighter graph capture.

## Why this matters for the kernel work
If the trace shows MoE kernels at ~0.16 util, that is **direct on-box evidence that the K5 work (100× over
scalar, e=0.46) is the lever** — vLLM's `fused_moe` is leaving ~3× on the table at B=1, and wiring a tuned
expert GEMV (via the cudarc Rust engine, or a vLLM custom kernel) recovers it. The K5 microbench already
measured the achievable e; `E-attr` measures vLLM's actual e to size the gap.

## Updated lever order (data-grounded)
1. **TP8 layout** (structural; EP weight 3.41 + comms 4.51 vs TP8 1.61 + 3.01) — already the plan.
2. **The 7 ms overhead** — attribute via `E-attr`, then attack comms (E0b) and/or kernels (K5) per the trace.
3. **Comms tuning (E0b)** — overlaps (2) if the trace says comms.
4. **Weight: fp8 (E2b) then int4 (E7)** — real but the *smallest* term (~14%); grows in share as 2–3 shrink.
5. **Spec decode** — orthogonal multiplier on whatever the above achieve.
