# The persistent megakernel — the endgame B=1 architecture (how you actually reach the ~2000 ceiling)

`absolute-ceiling.md` says the hard residual (750→2000 tok/s) is killing the 60% overhead + driving comms→0.
The CUDA-graph fast-path (`b1-fast-path-design.md`) removes per-*step* host work but still launches ~188
kernels per step inside the graph, each with its own grid-launch + tail + the host-launched NCCL between them.
The **persistent megakernel** removes that too: **one kernel for the entire 94-layer decode step.** It's the
most aggressive attack on the floor and the architecture the cudarc engine should ultimately target.

## The idea (one kernel, all layers, no relaunch)
A single grid is launched **once** and stays resident, looping over the 94 layers internally. Within the
kernel, for each layer: load this layer's weights HBM→regs/smem, do attention + MoE in-place, do the
collectives **device-side**, advance to the next layer — never returning to the host, never relaunching.

What that deletes from the floor:
| floor component (overhead-attribution.md) | how the megakernel removes it |
|---|---|
| **per-kernel launch × ~188/step** | one launch per step (or per *token*, persistent across tokens) → ~0 |
| **inter-kernel idle gaps / scheduler** | none — the kernel never yields to the host |
| **host-launched NCCL all-reduce** | **device-initiated** NVSHMEM/multimem all-reduce *inside* the kernel |
| **graph replay overhead** | no graph — the kernel is the loop |
| **HBM round-trip of the tiny B=1 activation between layers** | activation stays in registers/smem across layers |

What's left is the **irreducible physics**: the weight read (HBM-bandwidth-bound) + the device-side collective
latency. That's the roofline — i.e. this is the mechanism that takes the engine from ~37% to ~100% of the
fp8+spec ceiling.

## Why B=1 is uniquely suited to it
- **The activation is tiny** (1 token × 4096 = 8 KB) → it fits in registers/smem, so the whole residual stream
  lives on-chip between layers. At large batch this is impossible (activations are huge) — but B=1 is exactly
  where a megakernel is feasible. (This is the *inversion* theme again: the thing that's hard at throughput is
  easy at B=1.)
- **The work is memory-streaming** (AI≈1), so the kernel is a structured HBM→compute pipeline — `cp.async`
  double-buffering the next layer's weights while computing the current (same MLP lever as `k5-tuning-roadmap.md`,
  now spanning layer boundaries).
- **No batch to schedule** → the per-step host scheduler (the ~1–3 ms fixed floor, `fixed-overhead-floor.md`)
  simply doesn't exist; the host does one launch and waits.

## The hard parts (why it's the endgame, not the first step)
1. **Device-side collectives.** The all-reduce must run *inside* the kernel without a host launch — NVSHMEM /
   `multimem` / NVLS one-shot with grid-wide synchronization. This is the crux and the highest-risk piece
   (`comms_floor.md` lever #1, but now *in-kernel*). Get this and comms leaves the host critical path entirely.
2. **Occupancy vs weights resident.** The kernel must keep enough warps live to hide HBM latency while
   streaming 22B active params/step through smem — careful tiling so no layer's weights overflow the budget.
3. **MoE control flow in-kernel.** The router picks 8 of 128 experts per token; the megakernel must gather the
   selected expert weights (data-dependent addresses) inside the loop — a persistent grid with a work queue.
4. **Spec composition.** The verify forward (the W×D tree) is the same megakernel run over W×D positions —
   batched, so it *raises* AI (tensor-core-friendly, `why-spec-wins.md`). The megakernel + spec compose: one
   resident kernel verifies the tree, emits τ tokens, no host round-trip per token.

## Where it sits in the plan
- **Last and hardest** (the 750→2000 closer), after spec/comms/kernels land and the CUDA-graph fast-path is in.
- **It's the cudarc engine's true endgame** — vLLM cannot express a whole-model persistent megakernel; a custom
  engine can. It subsumes K5 (the expert GEMV becomes the MoE stage of the megakernel) and the fast-path (the
  loop *is* scheduler-free) and the in-graph comms (now in-kernel).
- **Staging:** (a) CUDA-graph fast-path (captures kernels, host-side); (b) fuse attention+MoE per layer into one
  kernel (halve the launch count); (c) device-side all-reduce (the crux); (d) full persistent megakernel over
  all 94 layers. Each stage is independently measurable (`E-attr` idle-gap delta) and bankable.

## One line
The megakernel is the architecture where the floor *becomes* the roofline: one resident kernel, weights streamed
with cp.async across layer boundaries, activations on-chip, collectives device-side — the only way the 7 ms
overhead actually goes to ~0 and B=1 reaches physics.
