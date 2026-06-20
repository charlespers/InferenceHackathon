# First-principles frontier to 1000+ — what kog.ai proves, and the levers beyond the plan

## What kog.ai actually does (researched, 2026-06)
Kog Inference Engine (KIE): **3,000 tok/s/request on 8×MI300X, 2,100 on 8×H200, batch-1.** Two pillars:
1. **Monokernel runtime** — the whole token generation is ONE persistent GPU program (a loop on-device),
   not a sequence of per-op kernel launches. Eliminates launch overhead AND lets comms/compute overlap
   *inside* the kernel.
2. **KCCL** — their own in-kernel collectives (not NCCL), hand-written CUDA+PTX. Budget: a few hundred µs/token.
Refs: blog.kog.ai "single-kernel latency-optimized engine on MI300X"; blog.kog.ai "3,000 tok/s per request"; kog.ai.

## First-principles ceiling (from our `absolute-ceiling.md`)
B=1 decode is HBM-byte-bound: t = (active bytes / 8) / (HBM·MBU). Sharded fp8 floor (e=1, comms=0): **~1280
tok/s**; int4 **~2560**. So:

- **Reaching 1000 is ASSEMBLY, not invention** — get to ~78% of the fp8 floor. The vehicle is the **monokernel**
  (kog's pillar 1) + our NVLS (pillar 2). We have every piece already:
  - NVLS in-switch all-reduce **3.84µs** (validated) — the KCCL analog.
  - k1–k5 sharded kernels validated; k6 whole-step CUDA-graph compiles+runs (755 nodes).
  - LOOP-C's deferred-overlap schedule (NVLink reduce ∥ HBM weight stream).
  → Fuse them into ONE persistent kernel: no per-token launches, comms hidden in-kernel. This is the
    74.5 → ~1000 path, and it's exactly kog's architecture.

- **Exceeding it (kog's 3000-class on OUR 22B-active model) requires CUTTING THE BYTE FLOOR.**

## Novel lever A (beyond the settled int4 plan): sub-int4 EXPERT quantization
First-principles, MoE-specific:
- **Experts are 66% of the bytes** (14.2 of 21.6 GB/token) and the **most quant-tolerant** part: 128 experts
  each fire ~6% of the time → huge inter-expert redundancy that dense weights lack.
- Quantize **experts to ~2 bits** (per-group scales + a few fp8 outliers; keep attention/router/lm_head at
  fp8 — small + accuracy-sensitive). Expert bytes 14.2 → ~3.6 GB → total ~21.6 → ~11 GB/token →
  **floor ≈ 2400 fp8-equivalent tok/s** → 1000 at <45% MBU, and the 3000-class opens.
- At B=1 (byte-bound) the byte cut translates ~directly to tok/s — unlike batched serving where it wouldn't.

**Why it's not "just quant harder":** the kernel decides whether the byte-win is realized. My on-box int4
measurement: naive int4 GEMV was **0.55× fp8 (issue-bound — the nibble-unpack ate the win)**. 2-bit is
*more* unpack-bound, so it demands a real Hopper dequant kernel (LUT-based or AQLM-lattice dequant fused into
the wgmma mainloop, TMA-staged). This is quant×kernel co-design, exactly the hand-tuned-PTX regime kog operates in.

## Novel lever B (a first-principles connection we haven't named): spec verify is *doubly* good for MoE B=1
At B=1, every expert is an **M=1 GEMV** — which is bad on *two* axes at once: bandwidth-bound (read the whole
weight for one output) AND low tensor-core utilization (M=1 wastes the MMA → this is *why* K5 sits at e≈0.28).
A speculative **tree verify** routes **W tree-tokens** through each expert → it becomes an **M=W GEMM**, which:
1. **amortizes** the dominant expert read across W tokens (the known spec win), AND
2. **raises e** — M>1 finally uses the tensor cores, curing the M=1 GEMV inefficiency that no GEMV-kernel
   tuning can fix.
Route-aware drafting (already designed) keeps the expert **union** small so the M=W stays cheap. So the spec
verify attacks *both* the byte floor and the e=0.28 floor simultaneously — the single highest-leverage MoE-B=1
move. The novelty is the framing + sizing: pick W to push the expert GEMM off the GEMV efficiency cliff while
the route-aware union keeps the read bounded. Stacks under 2-bit experts (lever A cuts the bytes the GEMM reads).

## Validation plan (prototype like we did NVLS — one number each)
1. **2-bit expert GEMV microbench** (box): LUT/lattice-dequant W2 expert — does it hit byte-proportional
   speedup (≥1.6× fp8, ideally ~3.5×) or go issue-bound like naive int4? (the NVLS-style single measurement)
2. **M-sweep on the expert kernel:** measure e(M) for M=1,2,4,8,16,32 — find the W where the GEMV cliff ends
   (confirms lever B and sizes the spec tree against the kernel, not just the model).
3. **Quality gate** (bench suite): PPL/task-delta of 2-bit experts vs fp8; sweep group size + outlier fraction;
   confirm the rollback ladder (2-bit → int4 → fp8 on sensitive layers).

## The stack (honest, end to end)
monokernel (assemble, kog-proven) → ~1000 at fp8 · + 2-bit experts (cut the floor) → ~2000–2400 single-pass ·
+ route-aware big-tree spec (amortize **and** fix e via M=W) → kog's 3000-class. Monokernel is the next
*execution*; 2-bit experts + spec-as-GEMM is the next *invention*.
