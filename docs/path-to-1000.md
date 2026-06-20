# The path to 1000 tok/s (B=1, Qwen3-235B-A22B, 8×H100) — from first principles

**The target is 1000 tok/s = 1.0 ms/token.** We are at 85.7 (11.67 ms). That's ~12×. This derives, from
physics, exactly what is *required* — and shows that most of what the team has been measuring (fp8 alone,
env-comms, spec alone) **cannot get there**; a specific, narrow combination can.

## Verdict (the three non-negotiables + the linchpin)
A 1.0 ms token has room for only three things, and **all three are mandatory** — drop any one and 1000 is
physically impossible:
1. **fp8 weights** — bf16's weight read alone is **1.56 ms (→ 641 tok/s ceiling)**. bf16 *cannot* hit 1000, ever.
   fp8 is 0.78 ms (→1282 ceiling). **Non-negotiable.** (int4 = 0.39 ms, the comfort path.)
2. **In-switch NVLS comms ≤ ~1 µs/collective** — the 188 serial all-reduces cost **3.0 ms at today's 16 µs**;
   they must drop to **0.19 ms** (1 µs each). 4 µs already blows the budget (0.75 ms). **Non-negotiable.**
3. **~Zero overhead** — the **~7 ms** of launch/host/scheduler/sub-roofline must go to ~0. **Non-negotiable.**

`0.78 (fp8) + 0.19 (NVLS) + ~0 (overhead) = 0.97 ms → ~1033 tok/s.` That's the whole budget. There is **no slack
for the current floor.** The one mechanism that delivers all three at once is the **persistent megakernel**
(`megakernel-b1.md`): fp8 weights streamed + dequant fused (free) + device-side NVLS all-reduce in-kernel +
no launches/host = zero overhead. **The megakernel is the linchpin of 1000 tok/s.**

## The budget, term by term (what it is, what it must be, what gets it there)
| term | now (bf16-TP8) | must be for 1000 | what gets it there | proven? |
|---|---|---|---|---|
| **weight read** | 1.56 ms | **≤ 0.8 ms** | **fp8** (fused dequant) | physics; fp8-otf measured SLOWER *because floor-bound* — needs the floor gone + fusion |
| **comms** (188 coll) | 3.0 ms @16µs | **≤ 0.2 ms** | **in-switch NVLS, in-kernel** | NCCL env defaults DON'T NVLS at 8 KB (Alyssa) → must force it device-side |
| **overhead** | ~7.0 ms | **~0** | **megakernel** (or graphs+fast-path, partial) | E-attr will split launch vs host vs kernel |
| **KV** | ~0 (short ctx) | ~0 | (grows w/ ctx → fp8 KV) | — |

## Reality check — realistic NVLS, and why LOSSLESS 1000 is still reachable (via small-tree spec)
My "0.97 ms" used an optimistic **1 µs** all-reduce. In-switch NVLS on H100 realistically lands ~**2–4 µs** for
8 KB (188 collectives → 0.38–0.75 ms). At that, **plain** fp8 decode (megakernel, zero overhead) tops at:
`fp8 weight 0.78 + NVLS@2µs 0.38 = 1.16 ms → 865 tok/s` (744 @3µs). Short of 1000 — *for plain decode*.

**But spec applies at the weight-bound limit too** — and this is the key correction. Once the megakernel removes
the floor, a **small** tree still amortizes the comms + the read-once non-expert weight over τ, while a *small*
union keeps the expert-weight tax low (the F→0 column of `tree_spec_optimizer.py`: **shrink** the tree here, don't
grow it). On the floor-removed fp8 engine, **W1×D2 EAGLE3 spec** (2 draft positions, union ~16, τ≈2.2, incl. the
draft cost):

| config (megakernel, fp8, zero-overhead) | NVLS 2µs | NVLS 3µs |
|---|---|---|
| plain decode | 865 | 744 |
| **+ W1×D2 spec** | **1169** | **1062** |
| + W1×D3 spec | 1034 | 960 |

**So LOSSLESS 1000 IS reachable:** fp8 + megakernel + in-kernel NVLS (≤~3 µs) + **small-tree** EAGLE3 spec → ~1060–1170.
No int4 required. (This is spec composing with the megakernel — the verify *is* a megakernel run over the few
tree positions; consistent with `stack-interaction-check.md`: small tree at the weight-bound limit.)

**int4 experts are MARGIN, not a requirement** — they buy robustness/headroom: if NVLS floors at ~4 µs, or the
megakernel's overhead isn't truly zero, or spec under-delivers, int4 experts (weight → 0.51 ms, small quality
gate) restore 1000. Keep them as the **safety net**, validated in parallel, not the primary path.

**Today (no NVLS, 16 µs): stuck ~250 regardless of weight precision** — the comms wall dominates everything.
**Stage 3 (the NVLS all-reduce kernel) is the single make-or-break experiment** (`megakernel-build-plan.md`):
≤3 µs → lossless 1000 is on; ~16 µs → nothing else matters.

Net: **1000 lossless = megakernel (overhead→0) + NVLS kernel (≤3 µs) + small-tree EAGLE3 spec.** int4 experts are
the safety net for the last bit if any of those three slips.

## Why the cheap levers DON'T reach 1000 (and what they're actually for)
- **Spec decode alone caps at ~300 tok/s.** It *amortizes* the floor over τ; at EAGLE3's τ≈3.5 that's 85.7×3.5 ≈
  300. It **cannot** reach 1000 because τ is capped (~3.5) and, worse, a big tree *reads the expert union* —
  growing the weight term. **Spec amortizes the floor; it does not remove it.** To get past ~300 the floor must
  be *structurally eliminated*, not divided. → Spec is the **cheap first 3.5×** (ship it now) and, once the floor
  is gone, a **small-tree ~1.5× topping** that takes 1033 → ~1500.
- **fp8 alone is *negative* today** (Alyssa: −19%) — the saving is invisible while floor-bound and the dequant
  adds critical-path cost. fp8 only pays **after** the floor is removed and the dequant is fused. Order matters.
- **NCCL env-tuning is dead** (defaults already chosen) — the 1 µs comms needs *structural* in-switch NVLS issued
  from inside the kernel, not env vars.
- **K5 kernels (e→1)** remove the sub-roofline part of the overhead but not the launch/host/comms — necessary,
  not sufficient.

## The sequenced roadmap (each step measurable, banked, and ordered by dependency)
1. **Now — spec on bf16-TP8 + prefix-cache → ~300 tok/s, lossless.** The cheap 3.5×. (EAGLE3/n-gram, `E6`.)
2. **`E-attr` — split the 7 ms overhead.** Decides how much is graph-removable (launch) vs host (fast-path) vs
   kernel (K5). This sizes the megakernel's job and tells us if vLLM+graphs can get *partway* (maybe ~600–800)
   before a custom engine is required.
3. **The floor demolition (the hard 300→1000 leap), in one engine (cudarc megakernel):**
   - fp8 weights streamed with **fused dequant** (free, overlapped — `k5-tuning-roadmap.md` cp.async).
   - **device-side NVLS/multimem all-reduce in-kernel** (`comms_floor.md` lever #1, now in-kernel) → 1 µs.
   - **one persistent kernel** over all 94 layers → launches/host/scheduler → 0 (`b1-fast-path-design.md` is the
     vLLM-side partial; the megakernel is the full version).
   - Result: 0.78 + 0.19 + ~0 ≈ **1.0 ms → ~1000 tok/s, lossless.**
4. **Past 1000:** small-tree route-aware spec (~1.5× → ~1500) and/or **int4 experts** (0.39 ms weight → ~2000) —
   the latter quality-gated, the former lossless. This is the `absolute-ceiling.md` ~2000 band.

## The lossy cushion (if a lossless constraint slips)
1000 is *tight* (0.97 ms, ~3% margin) — it assumes perfect fp8 dequant fusion AND a true 1 µs in-switch reduce.
If either slips, buy margin with **lossy** levers, quality-gated:
- **adaptive-top-k** (8→4 active experts): expert weight halves → fp8 weight ~0.5 ms. Big margin.
- **depth reduction** (skip layers): fewer layers → *less weight AND fewer collectives* (both terms drop
  proportionally). The single most leveraged lossy knob (47 layers ≈ halves weight + comms).
- **fp8/int4 KV** for long context (keeps the KV term from eating the budget as chats grow).

## What this means for the team, today
- The **09:45 EAGLE3** run confirms the spec ~3.5× → ~300 (the cheap rung) and, via the **bf16-vs-FP8** pair,
  whether fp8 helps the weight-heavy verify (the fp8-after-floor question, early).
- **The real prize is the megakernel**, and the team's research already converged on it (`fast_decode_research.md`
  Kog single-kernel + DTP). **It is the only thing that makes 1000 physically reachable** — fp8 makes the weight
  fit, NVLS makes the comms fit, and the megakernel is what delivers fp8-fused + in-kernel-NVLS + zero-overhead
  simultaneously. Everything else (spec, prefix-cache, K5) is necessary scaffolding and the cheap first 3.5×,
  but **the 300→1000 leap is the megakernel or nothing.**

## One line
**1000 tok/s is reachable and lossless: a persistent megakernel (overhead→0) + an in-kernel NVLS all-reduce
(≤3 µs) + small-tree EAGLE3 spec → ~1060–1170. Spec is also the cheap first ~3.5× to ~300 today; the 300→1000 leap
is the megakernel + NVLS kernel (the floor must be *removed*, not just *amortized*). int4 experts are the quality-
gated safety net, not the primary path. The NVLS all-reduce kernel (≤3 µs) is the single make-or-break.**
