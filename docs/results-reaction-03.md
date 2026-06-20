# Results reaction 03 — Alyssa's config-sweep: fp8 is net-NEGATIVE at B=1, NCCL-tuning is dead

Third data round (via LOOP-A's 09:00 synthesis of Alyssa's `config-sweep.md`). Two measured findings that
**correct my projections** and sharpen the plan. Both confirm "the floor is the game" harder than before.

## The data
| config | tok/s | vs bf16-TP8 |
|---|---|---|
| **bf16-TP8** | **85.7** | — (still the best real number) |
| FP8 on-the-fly (no EP) | 69.0 | **−19%** |
| FP8 + EP | 64.5 | −25% |
Plus: **NCCL env sweep (PROTO/NVLS/channels) = no usable gain** — the defaults are already near-optimal.

## Reaction 1 — fp8 is a LOSS for plain B=1 decode (my fp8 headroom was optimistic)
fp8 halves the weight bytes, but at B=1 the weight is only 14% of the step and the step is floor-bound — so the
saving is nearly invisible, while the **fp8→bf16 dequant adds cost on the critical path**. Net: **fp8 is ~19%
SLOWER**, not faster. This is the strongest confirmation yet of `results-reaction-02.md` ("weight levers are
invisible — or negative — while floor-bound"; `ab_adaptive` already hinted it, now fp8 confirms it directly).

**Correction to `single-user-latency-budget.md`:** the ~508/754 tok/s used an fp8 base. **Use bf16.** The safe
cheap-wins stack is **bf16-TP8 + prefix-cache + spec** → at the corrected EAGLE3 τ≈3.8 (`spec_predict.py`), that's
85.7 × ~3.8 ≈ **~325 tok/s** (decode), not 508. The extra to ~750 comes from the *floor* reduction (kernels +
structural comms), not from fp8. I'll hedge the budget headline accordingly.

## Reaction 1b — BUT fp8 may still help the spec VERIFY (the open question the 09:45 pair answers)
The verify forward reads the **expert union** (~52–126 experts), so it's far more weight-heavy than plain decode
(8 experts). fp8 halving that large union read could pay *in the verify* even though it loses in plain decode.
So **fp8 is regime-dependent**: a loss for plain decode, possibly a win for big-tree spec. **This is exactly the
`backout_floor.py` ΔF comparison we set up:** LOOP-A's **FP8+EP** EAGLE3 vs my **bf16-TP8** EAGLE3, same head,
same k-sweep → if FP8's spec-S beats bf16's despite the −19% plain-decode handicap, fp8 helps the verify. (LOOP-A's
analyzer already reports EAGLE3 *abs* vs bf16-best 85.7, so fp8's handicap isn't hidden — good.)

## Reaction 2 — NCCL env-tuning is a DEAD lever → comms needs STRUCTURE, not flags
The env sweep (NCCL_PROTO=LL, NVLS_ENABLE, channel counts) gave nothing — vLLM's defaults already pick the good
path at 8 KB. **So `E0b` (env-level comms tuning) is downgraded to ~done/null.** The 3 ms comms wall does **not**
fall to environment variables; it needs the *structural* levers: **device-side NVLS/multimem all-reduce captured
in a megakernel** (`megakernel-b1.md`, `comms_floor.md`) — i.e. comms must leave the host critical path, which is
engine surgery, not a flag. This re-points the comms effort from "sweep env vars" (dead) to "build the in-kernel
deferred collective" (the real, hard lever).

## Updated priority (post-3rd-round)
1. **bf16-TP8 + prefix-cache + spec** — the safe cheap stack (~325 tok/s decode + ~50–100× TTFT). Ship it.
2. **`E-attr`** — still the key diagnostic (split the ~7 ms overhead: kernel-inefficiency vs host).
3. **Kernel efficiency (K5 e→1, `k5-tuning-roadmap.md`)** and **structural comms (in-kernel NVLS / megakernel)** —
   the floor reduction that takes ~325 → ~750+. *Not* env-tuning (dead), *not* fp8 (negative for decode).
4. **fp8 — only inside the spec verify, IF the 09:45 FP8-vs-bf16 comparison shows it helps the union read.**
   Otherwise bf16 throughout. int4 stays last.

## One line
fp8 and NCCL-flags are both dead/negative at B=1 — the floor is *even more* the whole game. The cheap win is
**bf16 + prefix-cache + spec (~325, lossless)**; the rest is the floor (kernels + in-kernel comms), and fp8 only
earns its place inside the weight-heavy spec verify, which the 09:45 bf16-vs-FP8 pair will settle.
