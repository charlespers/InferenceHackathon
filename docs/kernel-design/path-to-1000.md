# Path to 1000 tok/s — how the levers stack (Qwen3-235B-A22B, B=1 decode, 8×H100)

**Target:** 1000 tok/s, benched, B=1 decode, Qwen3-235B-A22B (fp8 e4m3), 8×H100 (80 GB, 3.35 TB/s each).
**Date:** 2026-06-20.

This is the concrete, multiplicative chain from where we *measured* ourselves today to the goal,
which lever buys what, the order to land them, and the realistic best case. Every number is tagged
**[M] measured** (real H100, this session), **[B] baseline** (vLLM, real H100), or **[P] projected**
(roofline / analytical model — validate on box before trusting the absolute value).

---

## 1. Where we are (the starting line)

| Point | tok/s | Source | Note |
|---|---:|---|---|
| Single-GPU fused decode proxy | **30.9** | **[M]** | reads the full 21.96 GB active weights on ONE GPU; HBM-capped ~153 |
| vLLM fp8 / EP | **65.8** | **[B]** | production baseline, expert-parallel |
| vLLM bf16 / TP | **85.7** | **[B]** | production baseline, tensor-parallel, greedy |
| Single-GPU HBM cap (fp8, 1 GPU) | ~153 | **[P]** | 21.96 GB ÷ 3.35 TB/s × (1/eff); hard ceiling for 1-GPU |
| fp8 roofline, 8 GPU, no spec | ~1240 | **[P]** | 21.96 GB ÷ 8 ÷ (3.35 TB/s × ~0.85 eff); the bandwidth wall |

**The physics.** B=1 decode is **memory-bandwidth bound** — each token streams every active
weight byte once and does almost no arithmetic per byte (AI ≈ 2 FLOP/B vs an H100 ridge of ~591).
So tok/s ≈ (aggregate HBM bandwidth × bandwidth-utilization) ÷ (active bytes read per token). Every
lever below changes exactly one of those three terms, which is why they **multiply** instead of add.

Measured bottleneck breakdown: the **MoE experts are 65% of per-token weight bytes** (14.2 B of the
21.6 B active params), and the fused fp8 expert kernel (**K5**) runs at **45.7% of HBM peak (1530 GB/s)**
**[M]** — already the fast path. The prologue (**K1**) is at 27% (904 GB/s) **[M]** after a rewrite;
flash-decode (**K2**) is tiny at 95 GB/s **[M]** (KV is <1% of bytes at typical context). So the experts
are both the biggest byte sink and the closest to roofline — they set the pace.

---

## 2. The four levers and what each multiplies

Each lever targets a different term in `tok/s ≈ BW × util ÷ bytes_per_token`. They compose because
they are independent factors:

```
tok/s  ≈  base_1gpu
          ×  Shard        (more aggregate BW: 8 GPUs read in parallel)
          ×  Kernels      (higher util: drive each GPU nearer its roofline)
          ×  Int4         (fewer bytes: half the expert weight bytes)
          ×  Spec         (fewer weight reads per ACCEPTED token)
```

### Lever A — Shard across 8 GPUs (TP=8 or EP=8): **~8×** aggregate bandwidth
Splitting the model so each GPU reads only ~1/8 of the per-token weight volume turns 1 × 3.35 TB/s
into 8 × 3.35 TB/s of *parallel* read. This is the single biggest structural win and the precondition
for everything else.

- **TP=8 (intermediate-shard MoE):** every GPU holds 192/1536 of every expert's intermediate columns,
  reads exactly 1/8 of each active expert — **no balls-in-bins imbalance**. Per-GPU weight read drops to
  ~3.08 GB/token (NOTE: the reconstructed-×8 = ~24.7 GB *exceeds* the 21.96 GB single-GPU figure because
  the 4 KV-projection heads are **replicated** across all 8 ranks, N_KV_HEADS=4 < TP=8). Cost: **188
  small all-reduces/layer-pair** on ~16 KB [HIDDEN] payloads — latency-bound, ~7–16 us each **[P]**.
  - Kernel: `kernels/decode_step_tp8.cu` (NCCL, one host thread per rank — see §5 fix).
- **EP=8 (expert-parallel):** 16 experts/rank, each rank does the *full* expert for the active experts it
  owns, one all-reduce/layer. Cheaper comms, but **balls-in-bins tail risk**: balanced ≈ TP=8, but the
  adversarial case (all 8 active experts on one rank) collapses to single-GPU MoE bandwidth.
  - Kernel: `kernels/ep_moe_sharded.cu` (benches balanced / skewed / adversarial routings).

**Verdict:** TP=8 for the MoE term (no routing gamble); the choice is benched head-to-head. Effective
shard factor is **< 8** because of (i) replicated KV, (ii) the all-reduce latency tax, (iii) launch
overhead on skinny per-rank GEMVs. Realistic sharded base: **~450–545 tok/s [P]** (the TP=8 weight-only
ideal at ~45% peak is ~489 tok/s [P]; vLLM bf16-TP already shows 85.7 [B], and fp8 sharding should clear
the fp8/EP 65.8 [B] comfortably once kernels are near-roofline).

### Lever B — Near-roofline kernels: util **27–45% → target ~50–60%** of peak
This is a *utilization* multiplier on whatever the sharded base is, not a separate stage. The expert
kernel already proves the recipe works: **warp-per-output-row + coalesced uint4 fp8 + fp8x2→half2
dequant** hits 45.7% **[M]**. Bringing the rest of the chain to the same level is the lever:

- **K1 prologue** 27% → ~45%: already rewritten, more headroom **[M]**.
- **lm_head GEMV** (622 MB/token, the end-of-step tail once experts are sharded): the same warp-per-row
  fp8 idiom should land near 45% peak (~0.41 ms) vs a naive thread-per-row path (~1 ms) — a **~2.5–4×**
  speedup on that stage with the **argmax fused** into the epilogue (no second full-logits HBM read).
  - Kernel: `kernels/lmhead_k3_bench.cu` (also fuses the O-proj residual add).
- **TP=8 down-proj** caveat: the 192-wide contraction only engages 12 of 32 warp lanes (`192>>4=12`),
  so it runs at ~37% lane efficiency — a known **[P]** drag on the 489 tok/s ideal; a multi-row-per-warp
  or sub-warp layout reclaims it.

This lever's gain is **already partly priced into** the sharded base; the residual upside is closing the
27%→45% gap on K1 and the lm_head/down-proj tails. Treat it as **× ~1.1–1.3** on top of shard, not a
clean 2×.

### Lever C — Int4 experts (W4A16): **~2×** on the dominant byte term
The experts are 65% of the bytes. Storing expert weights in **group-wise symmetric int4** (GROUP=128
along K, fp16 scales) halves *those* bytes. Since fp16 scales add ~3%, the real factor is **1.94×** on
the expert traffic (77.9 MB int4+scales vs 151 MB fp8 per the comparable shape), i.e. roughly:

```
new_bytes_per_token ≈ 0.35 × (full)  +  0.65 × (full) / 1.94  ≈ 0.685 × full   →  ~1.46× tok/s
```

if (and only if) int4 stays **bandwidth-bound**. The first int4 attempt (`k5_experts_int4.cu`) was
**issue-bound at 0.57× fp8** — the unpack did ~32 scalar ALU ops + 32 int→float converts per uint4 load,
swamping the HBM load. The fix is the **LOP3-based int4→half2 fast-dequant idiom** (AWQ/Marlin/FT style),
structurally identical to the proven k5 fp8x2→half2 path: build `(n & mask) | 0x6400` (fp16 1024.0
magic-exponent OR), one half2 subtract of 1024, constant op-count, no per-element I2F.

- Kernel: `kernels/k5_experts_int4_v2.cu`. **[P]** target: int4 reports ~46% peak on the *packed* bytes
  (parity with fp8 %peak), i.e. ~2× *effective* (fp8-equivalent) GB/s.
- **Risk [P]:** the new path still converts each half2 to float2 and does 8 scalar FMAs + a 7-add
  reduction per word — it does *not* contract entirely on the half2 datapath, so the ALU tail is ~2× the
  fp8 kernel's. Whether it stays bandwidth-bound is **plausible but unverified**; the bench prints the
  int4-vs-fp8 ratio as the diagnostic (2.0× = bandwidth-bound ideal). Validate before banking the 2×.
- **Correctness:** the analytical-suite ladder predicts int4 ≈ **1.5× over fp8 [P]** (380 vs 260 tok/s
  in the roofline model), consistent with the ~1.46× above — *not* a clean 2× on end-to-end tok/s,
  because only 65% of bytes are experts and there's a comms/launch floor.

### Lever D — Speculative decode: **×2–3** on accepted tokens (the only way past the BW wall)
Sharding + kernels + int4 all push *toward* the bandwidth roofline (~1240 tok/s fp8 [P], higher with
int4). They cannot go *past* it — that requires reading fewer weight bytes **per accepted token**.
Speculative decode does exactly that: a cheap drafter proposes γ tokens, the big model **verifies all
γ+1 in one pass**, and the key physics is that a (γ+1)-row verify costs **~one single-token weight read**
because decode is HBM-bound (the weights are read once; the extra rows are nearly-free arithmetic).

- Microbench `kernels/spec_decode_bench.cu` measures `slowdown(B) = time(B-rows)/time(1-row)` and feeds
  it into the model (no hardcoded assumption); the premise is `slowdown ≈ 1.0` across B=1..9 **[P]**.
- Model: `E[tokens/pass] = (1 − α^(γ+1))/(1 − α)`; effective tok/s = `base × E / slowdown`.
- **[P]** at base=545: α=0.7/γ=4 → ~1511 tok/s (**2.77×**); α=0.8/γ=8 → ~2359 tok/s (**4.33×**). The min
  α at γ=4 to clear 1000 from base=545 is ~0.42; nearly all reasonable (α,γ) clear 1000 except the
  α=0.5/γ=2 corner (~954).
- **MoE tax caveat [P]:** the verify pass reads the union of experts routed by *all* γ+1 draft tokens,
  which can exceed top-8 — the per-pass expert bytes grow sub-linearly but not flat. Route-aware drafting
  (`docs/route-aware-drafting-design.md`) mitigates this. α is the real unknown; re-run with measured α.

---

## 3. The stack — multiplicative chain to 1000

The clean way to see it (all factors on the **single-GPU 30.9 tok/s [M]** base):

```
30.9  [M single-GPU proxy]
  × ~8     shard (TP=8)                  → ~247    (raw); but launch+comms+replicated-KV losses…
  → ~450–545  realistic sharded fp8 base [P]   (≈ TP=8 weight-only ideal 489 [P])
  × ~1.46  int4 experts (1.94× on 65% of bytes, IF bandwidth-bound)  → ~660–800  [P]
  × ~2.0–2.8  spec decode @ α≈0.7, γ=4 (E/slowdown)                  → ~1300–2200 [P]
  ───────────────────────────────────────────────────────────────────────────────
  ≥ 1000 tok/s  cleared with margin once spec lands on top of sharded int4.
```

**Realistic best case (the number to aim at): ~1300–1500 tok/s [P]** at
sharded-int4 base ~660–800 × spec 2.0–2.8×, with α≈0.7 and γ=4 — comfortably over 1000 with headroom
for the comms floor and a sub-2× int4. **Stretch:** α≈0.8 / γ≈8 reaches ~2.3k+ [P] if acceptance holds.

### Where 1000 comes from without int4
Sharded fp8 base ~545 [P] × spec 2.77× (α=0.7, γ=4) = **~1511 [P]** — *spec alone on top of sharding
clears 1000.* Int4 is the cushion (and the cost/$ win), spec is the lever that crosses the line. If α
disappoints (say 0.6), int4 becomes load-bearing: 700 × (E(0.6,4)/slowdown ≈ 2.18) ≈ **~1525 [P]**.

### What can NOT get there alone
- **Shard only** → ~450–545 [P]. Hard-stops at the fp8 8-GPU roofline ~1240 [P] even at 100% util.
- **Shard + kernels + int4, no spec** → ~660–800 [P], still short of 1000. **Spec is required.**
- **Spec on 1 GPU** → capped by the 153 tok/s single-GPU HBM wall; sharding is required first.

This is why the brief says *"the only way past the raw bandwidth roofline"* is speculation — the other
three levers race you *to* the wall efficiently; spec is the one that scales accepted-tokens beyond
weight-reads.

---

## 4. Order to land them (dependency-first, cheapest-validation-first)

1. **Shard to TP=8 (or EP=8) — FIRST, it's the precondition.** Nothing else matters at 30.9 tok/s on
   one GPU. Land `decode_step_tp8.cu`, bench the real per-token latency + all-reduce overhead, confirm
   the cross-rank correctness gate passes (<1e-2). Decide TP vs EP from the head-to-head
   (`ep_moe_sharded.cu`): expect EP-balanced ≈ TP, EP-adversarial ≈ single-GPU. **Target ~450–545 [P].**
2. **Close the kernel gap (cheap, no quality change).** K1 27%→45%; fuse the lm_head GEMV + argmax
   (`lmhead_k3_bench.cu`); fix the TP=8 down-proj lane under-utilization. Pure util multiplier, +10–30%.
3. **Int4 experts (cheap to flip, must verify bandwidth-bound).** Land `k5_experts_int4_v2.cu`; the
   gate is the printed int4-vs-fp8 ratio ≥ ~1.9× *and* correctness <1e-2. If it's still issue-bound,
   fall back to fp8 — spec alone still clears 1000. **Target ~660–800 [P].**
4. **Speculative decode (highest effort, biggest multiplier).** Drafter + tree/batched verify; the
   `spec_decode_bench.cu` flat `slowdown(B)≈1` is the load-bearing empirical result. Tune γ to measured
   α; mind the MoE verify-tax. **This is the lever that crosses 1000.**

Rationale for the order: (1) is a hard dependency; (2) and (3) are low-effort, low-risk, and raise the
*base* that (4) multiplies (so they make spec's job easier and the margin fatter); (4) is the
highest-effort and the only one that can overshoot, so it lands last on a validated, near-roofline base.

---

## 5. Correctness fixes applied to the kernels (this pass)

Before the analysis above is bankable, two issues from the kernel review were fixed in-tree:

- **`decode_step_tp8.cu` — NCCL deadlock (was BROKEN), FIXED.** The driver wrapped *each* rank's
  `ncclAllReduce` in its own `ncclGroupStart/End` and a single host thread enqueued rank 0's entire
  189-collective step before rank 1's. Per NVIDIA's Group-Calls contract, `ncclGroupEnd()` on a
  single-thread driver issuing per-rank groups sequentially can **block** waiting on peers that were
  never enqueued → hang, no tok/s ever produced. **Fix:** added a `run_all_ranks()` helper that drives
  **one `std::thread` per rank/communicator** (the documented one-comm-per-thread exception), so all 8
  ranks reach their i-th collective concurrently and `ncclGroupEnd` cannot block. Applied to the warmup,
  the full-step timing loop, the all-reduce-only timing loop, and the correctness-check
  `sharded_one_layer_capture`. Added `#include <thread>`; events are now recorded inside rank 0's thread.
- **`k5_experts_int4_v2.cu` — int4 unpack n7 bug, already FIXED in-tree.** The top nibble's shift is
  `((w & 0xF0000000u) >> 12)` (lands n7 at bit 16 of the high half-word), confirmed against the CPU
  `get_nib` reference convention (n0=bits[3:0] … n7=bits[31:28]); all 8 nibble placements audit clean.
  Had it been `>> 12`'s predecessor `>> 20`, 1 of every 8 weight nibbles would silently zero while its
  −8 bias was still subtracted — a systematic ~10–15% error that would fail the 1e-2 gate.

Remaining **non-blocking** kernel notes (do not gate the path, see review verdicts): the int4 ALU tail
(may fall short of 2×), the TP=8 down-proj 12/32-lane efficiency, the TP=8 cross-rank argmax not yet
resolving the token id (harmless to the timing proxy), and the lm_head non-multiple-of-32 block-size
guard. None block the 1000-tok/s thesis; all are tracked for the on-box bench.

---

## 6. Measured vs projected — the honest ledger

| Claim | Status |
|---|---|
| Single-GPU 30.9 tok/s, K5 45.7% peak (1530 GB/s), K1 27%, K2 95 GB/s | **[M]** real H100 |
| vLLM fp8/EP 65.8, bf16/TP 85.7 | **[B]** real H100 |
| Experts = 65% of bytes (14.2/21.6 B params) | **[M]** (shapes from config) |
| 8× shard, TP=8 ~489 weight-only ideal / ~450–545 base | **[P]** roofline + comms model |
| Int4 ~1.46–2× on experts (1.94× on the byte slice) | **[P]** — *gate on the int4-vs-fp8 bench ratio* |
| Spec ×2–3 (α≈0.7, γ=4 → 2.77×); slowdown(B)≈1 | **[P]** — *gate on measured α and the slowdown curve* |
| **≥1000 tok/s end-to-end** | **[P]** — the stack clears it; final number is the on-box bench |

**Bottom line:** the multiplicative chain `shard × kernels × int4 × spec` reaches 1000 tok/s with a
realistic best case of **~1300–1500 tok/s [P]**, and even *without* int4 the **sharded-fp8 × spec** path
(~545 × 2.77 ≈ 1511 [P]) clears it. The two pieces that *must* hold are (i) sharding lands near its ~489
weight-only ideal and (ii) speculative acceptance α ≥ ~0.6 at γ=4. Both are projections to validate on
the H100; the single-GPU and vLLM anchors are measured.
