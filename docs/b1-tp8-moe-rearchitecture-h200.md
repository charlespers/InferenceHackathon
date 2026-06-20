# B=1 Latency Re-architecture — Pure-TP8 Column-Sharded MoE (Qwen3-235B-A22B, 8×H200)

> ⚠️ **Hardware correction:** the actual cluster is **8×H100 80GB (3.35 TB/s)**, confirmed on-box — *not*
> H200. The tok/s ceilings below are H200-native (4.8 TB/s); for this H100, divide every weight-/KV-read
> rate by **1.433** (e.g. TP8 FP8 ~1424 → **~994 tok/s** weight-only). The **break-evens, KV-crossover
> contexts, kernel design, and decision boundary are bandwidth-invariant and apply as-is.** The kernel
> efficiency `e₁₉₂` was treated as an estimate here; the full-width expert GEMV is now **measured at
> e≈0.46 on the H100** (`k5-kernel-results-h100.md`), and the vLLM `192%128` launch failure this spec
> predicted **was reproduced live** (§6.1).

> **Implementation-grade spec.** Single-user batch-1 decode. Companion to
> [`b1-latency-architecture.md`](./b1-latency-architecture.md) (the broad survey) — this is the deep
> dive on its highest-leverage finding: replacing expert-parallel (EP) with pure tensor-parallel (TP8)
> for the MoE, because at B=1 EP is **busiest-rank bound** and TP8 structurally eliminates the imbalance.
>
> **Method:** fan-out workflow (4 parallel designs → 2 adversarial red-teams → synthesis), re-grounded
> for **H200** (4.8 TB/s, 141 GB). The red-teams **tightened the claim twice** and their corrections are
> baked in below. The §-references to `kernels/k5_experts.cu`, `common.cuh`, `bench/*.py`, and
> `server/topology.py` were **verified against the live repo** — they are real files/lines, not sketches.

## TL;DR

Pure-TP8 column-sharded MoE for Qwen3-235B-A22B (8xH200, B=1 decode) eliminates EP's busiest-rank routing imbalance: every GPU holds a 192-intermediate-col slice of all 128 experts, so it reads exactly 8/8 of each active expert's share with zero balls-in-bins gamble. Weight-only ceiling rises to ~1424 tok/s on H200 (was ~994 on H100, x1.433 from HBM3e BW) vs EP8's busiest-rank ~321 tok/s (E[max bin]=2.597). BUT the inversion is CONDITIONAL, and both red-teams tightened it: (1) TP8 vs the REAL competitor TP4xEP2 turns on realized GEMV efficiency e_192/e_384 > ~0.60 (re-derived from corrected E[max]; the bundle's 0.651 is not reproducible from correct balls-in-bins and is slightly conservative). e_192 lands ~0.50-0.55 best-case (rewrites applied) and ~0.30 on the shipped scalar-deq/K-major/atomicAdd kernel; (2) the deciding boundary is NOT the bundle's ~28K KV crossover. Because 4 KV heads cannot split 8 ranks, TP8 must REPLICATE KV (94KB/pos/GPU) while TP4xEP2 splits it (23.5KB/pos/GPU) -- a 4x per-step KV-BW penalty that flips the win to TP4xEP2 at ctx ~4-8K, not 28K. Honest verdict: TP8 wins ONLY when ctx <= ~8K AND e_192 >= ~0.50 AND the N-major down-proj re-layout has shipped. Defensible inversion is ~1.1-1.3x in a narrow short-context window, NOT the 1.5x headline. Day-0 ship is TP4xEP2 (384 cols, 384%128=0, launches on the stock block-128 FP8 checkpoint, no strided trap); TP8 is a fast-follow gated on a clean Nsight e_192 >= 0.55 measurement and a block-64 requant.

> **Decision boundary (read this first):**
> TP8 WINS only when ALL THREE hold: (a) ctx <= ~8K, (b) measured e_192/e_384 > ~0.60 (equivalently e_192 >= ~0.50 at e_384~0.78), and (c) the down-proj N-major [192,4096] re-layout + dynamic TMA descriptors + DSMEM (not atomicAdd) cross-expert reduce + vectorized 128-bit FP8 dequant have all shipped. In that window TP8 delivers ~1.1-1.3x over TP4xEP2 (e.g. 2K ctx: ~627 vs ~583 tok/s), NOT the 1.5x headline. TP4xEP2 WINS (and is the day-0 default) when ANY of: ctx >= ~8K (the 4-KV-head/8-rank mismatch forces TP8 to replicate KV at 4x the per-step KV-BW of TP4xEP2's split KV; at 28K TP4xEP2 is ~22% faster, ~529 vs ~434 tok/s), OR e_192 < ~0.50 (skinny-GEMV dead zone -- includes ALL stock-kernel day-0 configs: Triton fused_moe + K-major down-proj pin e_192 ~0.30-0.36), OR the block-64 requant has not landed (TP8-FP8 is literally unlaunchable on the stock block-128 checkpoint, vLLM #17569: 192 % 128 != 0). EP8 NEVER wins at B=1 (busiest-rank ~321 tok/s, Zipf skew worsens E[max] to ~3.37 -> ~288 tok/s; no shared expert means no dispatch-overlap hiding); it is a last-resort fallback only. The single number to measure: e_192 via Nsight dram__bytes_read.sum/time on the ISOLATED kernel (NOT TPOT floor-subtraction -- the floor-model error band exceeds the ~2% break-even margin at the knife-edge). GO threshold for the thesis: e_192 >= 0.55 (not 0.508) to buy margin against measurement noise and the exposed-all-reduce tax.

---

## 1. Motivation & the inversion (one paragraph)

Qwen3-235B-A22B is 94 all-MoE layers, 128 experts, top-8, no shared expert, ~21.57B active params/token (attention ~6.70B with a large 8192 inner dim, routed experts ~14.19B, lm_head ~0.62B, router ~0.05B). At B=1 decode the model is pure-memory-bandwidth-bound and the latency of a step is set by the **busiest rank**, because all 8 GPUs run lockstep and the step cannot retire until the slowest does. Expert-parallel (EP8) hands one token's 8 experts to 8 GPUs as 8 balls into 8 bins: the busiest GPU reads E[max]=2.597 experts' worth while others idle, and with no shared expert EP loses its dispatch-overlap hiding trick — a ~321 tok/s weight-only ceiling on H200 that **never wins at B=1**. The inversion: **pure TP8 gives every GPU a column-slice of ALL 128 experts** (1536/8 = 192 intermediate cols/expert), so each GPU reads exactly 8/8 of each active expert's columns — routing imbalance is **structurally eliminated**, not statistically mitigated — for a ~1424 tok/s weight-only ceiling on H200 (the same column-reduce folds into the down-proj all-reduce already on the TP path; 2 collectives/layer vs EP8's ~4 incl all-to-all). The catch this spec exists to bound honestly: the win is conditional on a realized-efficiency ratio AND a short-context regime, and **both red-teams tightened the boundary below what the raw byte-ledger suggests** — TP8's defensible advantage is ~1.1–1.3x in a ctx≤8K window, not a 1.5x headline, and the day-0 shippable answer is TP4xEP2.

H100→H200 framing (carry throughout): the **only** changes are HBM BW (3.35→4.8 TB/s, x1.433) and capacity (80→141 GB, x1.76). Same GH100 die, 132 SMs, 50 MB L2, same 4th-gen NVLink/NVSwitch, same ~1979 TF8 compute, no native FP4. **Every BW-invariant quantity below — break-even efficiencies, KV-crossover contexts, the comms floor, all kernel tiling — is UNCHANGED from H100.** What H200 changes: weight-bound ceilings scale x1.433, but the comms+sampling+host floor (~0.3–0.5 ms/step) does NOT, so that floor is now a *larger* share of TPOT, making comms tuning and persistent/mega-kernel work *relatively more* valuable; and 141 GB makes KV-replication and a co-resident draft model OOM-free.

---

## 2. The byte-ledger & busiest-rank proof

All FP8 (1 B/param) unless noted. Hidden d=4096, 94 layers, 128 experts, top-8, intermediate I=1536, attention inner 8192, GQA 4 KV heads × head_dim 128 → kv_dim 512/layer. Usable HBM BW = 3.84 TB/s/GPU (0.80x of 4.8), aggregate 30.7 TB/s.

### 2.1 Per-expert / per-layer mass
One expert = gate+up+down = 3·d·I = 3·4096·1536 = **18.87M params = 18.87 MB FP8**. Active 8/layer = 150.9 MB demanded/layer; ×94 = **14.19 GB** (anchors the routed-expert active mass). Full expert tensor = 128 × 18.87 MB × 94 = **227.1 GB** (dominant model mass — this is what the N-major re-layout in §4 must transpose offline).

Non-expert (read once, then divided by TP degree):

| Term | Params | FP8 bytes |
|---|---|---|
| Attention (all layers) | 6.70B | 6.70 GB |
| lm_head | 0.62B | 0.62 GB |
| router (94×128×4096) | 0.049B | 0.049 GB |
| **Non-expert subtotal** | 7.37B | **7.37 GB** |

### 2.2 Busiest-rank proof (balls-in-bins) — with CORRECTED E[max]

At B=1, EP8's 8 active experts are 8 balls into 8 bins (uniform). The step finishes when the busiest rank finishes. **E[max] for n=m=8 = 2.597** (standard result; P(M≤1)=8!/8^8=0.0024). EP8 ceiling decomposition (this reproduces the GT 321):

- busiest expert read = 2.597 × 18.87 MB × 94 = 4.606 GB
- **plus** full replicated non-expert (EP does not shard attention/lm_head) = 7.37 GB
- total busiest = **11.98 GB** → 11.98e9 / 3.84e12 = 3.12 ms → **321 tok/s** ✓

Zipf routing skew raises E[max]→~3.37 → (5.977 + 7.37)/3.84 = 3.48 ms → **~288 tok/s**.

> **RED-TEAM CORRECTION (carry forward, do not re-derive):** the byte-ledger bundle's intermediate-config E[max] values are wrong and must be replaced. **8 balls / 4 bins = 3.538** (not 3.158); **8 balls / 2 bins = 5.094** (not 5.236). These feed the TP×EP hybrids. The uniform 8/8 = 2.597 is correct.

### 2.3 TP8 — imbalance structurally eliminated
Each GPU holds 192/1536 cols of every expert. Per-GPU/token/layer expert read = 8 experts × (192/1536) × 18.87 MB = 8 × 2.359 MB = 18.87 MB (= exactly one full-expert-equivalent, no max-bin gamble); ×94 = 1.774 GB. Plus TP-sharded non-expert = 7.37/8 = 0.921 GB. **Total per-GPU = 2.695 GB** → 2.695e9/3.84e12 = 0.702 ms → **1424 tok/s weight-only (H200)** (was 994 on H100, x1.433). BF16 = 5.39 GB → 712 tok/s.

### 2.4 TP4xEP2 — the REAL competitor (corrected)
TP4 splits 1536→**384 cols** (384%128=0 → launches on the stock checkpoint); EP2 over 2 bins. Busiest = E[max](8,2)=**5.094** × (384/1536=0.25) = 1.273 expert-equiv × 18.87 MB × 94 = 2.258 GB expert; + non-expert /4 = 1.842 GB → busiest ≈ 4.10 GB → **~936 tok/s weight-only** (red-team's number; the bundle never stated it cleanly). Critically, TP4xEP2 reads **contiguous 384-col blocks** → high e_384 ≈ 0.75–0.78 and **no strided down-proj trap**.

### 2.5 Master ledger (per-GPU bytes/token, B=1, FP8)

| Term | EP8 busiest (uniform) | EP8 (Zipf 3.37) | TP4xEP2 busiest | **TP8** |
|---|---|---|---|---|
| Routed-expert | 4.606 GB | 5.977 GB | 2.258 GB | **1.774 GB** |
| Non-expert | 7.37 (replicated) | 7.37 | 1.842 (/4) | **0.921 (/8)** |
| **Total weight/GPU** | **11.98 GB** | 13.35 GB | 4.10 GB | **2.695 GB** |
| Weight-only ceiling (÷3.84 TB/s) | **321 tok/s** | 288 | ~936 | **1424** |

### 2.6 Break-even algebra (BW-invariant; UNCHANGED on H200) — RE-DERIVED
- **vs EP8 strawman (skewed):** TP8 wins when e_192 > 1.774 / (3.37×18.87×94/1000) = 1.774/7.09 = **0.25**. Huge margin, necessary not sufficient.
- **vs TP4xEP2 (the decider):** folding the non-expert TP-shard asymmetry (TP8 /8 vs TP4 /4), set TP8 (1.774/e_192 + 0.921) = TP4xEP2 (2.258/e_384 + 1.842). With corrected E[max]=5.094, **the break-even ratio re-derives to e_192/e_384 ≈ 0.60, NOT 0.651.** The bundle's 0.651 is not reproducible from correct balls-in-bins; 0.60 is the honest (slightly easier) bar. At e_384=0.78 this means **e_192 ≥ ~0.47**. To buy margin against measurement noise and the exposed-AR tax (§5), the engineering GO threshold is set higher: **e_192 ≥ 0.55**.

---

## 3. The KV-under-TP8 decision — THIS sets the real context boundary

4 KV heads < 8 TP ranks → KV cannot split 8 ways. KV/token-pos = 4×128×2×94 = 96,256 elems = **188 KB FP16 / 94 KB FP8 / 47 KB INT4**.

| Strategy | Bytes/GPU/pos (FP8) | Note |
|---|---|---|
| **(A) Replicate** (all 8 ranks hold all 4 heads) | **94 KB** | trivial, no KV comms; 141 GB makes it OOM-free; but every rank reads FULL KV — KV-read NOT parallelized |
| (B) TP4-split KV (4 ranks 1 head, pairs replicate) | 23.5 KB | KV-read /4; needs KV all-gather, asymmetric attn |

The bundle recommends (A) and quotes a KV/weight crossover at **~28K** (94 KB·C = 2.695 GB). **The decision-boundary red-team showed this is the WRONG comparison and must be corrected:**

> **CORRECTED BOUNDARY (load-bearing):** ~28K is TP8's KV-read vs TP8's *own* weight-read. The actual decision is TP8 vs **TP4xEP2**, and there TP4xEP2 **splits** KV cleanly across its 4 TP ranks (23.5 KB/pos/GPU) while TP8 must **replicate** (94 KB/pos/GPU) — a **4x per-step KV-BW penalty on TP8, paid every token.** After the realized-e haircut TP8's weight edge is razor-thin (e_192=0.52/e_384=0.78 → 5.18 vs 5.26 GB effective, a ~1.5% lead, not 4.4x), so the 4x KV penalty crosses over almost immediately:

| ctx | TP8 (e_192=0.52, tuned comms) | TP4xEP2 | winner |
|---|---|---|---|
| 2K | ~627 tok/s | ~583 | TP8 (+8%) |
| 4K | win | — | TP8 |
| 8K | tie | tie | knife-edge |
| 28K | ~434 | ~529 | **TP4xEP2 (+22%)** |

The one thing delaying TP8's loss: TP4xEP2's EP2 dimension needs all-to-all dispatch+combine (~376 collectives/step vs TP8's 188), doubling its comms floor (~0.56 ms vs ~0.28 ms stock). That offset buys TP8 the window from ctx ~1K (weight-only) out to ~8K (with comms) — but it does **not** save TP8 at long context. **Net: the TP8 win is ctx ≤ ~8K, not ~28K. The bundle overclaims the context range by ~3–4x.** The KV-crossover *contexts in isolation* (~28K TP8-replicated FP8, ~224K TP4-split+KIVI-INT4) are BW-invariant and unchanged on H200 — H200 only makes them OOM-reachable and 1.43x faster in the weight-bound portion — but they are not the competitive boundary.

**Decision:** ship TP8 KV-replicated only for ctx ≤ ~8K. Beyond that, TP4xEP2 (split KV) is the structurally-correct choice. Option (B) KV-split for TP8 is deferred — it neutralizes the 4x penalty but reintroduces a KV all-gather collective (~282 collectives/step) and is only worth it if a single long-context TP8 box is mandated.

---

## 4. The column-sharded MoE GEMV kernel (the §2a risk-killer)

**The risk:** 192-col shards are very skinny GEMVs (4096→192 gate/up; 192→4096 down), 1-of-128 gathered, with worse L2 reuse and more epilogue overhead than EP's fat contiguous 4096→1536 blocks. This is a **pure HBM-bandwidth** problem (~1 FLOP/byte); the only number that matters is e = achieved_BW / 4.8 TB/s. The shipped `kernels/k5_experts.cu` is in the LOSING regime and must be rewritten on four axes. Per-GPU MoE read = 18.87 MB → 4.9 µs ideal (e=1), ~9.4 µs at e=0.52.

### 4.1 Layout (the single most important decision)
- **Gate/Up: N-major [192,4096]** per expert → each CTA streams contiguous 4096-B rows (TMA-friendly, fully coalesced).
- **Down-proj: RE-LAYOUT N-major [4096,192]** (RED-TEAM FIX #1). The stock checkpoint stores down K-major (MOE_INTER=1536 contiguous) — confirmed at `k5_experts.cu:39` (`drow = Wd[e] + o*MOE_INTER`). Under a 192-col shard the live `a[j]` entries are **strided over a 1536-wide contraction → latency-bound, e_down ≈ 0.36.** Re-layout offline so the 192-contraction is contraction-minor contiguous; one-time O(weights) transpose at load, zero runtime cost.
- **Expert gather via DYNAMIC cuTensorMap descriptors** (FIX #1): the 8 active expert IDs are data-dependent; construct 8 descriptors per token on the device path. Do NOT bake gathered descriptors at CUDA-graph capture. The gather is a base-address select (`e = sel_idx[slot]`, no per-element divergence).

### 4.2 The fixes the bundle ASSUMED but the shipped kernel lacks (kernel red-team)
- **Vectorize the dequant.** `common.cuh:39` is scalar `float(v)*s`, one FP8/LDG (its own TODO: "vectorize to 128-bit loads, 16 fp8/thread"). Scalar byte loads waste ~93% of each 128-B sector and cap on LSU issue, not HBM → **shipped e_gate/up ≈ 0.30–0.38, not 0.62.** Must vectorize to 128-bit FP8 loads. *This is the single biggest shipped-kernel e-killer and the bundle's e≈0.52 silently assumes it fixed.*
- **Hoist the activation.** `k5:31` re-reads `y[k]` from global each iteration; hoist to SMEM/registers.
- **DSMEM cross-expert reduce, NOT atomicAdd.** `k5:42` is `atomicAdd(&h_io[o], ...)` → L2 serialization tanks e_down to ~0.25. Replace with an 8-CTA cluster (CGA) DSMEM reduction (Hopper max-8-CTA fits the 8 experts exactly): each CTA writes its [TN] partial to its DSMEM slot, `cluster.sync()`, CTA-0 sums via remote DSMEM reads. No atomics.

### 4.3 Tiling & launch (fill 132 SMs at B=1)
```
=== Kernel 1: fused GATE+UP (CUDA-core split-K GEMV) ===
grid    = (6 N-tiles, 8 experts, 4 splitK, x2 gate|up) = 384 CTAs  (~2.9 waves)
block   = 128 threads; cluster=4 over splitK (DSMEM reduce, no atomics)
smem    = 4-deep TMA ring 4x(32 rows x 256 K x 1B)=32KB + x[4096] BF16=8KB + accs ~4KB ~= 44KB
=== Kernel 2: DOWN (CUDA-core GEMV + 8-CTA cluster cross-expert reduce) ===
grid    = (64 N-tiles, 8 experts) = 512 CTAs as 64 clusters x 8 CTAs  (~3.9 waves)
smem    = 4-deep TMA ring 4x(64 x 192 x 1B)=48KB + h[192]=0.4KB + DSMEM slots 2KB ~= 52KB
```
Both captured in ONE CUDA graph with the dynamic-TMA prologue so expert IDs float per token without re-capture.

**Why CUDA cores, NOT wgmma:** padding M=1→16 for tensor cores costs 16x phantom FLOPs (free, it's BW-bound) BUT forces a k-major swizzled SMEM layout that **refragments the contiguous TMA stream** engineered in §4.1, and the large idle accumulator tile collapses occupancy / in-flight TMA buffers → e drops to ~0.30–0.35. Use CUDA-core streaming dot-product (warp-reduce via `__shfl_down`); this is the rare case where wgmma is the wrong tool.

### 4.4 Realized-e estimate — HONEST range (both red-teams)
Multiplicative loss model, **rewrites applied**:

| factor | value | reasoning |
|---|---|---|
| coalescing/contiguity | 0.92 | N-major re-layout → fat contiguous TMA loads |
| TMA pipeline | 0.90 (gate/up), ~0.82 (down) | down K=192 short → drains often |
| occupancy/tail | 0.88 | 2.9 / 3.9 waves |
| scale/epilogue + dynamic-descriptor prologue | 0.95 → 0.93 | ~0.3µs descriptor build is ~6% of 4.9µs (kernel red-team) |

Best-case blend: e_gate/up ≈ 0.62–0.69, e_down ≈ 0.50 → **e_192 ≈ 0.52** best-case.

> **RED-TEAM HONEST RANGE (do not overclaim):** The 0.50–0.55 is a *best case assuming all four rewrites land and each hits its ceiling simultaneously.* The **defensible worst case (rewrites applied) is e_192 ≈ 0.40–0.43** (e_down ≈ 0.40 even N-major, from short-K drain + the cluster.sync() barrier on the B=1 critical path). The **shipped kernel as-written (scalar deq, K-major down, atomicAdd) is e_192 ≈ 0.30** — right at the death line. Break-even (e_192/e_384 > 0.60, e_384≈0.78 → e_192≥~0.47): 0.52 → WINS by ~2% (knife-edge, inside floor-subtraction noise); 0.43 → **LOSES**; 0.30 → loses badly. **The inversion does NOT hold at the defensible worst case.** It holds only at the optimistic ~0.52, by a margin below the noise of TPOT-based e extraction.

**Failure modes that pin e below 0.30 (prevent all):** (1) down left K-major; (2) descriptors baked at capture; (3) M-padding for wgmma; (4) L2 atomicAdd reduce; (5) block-128 scales with 192 shard → **unlaunchable** (needs block-64 requant, 192/64=3). All five are real defaults in stock vLLM/SGLang (Triton fused_moe, K-major down) → **day-0 TP8 with stock kernels is in the losing regime.**

---

## 5. Comms & per-layer schedule (TP8)

Every layer: `RMSNorm → attn(GQA) → +res → RMSNorm → MoE → +res`. The only cross-GPU tensors are the **O-proj output** and the **MoE-down output**, both row-parallel partial sums.

### 5.1 Exactly 2 all-reduces/layer
| Collective | Tensor | Why unavoidable |
|---|---|---|
| AR#1 | O-proj [4096] | row-parallel over Q-head shards |
| AR#2 | MoE-down [4096] | row-parallel over 192-col expert shards |

**= 188 all-reduces/step.** Router is replicated (input identical after AR#1 → all ranks compute the same top-8, zero comms).

### 5.2 AR#2 does NOT fold into AR#1 (serial), but the expert-sum folds into the kernel
Strict dataflow: `attn partial → AR#1 → full residual → RMSNorm → router top-8 → MoE GEMV → partial → AR#2`. AR#1's result is the router/up-proj input, so the two AReduces are distinct critical-path points — **not** fusible. What DOES fold: the 8-expert cross-expert sum folds into the down-proj **DSMEM epilogue** (§4.2), so only ONE [4096] partial/rank enters AR#2. Net: still 2 network collectives/layer, and crucially **no all-to-all** (EP8's ~4/layer penalty).

### 5.3 Message size & algorithm
[4096] × 2 B (keep comms BF16 even with FP8 weights) = **8 KB/all-reduce**; 188 × 8 KB ≈ 1.5 MB/step — **pure latency, not bandwidth** (8 KB crosses a 450 GB/s NVLink link in ~18 ns vs ~1.5 µs fixed launch+sync). Use **one-shot (not two-shot/ring)** — single-hop single-sync on full NVSwitch; **LL protocol (not LL128)**; **1–2 channels** (`NCCL_MAX_NCHANNELS=2`); keep **NVLS** enabled (collapses to 1 hop, offloads the reduce; bandwidth benefit marginal at 8 KB).

### 5.4 Budget & the H200 delta
- **Stock** (per-call launch): ~1.5 µs × 188 = **~0.28 ms/step**.
- **Tuned** (CUDA-graph + LL/NVLS one-shot + 1–2 channels + NVSHMEM device-initiated put+signal from the down-proj epilogue to overlap the reduce with the GEMV tail): ~0.75 µs × 188 = **~0.14 ms/step**.

> The wire/protocol numbers are **UNCHANGED H100→H200** (same NVLink/NVSwitch/132 SMs/L2). Since the weight ceiling rose x1.433 but this floor did not, comms is now a **larger TPOT fraction** (~10–20% at 700–850 tok/s) → the NVSHMEM-overlap and one-shot tuning are **relatively more valuable on H200.** Note the exposed-AR tax (kernel red-team): at B=1 there is no second token to hide AR latency against; NVSHMEM hides the *launch*, not the wire+sync — `GEMV→AR→next-layer` is a hard serial chain, and the faster the weight stream, the more this fixed floor dominates and the *less* the e advantage matters. **Do not exceed TP8** — TP16 crosses the NVSwitch domain (per-hop latency 1.5 µs → tens of µs; the 188-floor explodes). Use 141 GB KV-replication for long context within TP8, not more TP.

---

## 6. Engine integration + gating measurement loop + eval gates

### 6.1 Engine assessment
- **vLLM:** default (no `--enable-expert-parallel`) IS pure TP-sharded MoE — no patch to *select* TP8. **But TP8-FP8 is UNLAUNCHABLE on the stock block-128 checkpoint** — vLLM **#17569**: `output_size ... = 192 is not divisible by ... block_n = 128` (also #30934 on Qwen3-235B-FP8). Official remediation: TP4+expert-parallel *or* requant — confirms red-team correction #2 verbatim. Default Triton `fused_moe` + K-major down-proj → worst-case e_192 path.
- **SGLang:** same default TP-MoE; richer knobs — `--moe-runner-backend {triton,cutlass,flashinfer}` is the cheap lever to lift e_192 above Triton without a from-scratch kernel. Same block-128 ceiling.
- **Custom (`kernels/k5_experts.cu`):** only if Triton/CUTLASS e_192 lands in the 0.30–0.45 dead zone. The three+one fixes of §4.2 are mandatory before trusting any custom e_192.

### 6.2 Frozen comparison window (change NOTHING between runs)
512 prompt / 128 decode / greedy / temp 0 / seed 0 / **B=1** / **ctx 2K** (weight-bound, KV negligible — keeps TPOT measuring *weight* e, since KV crosses weight only at ~28K). TPOT = mean inter-token over tokens 2–128 (`bench/measure.py` already does exactly this and streams the same OpenAI SSE contract as the UI, so demo == benchmark).

### 6.3 Measuring e (the go/no-go)
**PRIMARY (kernel red-team makes this the gate, not a cross-check):** Nsight Compute on the ISOLATED MoE GEMV → `dram__bytes_read.sum / gpu__time_duration`, then e = achieved_BW / 4.8e12. **CROSS-CHECK:** TPOT floor-subtraction via `bench/roofline.analyze()`: `e = ideal_weight_read_ms / (TPOT_ms − comms_ms − host_ms)`, subtracting the **unchanged ~0.3–0.5 ms/step** floor (must subtract or you under-report e). If the two disagree >15%, trust Nsight — the floor-model error band exceeds the ~2% break-even margin at the knife-edge, so floor-subtraction alone cannot decide the gate. Wire measured e into `bench/sweep.py → bench/results.jsonl`.

### 6.4 Change-one-thing order (`bench/sweep.py` DoF)
1. **layout** ep8 → tp4ep2 → tp8 (decides the gate; yields e_384 then e_192) ← **decision here**
2. weight_dtype bf16 → fp8 (NOT int4/FP4 — no TP8-compatible block checkpoint; anchor FP8)
3. graph off → on
4. moe_backend (SGLang) triton → cutlass → flashinfer
5. kv_dtype fp16 → fp8 (only >~28K; out of the e-window)
6. spec off → ngram → eagle3, draft_len (141 GB makes a resident draft free; measure AFTER the layout gate, never mixed into e)

### 6.5 Quality gates (precision in play → run BEFORE accepting any speed win)
Order, fail-fast: **bitwise** (temp 0, seed 0, 50 prompts; requant must reproduce the BF16 token sequence, first-divergence ≥ 64) → **PPL** (Δ < 0.5% on a 2K slice; hard fail >1%) → **task** (200×GSM8K or MMLU subset, ±1 pt). Check weight_scale sharding axis after block-64 requant (vLLM **#41511** is a known FP8/W4 TP scale-sharding bug). Spec-decode must be **output-identical** to greedy non-spec.

---

## 7. Honest decision boundary & fallbacks

> **TP8 WINS only when ALL hold:** ctx ≤ ~8K **AND** e_192/e_384 > ~0.60 (e_192 ≥ ~0.50, GO-threshold 0.55 for margin) **AND** the four kernel rewrites have shipped. In-window advantage: **~1.1–1.3x** over TP4xEP2 (NOT 1.5x).
> **TP4xEP2 WINS (day-0 default)** when ANY of: ctx ≥ ~8K (4-KV-head/8-rank mismatch forces TP8 KV-replication at 4x KV-BW; 28K → TP4xEP2 +22%), OR e_192 < ~0.50 (skinny-GEMV dead zone — includes all stock-kernel configs), OR block-64 requant not landed (TP8-FP8 unlaunchable).
> **EP8 NEVER wins at B=1** (321 → 288 tok/s skewed; no shared-expert dispatch hiding). Last-resort fallback only.

| Trigger | Fallback | Day-0? | Ceiling (H200 FP8) |
|---|---|---|---|
| e_192/e_384 ≤ 0.60, e_384 healthy | **TP4xEP2** (384 cols) | **YES — stock checkpoint, no requant/patch** | ~936 weight-only; ~450–560 e2e; the safe ship |
| e_192 < 0.50 | grouped-expert column tiling (CUTLASS / SGLang `--moe-runner-backend cutlass`) | partial (flag day-0; custom kernel research-grade) | recover e toward 0.50–0.55 |
| both TP8 paths fail e or quality | EP8 | YES | ~321 weight-only; never wins; last resort |

**Launchability gate precedes everything:** no block-64 requant ⇒ TP8-FP8 cannot launch ⇒ TP4xEP2 is the automatic day-0 ship regardless of e. If requant slips, stay on TP4xEP2 — it is a complete product, not a degraded mode.

**BW-invariant, explicitly UNCHANGED from H100 (state every time):** the break-even efficiencies (e>0.25 vs EP8; e_192/e_384>~0.60 vs TP4xEP2), the KV/weight crossover *contexts* (~28K, ~224K), the comms floor (~0.14–0.28 ms), 132 SMs, 50 MB L2, all kernel tiling. **Defensible headline: ~1.1–1.3x in a narrow short-context window, NOT 2.4–3.2x and NOT 1.5x.**

---

## 8. Phased implementation checklist

| Phase | Action | Expected tok/s | Exit criterion |
|---|---|---|---|
| **0 — Day-0 config (TP4xEP2)** | Launch TP4xEP2 FP8 on STOCK block-128 (vLLM TP4+EP, or SGLang). CUDA-graph, `NCCL_PROTO=LL`, `NCCL_MAX_NCHANNELS=2`, KV split /4. Update `server/topology.py` H100/81920→H200/141GB + add tp4ep2 mode. | ~450–560 e2e (ctx≤8K); ceiling ~936; e_384 ~0.75–0.78 | Stable B=1 stream; passes bitwise/PPL/task gates; e_384 measured via Nsight, recorded as the break-even denominator |
| **1 — Measure e** | Block-64 requant (192/64=3) → TP8 launches. Re-run quality gates (check scale-sharding axis). Measure e_192 PRIMARY via Nsight isolated, CROSS-CHECK via TPOT floor-subtract at ctx 2K. | stock-kernel TP8 ~0.30 → loses; this phase only produces the number | e_192 & e_384 both measured on isolated kernel, frozen window. GATE: proceed only if e_192 ≥ 0.55 is reachable; if stock <0.40 with no path >0.50, STOP, stay TP4xEP2 |
| **2 — Kernel work** | Apply 4 non-negotiables to `k5_experts.cu`/`common.cuh`: N-major down-proj re-layout (kills :39 strided read), DSMEM cluster reduce (replaces :42 atomicAdd), vectorize scalar `deq` (common.cuh:39, 128-bit/16-fp8), dynamic cuTensorMap descriptors. CUDA-core (not wgmma), one CUDA graph. | ~700–850 e2e (ctx≤8K) IF e_192≥0.52; ~1.1–1.3x over TP4xEP2 | Nsight e_192 ≥ 0.55 on rewritten isolated kernel AND ratio >0.60 AND quality gates pass. If stalled 0.40–0.50, try CUTLASS grouped-expert tiling before conceding |
| **3 — Long-ctx TP4 + console** | ctx>~8K → route to TP4xEP2 (split KV). Do NOT raise TP past 8 (NVSwitch cliff). Optional TP8 KV-split (option B, +KV all-gather) only if single long-ctx TP8 box mandated. Wire console telemetry (§9). Add resident EAGLE/MTP draft after the layout gate. | TP4xEP2 long-ctx ~345–529 (28K); TP8+spec in-window additive | Console live-asserts busiest/mean≈1.0 under tp8 and realized_fraction×1424≈measured; both layouts selectable; long-ctx routed to TP4xEP2; spec output bitwise-identical |

---

## 9. How the console validates it live

Emit in the SSE `x_telemetry` (per-step) / `x_summary` (per-request) frames the harness already carries (`server/mock_engine.py`, `bench/measure.py`):

1. **`busiest_rank_bytes` vs `mean_rank_bytes`** (GB FP8/GPU). The **direct visual proof of the inversion**: ratio ~2.597 under EP8 (→3.37 Zipf, one GPU hot while others idle) **collapsing to ~1.00 under TP8** (every GPU reads 8/8 of each active expert's 192 cols — imbalance structurally eliminated). This requires `server/topology.py` to gain a TP8 placement mode (every expert's columns on every GPU) replacing the current round-robin EP map.
2. **`realized_fraction_of_roofline`** = measured tok/s ÷ `roofline.roofline_tok_s(ctx)`, plotted against the **1424 (TP8)** and **321 (EP8)** weight-only horizons — surfaces e continuously and flags the >0.60 ratio gate.
3. **`comms_ms` share** — critical on H200: because the weight ceiling rose x1.433 but comms did NOT, `comms_ms/TPOT` is a larger fraction; the console shows when comms tuning (0.28→0.14 ms) is the binding lever vs bandwidth.

Supporting fields: `ctx`, `weight_dtype`, `kv_dtype`, `layout` (tp8/tp4ep2/ep8), `tpot_ms`, **`kv_read_GB` vs `weight_read_GB`** (so the KV/weight crossover is visible live — and so the operator can SEE TP8 cross into the KV-bound losing regime against TP4xEP2 at ctx ~8K), `e_192`/`e_384` once measured.

**Live assertions:** `assert busiest/mean ≈ 1.0 under tp8` and `realized_fraction × 1424 ≈ measured tok/s`. If either breaks, the box is NOT realizing spec'd TP8 and the telemetry says so immediately. Add a third: `assert tpot(tp8) < tpot(tp4ep2) only while ctx ≤ ~8K` — the console should flip the recommended layout to TP4xEP2 the moment context crosses the KV boundary, making the honest decision boundary operationally enforced rather than merely documented.

---

### Files to touch (absolute paths)
- `/Users/charles/Desktop/InferenceHackathon/bench/measure.py` — frozen-window streaming harness (TPOT over 2–128); implements §6.2.
- `/Users/charles/Desktop/InferenceHackathon/bench/roofline.py` — H200 constants present (`HBM_BW=4.8e12`, `HBM_CAP=141e9`); docstring/`--help` still say "8xH100" — update text only.
- `/Users/charles/Desktop/InferenceHackathon/bench/sweep.py` — change-one-thing DoF + `NEXT_LEVER` tree → `bench/results.jsonl`; DoF already has tp8/tp4ep2/ep8.
- `/Users/charles/Desktop/InferenceHackathon/kernels/k5_experts.cu` — line 39 K-major strided down-proj read, line 42 atomicAdd reduce; line 31 un-hoisted global `y[k]` — the e-killers of §4.2.
- `/Users/charles/Desktop/InferenceHackathon/kernels/common.cuh` — line 32 "stored K-major" contract; line 39 scalar `deq` (its TODO: vectorize to 128-bit) — the biggest shipped-kernel e-killer.
- `/Users/charles/Desktop/InferenceHackathon/server/topology.py` — hardcodes `H100`/`81920 MB`/round-robin EP; MUST add a TP8 column-on-every-GPU mode and update to H200/141 GB to drive console signal #1.