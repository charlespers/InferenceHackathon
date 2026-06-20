# 2-bit experts — the quality gate + probe design (the QUALITY side of the frontier lever)

**LOOP-C, 2026-06-20.** `first-principles-frontier.md` (Charles) proposes quantizing the experts to ~2 bits
(experts = 66% of bytes, claimed "most quant-tolerant") → byte floor ~2400 tok/s, opening the 3000-class. That
doc owns the **speed/kernel** side (2-bit GEMV microbench, M-sweep). This doc owns the **quality** side, which
is unowned and is the gate that decides GO/NO-GO. Extends `int4-experts-quality-gate.md`'s methodology. No GPU.

> **VERDICT (literature-confirmed): NO-GO for uniform 2-bit experts. CONDITIONAL only at ~3-bit (≈2.7–3.0
> effective bits, mixed-precision). And it's a CUSHION, not the path.** fp8 + spec already projects ≥1000
> (`ladder_to_1000.py`), so 2-bit experts is a *cushion* (relaxes the MBU/spec requirement), exactly like int4
> — and int4 was KILLED at B=1 (0.55× fp8, issue-bound unpack, `results-reaction-04.md`). 2-bit is **more**
> unpack-bound, so it faces a DOUBLE gate. The literature (below) now resolves the quality gate **before any
> probe**: uniform 2-bit on a large MoE drops 12–29 accuracy points and Qwen3 specifically collapses to ~random
> at ≤2-bit — it will not pass a 98% greedy-parity bar. **Do not build the 2-bit dequant kernel.** If a cushion
> is wanted, the viable target is **~3-bit mixed-precision** (router fp8/fp32, experts 3-bit + ~1% fp8 outliers),
> which the probe below should still *validate* (Qwen3 is documented to be quant-fragile), and which yields a
> ~2100 tok/s floor — a real but modest cushion that is **not needed** if fp8+spec lands.

## Where it sits: a cushion, not the path
The honest byte math (corrected for per-group scales + outliers, which the "3.6 GB / 2400 tok/s" headline omits):

| scheme | eff bits/wt | expert GB | total GB | byte floor | MBU to hit 1000 |
|---|---|---|---|---|---|
| frontier pure-2-bit (headline) | 2.00 | 3.55 | 10.95 | ~2524 | 40% |
| 2-bit, group-64, +1% fp8 outliers | 2.39 | 4.24 | 11.64 | ~2375 | 42% |
| 2-bit, group-128, +1% outliers | 2.27 | 4.02 | 11.42 | ~2421 | 41% |
| **2.5-bit + 2% outliers (SOTA near-lossless realistic)** | **3.02** | **5.36** | **12.76** | **~2167** | **46%** |
| 3-bit, group-128, +1% outliers | 3.25 | 5.78 | 13.18 | ~2098 | 48% |

**The byte accounting is roughly fine** — even at a realistic SOTA-near-lossless ~2.5–3 eff-bits, the floor is
~2100–2200 and 1000 needs only ~46% MBU (vs fp8's ~78%). So 2-bit's value is **relaxing the kernel-MBU climb +
giving margin if spec/e under-deliver** — a cushion. It is NOT needed for 1000 if fp8+spec lands. Don't bank the
2400 headline (it ignores scale+outlier overhead); the realistic prize is ~2100–2400.

## The DOUBLE gate (both must pass; either kills it)
1. **Quality gate (this doc) — the binding risk.** 2-bit is at the bleeding edge of weight quantization.
2. **Kernel gate (frontier doc / Charles) — already a RED FLAG.** Naive int4 GEMV measured **0.55× fp8** at B=1
   (the nibble-unpack is ALU/issue-bound, not BW-bound). 2-bit packs 4 weights/byte → **more** unpack work per
   byte → strictly worse on the same axis *unless* a LUT/lattice dequant fused into the wgmma mainloop (TMA-
   staged) hides it. So the speed win is unproven AND the most-likely-failing kernel regime. The quality probe
   is cheaper to run, so run it first — no point building a hard kernel for weights that don't pass quality.

## Quality: first-principles risk (why 2-bit is much riskier than int4)
- **Coarseness.** 2-bit = **4 levels per weight**. Even with per-group scales, plain RTN at 2-bit is known-broken
  on LLMs (multi-point perplexity blowups). Usable 2-bit requires **codebook/lattice + calibration** (AQLM,
  QuIP#) — i.e., it is NOT a drop-in recipe like int4-AWQ; it's a research-grade quantizer with hours of
  calibration. The "few fp8 outliers" in the frontier scheme is load-bearing, not a footnote.
- **"Experts are quant-tolerant" — partially true, easy to overstate.** The redundancy argument (128 experts,
  each fires ~6%) is about *inter-expert* averaging across tokens. But within a single token, each of the top-8
  experts is **fully load-bearing** for its share of that token's FFN output — there is no averaging that hides
  a mis-quantized *active* expert. So tolerance is real but bounded; it does not obviously survive to 2-bit.
- **Depth compounding (the stale-TP lesson).** Qwen3-235B is **94 layers**. Small per-layer expert-output error
  feeds the residual stream and the **next layer's router** (router stays fp8, but its *input* is the perturbed
  hidden) → small errors can compound and/or flip routes across depth. This is exactly the mechanism that made
  stale-TP catastrophic (route flip → gibberish). It is **unquantified** for 2-bit experts and must be measured,
  not assumed. A per-token-parity number alone won't catch slow downstream drift — see probe step 3.

## The probe (extends `int4-experts-quality-gate.md`; cheap to prep — quantize + eval, NO new kernel)
Run with the team's `quality_compare.py` / `quality_probe.py`, fp8 reference. **Sweeps, not a single 2-bit point**
— the goal is to find the cliff and size the outlier budget, so the team knows the real floor.

1. **Bit-width sweep 4 → 3 → 2.5 → 2 bit** (experts only; attn/router/lm_head fp8). Greedy token-parity on a
   held-out chat+code set; target **≥98–99%** at temp 0. This locates the cliff (likely between 3 and 2 bit) —
   reporting *where it breaks* is the deliverable even if 2-bit fails.
2. **Outlier-fraction sweep** at the target bit-width (0%, 0.5%, 1%, 2%, 5% kept in fp8). Find the **minimum
   outlier budget that passes** → that sets the *true* effective bits (row in the table above) and the *real*
   byte win. If you need ≥5% outliers, the win shrinks toward 3-bit and the lever weakens.
3. **Depth-compounding probe (the key addition).** Log per-layer expert-output **relative MSE** vs the fp8
   reference, layer 0→93, AND the greedy divergence position over a 256-token generation. GO requires the error
   does **not** grow super-linearly with depth and divergence onset is late/absent. This catches the stale-TP
   failure mode that per-token parity on short outputs misses.
4. **Downstream eval delta**: small MMLU/GSM8K/needle slice, within **~1%** of fp8 (catches reasoning/recall loss).
5. **Per-expert sanity / mixed precision**: log worst-expert MSE; if one hot expert is an outlier, keep *it* in
   fp8 (cheap insurance — mixed-bit experts). Report how many experts need the fp8 fallback.

## Decision / order
- **fp8 + NVLS + spec is the path; 2-bit experts is a shelf cushion.** Prep is cheap (a quantization recipe +
  the probe above), so it's worth *preparing*, but **only ship if** the C/MBU/spec measurements say fp8 is short
  AND the quality probe passes AND the 2-bit dequant kernel beats fp8 (it didn't for int4).
- **Run the quality probe before the kernel.** It's the cheaper of the two gates and the more likely to fail. A
  clean NO-GO here saves the expensive Hopper 2-bit-dequant kernel work.
- **Conclusion (literature-confirmed, not just predicted):** uniform 2-bit experts FAIL the 98% parity gate —
  **NO-GO**. ~3-bit mixed-precision is the realistic near-lossless floor (CONDITIONAL, still validate on Qwen3),
  with a modest ~2100 floor that fp8+spec doesn't need. The bit-sweep probe is now mainly to *confirm the cliff
  location on Qwen3* and size a 3-bit config if a cushion is ever wanted — not to rescue 2-bit.

## Literature anchors (focused review, arXiv ids inline; cross-verified against primary sources)
**The 3-vs-2-bit cliff is universal.** 4-bit ≈ lossless; ~3-bit near-lossless with any calibration; **2-bit is
catastrophic for simple methods** and only "usable with a real gap" for codebook+**fine-tuning** methods:

| method | eff bits | quality at 2-bit | needs |
|---|---|---|---|
| RTN | 2.0 | **unusable** (Llama-2-7B ppl ~432; L3-70B ~460k) | — |
| GPTQ (2210.17323) | 2.0–2.2 | **collapses** (~20–60 ppl) | calibration |
| QuIP# (2402.04396) | 2.0 | usable, +1.0 ppl | RHT + E8 lattice + **fine-tune** |
| AQLM (2401.06118) | 2.0 | usable, +1.0–1.5 ppl | learned codebooks + **fine-tune** |
| QTIP (2406.11235) | 2.0 | best-in-class, +0.74 ppl | trellis + incoherence |
| SpQR (2306.03078) | **3.35–4.7** | near-lossless ONLY here | ~1% fp16 outliers |

**MoE-specific — bigger MoEs collapse HARDER at 2-bit (the worst sign for 235B):**
- Mixtral-8x7B uniform 3-bit **70.85% → 2-bit 58.73% (−12 pts)**; **Mixtral-8x22B 65.48% → 36.53% (−29, ~random)**
  (2604.06515). MxMoE (2505.05799): 2.25-bit = **+1.8 (Mixtral) to +4.4 (Qwen1.5-MoE) ppl**.
- **The "experts are quant-tolerant" claim traces to MoQE (2310.02410) — which is encoder-decoder NMT, weight-only,
  and uses QAT (training-aware), NOT PTQ on a decoder LLM.** It explicitly says *"2-bit still requires QAT; 3-bit
  does not."* The favorable result does **not** transfer to our PTQ pipeline. (Qwen3-235B has **no shared expert**,
  so that particular MoE risk is moot — but the router must stay fp8, which the scheme already does.)
- Recovery to ~3-bit quality costs **~2.75–3.5 effective bits** (mixed precision) — i.e. you're no longer at 2 bits.

**Depth compounding is real and published** (validates probe step 3): QEP (2504.09629) — *"growth of quantization
errors across layers significantly degrades performance,"* worst at extreme-low-bit. Greedy parity is *stricter*
than perplexity ("Accuracy is Not All You Need" 2407.09141 — token "flips"); **distribution-lossless parity with
fp16 (~99% token agreement) needs 5–6 bits** ("Statistically-Lossless Quantization," 2605.02404). 94 layers is
exactly where accumulation bites.

**200B+ MoE at ~2-bit — every working case is MIXED-precision and the labels mislead:**
- **DeepSeek-R1 671B**, Unsloth dynamic 1.58-bit: *"naively quantizing all layers breaks the model entirely →
  gibberish."* **Uniform 1.58-bit scored 0%**; only mixed (router fp32, attn/down/embed/head 4–6 bit, bulk
  experts 1.58) reached 69.2%.
- **Qwen3-235B** Unsloth "UD-Q2_K_XL" is **actually ~2.9–3.0 effective bpw** (quality data only at Q3-class).
  The Qwen3 quant study (2505.02214): **uniform 2-bit AWQ/GPTQ on Qwen3 = ~24% MMLU (random)**, and Qwen3
  *"exhibits more pronounced degradation under low-bit (≤3-bit)"* than other models. → 235B is doubly at risk.

**Effective-bits correction confirmed:** "3.6 GB" back-solves to 2.03 bits (bare payload, zero metadata).
Realistic near-lossless is **~2.3–2.9 eff bits = 4.1–5.2 GB → the 3.6 GB headline is over-optimistic by 15–45%**
(matches the table above). AQLM/QuIP# "2.0-bit" headlines *exclude* their fp16 codebooks/scales from the count.

**Net:** the quality gate is resolved adversarially in the literature — **uniform 2-bit experts is a NO-GO for a
lossless B=1 engine** (kill it before the kernel); ~3-bit mixed-precision is the only viable cushion, validate on
Qwen3, and it's not on the critical path.
