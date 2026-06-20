# Speculative Decoding × CUDA Graph (the crux) — B=1, 8×H100

The single biggest B=1 lever, and the hardest to combine with graph capture. Public model +
standard speculative-sampling techniques only.

## 1. Why it's the biggest lever at B=1

Decode is memory-bandwidth bound: one token = one full ~21.6 GB weight read. **Verify K draft
tokens in ONE forward** and the weights are read once for all K+1 positions (the K+1 rows are a
tiny "batch" — still memory-bound, but the weight read is amortized). Accepting m tokens per
verify ⇒ ~m× fewer weight-read passes ⇒ ~m× tok/s, minus draft cost.

**Speedup math** (per-position accept prob α, draft length γ): expected accepted per verify
`E = (1 − α^(γ+1)) / (1 − α)`.
- α=0.7, γ=4 → **2.77 tokens/pass**; α=0.8, γ=5 → **3.69**.
- Net ≈ E × (target_pass / (target_pass + draft_cost)). EAGLE-style draft ≈ 5–15% of target
  ⇒ **~2.0–2.7× realistic** at B=1. Prompt-lookup draft is ~free but lower α.

## 2. Draft method (start → upgrade)

1. **Day-1: prompt-lookup / n-gram (zero training).** Draft the next γ tokens by matching the
   last n-gram (n=3) against the prompt+generation; emit up to γ=6. Accept 20–40% on
   structured/agentic/quoting prompts (high when output echoes context). Ship this first.
2. **Upgrade: EAGLE-3** (SGLang supports EAGLE speculative for Qwen-family). A small draft head
   conditioned on the target's hidden states; α≈0.7–0.8, γ=4–5. Biggest real win.
3. **If the Qwen3 variant ships an MTP head**, use it as the drafter (no separate model). Verify
   on the box; Qwen3-235B base does not obviously ship MTP, so plan on EAGLE-3.

First guess: **prompt-lookup now (n=3, γ=6); EAGLE-3 γ=4, accept-threshold tuned next.**

## 3. The capture problem and the fix

Per step, the model accepts m ∈ [0, γ] tokens — **variable**. A naïve CUDA graph captures
fixed shapes, so variable output length appears to break replay. It doesn't have to:

**Fix — fixed max-draft window + masked commit (single static graph):**
- Always run the verify forward over **exactly γ+1 positions** (pad the draft to γ). Shapes are
  static ⇒ one capturable graph (incl. the EP all-to-all carrying γ+1 rows).
- The only variable is *how many* of the γ+1 outputs you keep. That is **data, not shape**:
  - device computes, per position, accept/reject (greedy: `draft[i]==argmax(target_logits[i])`;
    sampled: speculative-sampling rule, accept w.p. `min(1, p_t/p_d)` else resample from
    `norm(max(p_t−p_d,0))` — all on device).
  - a **device-side counter** `n_accept` = length of the accepted prefix (first reject stops it).
  - **masked KV commit:** the KV-cache write for the γ+1 draft positions is gated by
    `pos < n_accept` (predicated store), so only accepted tokens persist; the KV write offset
    advances by `n_accept` via a device counter. No reshape, no realloc, no host round-trip.
- Rejected tail is simply never committed; next step re-drafts from the new accepted end.

**Alternatives if needed:** CUDA-graph **conditional nodes** (CUDA 12.4+) for an explicit
accept/redraft branch; or **bucketed graphs** (capture for a few common m). The fixed-window
approach is simplest and keeps one graph — start there.

## 4. Zero per-token D2H sync (keep it on-device)

Everything per step stays on device so the graph never needs a host hand-off:
- draft generation (EAGLE head / n-gram lookup table in device memory),
- target verify forward (the captured graph),
- accept/reject test + speculative-sampling resample,
- `n_accept` counter, masked KV commit, KV-offset advance,
- append accepted token ids to a **device output ring**; host drains the ring lazily (every N
  steps), never in the critical path.

This is what lets the graph replay back-to-back. (Greedy + on-device argmax is the simplest
first target; on-device categorical/spec-sampling for temperature>0 is the follow-up.)

## 5. Correctness

Speculative sampling preserves the target distribution **exactly** (Leviathan 2023 / Chen 2023):
accept token with prob `min(1, p_target/p_draft)`, else sample from the normalized positive
residual `max(p_target − p_draft, 0)`. Greedy is the degenerate case (accept iff draft == target
argmax). The fixed-window/masking changes nothing in this math — it only runs verify over a
padded window and discards the rejected tail before KV commit. **Output distribution == no-spec.**
Validate on the box with a greedy-equivalence test (spec vs non-spec greedy must be byte-identical).

## 6. Composition with EP and the whole-step graph

- The verify forward over γ+1 positions runs the full 94-layer model **including the EP
  dispatch/combine all-to-all** (now γ+1 rows, still KB-scale). Capture the entire thing —
  K1–K5 ×94 + lm_head over γ+1 positions + accept/commit — as **one graph** (see
  `ep-parallel-schedule.md` §6).
- The draft step: EAGLE head is tiny; capture it too (or run eagerly before the verify graph).
- Keep a **non-graph fallback path** (eager) so you always have a working number while the
  captured-spec path is being debugged.

## 7. First-guess parameters & the measurement

- prompt-lookup: n=3, γ=6. EAGLE-3 (when ready): γ=4, accept-threshold start 0.5 (sampled),
  exact-match (greedy).
- KV: speculative tokens append; on reject, roll back the KV write offset (device counter) — no
  realloc.
- **Measure:** accept rate α, mean accepted/verify E, tok/s with vs without spec, and that the
  captured-graph path shows no per-token D2H sync in Nsight (`bench/`).

**Open TODOs for the box:** confirm SGLang EAGLE-3 support for qwen3_moe + exact flags; verify
CUDA-graph capture of the comms backend; implement masked-commit + device counter; greedy
byte-equivalence test; tune γ / threshold against accept rate.
