# Native e2e spec-decode loop — wiring plan against Charles's existing kernels

**LOOP-A → Charles.** Goal: turn the *proven pieces* into a *running, lossless, measured* end-to-end
native spec decoder. Nothing here is a new engine — it's an assembly plan over your `.cu` files +
LOOP-A's proven-lossless accept/control. Build order is smallest-runnable-first so each milestone is a
real on-box number with a parity gate, not a projection.

## What exists (don't rebuild)
- `decode_step_tp8.cu` — WORKING TP8 plain decode step (76 tok/s w/ NVLS). Per layer: Wqkv row-shard →
  attn → Wo col-shard → **AR#1** → K4 router (replicated) → MoE expert-shard → MoE-down → **AR#2**;
  final replicated RMSNorm + VOCAB-sharded lm_head + cross-rank argmax. Uses the **M=1 GEMV** idiom and
  a single dummy layer reused 94× (latency proxy → meaningless logits).
- `spec_verify_forward_gemm.cu` — the **M=k fp8 GEMM verify primitive** (`LtGemm`, TN, opA=T/opB=N,
  pinned algo), PROVEN FLAT M=1..16 (T16/T1≈1.003). This is the GEMM that replaces the M=1 GEMV.
- `spec_decode_loop.cu` — the timing PROOF (`effective tok/s = E[accepted]/(T_verify(k)+T_draft)`) +
  the double-win measurement. It measures/models; it does **not** run with real weights/head/acceptance.
- LOOP-A engine (`engine/src/spec/`, djamoils-work): `Eagle3Engine::decode` + `accept_multi_drafter`,
  **proven lossless** (`decode_is_lossless_invariant_to_lambda_and_verify_depth`) — the accept/control
  reference + the parity oracle.

## The gap → 4 runnable milestones (each: real number + parity gate vs vLLM EAGLE3 greedy)

### M1 — Real M=k forward (the architecture flip; the bulk of the win)
Extend `decode_step_tp8` from "dummy 1 layer × 94, M=1 GEMV" → "real 94-layer fp8 weights, **M=k GEMM**".
- **Swap GEMV→GEMM** on every weight panel (Wqkv, Wo, router gate, MoE gate/up/down, lm_head) using the
  `LtGemm` recipe from `spec_verify_forward_gemm.cu`. Keep the SAME TP8 sharding + the 2 ARs/layer.
- **New kernel — M=k causal attention**: the k query rows attend to the growing KV cache **and causally
  to each other** (k×k lower-triangular mask). This is the only genuinely new compute vs the plain step
  (the proj/MoE are just GEMV→GEMM swaps). Append the k accepted keys/values to the KV cache after accept.
- **MoE at M=k**: the k rows route to the UNION of their top-8 experts; load that union once and GEMM all
  k rows through it. (Your flatness result says this union read is amortized to the 16-tile → no extra
  cost ≤16-wide. This is also why route-aware union-shrinking is NO-GO at B=1 — the union is read flat.)
- **Real weights**: all 94 layers' fp8 panels resident (235B FP8 is cached on box). Drop the latency-proxy.
- **Gate**: feed k=1 and assert the produced logits == `decode_step.cu` single-token reference (your
  existing `run_correctness_check` extended to k rows); the M=k=1 path must match plain decode exactly.

### M2 — EAGLE3 draft head (produces the k candidates)
- Load the RedHat head (`...speculator.eagle3`, 2.3 GB, draft_vocab 64000), draft_tp=8 if it shards.
- Head input = the target's **aux hidden states at layers [1, 46, 90]** (the EAGLE3 side-tap) — tap these
  out of the M=k forward in M1 and feed the head. Head autoregresses k steps → k draft tokens + logprobs.
- **draft_vocab map**: head logits are in 64000-draft-space → map to target token ids via the d2t table
  (LOOP-A's `DraftVocabMap`, the exact 1.4-bug fix: draft-space log-softmax + d2t). Get this right or
  acceptance collapses (the τ=1.4 failure mode).
- **Gate**: head first-position accept ≈ 0.75, mean accept length τ ≈ 2.7 (the RedHat-head published range).

### M3 — Acceptance (LOOP-A's proven-lossless accept) — exact losslessness lives or dies here
Port `accept_multi_drafter` to the device (or run it host-side on the k logit rows — cheap at B=1). TWO
contracts that the GEMM verify MUST satisfy (`engine/docs/spec-accept-correctness-notes.md`):
1. **Verify layout (off-by-one):** the row used to score `d_pos` must be `P(· | context + d[0..pos])`,
   i.e. the forward output **at the slot of `d_{pos-1}`** (and at `ctx_last` for pos 0) — NOT the output
   sitting at `d_pos` (which predicts `d_{pos+1}`). Read one slot to the LEFT.
2. **Bonus on full accept:** verify **k+1** positions (append one slot) OR do one extra forward, then
   sample `P(· | context + accepted)`. Do NOT reuse the last row (greedy stand-in → duplicates).
- **Gate**: greedy output must be **bit-identical to vLLM EAGLE3 greedy** on the same prompts (parity 1.0,
  the same gate vLLM passed). LOOP-A's `Eagle3Engine::decode` is the CPU oracle to diff against.

### M4 — The loop + measured e2e tok/s
- Assemble: head (M2) → M=k GEMM verify forward (M1) → accept (M3) → emit accepted+bonus → append to
  context + KV cache → advance head → repeat. ARs use **out-of-place NVLS** (your 1-barrier follow-up →
  full 3.84 µs comms win).
- **Gate / deliverable**: real end-to-end **spec'd tok/s** (the thing `spec_decode_loop.cu` projected at
  923–1452) — now measured, not modeled — with parity 1.0. This is the path-to-1000 proof closed.

## Division of labor (no duplication)
- **Charles (.cu):** M1 (GEMV→GEMM swap + M=k causal attention + real weights), the NVLS ARs, the head
  forward kernels, the device loop. The hard kernel work — your lane.
- **LOOP-A:** M3 accept/control as the **reference + parity oracle** (proven lossless), the **two verify
  contracts**, the **`DraftVocabMap` d2t** logic, and **max-τ config** (`slot_spec_tune.sh`: k=8/draft_tp=8).
- **vLLM EAGLE3:** the lossless reference every milestone diffs against.

## De-risking note
The single unproven link between "76 tok/s plain + flat verify proven" and "1000+" is **M1's M=k causal
attention over real KV + the head's real τ in the live loop**. Everything else is a GEMV→GEMM swap or a
ported CPU routine. So: build M1 first against the k=1 parity gate — if M=k=1 matches plain decode and
M=k=8 stays flat on real weights, the architecture is de-risked and M2–M4 are assembly.
