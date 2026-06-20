# Long-context chat: KV is the only term that grows (and it's a *post-floor* lever)

The console is a **chat** — context accumulates every turn (system prompt + history). But every benchmark is
512 tokens (short), where the floor dominates and KV is negligible. As a real chat grows to 8K–128K, KV-read
grows ∝ ctx and becomes the term that degrades tok/s. Quantified on the calculator (`tools/latency_budget.py
--ctx-sweep`, proven levers: prefix-cache + spec τ=2 + comms 8µs + eff 0.30):

| ctx | TPOT (bf16 KV) | tok/s | TPOT (fp8 KV) | tok/s |
|---|---|---|---|---|
| 512 | 3.45 ms | 290 | 3.44 | 291 |
| 4K | 3.53 | 283 | 3.48 | 287 |
| 16K | 3.83 | 261 | 3.63 | 275 |
| 32K | 4.22 | 237 | 3.83 | 261 |
| 64K | 5.00 | 200 | 4.22 | 237 |
| 128K | 6.57 | 152 | 5.00 | 200 |

## Reading it
- **At short ctx, KV is invisible** (the floor's 3.4ms dominates) — so, like the other weight levers, KV quant
  does nothing until the floor is fixed. (At the *current* floor-bound 11.67ms TPOT, KV is even more invisible.)
- **Post-floor, KV-read is the only growing term:** bf16 KV costs ~half the tok/s by 128K (290→152). fp8/INT8
  KV **halves that growth** (128K: 152→200 tok/s, ~+30%) — the right long-context lever, and accuracy-cheap
  (per-channel-K / per-token-V, KIVI-style; `b1-latency-architecture.md` §KV).
- **Crossover:** KV overtakes the (post-floor) weight+comms term around **~32–64K** — below that, don't bother
  with KV quant; above, it's the main lever. (Beyond ~64K, also consider attention sparsity / eviction.)

## Chat-specific structure (why prefix caching + KV quant pair up)
A T-turn chat at turn T has ctx ≈ T·(prompt+answer):
- **Prefix caching is mandatory for chat** — turn T re-prefills only the *new* user message, reusing the
  cached KV of the whole history → TTFT stays ~one decode step every turn (not re-prefilling T turns). This is
  the dominant chat-TTFT lever (`ttft-analysis.md`), and it's what *holds* the growing KV in cache.
- But the cached KV **is** the growing context, so **decode** TPOT still rises with turn T per the table →
  **fp8 KV** keeps per-turn tok/s up as the chat lengthens. Prefix-cache (TTFT) + fp8 KV (TPOT) are the chat
  pair; `latency_budget.py --turns T --ctx <history>` projects the per-turn budget.

## Spec decode amortizes the KV-read too → it's a long-context lever, not just a decode one
The batched verify (W×D positions in one forward) attends all those query positions against the **same
shared KV** — so the verify reads the KV **once** for the batch, exactly as it pays the floor once. So spec
amortizes **floor + weight + KV** by τ. At long context (KV-dominant) this matters more, not less: the growing
KV-read is divided by τ. So the chat lever stack is actually **prefix-cache (holds the KV, TTFT) + fp8 KV
(halve the per-read) + spec (amortize the read by τ)** — three multiplicative effects on the growing-KV term.
`latency_budget.py --proven --ctx-sweep` already reflects this (the τ=2 divides the whole TPOT incl. KV);
adding fp8 KV (`--kv-dtype 1`) stacks on top. Net: spec is the *universal* lever — floor-bound (short ctx) AND
KV-bound (long ctx) — which is another reason the team's EAGLE3 convergence is right.

## Priority placement
Long-context KV quant sits with the other **weight-ish levers: LAST in the floor-bound regime, then the #1
*long-context* lever once the floor is down and chats run long.** For the current short-prompt benchmarks it's
correctly untested; flag it as the lever that activates when the product runs real (long) conversations.
