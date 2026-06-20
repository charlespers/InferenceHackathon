# EAGLE3 at B=1: use `draft_tensor_parallel_size: 8`, not 1 (and why the draft cost matters)

`experiments/eagle3/INTEGRATION.md` recommends **`draft_tensor_parallel_size: 1`** — "sharding a 1-layer head
over 8 is pure overhead." That's correct for **throughput**, wrong for **B=1 latency**, for exactly the reason
this whole session is about: at B=1 you're **bandwidth-bound**, so sharding the *weights* across GPUs cuts the
read time; you're not compute-bound, so the "extra GPUs are wasted" intuition doesn't apply.

## The draft cost is real (and my earlier "over-delivers to ~3×" was optimistic without it)
EAGLE3's draft head is ~1B params (~2 GB bf16). The draft generates the tree with **D sequential forwards**
(D≈5 for `num_speculative_tokens=5`). Per-step cost is the **weight read** of the head, not compute:
| layout | per draft step | × D=5 |
|---|---|---|
| **draft_tp=1** (head on 1 GPU) | 2 GB / 3.35 TB/s ≈ **0.60 ms** | **~3.0 ms** |
| **draft_tp=8** (head sharded /8) | 0.25 GB/GPU + ~2 all-reduce@16µs ≈ **0.11 ms** | **~0.55 ms** |

So `draft_tp=1` adds **~3 ms of draft** to every spec round — comparable to the verify's floor (~3 ms)! That
caps EAGLE3 at ~2.5–2.6× on the floor-bound engine (not the ~3× the free-draft model projects). **`draft_tp=8`
cuts the draft to ~0.55 ms → the floor-amortization dominates again → ~3× is restored.**

Plus: EAGLE3 consumes **3 aux hidden states** from the target (`eagle_aux_hidden_state_layer_ids:[1,46,90]`).
Under `draft_tp=1` those must be **gathered** to the single draft GPU each step (extra comms + sync); under
`draft_tp=8` they're already sharded to match the target's TP8 — **no gather.** Two wins, same change.

## The general rule (the session's thesis, applied to the draft)
- **Throughput** (many requests): the draft head's *compute* is the cost; sharding a 1-layer head adds
  collective overhead for little compute benefit → `draft_tp=1`. (The INTEGRATION.md intuition.)
- **B=1 latency:** the draft head's *weight read* is the cost; sharding it /8 cuts the read 8× for a ~32µs
  all-reduce tax → `draft_tp=8` wins. Same EP→TP / DP-vs-TP logic as the target (`b1-tp8-moe-rearchitecture-h200.md`).
- vLLM allows `draft_tensor_parallel_size` ∈ {1, target_TP}. **Use the target TP (8).**

## Refines the spec models
`spec_floor_model.py` / `tree_spec_optimizer.py` assume a **free** draft (true for n-gram). For EAGLE3 add
`+ draft_ms/round` to the denominator: `speedup = E[accepted] / (verify_cost + draft_cost)`. With `draft_tp=1`
the draft_cost (~3 ms ≈ 0.26 verify-units) noticeably bites; with `draft_tp=8` (~0.55 ms ≈ 0.05 units) it's
negligible and the free-draft projection holds. **This is also why free n-gram can beat EAGLE3 on repetitive
content** (zero draft cost) while EAGLE3 wins on prose (higher acceptance) — content-dependent, per
`spec-in-production.md`.

## Action (E6)
Set `"draft_tensor_parallel_size": 8` in the EAGLE3 config (not 1). Measure τ AND the draft-phase ms (vLLM
logs draft/verify split) to confirm the draft is ~0.5 ms, not ~3 ms. If vLLM rejects draft_tp=8 for this head
(some heads pin draft_tp=1), the ~3 ms draft is the price → expect ~2.5× not ~3×, and n-gram becomes more
competitive for repetitive prompts.
