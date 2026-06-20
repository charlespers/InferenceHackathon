# Spec decode in production (chat) vs the benchmark — what τ actually delivers

Every on-box number is **greedy · ctx 512 · single-turn**. The product (the console) is **temperature>0 ·
long-context · multi-turn chat**. Spec's realized gain differs across that gap — in both directions. This
reconciles the lab τ with what a real user sees, so we don't over- or under-claim the convergent answer.

## 1. Temperature>0 LOWERS acceptance (the lab greedy number is optimistic)
EAGLE3 / spec is **lossless at any temperature** (speculative sampling preserves the target distribution),
but the **acceptance rate falls** as temperature rises: at temp 0 the draft only needs the argmax to match;
at temp 0.7 it must match a *sampled* draw, and the draft↔target distributions agree less per position. So:
- greedy benchmark: τ ≈ 3.0–3.5 (published) → ~3× (floor-bound).
- typical chat (temp 0.7, top-p 0.9): τ ≈ **2.2–2.8** → ~2.2–2.5× realized.
**Measure τ at the product's actual sampling params**, not just greedy — `run_eagle3.sh` should add a
`--temperature 0.7` arm. The headline "~3×" is a greedy ceiling; budget ~2.5× for chat.

## 2. Long context RAISES spec's value (the short-ctx benchmark is pessimistic)
Spec amortizes the per-step floor **and the KV read** over τ (the W×D verify shares the KV — `long-context-chat.md`).
As a chat grows (ctx 8K→128K), the KV-read term grows and spec divides it by τ. So spec's *relative* value
**increases** with conversation length — the opposite of the short-ctx benchmark's regime. Net: temp pulls the
realized gain down (~2.5×), context pushes it back up as chats lengthen.

## 3. Multi-turn: spec + prefix-cache are the chat pair
- **TTFT:** prefix caching makes turn-T re-prefill only the new user message (the whole history KV is cached) →
  TTFT ~one decode step every turn (`ttft-analysis.md`). Mandatory for chat.
- **decode:** spec gives ~2.5× per turn, growing as the cached history lengthens (#2).
- `latency_budget.py --turns T --ctx <history> --spec-tau 2.5 --prefix-cache` projects the per-turn budget.

## 4. Net production estimate (honest)
| | greedy bench (today) | chat production (temp 0.7) |
|---|---|---|
| spec τ | ~3.0–3.5 | **~2.2–2.8** |
| decode tok/s (cheap-wins stack) | ~508 | **~370–430** |
| TTFT (turn ≥2, prefix-cache) | ~10 ms | ~10 ms (same) |
| grows with chat length? | n/a | **yes** (KV-amortization) |

So the realistic single-user **chat** win from the convergent answer is **~2.5× decode + ~50–100× TTFT**, i.e.
perceived latency from ~2.3 s → ~0.5–0.6 s per 128-token turn, *improving* as the conversation lengthens. Still
the dominant lever — just don't quote the greedy ~3× to a temp-0.7 user.

## Action
- Add `--temperature 0.7` (and a long-`--ctx`) arm to the spec slot benches so the **product** τ is measured,
  not just the greedy ceiling.
- Surface the **live accept-rate** in the console (the contract already has `x_summary.spec_accept_rate`) —
  it's the knob that tells a user whether spec is paying on *their* prompt, and it varies with temp + content.
