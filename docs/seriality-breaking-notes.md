# Notes on `research/seriality_breaking.md` — the wall is 3.2× worse (measured), and spec is the amortizer

Two contributions to the team's seriality-breaking research (the comms-count attack), from the measured data
+ the floor-amortization analysis. Both *strengthen* its thesis.

## 1. Use the MEASURED collective latency: 16µs, not 5µs → the wall is 3.0ms, not 0.94ms
`seriality_breaking.md` §0 quotes comms = 2 AR/layer × 94 × **5µs** = 0.94ms (the model's `hardware.py`
default). But `nccl-tests` **measured all-reduce@8 ≈ 16µs** (`config-sweep.md`, `results-reaction-01.md`).
So the real wall is:

| collectives/layer | @5µs (model) | @**16µs (measured)** | @4µs (one-shot tuned) |
|---|---|---|---|
| 2.0 (today, TP) | 0.94 ms | **3.01 ms** | 0.75 ms |
| 1.0 | 0.47 | 1.50 | 0.38 |
| 0.5 | 0.235 | 0.75 | 0.19 |
| 0.0 (deferred) | → leaves critical path | → leaves critical path | → leaves critical path |

So comms isn't "the single largest term by a hair (0.94 vs 0.82)" — at fp8 it's **3.01ms vs 0.82ms weight,
~3.7× the weight term.** The seriality-breaking thesis is **understated 3.2×**; cutting the count is even
more the game than the doc says. (Also: two attacks compose — cut the *count* AND the *per-collective
latency* (E0b: 16→4µs). At 0.5/layer + 4µs that's **0.19ms** comms, a 16× cut from today's 3.0ms.)

## 2. Spec decode IS the "amortize" lever — quantified
The doc's taxonomy is reorder / amortize / remove. **n-gram/route-aware spec is the concrete amortizer** and
deserves a row: the verify is ONE batched forward → **one set of 188 collectives for E[accepted]=τ emitted
tokens** → the *per-emitted-token* collective count drops by τ:

```
comms per emitted token = (188 × latency) / τ      # τ = accepted tokens / round
```
At the measured 16µs and τ=2: **3.0ms → 1.5ms/token**; τ=3 → 1.0ms. This is "0.5–1.0 collectives/layer
*effective*" achieved **without touching the layer structure** — pure amortization. And it composes with
`spec_moe_model.py`'s route-aware drafting (which pushes τ up without the union tax exploding) and with the
count-reduction techniques (fewer collectives × fewer payments). See `spec-decode-floor-bound.md`.

**Why this is the cheapest seriality-break:** reorder/remove techniques (deferred all-reduce, sharded-norm,
fused residual-AR) are real engine surgery; **n-gram spec is a config flag** and already amortizes the count
by τ today. It should be the *first* seriality-break tried, in parallel with the structural ones.

## 3. A caution on the count-reduction targets
Getting below 2 AR/layer means deferring/removing a reduction the **RMSNorm depends on** (norm needs the full
post-residual hidden). Options like sharded-RMSNorm still pay a (tiny-payload) reduction → the *count* is
unchanged even if the payload shrinks (and at B=1 it's latency-bound, so payload doesn't matter). The
genuine count-cuts are: (a) **fuse the O-proj AR with the next op** if a layout keeps experts on the same
shard (avoids re-gathering), (b) **device-initiated NVSHMEM** to move the AR *off* the critical path
(overlap with the GEMV tail) — that's the "0.0/layer deferred" row and is the highest-ceiling but hardest.
Spec-amortization (#2) and E0b latency (16→4µs) get most of the win at a fraction of the engineering.

## Net
Update `seriality_breaking.md` §0 to **16µs** (comms 3.0ms, the dominant term by 3.7× at fp8), and add
**spec-decode as the config-level amortizer (÷τ collectives/token, ~2× today)** as the first seriality-break,
composing with route-aware drafting and the structural count-cuts.
