# Spec decode in the floor-bound regime — it amortizes the FLOOR, so it's a top lever NOW

A reframing that the floor-bound data (`overhead-attribution.md`, `results-reaction-02.md`) forces, and that
**reverses two of my earlier conclusions.** Spec decode is not "the orthogonal weight multiplier, do it
last" — in the current floor-bound regime it is one of the **two levers that actually help** (the other being
fixing the floor), and the MoE verify-tax I warned about (`spec-decode-moe-tax.md`) **barely bites here.**

## The mechanism: the verify pays the floor ONCE, amortized over τ
The per-step floor — **188 all-reduces (×16µs ≈ 3ms) + launch + host + sampling** — is paid **per forward
pass**, not per token. Normal decode of τ tokens = **τ forward passes** = τ× the floor. Spec decode's verify
is **one batched forward** over the N·k draft positions → **one set of 188 all-reduces**, paid once,
amortized over the τ accepted tokens. The draft, with **n-gram/prompt-lookup, is free** (no model).

So in the floor-bound regime (floor ≈ 10ms of the 11.67ms TPOT):
```
verify ≈ a normal step (~11.67ms — floor-dominated; the N·k batch adds little)
spec emits τ tokens for ~one step  =>  speedup ≈ τ
```
At τ=2 on bf16-TP8 (85.7 tok/s): **~11.67ms / 2 ≈ 5.8ms/token ≈ 170 tok/s (~2×)** — the single biggest
immediately-available lever, bigger than fp8 (~7%) or comms tuning alone.

## Why the MoE verify-tax barely bites while floor-bound
`spec-decode-moe-tax.md` derived (correctly, in *weight* units) that a big tree reads the expert *union*,
making break-even τ ≈ 4.6 at k=8. But that union growth adds to the **weight** term — which is **only 14% of
TPOT** right now. The verify's real cost = **floor (≈10ms, paid once) + weight×union_factor (≈1.6ms × growth)**.
With the floor dominating, the union growth on the small weight term **barely moves the total**, so:
- **While floor-bound, k can be moderate (4–6), not tiny** — the tax is a rounding error on a floor-dominated
  verify. The earlier "k must be ≤2–3" is the *weight-bound* conclusion.
- **As the floor is fixed** (comms tuning + kernel efficiency move us toward weight-bound), the tax re-asserts
  and the optimal **k shrinks back to 2–3.** → **Make k regime-adaptive:** larger k now, smaller k as the
  floor falls (gate on realized speedup, exactly as `engine/spec/ RoundStats` can).

## This resolves the apparent contradiction with "fix the floor first"
"Weight levers (fp8/int4/adaptive-top-k) are invisible while floor-bound" — **true**, because they shrink the
14% weight term. **Spec decode is NOT a weight lever** — it amortizes the *floor* (the 86% that dominates),
so it is exactly the kind of lever that **does** help while floor-bound. The two things that move the needle
now: **(a) reduce the floor** (comms tuning, kernel efficiency) and **(b) amortize the floor** (spec decode).
They **stack** — cut the floor *and* divide it over τ.

## Action (elevates E6)
- **Run n-gram/prompt-lookup spec NOW**, on the current floor-bound engine, with **moderate k (try 4)** —
  expect ≈τ× (~1.5–2.5× on structured prompts). Free draft, no requant, no kernel work. This is reprioritized
  from "after the byte levers" to **alongside the floor fixes** (it's a floor-amortization lever).
- **Gate on realized tok/s** (not acceptance) and **shrink k as the floor falls** — when the trace
  (`E-attr`) and comms tuning move us toward weight-bound, the verify-tax returns and k→2–3.
- **Caveat:** n-gram only fires on repetitive/structured text; for general prose, self-spec (`E9`) or a
  trained MTP head — but those have a draft cost, so they amortize the floor less cleanly than free n-gram.

## One line
In a floor-bound regime, **spec decode ≈ τ× because the verify pays the dominant floor once** — it jumps from
"last lever" to a top-2 lever, and the MoE verify-tax (a weight-term effect) is negligible until the floor is
fixed, at which point k shrinks. Update `E6`: run n-gram now, k≈4, gate on realized tok/s, make k adaptive.
