# Single-user latency budget — current vs achievable (the whole thesis, in one table)

The goal is single-user latency / max tok/s. Perceived latency for an N-token answer = **TTFT + N·TPOT**.
Here's the measured current state and the projected endpoint from the **proven/validated** levers (with
honest confidence), so the team can see where the ~5× is and which levers buy it.

## Current (measured, bf16 pure-TP8, N=128)
| component | value | what it is |
|---|---|---|
| TTFT | **777 ms** | ~770 ms overhead (prefill eager/cold), ~7 ms physics (`ttft-analysis.md`) |
| TPOT | **11.67 ms** | overhead ~7.0 / comms ~3.0 / weight ~1.6 ms (`overhead-attribution.md`) |
| **perceived (128 tok)** | **777 + 1494 = 2271 ms** | decode tok/s **85.7** (16% of roofline) |

## Achievable, lever by lever (proven or measured-adjacent)
| lever | effect | confidence | new TPOT / TTFT |
|---|---|---|---|
| **Layout = TP8** | avoid EP busiest-rank (2.53×) | ✅ measured (64.5→85.7) | already in baseline |
| **Comms tuning** (E0b: 16→4–8 µs) | TPOT comms 3.0→0.8–1.5 ms | med (env sweep) | TPOT → ~9.5–10 ms |
| **Kernel efficiency** (K5: vLLM ~0.16→0.46) | shrink ~half the 7 ms overhead* | med (E-attr decides) | TPOT → ~5–7 ms |
| **n-gram spec** (floor-amortization, τ≈2) | TPOT /τ on the dominant floor | med (accept-rate) | TPOT → ~3–4 ms |
| **fp8 weights** (E2b) | weight 1.6→0.8 ms | low (only ~7% now) | TPOT → ~2.5–3.5 ms |
| **Prefix caching** (E-ttft) | skip shared-prompt prefill | ✅ high (cache hit) | **TTFT → ~10–40 ms** |
| **Graph/compile prefill** | kill prefill-eager overhead | med | TTFT → ~40 ms (fresh) |

\*the 7 ms overhead splits into comms + kernel + host; `E-attr` (Nsight) sizes each. The kernel share is
direct K5 territory (vLLM `fused_moe` ~0.16 vs the tuned 0.46).

## Projected endpoint (if the proven levers land)
| | current | achievable | factor |
|---|---|---|---|
| TPOT | 11.67 ms | **~3–4 ms** | ~3–4× |
| decode tok/s | 85.7 | **~250–330** | ~3–4× |
| TTFT (cached / fresh) | 777 ms | **~10 / ~40 ms** | ~20–80× |
| **perceived (128 tok)** | **2271 ms** | **~410–550 ms** | **~4–5×** |

**UPDATE — with the convergent answer (EAGLE3 spec, τ≈3.5 vs n-gram's ~2):** the cheap-wins stack
(prefix-cache + EAGLE3 + TP8 + comms LL) → **~508 tok/s / 259 ms perceived (~8.8×)**; **+ the K5 kernel
(eff 0.46) + tuned comms (6µs) → ~754 tok/s / 174 ms (~13×)** (`tools/latency_budget.py --spec-tau 3.5`).
EAGLE3's accept length (~3.5) drives the bigger multiplier; that's why the team converged on it.

## The order that gets there (data-grounded, cheap-first)
1. **Prefix caching** — free, ~50–100× TTFT for repeated/structured prompts. Ship now (`E-ttft`).
2. **`E-attr`** — split the 7 ms floor; tells you whether comms (E0b) or kernels (K5) is the bigger decode win.
3. **n-gram spec** — amortizes the floor over τ (~2×), free draft, run now at k≈4 (`spec-decode-floor-bound.md`).
4. **Comms tuning** (E0b) and/or **kernel efficiency** (K5) per E-attr — the 7 ms + 3 ms.
5. **fp8 (E2b)** — the smallest decode lever (~7%); do it for the headroom once the floor is down.

## The one sentence
Today single-user latency is **~2.3 s for 128 tokens, 86 tok/s** — and it's ~95% overhead (TTFT eager +
decode floor), not physics. **Prefix caching + floor attribution + n-gram spec are cheap and get to ~0.4–0.5 s
/ ~250–330 tok/s (~4–5×)** before any quantization or exotic kernel work. The bytes were never the problem;
the floor is.
