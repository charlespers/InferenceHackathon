# Which spec variant, when — the B=1 decision guide

Spec is the convergent #1 lever; there are now four variants on the table. They all amortize the floor (one
batched verify pays it once); they differ in **acceptance (τ), draft cost, setup, and content sensitivity**.
This picks the right one per situation. All are modeled in `tools/spec_predict.py`.

## The four variants
| variant | τ (floor-bound) | draft cost | setup | best for |
|---|---|---|---|---|
| **EAGLE3** (trained draft head) | **~3–3.5** (highest) | ~0.5ms (draft_tp=8) / ~3ms (tp=1) | EAGLE3 head + vLLM≥0.10.2 + venv | the **headline** — max τ when the head exists |
| **Lookahead** (Jacobi, draft-free) | ~1.5–2.5 | **0** | config flag, any model/vLLM | **portable fallback** + the draft-free control |
| **n-gram** (prompt-lookup) | ~1–3 (content-dep) | **0** | config flag, works on 0.10.1 | **repetitive/structured** output (code, JSON, boilerplate) |
| **Self-spec** (layer-skip draft) | ~2 (but α collapses) | the shallow pass | none (same weights) | **loses on Qwen3** (pruning-fragile, `depth_reduction.md`) |

(No native **MTP**: Qwen3-235B uses an *external* EAGLE3 head — if it shipped multi-token-prediction layers
you'd use those instead. So the internal-draft branch is closed; the draft is either EAGLE3 or none.)

## The decision tree
1. **EAGLE3 head available + vLLM≥0.10.2 reachable?** → **EAGLE3** with `draft_tp=8`, big tree (W4–8×D3–4 on
   TP; biggest on EP). Highest τ. This is LOOP-A's 08:45 path.
2. **No head / stuck on system vLLM 0.10.1 / want a win today?** → **lookahead** (draft-free, config flag) —
   `spec_predict.py` puts it within ~5% of EAGLE3 in deep floor-bound (the floor amortization dominates the
   lower acceptance). The portable primary.
3. **Output is repetitive** (code-gen, JSON, retrieval, multi-turn boilerplate)? → **n-gram** — zero draft,
   τ rivals EAGLE3 on this content, works on 0.10.1. Cheapest possible.
4. **Always run lookahead/n-gram as the control:** EAGLE3-speedup ÷ draft-free-speedup = exactly what the
   *trained draft* buys over free self-verification. Tells you if the EAGLE3 setup cost is justified.
5. **Self-spec:** skip on Qwen3 (acceptance collapses at the shallow exits that make the draft cheap).

## Regime & layout modifiers (apply to whichever variant)
- **Floor-bound (now, F≈0.86):** go **big** on the tree/window (the union tax is on the 14% weight) — naive big
  trees win ~3× (`tree_spec_optimizer.py`). Route-awareness is **moot** here.
- **As the floor falls (F→0, after comms/kernels/megakernel):** **shrink** the tree and turn on **route-aware
  drafting** (`route-aware-drafting-design.md`) — the union tax starts to bite.
- **Layout:** TP8 has a tree sweet spot (W4×D8); **EP wants the biggest tree** (the big union balances the EP
  verify, `ep-balance-spec-verify.md`).
- **Temperature:** all variants are lossless, but acceptance drops ~25% at temp 0.7 → measure the *product* τ
  (`spec-in-production.md`), don't quote the greedy ceiling.

## One line
**EAGLE3 for max τ where the head exists; lookahead as the draft-free portable primary/control; n-gram for
repetitive content; never self-spec on Qwen3.** Go big on the tree while floor-bound; shrink + route-aware as
the floor falls. The choice is a config flag away in every case — run the control alongside the headline.
