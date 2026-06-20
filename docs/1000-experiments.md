# The make-or-break experiments for 1000 tok/s (post-squeeze-round, prioritized)

After `results-reaction-04.md` the picture is: **fp8 + graphs + fast-path + fp8-K5-e→1 + big-tree spec gets
~660–900 (spec amortizes the barrier-bound comms); the last ~100–350 to 1000 needs a comms reduction (count /
in-switch / stale-TP) — all uncertain.** So 1000 is at the margin and gated on a few measurements. Run them in
*this* order (cheapest/most-decisive first); each has a clean decision. Plug results into `ladder_to_1000.py`.

## 1. Comms C — `measure_collective.sh` (NO model load, seconds) — CHEAPEST + most decisive
**Measures:** the 8 KB all-reduce latency across NCCL Ring/Tree/**NVLS** + (if built) the multimem
`nvls_allreduce.cu`. **Decides:** is there ANY sub-16 µs "make-it-faster" comms lever?
- multimem in-switch **< ~8 µs** → real comms-latency win, stacks with everything. Build it.
- all ≈ 16 µs (NCCL already in-switch) → **no make-faster lever**; the comms term is barrier-floored → 1000 must
  come from **stale-TP (hide)** or **count (EP, verify it's lossless)** + spec. Settles the whole comms strategy.

## 2. EAGLE3 realized spec multiplier — the 09:45 slot (`slot_eagle3.sh`) — THE dominant lever
**Measures:** τ, S = tok/s(spec)/tok/s(baseline), V = τ/S on the real engine (`eagle3_analyze.py` + my
`backout_floor.py` for F). **Decides:** the actual spec speedup — the biggest single contributor to 1000.
- S ≈ 3.5–3.8 on the comms-dominated engine → spec carries most of the way (~900); the comms reduction is the
  finish. **draft_tp=8, big tree** (`eagle3-draft-tp.md`, `tree_spec_optimizer.py`). Measure at temp 0.7 too.
- S ≈ 2.5 → spec alone → ~700; need BOTH a comms reduction AND a good tree. (Watch the verify is correctly
  batched — the squeeze bench's per-row scaling was a bench bug; the real verify reads the union once.)

## 3. stale-TP quality probe — LOOP-C (`run_stale_probe.sh` + `DECISION.md`) — the comms UPSIDE
**Measures:** greedy parity vs exact at K∈{2,4} with stale/proxy all-reduce. **Decides:** can the comms be
*hidden* (overlapped) losslessly-enough?
- GO (A2 ≥ 0.99, control degraded) → comms → ~0 → **fp8 roofline ~1218**, 1000 comfortable. The big upside.
- NO-GO → comms stays barrier-bound; fall back to spec + count + (if it pans out) in-switch.

## 4. EP-decode count — verify it's LOSSLESS at B=1 (`reaction-04` flag) — the count lever
**Question:** does EP actually cut the *barrier* count vs NCCL's 1-barrier (16 µs) TP all-reduce? **Beware:**
DP-attn drops the attn-AR but replicates 6.7 B attention weight → **net loss at B=1** (+1.75 ms vs −1.5 ms);
EP-MoE *adds* a 2nd all-to-all (dispatch+combine). **Decide:** count the real barriers/layer of the EP-decode
against the NCCL TP baseline (16 µs, not the 51 µs 3-barrier NVSHMEM). Pair with spec (the verify balances EP's
imbalance — `ep-balance-spec-verify.md`); don't judge EP on plain decode.

## 5. fp8-K5 e→1 — `k5_experts_pipelined.cu` + `k5-tuning-roadmap.md` — the weight lever
**Measures:** the cp.async-pipelined kernel's `e` (0.46 → target ~0.85) vs `k5_microbench`. **Decides:** the
weight read at roofline (0.78 ms). Necessary partner to the comms work; e=0.85 still clears 1000 in the ladder.

## The decision tree (one line)
**Run #1 (comms C) and #2 (EAGLE3 S) first — they decide everything.** If #3 (stale-TP) GOes, 1000 is
comfortable (~1218). Else 1000 = spec(#2) + the best of {in-switch(#1), lossless-EP-count(#4)} + fp8-K5(#5),
and it's tight (~900–1050). **int4 is dead** (squeeze round, 0.58×). Cheap ship today: spec + prefix-cache ~300.

Update `ladder_to_1000.py --C <#1> --ncoll <#4> --tau-mult <#2 as S> --e <#5> [--stale-tp if #3]` after each.
