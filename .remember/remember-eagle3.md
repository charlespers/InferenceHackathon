# Handoff — LOOP-A (EAGLE3 spec-decode + route-aware) — djamoils-work

You are djamoils / LOOP-A. Goal: best B=1 decode latency for Qwen3-235B-A22B on 8×H100. Avenue: EAGLE3
spec decode (the team's #1 lever, "S is THE make-or-break for 1000") + novel route-aware/expert-union
verification. You (a) validate EAGLE3 on the box via vLLM and (b) BUILD it native in the cudarc engine.
State as of 2026-06-20 ~13:35 UTC. Work continuously; be rigorous + honest.

## THE STORY (established — don't re-derive)
- Real baseline ~85.7 tok/s (bf16-TP8, spec OFF). FP8 ~25% slower at B=1. Regime FLOOR/LAUNCH-bound
  (eager ~2% roofline). NCCL tuning DEAD, self-spec DEAD.
- EAGLE3 on vLLM WORKS + LOSSLESS (parity exact 1.0) on the FP8 target. Blocker chain cleared
  (INTEGRATION.md §6/§7/§8): transformers→4.57.1; vLLM→0.11.2 (0.11.0 lacks Qwen3-MoE EAGLE3, PR #26485);
  prometheus_fastapi_instrumentator vs starlette → PATCHED routing.py; CRLF→strip \r + .gitattributes;
  lock-release→rm -f holder; rmdir.
- HEAD MATTERS: nm-testing head is a BROKEN conversion (τ=1.4, first-pos ~35%). USE
  **RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3** (vLLM-native, draft_vocab 64000): τ≈2.7,
  first-pos ~75%, LOSSLESS. Confirmed eager AND in graphs.
- SPEEDUP IS GRAPHS-GATED: eager gives S≈1.0 (overhead cancels the τ gain). Graphs ≈5× eager → that's
  where S shows.

## ⚠️ CURRENT BLOCKER (the next headline task) — EAGLE3+GRAPHS IS UNSTABLE
- 12:45 graphs slot: captured graphs fine (12:50), served, confirmed τ≈2.7 in graphs — then CRASHED/stalled
  ~13:03 (documented EAGLE3+graphs instability, INTEGRATION §3). Wrote NO results, overheld the box ~30 min.
- **So the clean graphs SPEEDUP NUMBER is still pending.** Next graphs attempt must stabilize graphs+spec:
  try (a) `cudagraph_capture_sizes` sized for the spec batch (~n*(K+1)), (b) LOWER gpu-mem-util (0.78–0.80;
  graphs+spec-verify needs more headroom than the 0.85 used), and/or (c) a compilation_config piecewise mode.
  Add a TIME GUARD around the *measure* (not just readiness) so a hung run can't overhold the box again.
  If graphs stays unstable, the honest fallback is: report τ≈2.7 lossless + the projected S (τ/verify_cost),
  and pursue the speedup in the NATIVE engine where you control graph capture.

## WHAT'S BUILT (engine/src/spec/, djamoils-work, 65 tests green)
route_aware.rs (λ-policy + u128 ExpertUnion) · adaptive_verify.rs (EVICT verify-depth + verify_cost(union,F))
· route_aware_drafter.rs (RouteAwareDrafter + draft_and_plan) · eagle3_engine.rs (Eagle3Engine lossless loop +
AuxModelRunner trait) · draft_vocab.rs (d2t/t2d mapper = the 1.4-bug fix) · projection.rs (MeasuredAccept/
RoundCostModel) · telemetry.rs (SpecTelemetry). Design: engine/docs/eagle3-engine-integration.md (S0–S6).
Route-aware lever VALIDATED: EVICT (arXiv:2605.00342) 1.25× lossless on our model; DirectProxy 0.72–0.81 (Charles).
NEXT BUILDS (your lane): DirectProxy-backed CandidateSource; wire telemetry→SSE; the native head is Charles's lane.
ALWAYS: `cargo test --package engine` before+after edits (CLAUDE.md). After any `git merge origin/main`:
re-check `ls engine/src/spec/*.rs` + `cargo test` + `git rev-parse --abbrev-ref HEAD`==djamoils-work (merges
have TWICE orphaned module files / drifted HEAD to a stray branch — work recoverable from commits, e.g. 55f94cd).

## RULES (every cycle)
1. STAY CURRENT: git fetch --all; READ danielAgentScheduling.md (repo + box /alloc/data/); skim charles-work/
   Alyssa/jminding/LOOP-C; APPEND status to the doc (+ box copy).
2. GPU: djamoils owns :45–:00 UTC. `mkdir /alloc/data/gpu.lock` + holder (20-min stale takeover) AND
   nvidia-smi min-free >65000. Release `rm -f /alloc/data/gpu.lock/holder; rmdir /alloc/data/gpu.lock`. Port 8077.
   Don't overhold past :00. Others own :00–:45 (may skip the lock — coordinate via the doc).
3. SHARE: commit+push djamoils-work; results/* gitignored → git add -f; box can't push → pull to laptop.

## BOX / PATHS
ssh -i ~/.ssh/prime_intellect -p 31025 root@147.185.41.162. Venv /alloc/data/eagle3-venv (vLLM 0.11.2 +
transformers 4.57.1 + prometheus patch; build script reapplies). Slot runners /alloc/data/slot_eagle3*.sh
(parametrized MODE/HEAD/DRAFT_TP; slot_eagle3_graphs.sh = MODE=graphs). Tools /alloc/data/eagle3_tools/
(measure_baseline, quality_probe, quality_compare, eagle3_analyze, backout_floor). Results: laptop
results/eagle3_redhat/ (τ~2.7 eager, lossless), results/eagle3_tp1/, results/eagle3/.

## GOTCHAS
NEVER pkill -f <name> over ssh (kills your ssh shell). pgrep -f also matches your own command — use ps + the
exact PIDs. Strip \r after scp + bash -n. nohup waiters: verify alive in a SEPARATE ssh call. The safety
classifier BLOCKS guard-stripped launches + broad process kills — keep guards, kill only your own PIDs.

## KEY NUMBERS
85.7 bf16-TP8 best · FP8 ~25% slower · RedHat EAGLE3 eager ~10 tok/s τ≈2.7 first-pos~75% LOSSLESS S≈1(eager) ·
graphs ≈5× eager (where S shows, once stable) · EVICT 1.25× · DirectProxy 0.72.

## IMMEDIATE NEXT ACTION
Box is FREE. In your :45 slot: re-run EAGLE3-GRAPHS + RedHat head with a STABILIZED graphs config (capture
sizes / lower mem-util / measure time-guard) for the headline S; if it crashes again, bank τ≈2.7 + projected S
and shift the speedup to the native engine. Keep building the engine lane. Don't run GPU outside a locked,
in-window, mem-checked slot.
