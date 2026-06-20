# Conifer engine assessment — confidence-adaptive top-k for B=1 MoE decode

Date: 2026-06-20. Scope: read-only deep-dive of `C:\Users\danie\Conifer` for the
adaptive-k expert-reduction idea. Nothing under `C:\Users\danie\Conifer` was modified.

TL;DR — Conifer is a **near-ideal proof vehicle for the *kernel mechanism*** (its
per-expert gather grid has a real, clean `k` dimension with no padding/align floor,
so cutting k cuts wall-clock 1:1), but it is **not currently runnable as a
qwen3-30B-A3B end-to-end demo on this machine** (no A3B GGUF present; the 12 GB
laptop GPU can't hold the 17.5 GB expert set; the published A3B decode numbers were
all measured on an M3 Max / Metal, not on the local CUDA box). The strongest
hackathon play is to implement adaptive-k in the CUDA `moe_route_topk` +
`moe_ffn_decode_device` path and demonstrate the kernel-level byte/time reduction
on a small synthetic or a small-expert model, then port the *same* policy into the
vLLM/235B path which is where the headline win lives.

---

## 1. EXACT integration point for confidence-adaptive top-k

The decode MoE path is fully device-side and the expert loop is a real grid
dimension keyed on `k`. The whole chain lives in **one Rust function** that fires
five kernels in sequence:

**Host orchestration (the place to add the policy):**
`C:\Users\danie\Conifer\engine-lx\crates\conifer-cuda\src\lib.rs`
- `fn moe_ffn_decode_device(...)` — **lines ~9745-9943**. This is the B=1 device
  decode MoE. It: (1) router GEMV → `logits[n_expert]`; (2) launches
  `moe_route_topk` to write `sel_idx[k]` + `sel_w[k]`; (3) launches `moe_gather_q4_k`
  for gate and up with grid `(expert_ff, k, 1)`; (4) `ffn_act` SwiGLU over
  `k*expert_ff`; (5) `moe_gather_{q4_k,q6_k}` down with grid `(d_model, k, 1)`;
  (6) `moe_weighted_reduce` over `d_model` summing `k` contributions.
- The eligibility gate that routes qwen3moe onto this device path:
  `fn moe_decode_device_eligible(...)` — **lines ~9660-9694** (n_expert in 2..=256,
  k <= n_expert, expert_ff % 256 == 0, gate/up Q4_K, down Q4_K|Q6_K). qwen3-30B-A3B
  (128 experts, top-8, expert_ff 768, Q4_K gate/up + Q4_K/Q6_K down) satisfies this.

**The kernel that SELECTS the top-k experts (where the policy threshold goes):**
`C:\Users\danie\Conifer\engine-lx\kernels\cuda\moe_route_topk.cu`
- `extern "C" __global__ void moe_route_topk(...)` — single warp (32 lanes,
  grid `(1,1,1)` block `(32,1,1)`). It already computes the full softmax `probs[]`
  over all experts (lines 50-70 for `gating==0` Softmax — the qwen3moe path), then
  does `k` rounds of butterfly argmax to fill `sel_i_sh[0..k]`, then writes
  `sel_idx[s]` / `sel_w[s]` for `s in 0..k` (lines 91-148).
- **This is the exact and cleanest place to make k per-token-variable.** The softmax
  probabilities are already materialized, and the argmax rounds already select
  experts in descending-probability order. The cumulative-mass policy is a few lines:
  after each selected expert `s`, accumulate `mass += probs[best_i]`; stop early when
  `mass > 0.9` AND `s+1 ∈ {2,4,6,8}` (snap up to the next even k so the gather grid
  stays well-formed). Write the chosen `k_eff` to a new device output word.

**The kernel that LOOPS over experts (where k_eff must be read):**
`C:\Users\danie\Conifer\engine-lx\kernels\cuda\moe_gather_q4_k.cu` (and
`moe_gather_q6_k.cu`, `moe_weighted_reduce.cu`). The expert loop is the grid `.y`
dimension: `slot = blockIdx.y`, one block per (output row × selected expert). There
is **no padded expert grid, no align_block_size, no fixed capacity** — `k` is just
a launch parameter (`gather_cfg(out_rows)` sets `grid_dim = (out_rows, k, 1)` at
lib.rs:9858-9862; `moe_weighted_reduce` takes `k` as a scalar arg and loops
`for j in 0..k`). Cut k → fewer blocks launched → fewer expert-weight reads. 1:1.

### How hard is per-token-variable k? (concrete plan)

Two viable designs; both are small. The grids are launched **host-side** from
`moe_ffn_decode_device`, but the routing decision is **device-side** (the whole
point of the device path is no host readback). So:

- **Design A — host knows k_eff (simplest, costs one tiny sync):** have
  `moe_route_topk` write `k_eff` to a 1-word device buffer; the host reads it back
  (4-byte D2H) and uses it as the `.y` extent and the `k` arg for the gather/reduce
  launches. Cost: re-introduces ONE per-layer 4-byte sync that the device path was
  built to avoid (the decode map calls this sync the historic wall — see §2). Likely
  *not* worth it on its own, but trivial to wire for a first correctness demo.

- **Design B — device-only, launch the MAX grid and early-out per block
  (recommended, no sync):** keep launching `grid.y = 8` (static max k), but have each
  gather/reduce block read `k_eff` (or read its slot's `sel_w`/a validity flag) and
  `return` immediately if `slot >= k_eff`. The early-returned blocks do **zero global
  memory traffic** (they bail before the expert-weight loads at gather lines 71-104),
  so the dominant B=1 cost — the expert-weight READ — is skipped exactly as desired.
  This keeps the launch-count constant (a few wasted near-empty block launches, which
  the decode map shows are launch-floor-bound and cheap) while removing the
  byte-traffic of the skipped experts. **This is the clean win with no new sync and
  no padding floor.** `moe_weighted_reduce` already only needs `sel_w[j]` to be 0 for
  skipped slots (or to loop to `k_eff`); zeroing the dropped `sel_w` slots in
  `moe_route_topk` makes the reduce correct with no change to its signature.

Recommended: **Design B**, with `moe_route_topk` (a) computing `k_eff` from the
cumulative-mass policy, (b) zeroing `sel_w[k_eff..8]`, (c) optionally writing
`sel_idx[k_eff..8]` to a sentinel so the gather blocks can self-skip. Net engine
change: ~15-25 lines in `moe_route_topk.cu` + an early-out branch in the two gather
kernels. No host-side control-flow change, no new sync, parity-preserving for the
mass==1.0 (always-k=8) degenerate case.

**Policy note (be honest about ordering):** the policy "smallest k∈{2,4,6,8} whose
cumulative top-k mass > 0.9" is well-defined here because `moe_route_topk` selects
experts in strict descending (prob+bias) order with lowest-index tie-break (lines
91-113), so the running prefix sum of `probs[best_i]` *is* the cumulative top-k mass.
One subtlety: qwen3moe uses `norm_topk` (L1-renormalize the k weights, lines
133-138). The 0.9 threshold should be measured on the **pre-norm softmax mass** (the
`probs` values, which already sum to 1 over all experts), not post-norm — the code
has `probs[]` right there before the renorm step, so this is natural.

---

## 2. Is the wall-clock win REAL in Conifer, and how big?

**Real, yes — no padding floor (the core advantage over vLLM).** As shown in §1,
the expert dimension is a literal grid `.y` and a literal loop bound `k`; there is no
`moe_align_block_size`-style capacity pad. Dropping an expert drops its
gather block's weight read entirely (Design B early-out). This is precisely the
"per-expert loop lets adaptive-k skip cleanly" property you were looking for.

**Size of the win — from the decode map (`findings-qwen3moe-decode-map-2026-06-09.md`):**
- Active bytes/token ≈ **1995 MB**, of which the **expert chain (gate+up+down
  gathers) ≈ 1097 MB = ~55% of per-token bytes** (17.55 GB experts × 8/128). The
  rest is attention/dense 592 MB, lm_head 255 MB, router 50 MB.
- So the expert chain is the **single largest term (~55% of decode bytes)** and is
  exactly what adaptive-k cuts. Average k going 8 → ~5 (typical for a 0.9-mass
  policy on a concentrated router) would cut the expert chain ~37%, i.e. **~20% of
  total per-token bytes**, with a corresponding decode-time win *to the extent decode
  is byte-bound*.

**Big caveat the decode map forces us to state:** on the M3 Max the qwen3moe device
path is **NOT byte-bound — it is serial-dispatch / per-layer-sync bound.** The map's
headline: the entire conifer-vs-llama qwen3 gap was ONE kernel (`moe_route_topk`
running serially with per-candidate device re-reads, 306 µs/call → fixed to 30.6 µs),
and after the fixes "both paths converge ≈44 tok/s — the residual gap is the SHARED
serial attention chain, not the MoE chain." In that *dispatch-bound* regime, cutting
expert *bytes* yields **less than the byte fraction suggests**, because the gather
launches still happen (Design B) and the wall is launch/sync overhead, not the
weight reads. The win is largest where the engine is genuinely bandwidth-bound at the
expert gather — which is **more true on a big discrete GPU (H100) with the 128-expert
17.5 GB set than on the 270 GB/s M3 Max**. So:
- On a **bandwidth-bound** deployment (H100, full 128-expert A3B/235B): adaptive-k is
  close to a true ~(expert-byte-fraction × k-reduction) wall-clock win. This is the
  regime your vLLM/235B target lives in, and the regime where the conifer kernel
  design *proves the mechanism cleanly*.
- On the **M3 Max** numbers in the repo: smaller, because that path is sync-bound.

**Honest recommendation:** use Conifer to prove the *kernel mechanism* (real grid,
real byte reduction, parity-preserving) and to get a clean micro-benchmark of
gather-chain time vs k; carry the *end-to-end* wall-clock claim on the H100/vLLM
side where bandwidth dominates. Don't over-claim a big conifer end-to-end number from
the M3 Max data — that path is dispatch-bound.

---

## 3. Can Conifer run qwen3-30B-A3B as a baseline-vs-adaptive proof vehicle?

**Short answer: not as-is on this machine.** Three blockers:

1. **No A3B GGUF present.** A full-tree search found only:
   `qwen2.5-0.5b`, `qwen3-4b-instruct`, `qwen3-8b-instruct` (these last two are
   **dense**, not MoE), plus `qwen2.5-0.5b-instruct-q4_k_m.gguf`. There is **no
   `qwen3-30b-a3b` / `*a3b*` GGUF anywhere under `C:\Users\danie`**. The A3B decode
   numbers in `MoE-fastest.md` / the decode map were produced on an M3 Max with a
   GGUF that is not on this box.

2. **VRAM.** Local GPU is an **NVIDIA RTX 5070 Ti Laptop, 12 GB** (`nvidia-smi`).
   qwen3-30B-A3B Q4_K_M experts alone are ~17.5 GB — **will not fit in 12 GB.** The
   bench runner (`coniferbench/engines/conifer.py`) documents that the larger models
   only "fit" on a 12 GB card via `CONIFER_CUDA_USE_MMQ=1` (keeps weights quantized,
   drops the fp16 cache) and even then 4B/8B is the practical ceiling, not 30B.

3. **The published A3B numbers are Metal, not CUDA.** The CUDA `moe_ffn_decode_device`
   path has parity tests (`conifer-cuda/tests/parity.rs:4733`
   `ffn_moe_decode_device_softmax_matches_cpu`, qwen3moe-shaped softmax+norm_topk
   Q4_K/Q6_K) but they are `#[ignore = "requires CUDA GPU — conifer-jetson down"]` —
   i.e. the device MoE path was validated against a CPU oracle but its decode
   throughput on real CUDA hardware is not in the repo.

**What IS runnable here (the realistic demo):**
- The build is fresh: `C:\Users\danie\Conifer\engine-lx\target\release\conifer.exe`
  (built 2026-06-20 01:10, 10.6 MB), CUDA backend present (cudart/cublas DLLs in
  target). So `conifer.exe bench ... --backend cuda` works for models that fit.
- Bench command shape (from `coniferbench/engines/conifer.py`, the CLI `Bench`
  subcommand at `conifer-lx-cli/src/main.rs:420`):
  ```
  set CONIFER_CUDA_USE_MMVQ=1
  set CONIFER_CUDA_USE_MMQ=1
  set CONIFER_CUDA_USE_FA2_MMA=1
  C:\Users\danie\Conifer\engine-lx\target\release\conifer.exe bench ^
      <path-to.gguf> --prompt-tokens 64 --decode-tokens 96 --seed 0 ^
      --backend cuda --json
  ```
  Output is one JSON object incl. `decode_tok_s`, `prefill_tok_s`, `ttft_ms`,
  `bytes_per_token_mb`.
- **To get an actual A3B baseline-vs-adaptive on CUDA you must either:**
  (a) download a `qwen3-30b-a3b` GGUF and run on a **≥24 GB discrete GPU / a single
  H100** (the H100 is the right target anyway — it's bandwidth-bound there, the
  regime where adaptive-k wins most), or
  (b) demonstrate the mechanism on a **smaller MoE that fits 12 GB** — but note no
  small MoE GGUF (lfm2-8b-a1b etc.) is on this box either; lfm2-8b-a1b is the
  engine's fastest MoE (160 tok/s, `MoE-fastest.md`) and would fit, if its GGUF is
  fetched. lfm2moe is sigmoid+score_bias (different gating) but exercises the SAME
  `moe_route_topk` / gather / reduce kernels, so it's a valid mechanism demo.
  (c) run the **CUDA parity test** with adaptive-k added (synthetic 8-expert top-2
  shape, `parity.rs:4733`) to prove byte-identical-when-mass=1 + correct selection
  when k shrinks — fastest path to a credible correctness claim with zero new weights.

**Metal alternative:** the same kernels exist at
`C:\Users\danie\Conifer\conifer-engine\crates\conifer-metal\src\kernels\moe.metal`
(`moe_route_topk`, `moe_gather_*`, `moe_weighted_reduce`) and the M3 Max A3B GGUF
that produced the decode map exists on that machine. If djamoils has the Mac, an
A3B baseline-vs-adaptive there is immediately runnable — but per §2 that path is
dispatch-bound, so the wall-clock delta will understate the H100 win.

---

## 4. Other things pullable for the hackathon team

**Routing kernel design (directly reusable as the adaptive-k host/CPU reference):**
- `route_experts` (the ONE generic host router) in
  `C:\Users\danie\Conifer\conifer-engine\crates\conifer-core\src\backend.rs`
  (~line 244) and the `MoeRouting` struct (line 197). Param-driven, no arch names;
  this is the oracle the device kernel matches byte-for-byte. The cumulative-mass
  policy should be prototyped here first (CPU, trivial to test) then mirrored into
  `moe_route_topk.cu` — exactly the workflow the repo already uses (host oracle →
  device kernel parity).

**Device top-k kernel (the warp-parallel argmax pattern):** `moe_route_topk.cu`'s
butterfly-argmax with lowest-index tie-break (lines 91-113) is a clean, correct,
single-warp top-k over ≤256 experts. Reusable as a reference for a vLLM custom
routing kernel if you want device-side adaptive-k there too (vLLM's
`moe_align_block_size` is the floor you're fighting — a custom route+variable-k
launch modeled on this avoids it).

**Quant GEMV / expert gather (the "more bandwidth than MLX" kernels):**
- `moe_gather_q4_k.cu` / `moe_gather_q6_k.cu` — batched single-dispatch expert gather
  with in-kernel Q4_K/Q6_K dequant (affine factoring `d·sc·Σ(q4·x) − dmin·m·Σx`,
  byte-identical to the standalone `gemv_q4_k.cu`). The "expert indirection via a
  cached `u64[n_expert]` pointer table" (`moe_expert_ptr_table`, lib.rs:9700) is the
  CUDA trick that makes per-expert separate allocations gatherable in one launch —
  relevant if the team wants per-expert weight layouts.
- `conifer-kernel-wins.md`: the production `gemv_q4_k_lm` beats MLX `qmv_fast` by
  +3-8% (FFN) / +13-17% (square) at equal bytes; mechanism is 2-rows/TG occupancy
  vs MLX's 8-rows/TG underfill on narrow shapes. The standalone MLX replica lives at
  `crates/conifer-metal/tests/qmv_mlx_replica.rs` (Metal). Useful as a bandwidth-MBU
  reference if benchmarking kernel efficiency.

**Fusion ideas already shipped/scouted (decode map "campaign plan" + Lever-3 map):**
- `moe_down_reduce_q4k/q6k` fused down+weighted-reduce (one dispatch does
  `x += Σ_j w_j·f16(down_ej·g_j)`), env opt-out `CONIFER_MOE_DOWN_FUSED=0`. If you do
  Design B early-out, do it in the fused variant too.
- Env knobs that matter for any CUDA bench: `CONIFER_CUDA_USE_MMVQ=1` (int8 __dp4a
  decode GEMV), `CONIFER_CUDA_USE_MMQ=1` (int8 MMQ prefill + drops fp16 cache so
  models fit), `CONIFER_CUDA_USE_FA2_MMA=1` (FA2). MoE device route on Metal was
  `CONIFER_MOE_GPU_ROUTE=1`; on CUDA the device path is selected by
  `moe_decode_device_eligible` (n_rows==1 + gatherable quant), not an env var.
- `CONIFER_CUDA_SPEC=1` enables prompt-lookup speculative decode on CUDA (greedy
  only); reported +15% on qwen, up to ~1.95×/3.2× at high acceptance — orthogonal to
  adaptive-k and stackable.

**Bench/measurement harness (reusable as-is):** `Benching/coniferbench` is a clean
multi-engine harness (conifer / llama.cpp / MLX) that already emits decode_tok_s,
MBU, bytes/token, energy. `engines/conifer.py` shows the exact env + CLI invocation.
For an adaptive-k A/B you'd add a `CONIFER_MOE_ADAPTIVE_K` env gate (mirroring the
existing knob pattern) and run the same `conifer bench` twice (off/on).

---

## Bottom line for the action plan

1. **Implement** the cumulative-mass policy in `moe_route_topk.cu` (compute `k_eff`,
   zero dropped `sel_w`, write a self-skip sentinel) + early-out branch in the two
   `moe_gather_*` kernels (Design B, no new sync). Gate behind a new
   `CONIFER_MOE_ADAPTIVE_K` env so off == byte-identical to today.
2. **Prove correctness** with the existing CUDA parity test
   (`parity.rs:4733`, qwen3moe-shaped) extended for k-shrink cases.
3. **Demonstrate bytes/time** with a gather-chain micro-bench vs k (the regime where
   the win is unambiguous and machine-independent).
4. **Carry the end-to-end wall-clock claim on the H100/vLLM 235B path** (bandwidth-
   bound, where ~55% expert-byte share × k-reduction is a real decode win) — Conifer
   is the clean *mechanism* proof, not the e2e number on this 12 GB laptop.

All cited paths are absolute and under `C:\Users\danie\Conifer` (read-only; untouched).
