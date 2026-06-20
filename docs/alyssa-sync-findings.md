# Alyssa — synchronization-optimization findings (2026-06-20)

**Scope:** real on-box results from the `overlap_decode.cu` / `nvshmem_comms.cu` /
`decode_step_tp8.cu` synchronization work, plus the env/script bugs hit getting there.
Cross-referenced against `path-to-1000.md` / `b1-engine-push.md` / `results-reaction-04.md`
— some of this is already cited there; this doc is the source for those citations plus
detail that didn't make it in yet.

---

## 1. Real measured results

### `overlap_decode.cu` — comms/compute overlap, PASS, reproducible
Correctness: chunked-all-reduce SUM check, `max_abs=0.0` — PASS, every rerun.

| scheme | result | tok/s comms-cap |
|---|---|---|
| serial (collective fully exposed) | 70.3–71.6 µs | ~74–76 |
| **overlapped** (AR(L) \|\| next-layer K1 GEMV) | **60.4–62.4 µs** | **~85–88** |
| chunked (reduce-as-you-go, 4 chunks) | 151–155 µs — **worse** | — |

**This is the *lossless* version of "hide it"** — distinct from LOOP-C's stale-TP
(`research/n4_speculative_stale_tp.md`, `experiments/stale_tp/`). Stale-TP computes on
an approximate/old value and needs a quality gate; this overlaps the AR with the *next*
layer's QKV GEMV specifically because that GEMV is data-independent of the AR's result —
zero approximation, no gate needed. Different lever, same "hide it" family.

**Magnitude — checks out against `stale_tp_ceiling.py`'s logic, but reveals a second
problem.** The AR (70.3 µs) is bigger than the only compute we gave it to hide behind
(a single QKV GEMV, 41.7 µs) — partial, not full, hiding is exactly what their ceiling
model predicts. But the *actual* hidden amount (9.67 µs, 13.8% of the AR) is far below
the ~59% (41.7/70.3) a cleanly-concurrent execution should achieve. **There's real
overhead in the overlap scaffolding itself** (per-rank `cudaEventRecord`/
`cudaStreamWaitEvent` handoffs, the two-phase enqueue-then-`ncclGroupStart` structure) —
extending the hidden-behind window (full K1+K2+K3, not just K1) helps the first problem;
profiling/cutting the handoff overhead is a separate, second problem, not solved by a
bigger window alone.

### `decode_step_tp8.cu` — the real TP=8 94-layer pipeline, PASS
Correctness: cross-rank residual check vs single-GPU reference, `max|ref-shd| ~1e-7`
(tol 1e-2) — PASS.

- **189 NCCL all-reduces/token** confirmed (2/layer × 94 + 1 head).
- **17.5 µs/all-reduce** on a clean run — matches our independent `nccl-tests` baseline
  (~16 µs). (One run hit 83 µs — a ~5× outlier from GPU contention with concurrent
  failed NVSHMEM builds running at the same time; discard, trust 17.5.)
- **The real kernel chain itself: ~39.3 tok/s (25.5 ms/token).** Comms is only **13.0%**
  of that (3.31 ms/token); the other 87% (~22.2 ms/token) is the K1–K5 kernels' own
  unoptimized compute — scalar fp8 loads, `atomicAdd` instead of tree-reduce, sketch-level
  reductions (exactly what `kernels/README.md`'s `TODO(on-box)` list already flags).

**Implication: sync optimization is necessary but not sufficient.** Even a perfect
collective wouldn't make this chain beat vLLM yet — the GEMV/router/expert kernels need
their own tuning pass (K5 MBU work, K1/K2 restructure) independent of the comms work.
This is a complementary data point to `b1-engine-push.md`'s kernel-MBU table — that table
measures individual kernels in isolation; this is the first **end-to-end multi-GPU
sharded step** overhead split.

### NVSHMEM (`nvshmem_comms.cu`, `nvshmem_overlap_decode.cu`) — blocked, confirmed shared
`nvlink error: Uncompress failed / elfLink fatbinary error` — the NVSHMEM wheel
(`nvidia-nvshmem-cu13`) needs CUDA 13; the box's system `nvcc` is CUDA 12.6. **Same
blocker independently hit and logged in `b1-engine-push.md`'s "Known blockers"** — this
is real and shared, not specific to our setup.

Went a step further: found a pip-distributed CUDA 13 `nvcc`
(`nvidia-cuda-nvcc==13.2.78`, binary at
`/usr/local/lib/python3.10/dist-packages/nvidia/cu13/bin/nvcc`, full toolchain incl.
`nvlink`/`ptxas` alongside it). Switching to it traded the link error for a header
error: `"CUDA compiler and CUDA toolkit headers are incompatible"` — the pip package is
the compiler frontend alone, not a fully consistent installed toolkit. **Still
unresolved** — needs either the complete matching `nvidia-cuda-*-cu13` set installed and
made consistent, or an NVSHMEM wheel built for CUDA 12.

---

## 2. A correction that changes our own next step

`results-reaction-04.md` / `path-to-1000.md`'s newer finding: **comms is *barrier*-bound,
not launch-bound**, and critically — **naive recursive-doubling NVSHMEM does NOT beat
NCCL**: 3 barriers × ~17 µs ≈ 51 µs, *worse* than NCCL's single already-optimized ~16 µs
barrier. Only **NVLS in-switch reduction** (one hardware op, not 3 software rounds) can
actually beat the barrier (`kernels/nvls_allreduce.cu`).

**This means**: `kernels/nvshmem_overlap_decode.cu` (the file we wrote combining
`nvshmem_comms.cu`'s recursive-doubling AR with `overlap_decode.cu`'s double-buffered
pipeline) uses the wrong collective primitive. The **overlap architecture itself is
still reusable** — double-buffering + event-gating is primitive-agnostic — but once the
CUDA toolchain issue is sorted, the right move is wiring `nvls_allreduce.cu`'s primitive
into that same scaffold, not finishing the recursive-doubling version as originally
planned. Flagging this now rather than quietly shipping the original plan.

---

## 3. Env/script bugs found (useful for anyone else running kernel tests on this box)

1. Bare `nvcc` is **not on `PATH`** in a non-interactive shell (e.g. under `gpu-slot
   run`/`nohup`) — use the full path (`/usr/local/cuda/bin/nvcc`).
2. `bash -u` (`set -u`) aborts on a completely **unset** `$LD_LIBRARY_PATH` (not just
   empty) — use `${LD_LIBRARY_PATH:-}`.
3. `nvidia.nvshmem` is a **data-only namespace package** (no `__init__.py`) →
   `nvidia.nvshmem.__file__` is `None`. `nvshmem_comms.cu`'s own header comment uses
   `os.path.dirname(__file__)`, which crashes — use `nvidia.nvshmem.__path__[0]`
   instead. **Still wrong in `nvshmem_comms.cu`'s header** (didn't edit that file) —
   worth a one-line fix upstream so the next person doesn't hit it.
4. A fix applied directly on the box (via `ssh`+`sed`) got silently clobbered by a later
   `scp` redeploy from an un-updated local copy. Fix at the source, not the deployed
   copy — bit us twice before we caught it.
5. `decode_step_tp8.cu` `#include`s 5 other kernel files (`k1_attn_prologue.cu` through
   `k5_experts.cu`) — easy to forget copying all of them when staging just the files you
   think you need.

---

## 4. Net position

- Our FP8-regresses finding (−19 to −25%) and the NCCL-tuning-is-dead finding are already
  load-bearing in `path-to-1000.md`'s strategy.
- The NVSHMEM CUDA-toolchain blocker is real, shared, and still open — worth someone
  taking on the full pip-toolkit-consistency fix (or finding a cu12 NVSHMEM wheel) since
  two independent attempts have now hit the identical wall.
- The lossless comms-overlap result (~16%, single GEMV) is a small, safe existence proof
  for the "hide it" family LOOP-C is pursuing at a larger scale (stale-TP) — worth
  sharing with whoever owns that track as supporting evidence before they invest further
  in the lossy/quality-gated version.
- `nvshmem_overlap_decode.cu`'s architecture (double-buffered overlap scaffold) is reusable;
  its specific primitive (recursive-doubling) should be swapped for NVLS once buildable.
