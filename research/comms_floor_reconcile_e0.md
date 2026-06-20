# Reconciling E0's 35 µs with the ladder's 16 µs — what C does the model actually use?

**LOOP-C, 2026-06-20.** Adversarial validation of a fresh, load-bearing claim before the team builds on it.
Tool: `tools/comms_floor_reconcile.py` (re-runnable). Verdict: **E0's 35 µs is real but is NOT the engine's
effective all-reduce — keep the ladder at C ≈ 16 µs (in-engine band ~10–18 µs), and let E-attr pin it.**

## The contradiction
Three different per-collective all-reduce latencies are now live across the team's docs, a **7× spread on
the single most load-bearing comms parameter**:

| C | source | comms/token (188×C) | implied lever order |
|---|---|---|---|
| 5 µs | `hardware.py` original guess | 0.94 ms | comms minor |
| **16 µs** | ladder / `comms_floor.md` (nccl-tests, used as in-engine proxy) | **3.01 ms (26%)** | overhead ≫ comms ≫ weight |
| **35 µs** | **E0 today** (`onbox-collective-latency-e0.md`, stock NCCL ring) | **6.58 ms (DOMINANT)** | comms is THE floor |

E0's headline — *"188 × 35 µs = 6.6 ms ⇒ comms is the dominant floor term"* — if taken as the in-engine
number, **rewrites the ladder** (which assumes comms 3.0 ms and a 7.0 ms "overhead" elephant). It would also
roughly **double the apparent prize of LOOP-C's own exact-overlap/NVLS lever** (hide 6.6 ms, not 3.0 ms).
That is exactly the kind of self-flattering reading I must check before banking it.

## The cross-check (uses two OTHER on-box measurements, not opinion)
B=1 decode is serial — no comms/compute overlap exists yet (that's the whole reason exact-overlap is a lever):

```
TPOT = T_compute + T_comms + T_host ,   T_compute = weight_floor / e ,   T_comms = 188·C ,   T_host ≥ 0
```

Anchors, all measured/physics: **TPOT = 11.67 ms** (M1), **weight floor = 1.56 ms** (active 20.9 B ×2 B /8
/3.35 TB/s), **vLLM whole-model e ≈ 0.16–0.19** (M2, the K5 / overhead-attribution finding). For a
hypothesized C, the residual `R = TPOT − 188·C − kv` must cover `T_compute + T_host`; since `T_host ≥ 0`,
the hypothesis **forces** the kernels to run at `e ≥ weight_floor / R`. If that exceeds the measured e, the C
is inconsistent.

| C | comms ms | R (comp+host) | required e ≥ | verdict |
|---|---|---|---|---|
| 5 µs | 0.94 | 10.66 | 0.15 | consistent (comms negligible — the original under-count) |
| **16 µs** | 3.01 | 8.59 | **0.18** | **CONSISTENT** (≈ measured 0.16–0.19, host≈0) |
| **35 µs** | 6.58 | 5.02 | **0.31** | **INCONSISTENT** — forces e = 1.6–1.9× the measured kernel efficiency |

Inverting M2 directly (what C does the measured e *imply*, with T_host=0 as the upper bound):

```
e=0.16 → T_compute 9.75 ms → in-engine C ≤  9.9 µs
e=0.19 → T_compute 8.21 ms → in-engine C ≤ 18.0 µs
```

**The measured kernel efficiency alone bounds the in-engine all-reduce to ≤ ~18 µs.** 35 µs cannot fit.

## Why 35 µs is real yet not the engine's number
1. **It's stock NCCL ring, measured standalone.** `nccl-tests` launches each collective fresh; CUDA-graph
   decode amortizes that per-op launch/sync away. The same vLLM issue (#36481, `comms_floor.md`) clocks a
   ~16 KB all-reduce at ~10–11 µs on a graph-captured path — so even "NCCL" isn't one number; standalone vs
   in-graph differ by ~3×, which is the bulk of 35 → ~12 µs.
2. **vLLM doesn't run stock ring at 8 KB anyway.** 8 KB ≪ the 256 KB custom-AR cutoff on 8×H100, so the
   engine uses its **custom one-shot all-reduce**, not the ring E0 benched (`comms_floor.md`, vLLM source).
   E0 measured an *upper bound on a path the engine bypasses*.

**E0's structural conclusion stands and is correct** — env tuning (LL / NVLS-as-NCCL-algo / channels) does
NOT move the small-message floor, so the comms lever is structural (fused AR+RMSNorm, one-shot custom AR,
overlap). I reinforce that. The only thing I temper is the *magnitude* claim that comms is ~6.6 ms in the
real engine; the self-consistent in-engine value is the ~3 ms / 16 µs regime.

## Consequences (honest, including against my own avenue)
- **Ladder:** keep `--C 16` (band ~10–18 µs). Do **not** plug 35 µs into `ladder_to_1000.py` /
  `path-to-1000.md` as the engine's C — it would over-credit comms and (mechanically) shrink the "overhead"
  term to ~3.5 ms, contradicting the independently-measured kernel inefficiency.
- **Against LOOP-C's own lever:** this *reduces* the exact-overlap / NVLS comms prize (hide ~3 ms, not
  ~6.6 ms). The smaller number is the correct one to bank. Same direction as my earlier tempering — better to
  size it right than oversell it.
- **The overhead elephant survives.** With C≈16 µs the ~7 ms "overhead" (≈ kernel sub-roofline at e≈0.18 +
  residual host) is still the largest term — so the **K5 e→1 kernel work and E-attr remain the top floor
  levers**, ahead of comms tuning. E0 doesn't change that ordering; it sharpens why env-comms is dead.

## The one resolver (gate)
This whole reconciliation is a *bounding* argument resting on the measured e≈0.16–0.19. The definitive pin is
**E-attr** — `nsys profile -t cuda,nvtx,nccl` over ~20 decode steps, `nccl_sum` gives the engine's REAL
per-step NCCL time in one trace, collapsing the [10,18] µs band to a number and confirming/refuting this.
E-attr is on the roadmap and (per the scheduling doc) unowned; it needs one in-window GPU slot. Until it
runs, **C = 16 µs is the defensible ladder value, not 35**.
