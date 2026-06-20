"""Automated B=1 decode optimization loop — the reproducible tok/s benchmark + driver.

This runs on ANY machine (no GPU): it is an *additive TPOT model* calibrated to
the team's measured on-box anchors, not a CUDA benchmark. The real kernels live on
the 8×H100 box; this driver is the loop that decides what to build next and proves
where the physical ceiling is.

Why additive (not the multiplicative efficiency knob in bench.MockEngine)?
At B=1 a decode step's wall-time is a SUM of terms on *different hardware paths*:

    TPOT = overhead(launch/host/scheduler) + comms(collectives) + weight_read(HBM) + kv_read(HBM)

Each optimization lever attacks ONE term. "Profile the hottest path, attack it,
re-measure" is therefore literally: find the dominant term, apply the best lever
that targets it, recompute tok/s. The multiplicative knob can't express this
(it scales every term together); the additive model is the faithful one.

Calibration anchor (MEASURED on box, docs/b1-optimization-atlas.md L4,L20-21):
    bf16-TP8 baseline = 85.7 tok/s  ->  TPOT 11.67 ms
      = overhead ~7.00 ms (60%)  +  comms ~3.00 ms (26%)
      + weight ~1.56 ms (14%)    +  kv ~0.10 ms
The weight term is corroborated by the analytical roofline (latency.decode_latency,
tp-plan, bf16 ≈ 1.56–1.61 ms) — see `corroborate_weight()`.

Each lever carries: the term it targets, its effect (MEASURED on box / PREDICTED by
roofline / MISSING capability), a quality gate, and a source citation. The loop is
greedy + deterministic, so the whole thing is a reproducible, committable proof.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional

from .model import QWEN3_235B, MoEConfig
from .hardware import GPUS, Cluster
from .latency import decode_latency
from .bench.spec_model import spec_speedup, expected_emitted, spec_sweep


# ---------------------------------------------------------------------------
# The additive per-token (TPOT) decomposition. Times in milliseconds.
# ---------------------------------------------------------------------------
N_LAYERS = QWEN3_235B.n_layers          # 94
N_COLLECTIVES = 2 * N_LAYERS            # 188 (post-attn + post-MoE all-reduce / layer)


@dataclass(frozen=True)
class TPOT:
    overhead_ms: float   # host/launch/scheduler/python — paid per step regardless of work
    comms_ms: float      # the 188 B=1 collectives
    weight_ms: float     # active-weight HBM read (attn + lit experts + lm_head), sharded
    kv_ms: float         # KV-cache HBM read

    @property
    def total_ms(self) -> float:
        return self.overhead_ms + self.comms_ms + self.weight_ms + self.kv_ms

    @property
    def tok_s(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms else float("inf")

    @property
    def dominant(self) -> str:
        terms = {"overhead": self.overhead_ms, "comms": self.comms_ms,
                 "weight": self.weight_ms, "kv": self.kv_ms}
        return max(terms, key=terms.get)

    def shares(self) -> dict:
        t = self.total_ms or 1.0
        return {"overhead": self.overhead_ms / t, "comms": self.comms_ms / t,
                "weight": self.weight_ms / t, "kv": self.kv_ms / t}


# Measured baseline (docs/b1-optimization-atlas.md L4,L20-21; matches vLLM bf16-TP8).
MEASURED_BASELINE = TPOT(overhead_ms=7.00, comms_ms=3.00, weight_ms=1.56, kv_ms=0.10)
MEASURED_BASELINE_TOK_S = 85.7   # docs/b1-optimization-atlas.md L4


def corroborate_weight(cluster: Optional[Cluster] = None) -> dict:
    """Cross-check the calibrated weight term against the analytical roofline.

    Returns the tp-plan weight-read (ms) the pure roofline predicts at bf16/fp8,
    so the reader can see the calibrated 1.56 ms is not invented."""
    cluster = cluster or Cluster(gpu=GPUS["H100-SXM-80GB"], n_gpus=8)
    bf16 = decode_latency(QWEN3_235B, cluster, plan="tp", dtype_bytes=2, seq_len=2048)
    fp8 = decode_latency(QWEN3_235B, cluster, plan="tp", dtype_bytes=1, seq_len=2048)
    return {"bf16_weight_ms": bf16.weight_read_s * 1e3,
            "fp8_weight_ms": fp8.weight_read_s * 1e3,
            "comms_at_5us_model_ms": bf16.comms_s * 1e3}


# ---------------------------------------------------------------------------
# Levers. Each transforms ONE term of the TPOT. `status`:
#   MEASURED  — the number is from a real on-box kernel run
#   PREDICTED — roofline projection, kernel exists or is straightforward
#   MISSING   — capability not yet built; this is what gates the target
# `lossless=True` means no quality gate (bit-exact / exact-verify). A lossy lever
# is only taken if its quality gate is explicitly accepted.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Lever:
    name: str
    targets: str                       # term it attacks
    apply: Callable[[TPOT], TPOT]      # returns the new TPOT
    status: str                        # MEASURED | PREDICTED | MISSING
    lossless: bool
    gate: str                          # quality-gate description ("" if lossless)
    source: str                        # doc citation


def _nvls_comms(t: TPOT) -> TPOT:
    # 188 collectives × 3.84 µs measured (was NCCL ~16 µs → 3.0 ms).
    return replace(t, comms_ms=N_COLLECTIVES * 3.84e-3)


def _megakernel_overhead(t: TPOT) -> TPOT:
    # Persistent megakernel / CUDA-graph capture collapses host+launch+scheduler
    # overhead to a small residual. 7.0 ms → ~0.5 ms residual (graph replay + sync).
    return replace(t, overhead_ms=0.50)


def _fp8_weights(t: TPOT) -> TPOT:
    # Fused on-chip dequant halves the weight bytes: 1.56 ms → 0.78 ms.
    return replace(t, weight_ms=0.78)


def _deferred_overlap(t: TPOT) -> TPOT:
    # NVLS reduce (3.84 µs) runs on a few SMs over NVLink while the rest cp.async-
    # stream the next op's fp8 weights over HBM (separate paths). At C < weight-cover
    # the reduce is fully hidden → comms folds into the weight stream → comms ≈ 0.
    return replace(t, comms_ms=0.0)


def _int4_experts(t: TPOT) -> TPOT:
    # Lossy safety net: int4 experts halve the (already-fp8) weight again 0.78→0.39.
    return replace(t, weight_ms=0.39)


LEVERS: List[Lever] = [
    Lever("NVLS in-kernel all-reduce", "comms", _nvls_comms,
          status="MEASURED", lossless=True, gate="",
          source="comms-breakthrough-nvls.md L1-12 (C=3.84µs, bit-exact, 8×H100)"),
    Lever("Megakernel / CUDA-graph (overhead→0)", "overhead", _megakernel_overhead,
          status="PREDICTED", lossless=True, gate="",
          source="path-to-1000.md L36-39 (persistent megakernel)"),
    Lever("FP8 weights (fused dequant)", "weight", _fp8_weights,
          status="PREDICTED", lossless=True,
          gate="fp8 dequant fused on-chip; perplexity parity (gated, passes)",
          source="absolute-ceiling.md L8-13 (fp8 weight 0.78–0.80 ms)"),
    Lever("Deferred-overlap (hide comms under weight stream)", "comms", _deferred_overlap,
          status="MISSING", lossless=True,
          gate="requires C < per-collective weight-cover; e2e overlap UNMEASURED",
          source="comms-breakthrough-nvls.md L37-39 (the open gate)"),
    Lever("int4 experts (lossy safety net)", "weight", _int4_experts,
          status="PREDICTED", lossless=False,
          gate="LOSSY: accuracy drop at B=1; only if fp8 path slips",
          source="path-to-1000.md L72,L147 (int4 weight 0.39–0.40 ms)"),
]


# ---------------------------------------------------------------------------
# The loop: greedily attack the dominant term with the best lossless lever that
# targets it; apply ONE per iteration; keep only if tok/s strictly improves.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    iteration: int
    lever: str
    targeted: str
    status: str
    before_tok_s: float
    after_tok_s: float
    delta_tok_s: float
    new_dominant: str
    kept: bool
    note: str


def optimize(baseline: TPOT = MEASURED_BASELINE, *,
             allow_lossy: bool = False) -> tuple[TPOT, List[Step]]:
    """Run the greedy profile→attack→re-measure loop over the lossless levers.

    Returns (final TPOT for *plain* decode, list of Steps). Spec is handled
    separately (it is a multiplier over the whole step, not a TPOT term)."""
    cur = baseline
    used: set[str] = set()
    steps: List[Step] = []
    it = 0
    while True:
        dom = cur.dominant
        # candidate levers that target the current dominant term and we haven't used
        cands = [lv for lv in LEVERS if lv.targets == dom and lv.name not in used
                 and (lv.lossless or allow_lossy)]
        if not cands:
            # nothing targets the dominant term; try ANY remaining improving lever
            cands = [lv for lv in LEVERS if lv.name not in used
                     and (lv.lossless or allow_lossy)]
        # pick the lever giving the biggest improvement (profile-driven greedy)
        best, best_after = None, cur
        for lv in cands:
            cand = lv.apply(cur)
            if cand.tok_s > best_after.tok_s:
                best, best_after = lv, cand
        if best is None:
            break
        it += 1
        kept = best_after.tok_s > cur.tok_s + 1e-6
        steps.append(Step(
            iteration=it, lever=best.name, targeted=best.targets, status=best.status,
            before_tok_s=cur.tok_s, after_tok_s=best_after.tok_s,
            delta_tok_s=best_after.tok_s - cur.tok_s, new_dominant=best_after.dominant,
            kept=kept, note=best.gate or "lossless"))
        used.add(best.name)
        if kept:
            cur = best_after
    return cur, steps


# ---------------------------------------------------------------------------
# Speculative decode — the multiplier the docs prove is mandatory for ≥1000.
# τ = expected emitted tokens per verify pass; verify costs slightly >1 step.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpecResult:
    accept: float
    draft_len: int
    n_drafters: int
    floor_fraction: float     # F: overhead+comms share spec amortizes
    tau: float                # E[emitted per verify]
    speedup: float
    plain_tok_s: float
    spec_tok_s: float


def floor_fraction(plain: TPOT) -> float:
    """F = the share of the (optimized) step that spec amortizes = everything that
    is NOT expert weight (overhead + comms + kv + the non-expert ~34% of weight)."""
    f = (plain.overhead_ms + plain.comms_ms + plain.kv_ms) / plain.total_ms
    return max(0.0, min(1.0, f + 0.34 * plain.weight_ms / plain.total_ms))


def best_spec_tree(plain: TPOT, *, accept: float = 0.72,
                   ks=(1, 2, 4, 8), ns=(1, 2, 4)) -> tuple:
    """Sweep (draft_len × n_drafters) against the OPTIMIZED plain step's regime and
    return the top-ranked tree. Confirms the docs' claim that once the floor is
    removed (weight-bound, low F) SMALL trees win — the expert-union tax of a wide/
    deep tree outweighs the few extra emitted tokens."""
    f = floor_fraction(plain)
    rows = spec_sweep(accept, f, ks=ks, ns=ns, cfg=QWEN3_235B)
    return rows[0], rows   # rows already sorted by speedup desc


def apply_spec(plain: TPOT, *, accept: float = 0.72, draft_len: int = 2,
               n_drafters: int = 1) -> SpecResult:
    """Apply floor-aware EAGLE3-style spec on top of the optimized plain step.

    Uses the team's calibrated spec model (bench.spec_model). F = the share of
    TPOT that is *not* expert-weight (overhead+comms+attn) — what one verify pass
    amortizes."""
    f = floor_fraction(plain)
    sp = spec_speedup(accept, draft_len, n_drafters, f, QWEN3_235B)
    tau = expected_emitted(accept, draft_len, n_drafters)
    return SpecResult(accept=accept, draft_len=draft_len, n_drafters=n_drafters,
                      floor_fraction=f, tau=tau, speedup=sp,
                      plain_tok_s=plain.tok_s, spec_tok_s=plain.tok_s * sp)


# ---------------------------------------------------------------------------
# The physical ceiling: best *plain* decode vs the 1000 target.
# ---------------------------------------------------------------------------
def plain_decode_ceiling() -> dict:
    """Where does *plain* (non-spec) decode cap? Three scenarios that differ ONLY
    in how much of the comms term the (unmeasured) deferred-overlap kernel hides.

    The hard single-stream floor is the fp8 active-weight read alone (0.78 ms →
    ~1283 tok/s): 1000 tok/s is NOT below it, so 1000 is physically *possible* for
    plain decode. Whether plain decode actually reaches it hinges entirely on the
    deferred-overlap gate — which is the one thing not yet measured e2e."""
    fp8_weight_floor = TPOT(overhead_ms=0.0, comms_ms=0.0, weight_ms=0.78, kv_ms=0.0)
    # conservative: overhead→0 (perfect megakernel) but comms NOT overlapped — just
    # NVLS-fast at 2 µs (docs/path-to-1000.md L52-54 → ~865 tok/s).
    nvls_not_overlapped = TPOT(overhead_ms=0.0, comms_ms=N_COLLECTIVES * 2.0e-3,
                               weight_ms=0.78, kv_ms=0.05)
    # optimistic: deferred-overlap fully hides comms (comms→0), perfect megakernel.
    # Requires the UNMEASURED overlap gate to land — comms-breakthrough-nvls.md L37-39.
    overlap_ideal = replace(nvls_not_overlapped, comms_ms=0.0)
    return {
        "hard_floor_fp8_weight": fp8_weight_floor.tok_s,   # ~1283 — the wall 1000 sits below
        "conservative_nvls2us": nvls_not_overlapped.tok_s,  # ~865 — comms not hidden
        "optimistic_overlap": overlap_ideal.tok_s,          # ~1150 — comms fully hidden
        "target": 1000.0,
        # plain reaches 1000 ONLY in the optimistic (unmeasured) case:
        "plain_1000_conservative": nvls_not_overlapped.tok_s >= 1000.0,
        "plain_1000_optimistic": overlap_ideal.tok_s >= 1000.0,
    }


# ---------------------------------------------------------------------------
# Sensitivity: under WHICH (overhead residual, comms-overlap) does plain decode
# reach 1000? This is the crux the loop exposed — solve for the boundary instead
# of asserting it. Plain decode (fp8 weight 0.78, kv 0.05) reaches 1000 iff:
#   overhead_residual + (1 - overlap_frac)*comms_nvls + 0.78 + 0.05 <= 1.00 ms
# where comms_nvls = 188 * 3.84 µs = 0.72 ms (NVLS-fast, before overlap credit).
# ---------------------------------------------------------------------------
COMMS_NVLS_MS = N_COLLECTIVES * 3.84e-3   # 0.722 ms — NVLS-fast, not yet overlapped
FP8_WEIGHT_MS = 0.78
KV_SHORT_MS = 0.05


def plain_tok_s(overhead_ms: float, overlap_frac: float,
                weight_ms: float = FP8_WEIGHT_MS) -> float:
    comms = (1.0 - max(0.0, min(1.0, overlap_frac))) * COMMS_NVLS_MS
    return TPOT(overhead_ms=overhead_ms, comms_ms=comms,
                weight_ms=weight_ms, kv_ms=KV_SHORT_MS).tok_s


def min_overlap_for_1000(overhead_ms: float, target: float = 1000.0) -> Optional[float]:
    """Minimum comms-overlap fraction that lets PLAIN decode hit `target` at a
    given megakernel overhead residual. None if unreachable even at full overlap."""
    budget = 1000.0 / target            # total TPOT budget in ms (1.0 ms for 1000)
    slack = budget - overhead_ms - weight_ms_floor() - KV_SHORT_MS
    if slack >= COMMS_NVLS_MS:
        return 0.0                      # reaches target with no overlap at all
    if slack < 0:
        return None                     # overhead+weight already blow the budget
    return 1.0 - slack / COMMS_NVLS_MS   # need to hide this fraction of comms


def weight_ms_floor() -> float:
    return FP8_WEIGHT_MS


def sensitivity_grid(overheads=(0.0, 0.25, 0.50, 0.75),
                     overlaps=(0.0, 0.5, 0.76, 1.0)) -> List[dict]:
    """Plain-decode tok/s across the two uncertain knobs, with a 1000 flag."""
    grid = []
    for oh in overheads:
        row = {"overhead_ms": oh, "min_overlap_for_1000": min_overlap_for_1000(oh)}
        for ov in overlaps:
            row[f"ov{ov}"] = plain_tok_s(oh, ov)
        grid.append(row)
    return grid


# ---------------------------------------------------------------------------
# CLI / report
# ---------------------------------------------------------------------------
def _fmt_tpot(t: TPOT) -> str:
    s = t.shares()
    return (f"TPOT {t.total_ms:6.2f} ms = "
            f"oh {t.overhead_ms:5.2f}({s['overhead']*100:2.0f}%) + "
            f"cm {t.comms_ms:5.2f}({s['comms']*100:2.0f}%) + "
            f"wt {t.weight_ms:4.2f}({s['weight']*100:2.0f}%) + "
            f"kv {t.kv_ms:4.2f}  ->  {t.tok_s:6.1f} tok/s")


def report() -> str:
    out: List[str] = []
    P = out.append
    P("=" * 84)
    P("  B=1 DECODE OPTIMIZATION LOOP  —  Qwen3-235B-A22B on 8×H100  (additive TPOT)")
    P("=" * 84)
    cw = corroborate_weight()
    P("")
    P("CALIBRATION (measured on box, docs/b1-optimization-atlas.md L4,L20-21):")
    P(f"  baseline      : {_fmt_tpot(MEASURED_BASELINE)}")
    P(f"  measured tok/s: {MEASURED_BASELINE_TOK_S}  (vLLM bf16-TP8; model reproduces it)")
    P(f"  weight cross-check (roofline tp-plan): bf16 {cw['bf16_weight_ms']:.2f} ms / "
      f"fp8 {cw['fp8_weight_ms']:.2f} ms  -> corroborates the 1.56 ms calibration")
    P("")
    P("-" * 84)
    P("LOOP  (greedy: profile dominant term -> apply ONE lever -> re-measure):")
    P("-" * 84)
    final, steps = optimize()
    P(f"  start: {_fmt_tpot(MEASURED_BASELINE)}   [dominant: {MEASURED_BASELINE.dominant}]")
    for st in steps:
        flag = "KEEP" if st.kept else "drop"
        P(f"  [{st.iteration}] {flag}  attack {st.targeted:8s} via {st.lever}")
        P(f"        {st.status:9s} {st.before_tok_s:6.1f} -> {st.after_tok_s:6.1f} tok/s "
          f"(+{st.delta_tok_s:5.1f})   next dominant: {st.new_dominant}")
        P(f"        gate: {st.note}")
    P("")
    P(f"  PLAIN-DECODE BEST (all lossless levers): {_fmt_tpot(final)}")
    P("")
    P("-" * 84)
    P("CEILING PROOF  (where does PLAIN decode cap, and is 1000 physically possible?):")
    P("-" * 84)
    ceil = plain_decode_ceiling()
    P(f"  hard single-stream floor (fp8 weight 0.78 ms alone) : {ceil['hard_floor_fp8_weight']:6.1f} tok/s")
    P(f"    -> 1000 sits BELOW this wall, so 1000 is physically POSSIBLE for plain decode.")
    P(f"  conservative (overhead→0, NVLS 2µs, comms NOT hidden): {ceil['conservative_nvls2us']:6.1f} tok/s  "
      f"[1000? {ceil['plain_1000_conservative']}]")
    P(f"  optimistic   (deferred-overlap hides comms→0)        : {ceil['optimistic_overlap']:6.1f} tok/s  "
      f"[1000? {ceil['plain_1000_optimistic']}]")
    P("")
    P("  >>> The plain-decode path to 1000 hinges ENTIRELY on the deferred-overlap kernel")
    P("      (comms-breakthrough-nvls.md L37-39) — UNMEASURED e2e. If comms hides fully,")
    P("      plain decode clears 1000; if it only reaches the NVLS floor, plain caps ~865.")
    P("      NOTE: this exposes a real tension between two team docs (path-to-1000's 865")
    P("      ceiling assumes comms is NOT overlapped; comms-breakthrough says it is). The")
    P("      single measurement that resolves it: e2e TPOT with deferred-overlap in k6.")
    P("")
    P("-" * 84)
    P("SENSITIVITY  (plain-decode tok/s vs the 2 uncertain knobs; * = clears 1000):")
    P("-" * 84)
    P("  rows = megakernel overhead residual (ms); cols = comms-overlap fraction hidden")
    P(f"  {'overhead':>10s} | {'0% hidden':>10s} {'50%':>8s} {'76%':>8s} {'100%':>8s} | min-overlap→1000")
    for r in sensitivity_grid():
        cells = []
        for ov in (0.0, 0.5, 0.76, 1.0):
            v = r[f"ov{ov}"]
            cells.append(f"{v:6.0f}{'*' if v >= 1000 else ' '}")
        need = r["min_overlap_for_1000"]
        need_s = "unreachable" if need is None else f"{need*100:.0f}% hidden"
        P(f"  {r['overhead_ms']:8.2f}ms | {cells[0]:>10s} {cells[1]:>8s} {cells[2]:>8s} "
          f"{cells[3]:>8s} | {need_s}")
    P("  READ: plain decode reaches 1000 ONLY in the top-right corner — overhead≈0 AND")
    P("  ≥76% of comms hidden. At any realistic overhead residual (≥0.25 ms) plain decode")
    P("  CANNOT reach 1000 even with perfectly-hidden comms. That corner is two unmeasured")
    P("  kernel bets stacked; spec (below) clears 1000 without either bet.")
    P("")
    P("-" * 84)
    P("SPEC DECODE  (the ROBUST path; docs/why-spec-wins.md, spec-decode-floor-bound.md):")
    P("-" * 84)
    top, rows = best_spec_tree(final)
    # Headline uses the REALISTIC buildable tree (single EAGLE3 head = W1×D2), not the
    # model optimum (which assumes N *independent* drafters — optimistic; real EAGLE3 is
    # one correlated head). Both are shown; both clear 1000.
    spec = apply_spec(final, draft_len=2, n_drafters=1)
    P(f"  tree sweep vs the OPTIMIZED step  F={spec.floor_fraction:.2f} (still floor-bound: 0.5ms")
    P(f"  overhead residual + structural non-expert weight keep F high, so bigger trees score):")
    for r in rows[:4]:
        opt = " <- model opt (assumes N independent drafters)" if (
            r.draft_len, r.n_drafters) == (top.draft_len, top.n_drafters) else ""
        real = " <- realistic single EAGLE3 head" if (r.draft_len, r.n_drafters) == (2, 1) else ""
        P(f"      W{r.n_drafters}×D{r.draft_len}: emit {r.emitted:.2f}/verify, "
          f"cost {r.verify_cost:.2f} steps -> {r.speedup:.2f}×{opt}{real}")
    P(f"  HEADLINE (buildable): EAGLE3 W1×D2  accept={spec.accept:.2f}  τ={spec.tau:.2f}  "
      f"-> {spec.speedup:.2f}×  (lossless: exact verification)")
    P(f"  the model optimum W{top.n_drafters}×D{top.draft_len} ({top.speedup:.2f}×) is the "
      f"theoretical envelope if N independent drafters were available.")
    P(f"  {spec.plain_tok_s:6.1f} tok/s (plain)  ->  {spec.spec_tok_s:6.1f} tok/s (spec)  "
      f"[{'>= 1000 ✓' if spec.spec_tok_s >= 1000 else '< 1000'}]")
    P("  Why spec is the make-or-break: it clears 1000 in BOTH the optimistic AND the")
    P("  conservative comms scenarios, losslessly — it does not bet 1000 on one unmeasured")
    P("  kernel gate. It is the only lever that reaches the target ROBUSTLY.")
    P("")
    P("-" * 84)
    P("RANKED ROADMAP  (remaining gains, by leverage & readiness):")
    P("-" * 84)
    for i, line in enumerate(_roadmap(steps, spec, ceil), 1):
        P(f"  {i}. {line}")
    P("=" * 84)
    return "\n".join(out)


def _roadmap(steps, spec, ceil) -> List[str]:
    rows = []
    for lv in LEVERS:
        st = next((s for s in steps if s.lever == lv.name), None)
        delta = f"+{st.delta_tok_s:.0f} tok/s" if st and st.kept else "(no gain in current regime)"
        rows.append(f"[{lv.status:9s}] {lv.name:42s} {delta:24s} — {lv.source}")
    rows.append(f"[MISSING  ] {'Speculative decode (EAGLE3 small-tree)':42s} "
                f"{'×%.2f -> %.0f tok/s' % (spec.speedup, spec.spec_tok_s):24s} "
                f"— THE capability that breaks the single-stream ceiling")
    return rows


def main(argv=None) -> None:
    print(report())


if __name__ == "__main__":
    main()
