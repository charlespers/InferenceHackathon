import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.optimize import (
    TPOT, MEASURED_BASELINE, MEASURED_BASELINE_TOK_S, corroborate_weight,
    optimize, apply_spec, plain_decode_ceiling, report, LEVERS)


def test_baseline_reproduces_measured_tok_s():
    # The additive TPOT calibration must reproduce the on-box vLLM bf16-TP8 number
    # (85.7 tok/s) within 1% — this is what makes the loop's deltas trustworthy.
    assert abs(MEASURED_BASELINE.tok_s - MEASURED_BASELINE_TOK_S) / MEASURED_BASELINE_TOK_S < 0.01


def test_weight_term_corroborated_by_roofline():
    # The calibrated 1.56 ms weight term must sit between the analytical fp8 and
    # bf16 tp-plan weight reads (it is bf16, so ~1.6 ms; fp8 ~0.8 ms).
    cw = corroborate_weight()
    assert cw["fp8_weight_ms"] < MEASURED_BASELINE.weight_ms + 0.1 < cw["bf16_weight_ms"] + 0.1
    assert 1.4 < cw["bf16_weight_ms"] < 1.8


def test_loop_is_monotonic_and_deterministic():
    final, steps = optimize()
    assert steps, "loop must take at least one step"
    # every KEPT step strictly improves tok/s (greedy, no regressions committed)
    prev = MEASURED_BASELINE.tok_s
    for st in steps:
        if st.kept:
            assert st.after_tok_s > prev
            prev = st.after_tok_s
    # determinism: same inputs -> identical trajectory
    final2, steps2 = optimize()
    assert final.tok_s == final2.tok_s
    assert [s.lever for s in steps] == [s.lever for s in steps2]


def test_loop_attacks_dominant_term_first():
    # baseline dominant term is overhead (60%); the first lever taken must target it
    assert MEASURED_BASELINE.dominant == "overhead"
    _, steps = optimize()
    assert steps[0].targeted == "overhead"


def test_lossless_only_by_default_lossy_gated():
    # int4 is lossy; it must not appear among the levers the default loop keeps
    _, steps = optimize(allow_lossy=False)
    kept = {s.lever for s in steps if s.kept}
    assert "int4 experts (lossy safety net)" not in kept
    # but it IS available when lossy is explicitly allowed
    _, steps_lossy = optimize(allow_lossy=True)
    assert any("int4" in s.lever for s in steps_lossy)


def test_1000_is_below_the_hard_floor():
    # The fp8 single-stream weight floor exceeds 1000, so 1000 is physically possible
    # for plain decode — the blocker is overhead+comms, not the memory wall.
    ceil = plain_decode_ceiling()
    assert ceil["hard_floor_fp8_weight"] > 1000.0
    # but the conservative (comms-not-overlapped) plain path does NOT reach 1000
    assert ceil["plain_1000_conservative"] is False
    # the optimistic (deferred-overlap) path does — it hinges on the unmeasured gate
    assert ceil["plain_1000_optimistic"] is True


def test_spec_clears_1000_losslessly():
    final, _ = optimize()
    spec = apply_spec(final)
    assert spec.speedup > 1.0
    assert spec.spec_tok_s >= 1000.0          # spec reaches the target
    assert spec.tau > 1.0                     # emits >1 token per verify pass


def test_every_lever_has_status_and_source():
    for lv in LEVERS:
        assert lv.status in ("MEASURED", "PREDICTED", "MISSING")
        assert lv.source, f"{lv.name} missing a doc citation"
        assert lv.targets in ("overhead", "comms", "weight", "kv")


def test_report_renders():
    r = report()
    assert "OPTIMIZATION LOOP" in r and "CEILING PROOF" in r and "RANKED ROADMAP" in r


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
