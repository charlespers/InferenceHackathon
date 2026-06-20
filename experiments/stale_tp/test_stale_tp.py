"""Pure unit tests for the stale-TP decision logic (NO torch, NO GPU, NO vLLM).

De-risks the scheduler before any slot is spent: verifies refresh/substitute
sequencing, per-slot caching, period wrap, decode-only gating, and both modes.
Run: python experiments/stale_tp/test_stale_tp.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from stale_tp import StaleScheduler, REAL, STALE, LOCAL, EXACT, REAL_FALLBACK  # noqa: E402


class FakeCfg:
    def __init__(self, **kw):
        self.enable = kw.get("enable", True)
        self.K = kw.get("K", 2)
        self.mode = kw.get("mode", "layer")
        self.policy = kw.get("policy", "proxy")
        self.decode_only = kw.get("decode_only", True)
        self.period = kw.get("period", 4)            # 2 layers x 2 slots
        self.collectives_per_layer = kw.get("collectives_per_layer", 2)
        self.debug = 0


def run_pass(sched, token_count=1, n=None):
    """Drive one forward pass of `period` calls. real_reduce returns 'R{idx}',
    local partial is 'L{idx}'. Returns list of (kind, value)."""
    n = n or sched.period
    out = []
    for i in range(n):
        # capture i for the thunk
        kind, val = sched.route(
            real_reduce=(lambda i=i: f"R{i}"),
            local_value=f"L{i}",
            token_count=token_count,
        )
        out.append((kind, val))
    return out


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    print(f"  ok: {name}")


def test_layer_proxy():
    s = StaleScheduler(FakeCfg(mode="layer", K=2, policy="proxy", period=4))
    r = run_pass(s)  # 2 layers
    kinds = [k for k, _ in r]
    vals = [v for _, v in r]
    check("layer/proxy kinds", kinds == [REAL, REAL, STALE, STALE])
    # layer1 slot0 reuses layer0 slot0 (R0); slot1 reuses R1
    check("layer/proxy reuses per-slot cache", vals == ["R0", "R1", "R0", "R1"])
    # second pass: cache cleared at wrap, fresh real on refresh layer
    r2 = run_pass(s)
    check("layer/proxy pass2 kinds", [k for k, _ in r2] == [REAL, REAL, STALE, STALE])


def test_layer_local():
    s = StaleScheduler(FakeCfg(mode="layer", K=2, policy="local", period=4))
    r = run_pass(s)
    kinds = [k for k, _ in r]
    vals = [v for _, v in r]
    check("layer/local kinds", kinds == [REAL, REAL, LOCAL, LOCAL])
    check("layer/local returns local partial", vals == ["R0", "R1", "L2", "L3"])


def test_K1_is_exact():
    s = StaleScheduler(FakeCfg(K=1, period=4))
    r = run_pass(s)
    check("K=1 all exact", all(k == EXACT for k, _ in r))
    check("K=1 all real values", [v for _, v in r] == ["R0", "R1", "R2", "R3"])


def test_disabled_is_exact():
    s = StaleScheduler(FakeCfg(enable=False, K=4, period=4))
    r = run_pass(s)
    check("disabled all exact", all(k == EXACT for k, _ in r))


def test_decode_only_keeps_prefill_exact():
    s = StaleScheduler(FakeCfg(mode="layer", K=2, decode_only=True, period=4))
    r = run_pass(s, token_count=8)  # prefill
    check("prefill stays real (no stale)", all(v.startswith("R") for _, v in r))
    check("prefill no STALE/LOCAL", all(k not in (STALE, LOCAL) for k, _ in r))
    # following decode pass DOES stale
    r2 = run_pass(s, token_count=1)
    check("decode after prefill staled", any(k == STALE for k, _ in r2))


def test_K3_refresh_spacing():
    # period = 3 layers x 2 slots = 6 ; K=3 -> only layer 0 refreshes, layers 1,2 stale
    s = StaleScheduler(FakeCfg(mode="layer", K=3, policy="proxy",
                               period=6, collectives_per_layer=2))
    r = run_pass(s, n=6)
    kinds = [k for k, _ in r]
    check("K=3 layer0 real, layers1-2 stale",
          kinds == [REAL, REAL, STALE, STALE, STALE, STALE])


def test_temporal():
    s = StaleScheduler(FakeCfg(mode="temporal", K=2, policy="proxy", period=4))
    p0 = run_pass(s)  # step 0: full real
    check("temporal step0 all real", all(k == REAL for k, _ in p0))
    p1 = run_pass(s)  # step 1: reuse cache per idx
    check("temporal step1 all stale", all(k == STALE for k, _ in p1))
    check("temporal step1 reuses prev-step values",
          [v for _, v in p1] == ["R0", "R1", "R2", "R3"])
    p2 = run_pass(s)  # step 2: full real again
    check("temporal step2 all real", all(k == REAL for k, _ in p2))


def test_shape_guard_prevents_prefill_leak():
    # The real bug that crashed vLLM: a prefill-shaped cached tensor leaking into a
    # decode call (token-count mismatch). The shape guard must fall back to a REAL
    # reduce instead of returning the wrong-shape cached value.
    s = StaleScheduler(FakeCfg(mode="temporal", K=2, policy="proxy",
                               decode_only=False, period=4))
    for i in range(4):  # step 0: prefill-shaped, cached with shape (8,4096)
        s.route((lambda i=i: f"P{i}"), f"L{i}", token_count=8, shape=(8, 4096))
    out = []
    for i in range(4):  # step 1: decode-shaped -> shapes mismatch -> must NOT reuse P*
        k, v = s.route((lambda i=i: f"D{i}"), f"L{i}", token_count=1, shape=(1, 4096))
        out.append((k, v))
    check("shape mismatch -> real_fallback (no prefill leak)",
          all(k == REAL_FALLBACK for k, _ in out))
    check("returns fresh decode values, never the prefill tensor",
          [v for _, v in out] == ["D0", "D1", "D2", "D3"])


def test_stats_and_calibration():
    s = StaleScheduler(FakeCfg(mode="layer", K=2, period=4))
    run_pass(s)
    snap = s.snapshot()
    check("stats counts a full pass", snap[REAL] + snap[STALE] == 4)
    check("observed calls/pass calibrated", snap["observed_calls_per_pass"] == 4)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} stale-TP scheduler tests (no GPU)...")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
