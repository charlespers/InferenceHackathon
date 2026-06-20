import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.spec_model import (
    per_position_hit, expected_emitted, verify_weight_units, verify_cost,
    spec_speedup, spec_sweep, SpecRow,
)


def test_expected_emitted_includes_bonus_token():
    # one drafter, k=4, accept=0.7: (1-0.7^5)/(1-0.7) INCLUDING the +1 bonus
    p = 0.7
    expected = (1 - p ** 5) / (1 - p)
    assert abs(expected_emitted(0.7, 4, 1) - expected) < 1e-9
    # always >= 1 (the guaranteed bonus token), even at accept=0
    assert expected_emitted(0.0, 4, 1) >= 1.0 - 1e-9
    # more acceptance -> more emitted
    assert expected_emitted(0.8, 4, 1) > expected_emitted(0.3, 4, 1)


def test_multi_drafter_raises_hit_prob():
    assert per_position_hit(0.5, 4) > per_position_hit(0.5, 1)
    assert expected_emitted(0.5, 4, 4) > expected_emitted(0.5, 4, 1)


def test_verify_cost_plain_step_is_one():
    # a single-position verify (k=1,N=1) costs ~1 plain step at any floor
    assert abs(verify_weight_units(1, 1) - 1.0) < 1e-9
    assert abs(verify_cost(1, 1, 0.0) - 1.0) < 1e-9
    assert abs(verify_cost(1, 1, 0.9) - 1.0) < 1e-9
    # bigger trees cost more weight (more distinct experts)
    assert verify_weight_units(8, 4) > verify_weight_units(2, 1)


def test_floor_regime_flips_optimal_tree():
    # weight-bound (F=0): small tree wins; floor-bound (F=0.9): big tree wins
    assert spec_speedup(0.7, 2, 1, 0.0) > spec_speedup(0.7, 8, 4, 0.0)
    assert spec_speedup(0.7, 8, 4, 0.9) > spec_speedup(0.7, 2, 1, 0.9)


def test_sweep_ranks_and_high_floor_prefers_big():
    rows = spec_sweep(0.7, 0.9, ks=(2, 4, 8), ns=(1, 2, 4))
    assert all(isinstance(r, SpecRow) for r in rows)
    sp = [r.speedup for r in rows]
    assert sp == sorted(sp, reverse=True)
    # at high floor the top config is a large tree
    assert rows[0].draft_len >= 4 and rows[0].n_drafters >= 2


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
