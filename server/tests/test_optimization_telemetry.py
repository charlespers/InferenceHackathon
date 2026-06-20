from server.optimization_telemetry import floor_breakdown, summary_fields, kv_ms


def test_breakdown_sums_to_tpot_at_measured_point():
    # The measured bf16-TP8 point: 11.67ms, 16µs collectives, ctx 512.
    bd = floor_breakdown(11.67, "bf16", "bf16", 16.0, 512)
    assert abs(bd["weight"] - 1.61) < 0.01
    assert abs(bd["comms"] - 3.008) < 0.01        # 188 × 16µs
    assert bd["overhead"] > 6.0                   # the dominant residual (kernel+host+launch)
    assert abs(sum(bd.values()) - 11.67) < 0.05   # parts sum to the whole


def test_regime_floor_bound_today():
    s = summary_fields(11.67, 85.7, "bf16", "bf16", 16.0, 512)
    assert s["regime"] == "floor-bound"           # comms+overhead >> weight
    assert "spec" in s["next_lever"]
    assert s["pct_of_ceiling"] == round(100 * 85.7 / 2000, 1)   # ~4.3%
    assert 0 < s["pct_of_roofline"] < 100


def test_regime_flips_weight_bound_when_floor_removed():
    # Hypothetical fp8 engine near its roofline (floor mostly gone): weight should dominate.
    s = summary_fields(1.1, 900.0, "fp8", "fp8", 4.0, 512)
    assert s["regime"] == "weight-bound"
    assert "quant" in s["next_lever"] or "route" in s["next_lever"]


def test_kv_grows_with_context_and_halves_with_fp8():
    assert kv_ms(512, "bf16") < kv_ms(128_000, "bf16")        # grows with ctx
    assert abs(kv_ms(128_000, "fp8") - kv_ms(128_000, "bf16") / 2) < 1e-6   # fp8 halves it


def test_overhead_never_negative():
    # an implausibly low tpot shouldn't produce a negative residual
    assert floor_breakdown(0.5, "fp8", "fp8", 4.0, 512)["overhead"] == 0.0
