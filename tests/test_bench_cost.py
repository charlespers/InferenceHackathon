import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.cost import energy_metrics, rental_usd_per_mtok


def test_rental_cost_formula():
    # 8 GPUs, $3/hr, 200 tok/s: (8*3/3600)/200*1e6
    c = rental_usd_per_mtok(200.0, 8, usd_per_gpu_hr=3.0)
    assert abs(c - (8 * 3 / 3600) / 200 * 1e6) < 1e-6
    assert rental_usd_per_mtok(None, 8) is None
    # faster decode -> cheaper per Mtok
    assert rental_usd_per_mtok(400.0, 8) < rental_usd_per_mtok(200.0, 8)


def test_energy_metrics_known_values():
    em = energy_metrics(200.0, 4000.0, usd_per_kwh=0.12)
    assert abs(em["joules_per_token"] - 20.0) < 1e-9        # 4000/200
    assert abs(em["tok_s_per_watt"] - 0.05) < 1e-9          # 200/4000
    assert abs(em["kwh_per_mtok"] - (20 * 1e6 / 3.6e6)) < 1e-9
    assert abs(em["usd_per_mtok_energy"] - em["kwh_per_mtok"] * 0.12) < 1e-12


def test_energy_none_when_unmeasured():
    em = energy_metrics(None, None)
    assert all(v is None for v in em.values())


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
