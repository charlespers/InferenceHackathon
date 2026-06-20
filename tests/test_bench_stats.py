import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.stats import (
    Stat, summarize, percentile, t95, stat_to_dict, stat_from_dict, means_differ,
)


def test_percentile_interpolates():
    assert percentile([0.0, 1.0, 2.0, 3.0], 0.5) == 1.5
    assert percentile([], 0.5) is None
    assert percentile([7.0], 0.9) == 7.0


def test_empty_is_all_none():
    s = summarize([])
    assert s.n == 0 and s.mean is None and s.ci95_lo is None


def test_single_sample_has_no_std_or_ci():
    s = summarize([42.0])
    assert s.n == 1 and s.mean == 42.0
    assert s.std is None and s.ci95_lo is None
    assert s.p50 == 42.0 and s.min == 42.0 and s.max == 42.0


def test_known_mean_std_cv():
    s = summarize([90.0, 110.0])
    assert s.mean == 100.0
    # sample std (ddof=1) of [90,110] = sqrt(((10^2)+(10^2))/1) = sqrt(200)
    assert abs(s.std - (200 ** 0.5)) < 1e-9
    assert abs(s.cv - (s.std / 100.0)) < 1e-12


def test_ci_uses_student_t():
    # n=2, df=1, t=12.706; se = std/sqrt(2); half = t*se
    s = summarize([90.0, 110.0])
    se = (200 ** 0.5) / (2 ** 0.5)
    half = 12.706 * se
    assert abs((s.ci95_hi - s.ci95_lo) / 2.0 - half) < 1e-6
    assert s.ci95_half_width is not None


def test_t95_table_and_fallback():
    assert t95(1) == 12.706
    assert t95(30) == 2.042
    assert t95(100) == 1.96            # beyond table -> normal z
    assert t95(0) == float("inf")


def test_roundtrip_dict():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    back = stat_from_dict(stat_to_dict(s))
    assert back.n == s.n and back.mean == s.mean
    assert back.samples == s.samples
    assert stat_from_dict(None) is None


def test_means_differ_on_disjoint_cis():
    a = summarize([100.0, 101.0, 99.0, 100.0])
    b = summarize([200.0, 201.0, 199.0, 200.0])
    assert means_differ(a, b) is True
    # overlapping -> not distinguishable
    c = summarize([100.0, 105.0, 95.0, 100.0])
    assert means_differ(a, c) is False
    # n<2 -> conservative False
    assert means_differ(summarize([100.0]), b) is False


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
