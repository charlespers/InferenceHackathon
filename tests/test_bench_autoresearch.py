import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import roofline
import sweep as bsweep   # bench/sweep.py (standalone autoresearch driver)


def test_roofline_dominant_term_recognized_by_classify():
    # the standalone roofline + autoresearch decision tree must share one vocabulary
    a = roofline.analyze(ctx=4096, tpot_ms=3.8, weight_bytes=2, kv_bytes=2)
    assert a["dominant_term"] in bsweep.NEXT_LEVER
    levers = bsweep.classify(a["dominant_term"])
    assert levers and "unclear" not in levers[0]


def test_classify_maps_every_principled_term():
    for term in ("weight_bw", "kv_bw", "comms", "kernel_gap", "compute"):
        out = bsweep.classify(term)
        assert out and "unclear" not in out[0]
    assert "unclear" in bsweep.classify("nonsense")[0]


def test_public_api_exports_rigorous_suite():
    import inferutil.bench as b
    for name in ("diagnose", "recommend", "compute_efficiency", "summarize",
                 "depth_sweep", "layout_grid", "full_grid", "rental_usd_per_mtok",
                 "build_manifest", "regression_gate", "export_markdown",
                 "prefill_latency"):
        assert hasattr(b, name), name


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
