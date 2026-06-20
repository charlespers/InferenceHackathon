import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.speculative import sweep as spec_sweep, memory_feasibility
from inferutil.bench.report import format_spec_sweep


def test_format_spec_sweep_marks_best_and_projects_tokps():
    rows = spec_sweep(alphas=[0.7], ks=[4, 8], n_drafters_list=[1, 2, 4, 8])
    feas = memory_feasibility(use_fp8_target=True)
    out = format_spec_sweep(rows, feas, base_tok_s=260.0)
    assert "SPEC-DECODE SWEEP" in out
    assert "best feasible" in out
    assert "tok/s" in out                       # base_tok_s projection rendered
    assert "*best" in out


def test_bf16_target_has_less_headroom():
    fp8 = memory_feasibility(use_fp8_target=True)
    bf16 = memory_feasibility(use_fp8_target=False)
    assert bf16["max_drafters"] < fp8["max_drafters"]


def test_infeasible_when_draft_model_huge():
    # a 1 TB "draft" model can't fit -> no feasible drafters, no best line
    feas = memory_feasibility(draft_model_gb=1000.0, use_fp8_target=True)
    rows = spec_sweep(alphas=[0.7], ks=[4], n_drafters_list=[1, 2])
    out = format_spec_sweep(rows, feas)
    assert "fits <= 0 drafters" in out
    assert "best feasible" not in out


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
