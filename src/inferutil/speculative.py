"""Multi-drafter speculative decoding analysis for B=1 MoE inference.

Classic spec decode (1 drafter):
  - Draft model proposes k tokens autoregressively.
  - Target verifies all k in one parallel forward pass.
  - Expected tokens accepted per round: E[acc] = (1 - α^(k+1)) / (1 - α)
    where α = per-token acceptance rate (fraction of draft tokens the target
    would have sampled itself).
  - Speedup ≈ E[acc] / (draft_cost + verify_cost) vs naive.

Multi-drafter (N independent draft models, tree verification):
  - N drafters each independently propose k tokens.
  - Target verifies the whole N-branch tree in one parallel pass.
  - At decode position i, at least one drafter matches iff any of the N
    independent proposals matches the target distribution.
  - Assuming proposals are independent and each has acceptance prob α:
      P(position i accepted) = 1 - (1 - α)^N
  - E[acc] = sum_{i=0}^{k-1} prod_{j=0}^{i} (1 - (1-α)^N)
             (walk the tree until we hit the first position where all N miss)
  - Verify cost: N*k tokens in one target fwd pass (vs k for single drafter).
    But at B=1 the target is BW-bound and batching N*k is essentially free up
    to ~32 tokens without hitting compute bottleneck.
  - Drafter cost: N drafters run *in parallel* on the same GPUs, each reading
    their own weights. If each draft model fits on ≤1 GPU the drafters run
    truly concurrently and the drafter wall-clock = one_drafter_cost.

Memory feasibility for Qwen3-235B-A22B on 8×H100 (640 GB HBM):
  - Target weights: ~470 GB bf16 / ~235 GB fp8
  - Headroom: ~170 GB bf16 / ~405 GB fp8
  - A Qwen3-4B draft model: ~8 GB bf16 → fits 21 copies in fp8 headroom
  - Practical N: 4-8 drafters with Qwen3-1.7B-ish models, all on-chip.

The speedup over baseline (no speculative decoding) is:
  speedup = E[acc] / (N * drafter_bw_fraction + verify_overhead)
where drafter_bw_fraction is drafter weight bytes / target active weight bytes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecConfig:
    alpha: float       # per-token acceptance rate for a single drafter
    k: int             # draft length (tokens proposed per round)
    n_drafters: int    # number of independent draft models
    # Cost of running one drafter forward pass relative to one target forward
    # pass (in bandwidth terms). Drafter/target active param ratio.
    drafter_cost_ratio: float = 0.05   # e.g. 1.7B drafter vs 22B active target


def expected_accepted_single(alpha: float, k: int) -> float:
    """E[tokens accepted per round], one drafter, sequential draft."""
    if alpha >= 1.0:
        return float(k)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha) - 1
    # -1 because we only count the draft tokens, not the +1 bonus target token
    # (we're measuring drafting efficiency, not total output)


def expected_accepted_multi(alpha: float, k: int, n: int) -> float:
    """E[tokens accepted per round] with N independent drafters, tree verify.

    At each draft position i the probability at least one of N drafters matches
    is p_i = 1-(1-alpha)^N. We accept a run of positions until the first miss.
    The expectation is sum_{i=0}^{k-1} prod_{j=0}^{i-1} p_j (survival to pos i).
    """
    if n <= 0:
        return 0.0
    p_hit = 1.0 - (1.0 - alpha) ** n  # P(accept at any one position)
    # Geometric-like: E[accepted] = sum_{i=0}^{k-1} p_hit^i
    if p_hit >= 1.0:
        return float(k)
    return (1.0 - p_hit ** k) / (1.0 - p_hit)


def rounds_per_target_token(cfg: SpecConfig) -> float:
    """Target forward passes needed per accepted output token."""
    e = expected_accepted_multi(cfg.alpha, cfg.k, cfg.n_drafters)
    return 1.0 / max(e, 1e-9)


def wall_clock_speedup(cfg: SpecConfig, target_bw_s: float = 1.0) -> float:
    """Speedup over naive (no spec decode), ignoring overhead.

    Naive baseline: 1 target forward pass per token.
    Spec decode round cost: drafter(s) + one target verify pass.
      - N drafters run in parallel, so drafter wall-clock = 1x drafter_cost.
      - Verify pass reads k*N tokens against target, but BW-bound at B=1 so
        verify ≈ 1x target pass (reading active weights once regardless of k*N).
    """
    e_acc = expected_accepted_multi(cfg.alpha, cfg.k, cfg.n_drafters)
    # Round cost = drafter pass + verify pass (both BW-bound, drafters parallel)
    round_cost = cfg.drafter_cost_ratio + 1.0  # in units of target_bw_s
    return e_acc / round_cost  # tokens accepted per unit time, normalised


@dataclass
class SpecSweepRow:
    n_drafters: int
    alpha: float
    k: int
    e_acc: float
    speedup: float
    verify_tokens: int   # tokens in the tree verification batch


def sweep(
    alphas: list[float] | None = None,
    ks: list[int] | None = None,
    n_drafters_list: list[int] | None = None,
    drafter_cost_ratio: float = 0.05,
) -> list[SpecSweepRow]:
    alphas = alphas or [0.5, 0.6, 0.7, 0.8]
    ks = ks or [4, 8, 16]
    n_drafters_list = n_drafters_list or [1, 2, 4, 8]
    rows = []
    for alpha in alphas:
        for k in ks:
            for n in n_drafters_list:
                cfg = SpecConfig(alpha=alpha, k=k, n_drafters=n,
                                 drafter_cost_ratio=drafter_cost_ratio)
                rows.append(SpecSweepRow(
                    n_drafters=n, alpha=alpha, k=k,
                    e_acc=round(expected_accepted_multi(alpha, k, n), 2),
                    speedup=round(wall_clock_speedup(cfg), 2),
                    verify_tokens=n * k,
                ))
    return rows


def memory_feasibility(
    target_total_gb: float = 470.0,   # bf16
    target_active_gb: float = 43.2,   # 21.6B * 2
    hbm_total_gb: float = 640.0,      # 8x H100 80GB
    draft_model_gb: float = 3.4,      # Qwen3-1.7B bf16
    use_fp8_target: bool = True,
) -> dict:
    target_gb = target_total_gb / 2 if use_fp8_target else target_total_gb
    headroom = hbm_total_gb - target_gb
    max_drafters = int(headroom // draft_model_gb)
    return {
        "target_weight_gb": round(target_gb, 1),
        "hbm_headroom_gb": round(headroom, 1),
        "draft_model_gb": draft_model_gb,
        "max_drafters": max_drafters,
        "use_fp8_target": use_fp8_target,
    }
