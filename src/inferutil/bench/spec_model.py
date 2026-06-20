"""Floor-aware speculative-decode sizing — the corrected model.

Spec decode wins by amortizing the dominant per-step **floor** (host/launch/comms
that a decode step pays regardless of work) over the tokens emitted per verify
pass. The regime decides the optimal tree:

  - **floor-bound** (F high): one verify pass pays the floor once and re-reads the
    weights ~once, so it costs ≈1 plain step no matter the tree size → speedup ≈
    tokens emitted per round → **big trees win**.
  - **weight-bound** (F→0): the verify re-reads the experts the tree lights up, so
    a wide/deep tree's expert-union tax dominates → **small trees win**.

Three things the team's `tools/spec_*` models got wrong (per the measured-data
cross-check) and that this model fixes:
  1. carry the floor fraction **F explicitly** (they hard-coded F=0, weight-bound);
  2. include the **guaranteed bonus token** the target always emits
     (`engine/src/spec/accept.rs`);
  3. price the verify by the **expected distinct experts** the tree activates
     (so the tax has a real, saturating cost — no unbounded-depth artifact).

F is a parameter: feed the measured/diagnosed floor fraction (overhead+comms share
of TPOT). Predictions are analytical; gate go/no-go on realized tok/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..model import MoEConfig, QWEN3_235B
from .prefill import expected_distinct_experts

_ROUTED_BYTE_SHARE = 0.66   # share of per-token weight bytes that are routed experts


def per_position_hit(accept: float, n_drafters: int) -> float:
    """P(at least one of N independent drafters matches) at a draft position."""
    a = max(0.0, min(0.999, accept))
    return 1.0 - (1.0 - a) ** max(1, n_drafters)


def expected_emitted(accept: float, draft_len: int, n_drafters: int = 1) -> float:
    """E[tokens emitted per verify], INCLUDING the guaranteed bonus token.

    Accept a run of draft positions until the first miss (each hits w.p. p), then
    the target always emits one bonus token: sum_{i=0}^{k} p^i = (1-p^(k+1))/(1-p)."""
    p = per_position_hit(accept, n_drafters)
    if p >= 1.0:
        return float(draft_len + 1)
    return (1.0 - p ** (draft_len + 1)) / (1.0 - p)


def verify_weight_units(draft_len: int, n_drafters: int, cfg: MoEConfig = QWEN3_235B,
                        routed_share: float = _ROUTED_BYTE_SHARE) -> float:
    """Verify weight cost in units of a plain decode step. A plain step (1 position)
    = 1.0; a tree of `draft_len*n_drafters` positions lights up more distinct
    experts and so reads proportionally more routed weight (saturating at all
    experts)."""
    positions = max(1, draft_len * n_drafters)
    union = expected_distinct_experts(cfg.n_experts, cfg.top_k, positions)
    return (1.0 - routed_share) + routed_share * (union / cfg.top_k)


def verify_cost(draft_len: int, n_drafters: int, floor: float,
                cfg: MoEConfig = QWEN3_235B,
                routed_share: float = _ROUTED_BYTE_SHARE) -> float:
    """Cost of one verify pass in plain-step units: the floor is paid ONCE, the
    (1-floor) non-floor part scales with the tree's expert-union weight tax."""
    f = max(0.0, min(1.0, floor))
    return f + (1.0 - f) * verify_weight_units(draft_len, n_drafters, cfg, routed_share)


def spec_speedup(accept: float, draft_len: int, n_drafters: int, floor: float,
                 cfg: MoEConfig = QWEN3_235B) -> float:
    """Wall-clock speedup over plain decode = emitted-per-round / verify-cost."""
    vc = verify_cost(draft_len, n_drafters, floor, cfg)
    return expected_emitted(accept, draft_len, n_drafters) / vc if vc else 0.0


@dataclass(frozen=True)
class SpecRow:
    draft_len: int
    n_drafters: int
    emitted: float
    verify_cost: float
    speedup: float


def spec_sweep(accept: float, floor: float, ks=(2, 4, 8), ns=(1, 2, 4),
               cfg: MoEConfig = QWEN3_235B) -> List[SpecRow]:
    """Rank (draft_len x n_drafters) by floor-aware speedup, descending."""
    rows = [SpecRow(k, n, expected_emitted(accept, k, n),
                    verify_cost(k, n, floor, cfg), spec_speedup(accept, k, n, floor, cfg))
            for k in ks for n in ns]
    rows.sort(key=lambda r: -r.speedup)
    return rows
