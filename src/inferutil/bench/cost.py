"""Energy and cost-per-token metrics — the serving economics view.

Two angles on cost, both per million tokens (the unit serving is priced in):
  - energy:  from measured power -> joules/token, tok/s/W, kWh/Mtok, $/Mtok(energy)
  - rental:  from GPU-hour price -> $/Mtok(rental), independent of telemetry

When a lever's speedup is marginal, cost/token often breaks the tie: int4 that
gives 1.4x throughput also cuts $/Mtok by ~30%. These let `recommend` and the
sweep rank on economics, not just raw tok/s.
"""

from __future__ import annotations

from typing import Optional

_J_PER_KWH = 3.6e6
_S_PER_HR = 3600.0


def energy_metrics(decode_tok_s: Optional[float], power_w: Optional[float], *,
                   usd_per_kwh: float = 0.12) -> dict:
    """Energy economics from measured total power. None fields when unmeasured."""
    if not decode_tok_s or not power_w:
        return {"joules_per_token": None, "tok_s_per_watt": None,
                "kwh_per_mtok": None, "usd_per_mtok_energy": None}
    j_per_tok = power_w / decode_tok_s
    kwh_per_mtok = j_per_tok * 1e6 / _J_PER_KWH
    return {
        "joules_per_token": j_per_tok,
        "tok_s_per_watt": decode_tok_s / power_w,
        "kwh_per_mtok": kwh_per_mtok,
        "usd_per_mtok_energy": kwh_per_mtok * usd_per_kwh,
    }


def rental_usd_per_mtok(decode_tok_s: Optional[float], n_gpus: int, *,
                        usd_per_gpu_hr: float = 3.0) -> Optional[float]:
    """$/Mtok from GPU rental: (n_gpus * $/gpu-hr / 3600) / tok_s * 1e6.

    Telemetry-free, so it works on the analytical model too. Default rate is a
    typical on-demand 8xH100 figure (~$3/GPU-hr); override for your contract."""
    if not decode_tok_s:
        return None
    usd_per_s = n_gpus * usd_per_gpu_hr / _S_PER_HR
    return usd_per_s / decode_tok_s * 1e6
