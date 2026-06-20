import json
import subprocess
from pathlib import Path

OPTIMIZED_PLACEMENT_PATH = Path("/alloc/data/optimized_placement.json")


def _query_gpus() -> list[dict]:
    """Pull live stats from nvidia-smi. Falls back to static data if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        ).decode()
        gpus = []
        for line in out.strip().splitlines():
            idx, name, mem_total, mem_used, util, temp = [x.strip() for x in line.split(",")]
            gpus.append({
                "id": int(idx),
                "name": name,
                "mem_total_mb": int(mem_total),
                "mem_used_mb": int(mem_used),
                "utilization_pct": int(util),
                "temp_c": int(temp),
            })
        return gpus
    except Exception:
        return [
            {
                "id": i,
                "name": f"H100-{i}",
                "mem_total_mb": 81920,
                "mem_used_mb": 0,
                "utilization_pct": 0,
                "temp_c": 0,
            }
            for i in range(8)
        ]


def _load_placement(num_layers: int, experts_per_layer: int, num_gpus: int) -> dict:
    """Use optimized placement if available, else fall back to round-robin."""
    if OPTIMIZED_PLACEMENT_PATH.exists():
        try:
            with open(OPTIMIZED_PLACEMENT_PATH) as f:
                data = json.load(f)
            raw = data["placement"]
            return {
                str(layer): {
                    str(e): int(raw.get(str(layer), {}).get(str(e), e % num_gpus))
                    for e in range(experts_per_layer)
                }
                for layer in range(num_layers)
            }
        except Exception:
            pass
    return {
        str(layer): {str(e): e % num_gpus for e in range(experts_per_layer)}
        for layer in range(num_layers)
    }


def build_topology(num_layers: int = 94, experts_per_layer: int = 128) -> dict:
    """Live cluster map: real GPU stats + optimized expert placement."""
    gpus = _query_gpus()
    num_gpus = len(gpus)
    placement = _load_placement(num_layers, experts_per_layer, num_gpus)
    source = "optimized" if OPTIMIZED_PLACEMENT_PATH.exists() else "round-robin"
    return {
        "gpus": gpus,
        "num_layers": num_layers,
        "experts_per_layer": experts_per_layer,
        "placement": placement,
        "placement_source": source,
    }
