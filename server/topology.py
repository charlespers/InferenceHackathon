def build_topology(num_gpus: int = 8, num_layers: int = 94, experts_per_layer: int = 128) -> dict:
    """Static cluster map. Defaults model Qwen3-235B-A22B (94 layers, 128 experts/layer)
    spread across 8 H100s via round-robin expert parallelism (~16 experts/GPU)."""
    gpus = [{"id": i, "name": f"H100-{i}", "mem_total_mb": 81920} for i in range(num_gpus)]
    placement = {
        str(layer): {str(e): e % num_gpus for e in range(experts_per_layer)}
        for layer in range(num_layers)
    }
    return {
        "gpus": gpus,
        "num_layers": num_layers,
        "experts_per_layer": experts_per_layer,
        "placement": placement,
    }
