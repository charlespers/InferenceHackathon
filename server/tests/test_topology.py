from server.topology import build_topology


def test_topology_shape():
    t = build_topology(num_layers=4, experts_per_layer=16)
    assert len(t["gpus"]) == 8
    assert t["num_layers"] == 4
    assert t["experts_per_layer"] == 16
    # expert 9 -> gpu 1 (9 % 8)
    assert t["placement"]["0"]["9"] == 1
    for layer in t["placement"].values():
        for gpu in layer.values():
            assert 0 <= gpu < 8


def test_topology_gpu_fields():
    t = build_topology(num_layers=2, experts_per_layer=8)
    gpu = t["gpus"][0]
    assert "mem_total_mb" in gpu
    assert "mem_used_mb" in gpu
    assert "utilization_pct" in gpu
    assert "temp_c" in gpu


def test_topology_defaults_qwen():
    t = build_topology()
    assert len(t["gpus"]) == 8
    assert t["num_layers"] == 94
    assert t["experts_per_layer"] == 128
