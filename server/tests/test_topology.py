from server.topology import build_topology


def test_topology_shape():
    t = build_topology(num_gpus=8, num_layers=4, experts_per_layer=16)
    assert len(t["gpus"]) == 8
    assert t["gpus"][0]["name"] == "H100-0"
    assert t["num_layers"] == 4
    assert t["experts_per_layer"] == 16
    # expert 9 -> gpu 1 (9 % 8)
    assert t["placement"]["0"]["9"] == 1
    # every expert maps to a valid gpu
    for layer in t["placement"].values():
        for gpu in layer.values():
            assert 0 <= gpu < 8


def test_topology_defaults_qwen():
    t = build_topology()
    assert len(t["gpus"]) == 8
    assert t["num_layers"] == 94
    assert t["experts_per_layer"] == 128
