from server.schemas import ChatRequest, ChatMessage
from server.topology import build_topology
from server.mock_engine import mock_stream


def _req(n=5):
    return ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=n)


def test_emits_content_chunks_then_summary():
    topo = build_topology(num_layers=4, experts_per_layer=16)
    out = list(mock_stream(_req(5), topo))
    content_chunks = [c for c in out if "choices" in c]
    assert len(content_chunks) == 5
    for i, c in enumerate(content_chunks):
        assert isinstance(c["choices"][0]["delta"]["content"], str)
        tel = c["x_telemetry"]
        assert tel["token_index"] == i
        assert tel["t_ms"] > 0
        assert len(tel["experts"]) == 8  # Qwen3 top-8 routing
        for ex in tel["experts"]:
            assert 0 <= ex["gpu"] < 8
        assert tel["spec"]["accepted"] <= tel["spec"]["proposed"]
    assert "x_summary" in out[-1]
    s = out[-1]["x_summary"]
    assert s["completion_tokens"] == 5
    assert s["ttft_ms"] > 0
    assert 0.0 <= s["spec_accept_rate"] <= 1.0


def test_deterministic():
    topo = build_topology(num_layers=4, experts_per_layer=16)
    a = list(mock_stream(_req(3), topo))
    b = list(mock_stream(_req(3), topo))
    assert a == b
