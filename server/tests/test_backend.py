import pytest
from server.schemas import ChatRequest, ChatMessage
from server.topology import build_topology
from server.backend import get_backend, MockBackend, RealEngineBackend


def test_default_is_mock(monkeypatch):
    monkeypatch.delenv("BACKEND", raising=False)
    assert isinstance(get_backend(), MockBackend)


def test_real_selected_by_env(monkeypatch):
    monkeypatch.setenv("BACKEND", "real")
    assert isinstance(get_backend(), RealEngineBackend)


def test_mock_backend_streams():
    topo = build_topology(num_gpus=8, num_layers=4, experts_per_layer=16)
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=2)
    out = list(MockBackend().stream(req, topo))
    assert any("choices" in c for c in out)


def test_real_backend_not_implemented():
    topo = build_topology()
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(NotImplementedError):
        list(RealEngineBackend().stream(req, topo))
