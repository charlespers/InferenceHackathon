import pytest
from server.schemas import ChatRequest, ChatMessage
from server.topology import build_topology
from server import backend as backend_mod
from server.backend import get_backend, MockBackend, VLLMBackend, Backend


def test_default_is_mock_when_vllm_absent(monkeypatch):
    # No BACKEND override + no healthy vLLM -> mock (the GPU-free demo path).
    monkeypatch.delenv("BACKEND", raising=False)
    monkeypatch.setattr(backend_mod, "_vllm_healthy", lambda: False)
    assert isinstance(get_backend(), MockBackend)


def test_mock_forced_by_env(monkeypatch):
    # BACKEND=mock forces mock even if a vLLM server is up.
    monkeypatch.setenv("BACKEND", "mock")
    monkeypatch.setattr(backend_mod, "_vllm_healthy", lambda: True)
    assert isinstance(get_backend(), MockBackend)


def test_vllm_selected_when_healthy(monkeypatch):
    # Auto-detect: a healthy vLLM (and no mock override) -> the real proxy backend.
    monkeypatch.delenv("BACKEND", raising=False)
    monkeypatch.setattr(backend_mod, "_vllm_healthy", lambda: True)
    assert isinstance(get_backend(), VLLMBackend)


def test_mock_backend_streams():
    topo = build_topology(num_layers=4, experts_per_layer=16)
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=2)
    out = list(MockBackend().stream(req, topo))
    assert any("choices" in c for c in out)


def test_vllm_backend_is_backend():
    # VLLMBackend proxies to a real vLLM server (no longer NotImplementedError); it's a Backend.
    assert issubclass(VLLMBackend, Backend)
    assert callable(VLLMBackend().stream)
