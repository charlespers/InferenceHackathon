import io
import json
from server.schemas import ChatRequest, ChatMessage
from server.topology import build_topology
from server import backend as backend_mod
from server.backend import get_backend, MockBackend, VLLMBackend


def test_default_is_mock_when_vllm_unhealthy(monkeypatch):
    monkeypatch.delenv("BACKEND", raising=False)
    monkeypatch.setattr(backend_mod, "_vllm_healthy", lambda: False)
    assert isinstance(get_backend(), MockBackend)


def test_uses_vllm_when_healthy(monkeypatch):
    monkeypatch.delenv("BACKEND", raising=False)
    monkeypatch.setattr(backend_mod, "_vllm_healthy", lambda: True)
    assert isinstance(get_backend(), VLLMBackend)


def test_forced_mock_via_env_even_if_vllm_healthy(monkeypatch):
    monkeypatch.setenv("BACKEND", "mock")
    monkeypatch.setattr(backend_mod, "_vllm_healthy", lambda: True)
    assert isinstance(get_backend(), MockBackend)


def test_mock_backend_streams():
    topo = build_topology(num_layers=4, experts_per_layer=16)
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=2)
    out = list(MockBackend().stream(req, topo))
    assert any("choices" in c for c in out)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sse_bytes(*chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def test_vllm_backend_streams_and_builds_telemetry(monkeypatch):
    topo = build_topology(num_layers=4, experts_per_layer=16)
    chunks = [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {"content": " there"}}]},
    ]
    monkeypatch.setattr(
        backend_mod.urllib.request, "urlopen",
        lambda request, timeout=120: _FakeResponse(_sse_bytes(*chunks)),
    )
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=2)
    out = list(VLLMBackend().stream(req, topo))

    content = [c for c in out if "choices" in c]
    assert len(content) == 2
    for c in content:
        tel = c["x_telemetry"]
        assert len(tel["experts"]) == 8
        for ex in tel["experts"]:
            assert 0 <= ex["gpu"] < 8

    summary = out[-1]["x_summary"]
    assert summary["completion_tokens"] == 2
    assert summary["ttft_ms"] >= 0.0
