import json
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_models():
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == "qwen3-235b-a22b"


def test_topology():
    t = client.get("/v1/topology").json()
    assert len(t["gpus"]) == 8


def _parse_sse(text):
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                events.append("DONE")
            else:
                events.append(json.loads(payload))
    return events


def test_chat_stream():
    body = {"model": "qwen3-235b-a22b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3, "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        events = _parse_sse("".join(r.iter_text()))
    assert events[-1] == "DONE"
    assert "x_summary" in events[-2]
    content = [e for e in events if isinstance(e, dict) and "choices" in e]
    assert len(content) == 3
    assert content[0]["x_telemetry"]["experts"]


def test_chat_non_stream():
    body = {"model": "qwen3-235b-a22b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3, "stream": False}
    data = client.post("/v1/chat/completions", json=body).json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"]
