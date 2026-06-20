import json
import os
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Iterator

from server.schemas import ChatRequest
from server.mock_engine import mock_stream

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8001")


class Backend(ABC):
    @abstractmethod
    def stream(self, req: ChatRequest, topo: dict) -> Iterator[dict]:
        ...


class MockBackend(Backend):
    def stream(self, req: ChatRequest, topo: dict) -> Iterator[dict]:
        yield from mock_stream(req, topo)


class VLLMBackend(Backend):
    """Proxies to a vLLM OpenAI-compatible server, injects x_telemetry and x_summary."""

    def stream(self, req: ChatRequest, topo: dict) -> Iterator[dict]:
        payload = json.dumps({
            "model": "/alloc/data/Qwen3-235B-A22B",
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": True,
        }).encode()

        request = urllib.request.Request(
            f"{VLLM_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.time()
        n_tokens = 0
        buf = b""

        with urllib.request.urlopen(request, timeout=120) as resp:
            for raw in resp:
                buf += raw
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    line = frame.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        t_tok = (time.time() - t_start) * 1000
                        chunk["x_telemetry"] = _make_telemetry(n_tokens, t_tok, topo)
                        n_tokens += 1

                    yield chunk

        elapsed = time.time() - t_start
        prefill_tokens = sum(len(m.content.split()) for m in req.messages)
        yield {
            "x_summary": {
                "ttft_ms": round(elapsed * 1000 / max(n_tokens, 1), 1),
                "decode_tok_per_s": round(n_tokens / max(elapsed, 1e-3), 1),
                "prefill_tokens": prefill_tokens,
                "completion_tokens": n_tokens,
                "spec_accept_rate": 0.0,
            }
        }


def _make_telemetry(token_index: int, t_ms: float, topo: dict) -> dict:
    """Derive synthetic per-token expert routing from the placement map.

    Uses the placement to show which GPU each expert routes to. Expert selection
    cycles deterministically through the placement so the GPU cards light up
    in a pattern reflecting the actual optimized assignment.
    """
    placement = topo.get("placement", {})
    num_layers = topo.get("num_layers", 94)
    experts_per_layer = topo.get("experts_per_layer", 128)
    top_k = 8

    layer = token_index % num_layers
    layer_placement = placement.get(str(layer), {})
    experts = []
    for k in range(top_k):
        expert_id = (token_index * 7 + k * 13) % experts_per_layer
        gpu = layer_placement.get(str(expert_id), expert_id % 8)
        experts.append({"layer": layer, "expert_id": expert_id, "gpu": gpu})

    return {
        "token_index": token_index,
        "t_ms": round(t_ms, 2),
        "experts": experts,
        "spec": {"proposed": 8, "accepted": 0},
    }


def _vllm_healthy() -> bool:
    try:
        urllib.request.urlopen(f"{VLLM_URL}/health", timeout=2)
        return True
    except Exception:
        return False


def get_backend() -> Backend:
    if os.environ.get("BACKEND") == "mock":
        return MockBackend()
    if _vllm_healthy():
        print(f"vLLM detected at {VLLM_URL} — using real backend")
        return VLLMBackend()
    print("vLLM not available — falling back to mock")
    return MockBackend()
