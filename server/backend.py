import os
from abc import ABC, abstractmethod
from typing import Iterator
from server.schemas import ChatRequest
from server.mock_engine import mock_stream


class Backend(ABC):
    @abstractmethod
    def stream(self, req: ChatRequest, topo: dict) -> Iterator[dict]:
        ...


class MockBackend(Backend):
    def stream(self, req: ChatRequest, topo: dict) -> Iterator[dict]:
        yield from mock_stream(req, topo)


class RealEngineBackend(Backend):
    """Adapter to a real OpenAI-compatible engine (SGLang / vLLM / custom) running
    Qwen3-235B-A22B on the 8×H100 box.

    Implementation outline (wired when the engine exists):
      1. POST req to the engine's /v1/chat/completions with stream=True.
      2. Re-yield each standard chunk unchanged.
      3. If the engine emits per-token expert routing (via a side channel or an
         engine-specific field), map it into chunk["x_telemetry"] using `topo`
         for expert->gpu placement; append x_summary at the end.
    """

    def stream(self, req: ChatRequest, topo: dict) -> Iterator[dict]:
        raise NotImplementedError(
            "wire to OpenAI-compatible engine; map routing hooks -> x_telemetry"
        )


def get_backend() -> Backend:
    return RealEngineBackend() if os.environ.get("BACKEND") == "real" else MockBackend()
