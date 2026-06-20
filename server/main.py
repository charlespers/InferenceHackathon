import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from server.schemas import ChatRequest, sse, sse_done
from server.topology import build_topology
from server.backend import get_backend
from server.mock_engine import engine_profile

app = FastAPI(title="multi-gpu-inference-console")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

TOPO = build_topology()
BACKEND = get_backend()
# Pace the SSE stream at the engine's real per-token latency so the UI measures the
# speedup on the wall clock instead of trusting a reported number. STREAM_SCALE warps
# real time (1.0 = real time; lower = faster demo); STREAM_DELAY adds a flat per-token
# delay on top (kept for backward compatibility).
STREAM_SCALE = float(os.environ.get("STREAM_SCALE", "1"))
STREAM_DELAY = float(os.environ.get("STREAM_DELAY", "0"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "qwen3-235b-a22b", "object": "model"}]}


@app.get("/v1/topology")
def topology():
    return TOPO


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    if not req.stream:
        text = "".join(
            c["choices"][0]["delta"]["content"]
            for c in BACKEND.stream(req, TOPO) if "choices" in c
        )
        return JSONResponse({
            "id": "chatcmpl-mock", "object": "chat.completion", "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
        })

    prof = engine_profile(req.engine)

    def gen():
        first = True
        for chunk in BACKEND.stream(req, TOPO):
            if "choices" in chunk:
                # TTFT before the first token (prefill), then inter-token latency.
                if first:
                    time.sleep(prof["ttft_ms"] / 1000.0 * STREAM_SCALE + STREAM_DELAY)
                    first = False
                else:
                    t_ms = chunk.get("x_telemetry", {}).get("t_ms", prof["per_tok_ms"])
                    time.sleep(t_ms / 1000.0 * STREAM_SCALE + STREAM_DELAY)
            yield sse(chunk)
        yield sse_done()

    return StreamingResponse(gen(), media_type="text/event-stream")
