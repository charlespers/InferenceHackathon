import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from server.schemas import ChatRequest, sse, sse_done
from server.topology import build_topology
from server.backend import get_backend

app = FastAPI(title="multi-gpu-inference-console")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

BACKEND = get_backend()
STREAM_DELAY = float(os.environ.get("STREAM_DELAY", "0"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "qwen3-235b-a22b", "object": "model"}]}


@app.get("/v1/topology")
def topology():
    return build_topology()


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    topo = build_topology()
    if not req.stream:
        text = "".join(
            c["choices"][0]["delta"]["content"]
            for c in BACKEND.stream(req, topo) if "choices" in c
        )
        return JSONResponse({
            "id": "chatcmpl-mock", "object": "chat.completion", "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
        })

    def gen():
        for chunk in BACKEND.stream(req, topo):
            yield sse(chunk)
            if STREAM_DELAY:
                time.sleep(STREAM_DELAY)
        yield sse_done()

    return StreamingResponse(gen(), media_type="text/event-stream")
