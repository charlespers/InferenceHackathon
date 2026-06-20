# Multi-GPU Inference Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A minimal web console that streams chat from a multi-GPU MoE backend over an OpenAI-compatible API and visualizes B=1 latency + live expert/GPU routing.

**Architecture:** Three units in one repo — a Python FastAPI `server/` (mock engine + adapter stub, the contract authority), a Vite/React/TS/Tailwind `ui/` SPA (the deliverable), and the API contract between them. The UI talks only to the HTTP contract; the mock lets the whole UI + viz be built and demoed with no real engine. Telemetry rides the standard SSE stream as optional `x_`-namespaced fields.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, httpx (tests); Node 18+, Vite, React 18, TypeScript, Tailwind 3, Vitest, Playwright, react-markdown.

## Global Constraints

- No code, kernels, or model internals from the Conifer engine/Tauri app may be copied. Build fresh; only generic glue (SSE parsing, markdown render) is reimplemented.
- API is OpenAI-wire-compatible. All non-standard fields are namespaced with `x_` and are OPTIONAL — the UI MUST render chat correctly when they are absent.
- Backend base URL is configurable in the UI (`VITE_API_BASE` env + in-UI override field). Server enables permissive CORS (hackathon scope).
- Target topology is 8 GPUs (H100). The UI reads real counts from `GET /v1/topology` and falls back to an 8-GPU default if unavailable.
- B=1 only. No batching features in the UI.
- Commit after every task (frequent commits).

---

### Task 1: Repo scaffold + server contract types

**Files:**
- Create: `server/requirements.txt`
- Create: `server/schemas.py`
- Create: `server/tests/__init__.py`
- Create: `server/tests/test_schemas.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: Pydantic models `ChatMessage{role:str, content:str}`, `ChatRequest{model:str, messages:list[ChatMessage], temperature:float=0.7, max_tokens:int=256, stream:bool=True}`. Helper `sse(data: dict) -> str` returning `"data: " + json + "\n\n"` and `sse_done() -> str` returning `"data: [DONE]\n\n"`.

- [ ] **Step 1: Write `.gitignore`**

```
node_modules/
dist/
__pycache__/
*.pyc
.venv/
.env
.DS_Store
playwright-report/
test-results/
```

- [ ] **Step 2: Write `server/requirements.txt`**

```
fastapi==0.115.*
uvicorn[standard]==0.32.*
httpx==0.27.*
pytest==8.*
```

- [ ] **Step 3: Write the failing test `server/tests/test_schemas.py`**

```python
from server.schemas import ChatRequest, ChatMessage, sse, sse_done


def test_chat_request_defaults():
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")])
    assert req.temperature == 0.7
    assert req.max_tokens == 256
    assert req.stream is True


def test_sse_framing():
    assert sse({"a": 1}) == 'data: {"a": 1}\n\n'
    assert sse_done() == "data: [DONE]\n\n"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/tests/test_schemas.py -v`
Expected: FAIL (ModuleNotFoundError: server.schemas)

- [ ] **Step 5: Write `server/schemas.py`**

```python
import json
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    model: str = "moe-200b"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = True


def sse(data: dict) -> str:
    return "data: " + json.dumps(data) + "\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/tests/test_schemas.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add .gitignore server/requirements.txt server/schemas.py server/tests/
git commit -m "feat(server): contract schemas + SSE framing"
```

---

### Task 2: Topology endpoint

**Files:**
- Create: `server/topology.py`
- Create: `server/tests/test_topology.py`

**Interfaces:**
- Produces: `build_topology(num_gpus:int=8, num_layers:int=48, experts_per_layer:int=64) -> dict` returning `{"gpus":[{"id","name","mem_total_mb"}], "num_layers", "experts_per_layer", "placement": {str(layer): {str(expert_id): gpu_id}}}`. Placement assigns expert `e` in any layer to GPU `e % num_gpus` (deterministic round-robin).

- [ ] **Step 1: Write the failing test `server/tests/test_topology.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/tests/test_topology.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `server/topology.py`**

```python
def build_topology(num_gpus: int = 8, num_layers: int = 48, experts_per_layer: int = 64) -> dict:
    gpus = [{"id": i, "name": f"H100-{i}", "mem_total_mb": 81920} for i in range(num_gpus)]
    placement = {
        str(layer): {str(e): e % num_gpus for e in range(experts_per_layer)}
        for layer in range(num_layers)
    }
    return {
        "gpus": gpus,
        "num_layers": num_layers,
        "experts_per_layer": experts_per_layer,
        "placement": placement,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/tests/test_topology.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/topology.py server/tests/test_topology.py
git commit -m "feat(server): /v1/topology cluster map builder"
```

---

### Task 3: Mock engine (token + telemetry generator)

**Files:**
- Create: `server/mock_engine.py`
- Create: `server/tests/test_mock_engine.py`

**Interfaces:**
- Consumes: `build_topology` (Task 2).
- Produces: generator `mock_stream(req: ChatRequest, topo: dict) -> Iterator[dict]`. Yields dicts that are either an OpenAI `chat.completion.chunk` (with `choices[0].delta.content` and `x_telemetry`), or a final `{"x_summary": {...}}` dict, then nothing. Telemetry per token: `top_k` (default 2) experts drawn deterministically from `(token_index, layer)`, mapped to GPUs via `topo["placement"]`. No real sleeping in the generator (timing values are synthetic in `t_ms`); the route layer adds optional real delay.

- [ ] **Step 1: Write the failing test `server/tests/test_mock_engine.py`**

```python
from server.schemas import ChatRequest, ChatMessage
from server.topology import build_topology
from server.mock_engine import mock_stream


def _req(n=5):
    return ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=n)


def test_emits_content_chunks_then_summary():
    topo = build_topology(num_gpus=8, num_layers=4, experts_per_layer=16)
    out = list(mock_stream(_req(5), topo))
    content_chunks = [c for c in out if "choices" in c]
    assert len(content_chunks) == 5
    # each content chunk carries delta.content and telemetry
    for i, c in enumerate(content_chunks):
        assert isinstance(c["choices"][0]["delta"]["content"], str)
        tel = c["x_telemetry"]
        assert tel["token_index"] == i
        assert tel["t_ms"] > 0
        assert len(tel["experts"]) == 2
        for ex in tel["experts"]:
            assert 0 <= ex["gpu"] < 8
        assert tel["spec"]["accepted"] <= tel["spec"]["proposed"]
    # last item is the summary
    assert "x_summary" in out[-1]
    s = out[-1]["x_summary"]
    assert s["completion_tokens"] == 5
    assert s["ttft_ms"] > 0
    assert 0.0 <= s["spec_accept_rate"] <= 1.0


def test_deterministic():
    topo = build_topology(num_gpus=8, num_layers=4, experts_per_layer=16)
    a = list(mock_stream(_req(3), topo))
    b = list(mock_stream(_req(3), topo))
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/tests/test_mock_engine.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `server/mock_engine.py`**

```python
from typing import Iterator
from server.schemas import ChatRequest

_WORDS = ("Routing across eight H100s with expert parallelism keeps "
          "per-token latency low at batch size one. ").split()


def _experts_for(token_index: int, topo: dict, top_k: int = 2) -> list[dict]:
    num_layers = topo["num_layers"]
    experts_per_layer = topo["experts_per_layer"]
    placement = topo["placement"]
    layer = token_index % num_layers
    chosen = []
    for k in range(top_k):
        expert_id = (token_index * 7 + k * 13) % experts_per_layer
        gpu = placement[str(layer)][str(expert_id)]
        chosen.append({"layer": layer, "expert_id": expert_id, "gpu": gpu})
    return chosen


def mock_stream(req: ChatRequest, topo: dict) -> Iterator[dict]:
    n = max(1, req.max_tokens)
    ttft_ms = 40.0
    per_tok_ms = 8.0
    accepted_total = 0
    proposed_total = 0
    for i in range(n):
        word = _WORDS[i % len(_WORDS)]
        piece = (word if i == 0 else " " + word)
        proposed = 4
        accepted = 3 if (i % 4 != 0) else 2
        accepted_total += accepted
        proposed_total += proposed
        yield {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": req.model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            "x_telemetry": {
                "token_index": i,
                "t_ms": per_tok_ms,
                "experts": _experts_for(i, topo),
                "spec": {"proposed": proposed, "accepted": accepted},
            },
        }
    yield {
        "x_summary": {
            "ttft_ms": ttft_ms,
            "decode_tok_per_s": round(1000.0 / per_tok_ms, 1),
            "prefill_tokens": sum(len(m.content.split()) for m in req.messages),
            "completion_tokens": n,
            "spec_accept_rate": round(accepted_total / proposed_total, 3),
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/tests/test_mock_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add server/mock_engine.py server/tests/test_mock_engine.py
git commit -m "feat(server): deterministic mock token+telemetry generator"
```

---

### Task 4: FastAPI app wiring all endpoints

**Files:**
- Create: `server/main.py`
- Create: `server/tests/test_api.py`
- Create: `server/README.md`

**Interfaces:**
- Consumes: `ChatRequest`, `sse`, `sse_done` (Task 1); `build_topology` (Task 2); `mock_stream` (Task 3).
- Produces: FastAPI `app` with `GET /v1/models`, `GET /v1/topology`, `POST /v1/chat/completions`, `GET /health`. Streaming responses set `media_type="text/event-stream"`. CORS allow-all. An env `STREAM_DELAY` (seconds/token, default `0`) lets the server pace the stream for demos; tests use 0. Launch: `uvicorn server.main:app --host 0.0.0.0 --port 8000`.

- [ ] **Step 1: Write the failing test `server/tests/test_api.py`**

```python
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
    body = {"model": "moe-200b",
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/tests/test_api.py -v`
Expected: FAIL (ModuleNotFoundError: server.main)

- [ ] **Step 3: Write `server/main.py`**

```python
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from server.schemas import ChatRequest, sse, sse_done
from server.topology import build_topology
from server.mock_engine import mock_stream

app = FastAPI(title="multi-gpu-inference-console")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

TOPO = build_topology()
STREAM_DELAY = float(os.environ.get("STREAM_DELAY", "0"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "moe-200b", "object": "model"}]}


@app.get("/v1/topology")
def topology():
    return TOPO


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    if not req.stream:
        text = "".join(
            c["choices"][0]["delta"]["content"]
            for c in mock_stream(req, TOPO) if "choices" in c
        )
        return JSONResponse({
            "id": "chatcmpl-mock", "object": "chat.completion", "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
        })

    def gen():
        for chunk in mock_stream(req, TOPO):
            yield sse(chunk)
            if STREAM_DELAY:
                time.sleep(STREAM_DELAY)
        yield sse_done()

    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/ -v`
Expected: PASS (all server tests)

- [ ] **Step 5: Write `server/README.md`**

````markdown
# server/ — mock inference backend + adapter stub

Implements the API contract the UI consumes. In mock mode it synthesizes streamed
tokens with fabricated 8×H100 expert routing and speculative-decode stats, so the
UI and visualization run with no real engine.

## Run

```bash
cd /Users/charles/Desktop/InferenceHackathon
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
# demo pacing: STREAM_DELAY=0.03 uvicorn server.main:app --port 8000
```

## Endpoints
- `GET /health`
- `GET /v1/models`
- `GET /v1/topology`
- `POST /v1/chat/completions` (OpenAI-compatible; SSE when `stream:true`)

## On the H100 box
Run there, then from your laptop: `ssh -L 8000:localhost:8000 <box>` and point the
UI at `http://localhost:8000`. Swap mock for the real engine via the adapter (Task 11).
````

- [ ] **Step 6: Commit**

```bash
git add server/main.py server/tests/test_api.py server/README.md
git commit -m "feat(server): FastAPI app with models/topology/chat SSE + non-stream"
```

---

### Task 5: UI scaffold (Vite + React + TS + Tailwind)

**Files:**
- Create: `ui/package.json`
- Create: `ui/vite.config.ts`
- Create: `ui/tsconfig.json`
- Create: `ui/tsconfig.node.json`
- Create: `ui/index.html`
- Create: `ui/postcss.config.js`
- Create: `ui/tailwind.config.js`
- Create: `ui/src/index.css`
- Create: `ui/src/main.tsx`
- Create: `ui/src/App.tsx`

**Interfaces:**
- Produces: a buildable SPA. `npm run dev` serves on 5173; `npm run build` type-checks and bundles; `npm run test` runs Vitest. `App` renders a placeholder header so build/dev verify.

- [ ] **Step 1: Write `ui/package.json`**

```json
{
  "name": "inference-console",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "e2e": "playwright test"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-markdown": "^9.0.1"
  },
  "devDependencies": {
    "@playwright/test": "^1.48.0",
    "@testing-library/react": "^16.0.1",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.3",
    "autoprefixer": "^10.4.20",
    "jsdom": "^25.0.1",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.15",
    "typescript": "^5.6.3",
    "vite": "^5.4.11",
    "vitest": "^2.1.5"
  }
}
```

- [ ] **Step 2: Write config files**

`ui/vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", globals: true, exclude: ["e2e/**", "node_modules/**"] },
});
```

`ui/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "types": ["vitest/globals"]
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

`ui/tsconfig.node.json`:
```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
```

`ui/postcss.config.js`:
```javascript
export default { plugins: { tailwindcss: {}, autoprefixer: {} } };
```

`ui/tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

- [ ] **Step 3: Write entry files**

`ui/index.html`:
```html
<!doctype html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Inference Console</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`ui/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body { @apply bg-neutral-950 text-neutral-100 antialiased; }
```

`ui/src/main.tsx`:
```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

`ui/src/App.tsx`:
```tsx
export default function App() {
  return (
    <div className="min-h-screen p-6">
      <h1 className="text-xl font-semibold tracking-tight">Inference Console</h1>
      <p className="text-neutral-400 text-sm">scaffold ok</p>
    </div>
  );
}
```

- [ ] **Step 4: Install and verify build**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm install && npm run build`
Expected: install succeeds; `tsc -b && vite build` produces `dist/` with no type errors.

- [ ] **Step 5: Commit**

```bash
git add ui/package.json ui/package-lock.json ui/*.ts ui/*.js ui/*.json ui/index.html ui/src/
git commit -m "feat(ui): Vite + React + TS + Tailwind scaffold"
```

---

### Task 6: Contract types + SSE-parsing API client

**Files:**
- Create: `ui/src/types.ts`
- Create: `ui/src/lib/apiClient.ts`
- Create: `ui/src/lib/apiClient.test.ts`

**Interfaces:**
- Produces:
  - `types.ts`: `Expert{layer:number; expert_id:number; gpu:number}`, `Telemetry{token_index:number; t_ms:number; experts:Expert[]; spec:{proposed:number; accepted:number}}`, `Summary{ttft_ms:number; decode_tok_per_s:number; prefill_tokens:number; completion_tokens:number; spec_accept_rate:number}`, `Topology{gpus:{id:number;name:string;mem_total_mb:number}[]; num_layers:number; experts_per_layer:number; placement:Record<string,Record<string,number>>}`, `ChatMessage{role:"user"|"assistant"|"system"; content:string}`.
  - `apiClient.ts`: `parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<any>` yielding parsed JSON objects (skips `[DONE]` and blank lines, buffers partial lines); `streamChat(base:string, body:{model:string;messages:ChatMessage[];temperature:number;max_tokens:number}, signal:AbortSignal): AsyncGenerator<any>`; `getTopology(base:string): Promise<Topology>`; `getModels(base:string): Promise<string[]>`.

- [ ] **Step 1: Write `ui/src/types.ts`**

```typescript
export interface Expert { layer: number; expert_id: number; gpu: number; }
export interface Telemetry {
  token_index: number;
  t_ms: number;
  experts: Expert[];
  spec: { proposed: number; accepted: number };
}
export interface Summary {
  ttft_ms: number;
  decode_tok_per_s: number;
  prefill_tokens: number;
  completion_tokens: number;
  spec_accept_rate: number;
}
export interface Topology {
  gpus: { id: number; name: string; mem_total_mb: number }[];
  num_layers: number;
  experts_per_layer: number;
  placement: Record<string, Record<string, number>>;
}
export interface ChatMessage { role: "user" | "assistant" | "system"; content: string; }
```

- [ ] **Step 2: Write the failing test `ui/src/lib/apiClient.test.ts`**

```typescript
import { describe, it, expect } from "vitest";
import { parseSSE } from "./apiClient";

function streamFrom(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
}

describe("parseSSE", () => {
  it("parses framed events and skips DONE", async () => {
    const s = streamFrom([
      'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
      'data: {"x_summary":{"ttft_ms":1}}\n\n',
      "data: [DONE]\n\n",
    ]);
    const out: any[] = [];
    for await (const e of parseSSE(s)) out.push(e);
    expect(out).toHaveLength(2);
    expect(out[0].choices[0].delta.content).toBe("a");
    expect(out[1].x_summary.ttft_ms).toBe(1);
  });

  it("reassembles events split across chunk boundaries", async () => {
    const s = streamFrom(['data: {"choices":[{"de', 'lta":{"content":"hi"}}]}\n\n']);
    const out: any[] = [];
    for await (const e of parseSSE(s)) out.push(e);
    expect(out[0].choices[0].delta.content).toBe("hi");
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test -- apiClient`
Expected: FAIL (cannot find ./apiClient)

- [ ] **Step 4: Write `ui/src/lib/apiClient.ts`**

```typescript
import type { Topology, ChatMessage } from "../types";

export async function* parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<any> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      if (!frame.startsWith("data:")) continue;
      const payload = frame.slice(frame.indexOf(":") + 1).trim();
      if (payload === "[DONE]" || payload === "") continue;
      yield JSON.parse(payload);
    }
  }
}

export async function* streamChat(
  base: string,
  body: { model: string; messages: ChatMessage[]; temperature: number; max_tokens: number },
  signal: AbortSignal,
): AsyncGenerator<any> {
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });
  if (!res.ok || !res.body) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(`chat request failed (${res.status}): ${msg}`);
  }
  yield* parseSSE(res.body);
}

export async function getTopology(base: string): Promise<Topology> {
  const res = await fetch(`${base}/v1/topology`);
  if (!res.ok) throw new Error(`topology failed: ${res.status}`);
  return res.json();
}

export async function getModels(base: string): Promise<string[]> {
  const res = await fetch(`${base}/v1/models`);
  if (!res.ok) throw new Error(`models failed: ${res.status}`);
  const data = await res.json();
  return (data.data ?? []).map((m: any) => m.id);
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test -- apiClient`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ui/src/types.ts ui/src/lib/apiClient.ts ui/src/lib/apiClient.test.ts
git commit -m "feat(ui): contract types + SSE-parsing API client"
```

---

### Task 7: useChatStream hook (turn lifecycle + telemetry accumulation)

**Files:**
- Create: `ui/src/lib/useChatStream.ts`
- Create: `ui/src/lib/useChatStream.test.ts`

**Interfaces:**
- Consumes: `streamChat` (Task 6), `Telemetry`, `Summary`, `ChatMessage` (Task 6 types). Test mocks the `./apiClient` module.
- Produces: hook `useChatStream(base:string)` returning `{ messages: ChatMessage[]; telemetry: Telemetry[]; summary: Summary|null; status: "idle"|"streaming"|"error"; error: string|null; send(text:string, opts:{model:string;temperature:number;max_tokens:number}): void; cancel(): void }`. On `send`: pushes the user message, appends an empty assistant message, then streams; each `choices[0].delta.content` is appended to the last assistant message; each `x_telemetry` is pushed to `telemetry`; `x_summary` sets `summary`. `cancel()` aborts. Errors set `status="error"` and `error`, preserving partial content.

- [ ] **Step 1: Write the failing test `ui/src/lib/useChatStream.test.ts`**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";

const mockStreamChat = vi.fn();
vi.mock("./apiClient", () => ({ streamChat: (...a: any[]) => mockStreamChat(...a) }));

import { useChatStream } from "./useChatStream";

async function* fakeStream() {
  yield { choices: [{ delta: { content: "Hel" } }] };
  yield { choices: [{ delta: { content: "lo" } }], x_telemetry: { token_index: 0, t_ms: 8, experts: [{ layer: 0, expert_id: 1, gpu: 1 }], spec: { proposed: 4, accepted: 3 } } };
  yield { x_summary: { ttft_ms: 40, decode_tok_per_s: 125, prefill_tokens: 1, completion_tokens: 2, spec_accept_rate: 0.75 } };
}

describe("useChatStream", () => {
  beforeEach(() => mockStreamChat.mockReset());

  it("accumulates assistant content, telemetry, and summary", async () => {
    mockStreamChat.mockReturnValue(fakeStream());
    const { result } = renderHook(() => useChatStream("http://x"));
    act(() => result.current.send("hi", { model: "m", temperature: 0.7, max_tokens: 8 }));
    await waitFor(() => expect(result.current.status).toBe("idle"));
    const msgs = result.current.messages;
    expect(msgs[0]).toEqual({ role: "user", content: "hi" });
    expect(msgs[1]).toEqual({ role: "assistant", content: "Hello" });
    expect(result.current.telemetry).toHaveLength(1);
    expect(result.current.summary?.decode_tok_per_s).toBe(125);
  });

  it("sets error status on stream failure but keeps prior messages", async () => {
    mockStreamChat.mockImplementation(async function* () { throw new Error("boom"); });
    const { result } = renderHook(() => useChatStream("http://x"));
    act(() => result.current.send("hi", { model: "m", temperature: 0.7, max_tokens: 8 }));
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toContain("boom");
    expect(result.current.messages[0].content).toBe("hi");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test -- useChatStream`
Expected: FAIL (cannot find ./useChatStream)

- [ ] **Step 3: Write `ui/src/lib/useChatStream.ts`**

```typescript
import { useCallback, useRef, useState } from "react";
import { streamChat } from "./apiClient";
import type { ChatMessage, Telemetry, Summary } from "../types";

type Status = "idle" | "streaming" | "error";
interface SendOpts { model: string; temperature: number; max_tokens: number; }

export function useChatStream(base: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback((text: string, opts: SendOpts) => {
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setTelemetry([]);
    setSummary(null);
    setError(null);
    setStatus("streaming");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    (async () => {
      try {
        for await (const chunk of streamChat(base, { ...opts, messages: history }, ctrl.signal)) {
          const piece = chunk?.choices?.[0]?.delta?.content;
          if (piece) {
            setMessages((prev) => {
              const next = prev.slice();
              const last = next[next.length - 1];
              next[next.length - 1] = { ...last, content: last.content + piece };
              return next;
            });
          }
          if (chunk?.x_telemetry) setTelemetry((p) => [...p, chunk.x_telemetry]);
          if (chunk?.x_summary) setSummary(chunk.x_summary);
        }
        setStatus("idle");
      } catch (e: any) {
        if (e?.name === "AbortError") { setStatus("idle"); return; }
        setError(String(e?.message ?? e));
        setStatus("error");
      }
    })();
  }, [base, messages]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  return { messages, telemetry, summary, status, error, send, cancel };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test -- useChatStream`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/src/lib/useChatStream.ts ui/src/lib/useChatStream.test.ts
git commit -m "feat(ui): useChatStream turn lifecycle hook"
```

---

### Task 8: Chat pane (config bar, messages, prompt input)

**Files:**
- Create: `ui/src/config.ts`
- Create: `ui/src/components/ChatPane.tsx`
- Create: `ui/src/components/ChatPane.test.tsx`
- Modify: `ui/src/App.tsx` (replace placeholder; render two-column shell)

**Interfaces:**
- Consumes: `useChatStream` (Task 7). 
- Produces:
  - `config.ts`: `getDefaultBase(): string` returning `import.meta.env.VITE_API_BASE ?? "http://localhost:8000"`.
  - `ChatPane({ base, onTurn }: { base: string; onTurn: (t:{telemetry:Telemetry[];summary:Summary|null}) => void })` — renders model name input, temperature + max_tokens fields, scrollable message list (assistant content via `react-markdown`), prompt textarea, Send/Stop button. Calls `onTurn` whenever telemetry/summary change so `App` can feed the right-hand panels.
  - `App` renders a two-column grid: `ChatPane` left, a right column placeholder that will hold latency + viz (wired in Tasks 9–10). `App` owns `base` state + an in-UI base-URL field.

- [ ] **Step 1: Write the failing test `ui/src/components/ChatPane.test.tsx`**

```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

const send = vi.fn();
vi.mock("../lib/useChatStream", () => ({
  useChatStream: () => ({
    messages: [{ role: "user", content: "hi" }, { role: "assistant", content: "**yo**" }],
    telemetry: [], summary: null, status: "idle", error: null, send, cancel: vi.fn(),
  }),
}));

import { ChatPane } from "./ChatPane";

describe("ChatPane", () => {
  it("renders messages and sends on click", () => {
    render(<ChatPane base="http://x" onTurn={() => {}} />);
    expect(screen.getByText("hi")).toBeTruthy();
    expect(screen.getByText("yo")).toBeTruthy(); // markdown bold -> text node
    const input = screen.getByPlaceholderText(/message/i) as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "hello there" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));
    expect(send).toHaveBeenCalledWith("hello there", expect.objectContaining({ model: expect.any(String) }));
  });
});
```

- [ ] **Step 2: Add jsdom matchers shim + run test to verify it fails**

Append to `ui/vite.config.ts` test block: `setupFiles: []` is fine; `@testing-library/react` works with jsdom + globals. (No extra setup file needed because tests use truthiness, not jest-dom matchers.)

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test -- ChatPane`
Expected: FAIL (cannot find ./ChatPane)

- [ ] **Step 3: Write `ui/src/config.ts`**

```typescript
export function getDefaultBase(): string {
  return (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";
}
```

- [ ] **Step 4: Write `ui/src/components/ChatPane.tsx`**

```tsx
import { useEffect, useState } from "react";
import Markdown from "react-markdown";
import { useChatStream } from "../lib/useChatStream";
import type { Telemetry, Summary } from "../types";

interface Props {
  base: string;
  onTurn: (t: { telemetry: Telemetry[]; summary: Summary | null }) => void;
}

export function ChatPane({ base, onTurn }: Props) {
  const chat = useChatStream(base);
  const [text, setText] = useState("");
  const [model, setModel] = useState("moe-200b");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(256);

  useEffect(() => {
    onTurn({ telemetry: chat.telemetry, summary: chat.summary });
  }, [chat.telemetry, chat.summary, onTurn]);

  const streaming = chat.status === "streaming";
  const submit = () => {
    if (!text.trim() || streaming) return;
    chat.send(text.trim(), { model, temperature, max_tokens: maxTokens });
    setText("");
  };

  return (
    <div className="flex flex-col h-full border border-neutral-800 rounded-lg overflow-hidden">
      <div className="flex gap-3 items-center px-3 py-2 border-b border-neutral-800 text-xs text-neutral-400">
        <input className="bg-neutral-900 rounded px-2 py-1 w-32" value={model}
               onChange={(e) => setModel(e.target.value)} aria-label="model" />
        <label className="flex items-center gap-1">temp
          <input type="number" step="0.1" min="0" max="2" value={temperature}
                 onChange={(e) => setTemperature(+e.target.value)}
                 className="bg-neutral-900 rounded px-2 py-1 w-16" /></label>
        <label className="flex items-center gap-1">max
          <input type="number" min="1" value={maxTokens}
                 onChange={(e) => setMaxTokens(+e.target.value)}
                 className="bg-neutral-900 rounded px-2 py-1 w-20" /></label>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chat.messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-neutral-200" : "text-emerald-200"}>
            <div className="text-[10px] uppercase tracking-wide text-neutral-500 mb-1">{m.role}</div>
            <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap">
              {m.role === "assistant" ? <Markdown>{m.content}</Markdown> : m.content}
            </div>
          </div>
        ))}
        {chat.error && <div className="text-red-400 text-sm">error: {chat.error}</div>}
      </div>

      <div className="border-t border-neutral-800 p-3 flex gap-2">
        <textarea
          className="flex-1 bg-neutral-900 rounded px-3 py-2 text-sm resize-none h-12"
          placeholder="Message the model…" value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
        />
        {streaming
          ? <button className="px-4 rounded bg-red-600/80 text-sm" onClick={chat.cancel}>Stop</button>
          : <button className="px-4 rounded bg-emerald-600 text-sm" onClick={submit}>Send</button>}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Replace `ui/src/App.tsx`**

```tsx
import { useCallback, useState } from "react";
import { ChatPane } from "./components/ChatPane";
import { getDefaultBase } from "./config";
import type { Telemetry, Summary } from "./types";

export default function App() {
  const [base, setBase] = useState(getDefaultBase());
  const [turn, setTurn] = useState<{ telemetry: Telemetry[]; summary: Summary | null }>({
    telemetry: [], summary: null,
  });
  const onTurn = useCallback((t: { telemetry: Telemetry[]; summary: Summary | null }) => setTurn(t), []);

  return (
    <div className="h-screen flex flex-col p-4 gap-3">
      <header className="flex items-center justify-between">
        <h1 className="text-lg font-semibold tracking-tight">Inference Console
          <span className="text-neutral-500 text-sm font-normal"> · 8×H100 · B=1</span></h1>
        <input className="bg-neutral-900 border border-neutral-800 rounded px-2 py-1 text-xs w-72"
               value={base} onChange={(e) => setBase(e.target.value)} aria-label="backend url" />
      </header>
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_420px] gap-3 min-h-0">
        <ChatPane base={base} onTurn={onTurn} />
        <div className="flex flex-col gap-3 min-h-0" data-testid="right-rail">
          {/* LatencyPanel (Task 9) + GpuExpertViz (Task 10) mount here */}
          <pre className="text-[10px] text-neutral-600">{turn.summary ? "summary ready" : "no turn yet"}</pre>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test -- ChatPane && npm run build`
Expected: PASS and clean build.

- [ ] **Step 7: Commit**

```bash
git add ui/src/config.ts ui/src/components/ChatPane.tsx ui/src/components/ChatPane.test.tsx ui/src/App.tsx
git commit -m "feat(ui): chat pane + app shell"
```

---

### Task 9: Latency panel

**Files:**
- Create: `ui/src/lib/latency.ts`
- Create: `ui/src/lib/latency.test.ts`
- Create: `ui/src/components/LatencyPanel.tsx`
- Modify: `ui/src/App.tsx` (mount `LatencyPanel` in right rail)

**Interfaces:**
- Consumes: `Telemetry`, `Summary` (Task 6 types).
- Produces:
  - `latency.ts`: `liveStats(telemetry: Telemetry[]): { tokens:number; avgInterMs:number; tokPerSec:number; specAccept:number }` — `avgInterMs` is mean of `t_ms`; `tokPerSec` is `1000/avgInterMs` (0 if no tokens); `specAccept` is `sum(accepted)/sum(proposed)` (0 if none).
  - `LatencyPanel({ telemetry, summary }: { telemetry: Telemetry[]; summary: Summary|null })` — renders TTFT (from summary, "—" until present), live tok/s, total tokens, spec-accept %, and an inline SVG sparkline of `t_ms` over tokens.

- [ ] **Step 1: Write the failing test `ui/src/lib/latency.test.ts`**

```typescript
import { describe, it, expect } from "vitest";
import { liveStats } from "./latency";

const tel = (t: number, acc: number, prop: number) => ({
  token_index: 0, t_ms: t, experts: [], spec: { accepted: acc, proposed: prop },
});

describe("liveStats", () => {
  it("returns zeros for empty input", () => {
    expect(liveStats([])).toEqual({ tokens: 0, avgInterMs: 0, tokPerSec: 0, specAccept: 0 });
  });
  it("computes averages and acceptance", () => {
    const s = liveStats([tel(10, 3, 4), tel(10, 1, 4)] as any);
    expect(s.tokens).toBe(2);
    expect(s.avgInterMs).toBe(10);
    expect(s.tokPerSec).toBe(100);
    expect(s.specAccept).toBeCloseTo(0.5);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test -- latency`
Expected: FAIL (cannot find ./latency)

- [ ] **Step 3: Write `ui/src/lib/latency.ts`**

```typescript
import type { Telemetry } from "../types";

export function liveStats(telemetry: Telemetry[]) {
  const tokens = telemetry.length;
  if (tokens === 0) return { tokens: 0, avgInterMs: 0, tokPerSec: 0, specAccept: 0 };
  const avgInterMs = telemetry.reduce((a, t) => a + t.t_ms, 0) / tokens;
  const proposed = telemetry.reduce((a, t) => a + t.spec.proposed, 0);
  const accepted = telemetry.reduce((a, t) => a + t.spec.accepted, 0);
  return {
    tokens,
    avgInterMs,
    tokPerSec: avgInterMs > 0 ? Math.round(1000 / avgInterMs) : 0,
    specAccept: proposed > 0 ? accepted / proposed : 0,
  };
}
```

- [ ] **Step 4: Write `ui/src/components/LatencyPanel.tsx`**

```tsx
import { liveStats } from "../lib/latency";
import type { Telemetry, Summary } from "../types";

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <svg className="w-full h-10" />;
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1)) * 100},${30 - (v / max) * 28}`).join(" ");
  return (
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" className="w-full h-10">
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="1"
                className="text-emerald-400" />
    </svg>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-neutral-900 rounded p-2">
      <div className="text-[10px] uppercase tracking-wide text-neutral-500">{label}</div>
      <div className="text-lg font-mono text-emerald-300">{value}</div>
    </div>
  );
}

export function LatencyPanel({ telemetry, summary }: { telemetry: Telemetry[]; summary: Summary | null }) {
  const s = liveStats(telemetry);
  const ttft = summary ? `${summary.ttft_ms.toFixed(0)} ms` : "—";
  return (
    <div className="border border-neutral-800 rounded-lg p-3">
      <div className="text-xs text-neutral-400 mb-2">Latency · B=1</div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="TTFT" value={ttft} />
        <Stat label="tok/s" value={String(s.tokPerSec)} />
        <Stat label="tokens" value={String(s.tokens)} />
        <Stat label="spec accept" value={`${Math.round(s.specAccept * 100)}%`} />
      </div>
      <div className="mt-2 text-neutral-500">
        <div className="text-[10px] uppercase tracking-wide mb-1">inter-token ms</div>
        <Sparkline values={telemetry.map((t) => t.t_ms)} />
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Mount in `ui/src/App.tsx`** — replace the `<pre>` placeholder line with:

```tsx
          <LatencyPanel telemetry={turn.telemetry} summary={turn.summary} />
```
and add the import at the top: `import { LatencyPanel } from "./components/LatencyPanel";`

- [ ] **Step 6: Run tests + build to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test -- latency && npm run build`
Expected: PASS and clean build.

- [ ] **Step 7: Commit**

```bash
git add ui/src/lib/latency.ts ui/src/lib/latency.test.ts ui/src/components/LatencyPanel.tsx ui/src/App.tsx
git commit -m "feat(ui): latency panel with live stats + sparkline"
```

---

### Task 10: GPU/expert visualization

**Files:**
- Create: `ui/src/lib/gpuLoad.ts`
- Create: `ui/src/lib/gpuLoad.test.ts`
- Create: `ui/src/components/GpuExpertViz.tsx`
- Modify: `ui/src/App.tsx` (fetch topology, mount `GpuExpertViz`)

**Interfaces:**
- Consumes: `Telemetry`, `Topology` (Task 6 types); `getTopology` (Task 6).
- Produces:
  - `gpuLoad.ts`: `gpuHits(telemetry: Telemetry[], numGpus: number): number[]` — array length `numGpus`, counting expert activations per GPU across all telemetry; `recentGpus(telemetry: Telemetry[], window?: number): Set<number>` — set of GPU ids touched in the last `window` (default 1) tokens.
  - `GpuExpertViz({ telemetry, topology }: { telemetry: Telemetry[]; topology: Topology|null })` — renders 8 (or `topology.gpus.length`) GPU tiles; each tile shows name, a load bar proportional to its share of `gpuHits`, and pulses (ring) when in `recentGpus`. If `topology` is null, renders an 8-GPU fallback grid and a muted "topology unavailable" note. Below tiles, a compact expert-activation strip: the last token's `experts` listed as `L{layer}·E{expert}→GPU{gpu}` chips.

- [ ] **Step 1: Write the failing test `ui/src/lib/gpuLoad.test.ts`**

```typescript
import { describe, it, expect } from "vitest";
import { gpuHits, recentGpus } from "./gpuLoad";

const t = (gpus: number[]) => ({
  token_index: 0, t_ms: 8, spec: { proposed: 4, accepted: 3 },
  experts: gpus.map((g) => ({ layer: 0, expert_id: 0, gpu: g })),
});

describe("gpuLoad", () => {
  it("counts hits per gpu", () => {
    expect(gpuHits([t([0, 1]), t([1, 1])] as any, 4)).toEqual([1, 3, 0, 0]);
  });
  it("returns gpus from last token only", () => {
    expect([...recentGpus([t([0]), t([2, 3])] as any)].sort()).toEqual([2, 3]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test -- gpuLoad`
Expected: FAIL (cannot find ./gpuLoad)

- [ ] **Step 3: Write `ui/src/lib/gpuLoad.ts`**

```typescript
import type { Telemetry } from "../types";

export function gpuHits(telemetry: Telemetry[], numGpus: number): number[] {
  const hits = new Array(numGpus).fill(0);
  for (const t of telemetry)
    for (const e of t.experts)
      if (e.gpu >= 0 && e.gpu < numGpus) hits[e.gpu]++;
  return hits;
}

export function recentGpus(telemetry: Telemetry[], window = 1): Set<number> {
  const set = new Set<number>();
  for (const t of telemetry.slice(-window))
    for (const e of t.experts) set.add(e.gpu);
  return set;
}
```

- [ ] **Step 4: Write `ui/src/components/GpuExpertViz.tsx`**

```tsx
import { gpuHits, recentGpus } from "../lib/gpuLoad";
import type { Telemetry, Topology } from "../types";

export function GpuExpertViz({ telemetry, topology }: { telemetry: Telemetry[]; topology: Topology | null }) {
  const numGpus = topology?.gpus.length ?? 8;
  const names = topology?.gpus.map((g) => g.name) ?? Array.from({ length: 8 }, (_, i) => `H100-${i}`);
  const hits = gpuHits(telemetry, numGpus);
  const max = Math.max(...hits, 1);
  const hot = recentGpus(telemetry);
  const last = telemetry[telemetry.length - 1];

  return (
    <div className="border border-neutral-800 rounded-lg p-3 flex-1 min-h-0 flex flex-col">
      <div className="flex justify-between text-xs text-neutral-400 mb-2">
        <span>GPU / expert routing</span>
        {!topology && <span className="text-amber-500/70">topology unavailable — fallback</span>}
      </div>
      <div className="grid grid-cols-4 gap-2">
        {Array.from({ length: numGpus }, (_, i) => (
          <div key={i}
               className={`rounded p-2 bg-neutral-900 border transition-colors ${
                 hot.has(i) ? "border-emerald-400 shadow-[0_0_12px] shadow-emerald-500/40" : "border-neutral-800"}`}>
            <div className="text-[10px] text-neutral-400">{names[i]}</div>
            <div className="h-1.5 mt-1 rounded bg-neutral-800 overflow-hidden">
              <div className="h-full bg-emerald-500" style={{ width: `${(hits[i] / max) * 100}%` }} />
            </div>
            <div className="text-[10px] font-mono text-neutral-500 mt-1">{hits[i]}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-[10px] text-neutral-500">
        <div className="uppercase tracking-wide mb-1">last token experts</div>
        <div className="flex flex-wrap gap-1">
          {last?.experts.map((e, i) => (
            <span key={i} className="px-1.5 py-0.5 rounded bg-neutral-800 font-mono text-emerald-300">
              L{e.layer}·E{e.expert_id}→GPU{e.gpu}
            </span>
          )) ?? <span className="text-neutral-600">idle</span>}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Wire topology + viz into `ui/src/App.tsx`**

Add imports:
```tsx
import { useEffect } from "react";
import { GpuExpertViz } from "./components/GpuExpertViz";
import { getTopology } from "./lib/apiClient";
import type { Topology } from "./types";
```
Add state + effect inside `App` (after `base` state):
```tsx
  const [topology, setTopology] = useState<Topology | null>(null);
  useEffect(() => {
    let alive = true;
    getTopology(base).then((t) => alive && setTopology(t)).catch(() => alive && setTopology(null));
    return () => { alive = false; };
  }, [base]);
```
Mount under `LatencyPanel` in the right rail:
```tsx
          <GpuExpertViz telemetry={turn.telemetry} topology={topology} />
```

- [ ] **Step 6: Run tests + build to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon/ui && npm run test && npm run build`
Expected: all unit tests PASS, clean build.

- [ ] **Step 7: Commit**

```bash
git add ui/src/lib/gpuLoad.ts ui/src/lib/gpuLoad.test.ts ui/src/components/GpuExpertViz.tsx ui/src/App.tsx
git commit -m "feat(ui): live GPU/expert routing visualization"
```

---

### Task 11: Real-engine adapter stub

**Files:**
- Create: `server/backend.py`
- Create: `server/tests/test_backend.py`

**Interfaces:**
- Consumes: `ChatRequest` (Task 1).
- Produces: abstract `Backend` with `stream(req: ChatRequest, topo: dict) -> Iterator[dict]`; `MockBackend(Backend)` delegating to `mock_stream`; `RealEngineBackend(Backend)` whose `stream` raises `NotImplementedError("wire to OpenAI-compatible engine; map routing hooks -> x_telemetry")` with a docstring describing the mapping. `get_backend()` returns `MockBackend()` unless env `BACKEND=real`. This isolates the swap point without changing `main.py` behavior in mock mode.

- [ ] **Step 1: Write the failing test `server/tests/test_backend.py`**

```python
import pytest
from server.schemas import ChatRequest, ChatMessage
from server.topology import build_topology
from server.backend import get_backend, MockBackend, RealEngineBackend


def test_default_is_mock(monkeypatch):
    monkeypatch.delenv("BACKEND", raising=False)
    assert isinstance(get_backend(), MockBackend)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/tests/test_backend.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `server/backend.py`**

```python
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
    """Adapter to a real OpenAI-compatible engine (vLLM / SGLang / custom).

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/tests/test_backend.py -v`
Expected: PASS

- [ ] **Step 5: Route `main.py` through the backend** — in `server/main.py`, replace direct `mock_stream` use in `chat()` with the selected backend.

Add import: `from server.backend import get_backend` and at module level `BACKEND = get_backend()`. In `chat()`, replace `mock_stream(req, TOPO)` (both the non-stream join and the `gen()` loop) with `BACKEND.stream(req, TOPO)`. Remove the now-unused `mock_stream` import.

- [ ] **Step 6: Run full server suite to verify nothing broke**

Run: `cd /Users/charles/Desktop/InferenceHackathon && python -m pytest server/ -v`
Expected: PASS (all server tests, mock path unchanged)

- [ ] **Step 7: Commit**

```bash
git add server/backend.py server/tests/test_backend.py server/main.py
git commit -m "feat(server): pluggable backend with real-engine adapter stub"
```

---

### Task 12: End-to-end smoke test + top-level README

**Files:**
- Create: `ui/playwright.config.ts`
- Create: `ui/e2e/smoke.spec.ts`
- Create: `README.md`

**Interfaces:**
- Consumes: the running mock server (port 8000) + UI dev server (port 5173).
- Produces: a Playwright test that loads the SPA against the live mock, sends a prompt, and asserts streamed text + a populated latency stat + an activated GPU tile. Top-level README documents the two-process run + `ssh -L`.

- [ ] **Step 1: Write `ui/playwright.config.ts`**

```typescript
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  use: { baseURL: "http://localhost:5173" },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: true,
  },
});
```

- [ ] **Step 2: Write `ui/e2e/smoke.spec.ts`**

```typescript
import { test, expect } from "@playwright/test";

test("streams a turn and lights up telemetry", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("backend url").fill("http://localhost:8000");
  await page.getByPlaceholder(/message/i).fill("hello");
  await page.getByRole("button", { name: /send/i }).click();

  // assistant text streams in
  await expect(page.getByText(/Routing across eight H100s/i)).toBeVisible({ timeout: 10000 });
  // a latency stat populates (tokens > 0)
  await expect(page.getByText("tokens")).toBeVisible();
  // at least one GPU tile shows a nonzero hit count
  await expect(page.getByText("GPU / expert routing")).toBeVisible();
});
```

- [ ] **Step 3: Install Playwright browser + run the smoke test**

Run (mock server must be running in another shell):
```bash
# shell A
cd /Users/charles/Desktop/InferenceHackathon && source .venv/bin/activate && \
  STREAM_DELAY=0.02 uvicorn server.main:app --port 8000
# shell B
cd /Users/charles/Desktop/InferenceHackathon/ui && npx playwright install chromium && npm run e2e
```
Expected: 1 passed.

- [ ] **Step 4: Write top-level `README.md`**

````markdown
# Multi-GPU Inference Console

Minimal UI for latency-oriented, B=1 MoE inference on 8×H100. Streams chat over an
OpenAI-compatible API and visualizes per-token expert→GPU routing and latency.

Built fresh for the hackathon; no proprietary engine code is included. The UI talks
to any OpenAI-compatible backend; a Python mock ships so the whole thing runs locally.

## Run (two processes)

```bash
# 1. backend (mock)
python -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
STREAM_DELAY=0.03 uvicorn server.main:app --host 0.0.0.0 --port 8000

# 2. UI
cd ui && npm install && npm run dev   # http://localhost:5173
```

Point the backend-url field (top-right) at your server. For the H100 box:
`ssh -L 8000:localhost:8000 <box>`, run the server there, keep the UI pointed at
`http://localhost:8000`.

## Swap in the real engine
Run an OpenAI-compatible server (SGLang/vLLM/custom) and implement
`server/backend.py:RealEngineBackend` to forward + map routing into `x_telemetry`.
Set `BACKEND=real`. Until then chat + latency work; the viz runs on the mock.

## Tests
- `python -m pytest server/`
- `cd ui && npm run test`
- `cd ui && npm run e2e` (needs the mock running)
````

- [ ] **Step 5: Commit**

```bash
git add ui/playwright.config.ts ui/e2e/smoke.spec.ts README.md
git commit -m "test(e2e): playwright smoke against mock + top-level README"
```

---

## Self-Review

**Spec coverage:**
- §3 three units → Tasks 1–11 (server), 5–10 (ui), contract in 1/2/4/6. ✓
- §4 `/v1/models`, `/v1/topology`, `/v1/chat/completions` (SSE + non-stream), `x_telemetry`, `x_summary`, graceful degradation → Tasks 2, 4, 6, 10 (fallback layout), 7 (optional fields). ✓
- §5 chat pane, latency panel, GPU/expert viz, lib modules, configurable base URL → Tasks 6–10, 8 (base field). ✓
- §6 mock mode + adapter stub → Tasks 3, 11. ✓
- §7 error handling (connection/non-2xx, mid-stream, missing telemetry/topology) → Tasks 6 (throw on !ok), 7 (error status), 10 (fallback). ✓
- §8 Vitest units + Playwright smoke → Tasks 6/7/9/10 (vitest), 12 (playwright). ✓
- §9 real-engine recommendation → README + backend stub docstring (Task 11/12). ✓

**Placeholder scan:** No "TBD/TODO/handle edge cases" in steps; every code step shows complete code. The `RealEngineBackend` NotImplementedError is intentional (the documented swap point), not a plan gap.

**Type consistency:** `Telemetry`/`Summary`/`Topology`/`Expert`/`ChatMessage` defined once in Task 6 `types.ts` and consumed unchanged in Tasks 7–10. `x_telemetry`/`x_summary` field names match between server (Task 3) and UI (Tasks 6–10). `liveStats`, `gpuHits`, `recentGpus`, `build_topology`, `mock_stream`, `streamChat`, `parseSSE`, `useChatStream`, `get_backend` names are used consistently across tasks.
