# Multi-GPU Inference Console — Design Spec

**Date:** 2026-06-19
**Status:** Draft, pending user review
**Context:** Hackathon. Build a minimal UI utility for multi-GPU, latency-oriented MoE
inference (8×H100, batch size B=1). Target model: **Qwen3-235B-A22B** (235B total /
22B active params, 94 layers, 128 experts per MoE layer, top-8 routing). Source the UI
fresh; port the absolute minimum and keep the Conifer company's engine/wrapper internals
private.

## 1. Goal & Non-Goals

### Goal
A small, self-contained web console that:
- Streams a chat completion from a multi-GPU inference backend over a public-standard API.
- Surfaces the **latency story** for B=1 inference: TTFT, tokens/sec, inter-token latency,
  and speculative-decode acceptance rate ("prediction").
- Visualizes **expert/GPU activity** live: which experts on which of the 8 H100s light up
  as each token is produced.

### Non-Goals (deliberately excluded — YAGNI and IP protection)
- No Conifer Rust engine, Metal/CUDA kernels, or model internals.
- No Tauri desktop shell, web-llm browser fallback, auth, billing, Cloudflare functions.
- No studio/agents/research/marketing/docs surface from Conifer-Website.
- The UI does **not** implement multi-GPU inference; it consumes a backend that does.

## 2. Why this seam

Conifer's engine already speaks an **OpenAI-compatible HTTP API** (`/v1/chat/completions`
with SSE). The OpenAI wire format is a public standard, so building the UI against it means
**zero proprietary code** crosses into this repo, and the UI works against any compliant
server (vLLM, SGLang, a stub, or the team's custom engine) while the real engine is built.

The only thing the standard does not carry is per-token expert/GPU routing. We add that as a
**namespaced, optional** field on the existing SSE stream (see §4). Standard clients ignore
unknown fields; our UI reads them; absence degrades gracefully.

## 3. Components

```
InferenceHackathon/
  ui/        Vite + React + TypeScript + Tailwind SPA   (the deliverable)
  server/    Python FastAPI: mock backend + real-engine adapter stub
  docs/       this spec + API contract
```

Three units, each independently understandable and testable, communicating only through the
HTTP contract in §4.

| Unit       | Does                                                              | Depends on        |
|------------|------------------------------------------------------------------|-------------------|
| `ui/`      | Renders chat, latency panel, GPU/expert viz; streams from backend | the HTTP contract |
| `server/`  | Serves the contract; mock synthesizes data, adapter forwards real | engine (adapter only) |
| contract   | The API both sides agree on                                       | nothing           |

## 4. API contract (the seam)

Base URL configurable. CORS enabled on the server. Three endpoints.

### `GET /v1/models`
Standard OpenAI list response: `{ "object": "list", "data": [{ "id": "...", ... }] }`.

### `GET /v1/topology`
Static cluster map the visualization lays out from. Served once at UI load.
```jsonc
{
  "gpus": [{ "id": 0, "name": "H100-0", "mem_total_mb": 81920 }, ...],   // 8 entries
  "num_layers": 94,
  "experts_per_layer": 128,
  "placement": { "<layer>": { "<expert_id>": <gpu_id> } }   // expert→GPU mapping (Qwen3-235B-A22B: top-8 active)
}
```
If unavailable (non-augmented engine), the UI falls back to an 8-GPU default layout and an
inferred expert grid; the viz still renders.

### `POST /v1/chat/completions`  (`"stream": true`)
Standard OpenAI request body (`model`, `messages`, `temperature`, `max_tokens`, `stream`).
Response is SSE (`text/event-stream`), `data:`-prefixed JSON chunks, terminated by
`data: [DONE]`.

Each chunk is a standard `chat.completion.chunk` (with `choices[].delta.content`), optionally
augmented with `x_telemetry` for the token it carries:
```jsonc
"x_telemetry": {
  "token_index": 12,
  "t_ms": 8.4,                                                  // inter-token latency, ms
  "experts": [ { "layer": 3, "expert_id": 17, "gpu": 2 },       // experts activated this step
               { "layer": 3, "expert_id": 4,  "gpu": 5 } ],
  "spec": { "proposed": 4, "accepted": 3 }                      // speculative-decode stats
}
```
Immediately before `[DONE]`, one summary chunk carries `x_summary`:
```jsonc
"x_summary": {
  "ttft_ms": 41.2,
  "decode_tok_per_s": 118.3,
  "prefill_tokens": 532,
  "completion_tokens": 256,
  "spec_accept_rate": 0.74
}
```

**Contract rules**
- `x_telemetry` and `x_summary` are optional. The UI must render chat correctly without them.
- Field names are namespaced with `x_` so any OpenAI-standard client ignores them.
- Errors: non-2xx returns `{ "error": { "message": "..." } }`; mid-stream failure ends the
  stream and the UI shows a turn-level error without losing prior content.

## 5. UI design

Stack: Vite + React + TypeScript + Tailwind. Dark, dense, instrument-panel aesthetic.
Static SPA; no server-side rendering.

### Layout
- **Chat pane** (primary): message list with streaming markdown (`react-markdown`), prompt
  input, model picker (from `/v1/models`), settings (temperature, max tokens). A turn is
  cancelable mid-stream via `AbortController`.
- **Latency panel**: live TTFT, current tokens/sec, inter-token-latency sparkline, total
  tokens, speculative-decode acceptance rate. Resets per turn.
- **GPU/Expert viz**: 8 H100 tiles that pulse with load as tokens route through them; an
  expert-activation heatmap (layers × experts) that lights up per token; per-GPU utilization
  bars (rolling window). Rendered on Canvas/SVG (no heavy 3D); fed by `x_telemetry`.

### lib/
- `apiClient.ts` — typed fetch + SSE line parser (the one piece of generic glue worth
  lifting; not proprietary). Yields parsed chunks.
- `useChatStream.ts` — React hook owning a turn: issues the request, accumulates
  `delta.content`, accumulates `x_telemetry`, exposes derived latency stats, handles cancel
  and error.
- `topology.ts` — fetches and caches `/v1/topology`; provides the fallback layout.

### Connectivity
Backend base URL from `VITE_API_BASE` plus an in-UI override field. To reach the H100 box,
developer uses `ssh -L <localport>:localhost:<serverport> <box>` and points the UI at
`localhost:<localport>`. Documented in `ui/README.md`.

## 6. server/ design (Python, FastAPI + uvicorn)

Mirrors the real serving stack (vLLM/SGLang are Python) so the adapter path is a short hop.

- **mock mode** (default): synthesizes a streamed completion token-by-token with configurable
  TTFT and tokens/sec; fabricates plausible expert routing across 8 GPUs and speculative-decode
  stats; serves `/v1/topology` and `/v1/models`. Enables full UI + viz development and demo
  with **no real engine**.
- **adapter mode** (stub): a pluggable `Backend` interface with one method to forward a chat
  request to a real OpenAI-compatible engine and map its routing hooks into `x_telemetry`.
  Scaffolded with a clear TODO boundary; wired to the real engine when it exists.

## 7. Error handling

- Connection refused / DNS / non-2xx: UI shows a clear banner with the configured base URL;
  chat input disabled until resolved.
- Mid-stream error or cancel: current turn marked errored/canceled, partial content retained,
  app stays usable.
- Missing `x_telemetry` / `/v1/topology`: viz shows "telemetry unavailable" but chat + latency
  summary (if `x_summary` present) still render.

## 8. Testing

- **Vitest** units: `apiClient` SSE parsing (canned multi-chunk streams incl. `x_telemetry`,
  `x_summary`, `[DONE]`, and malformed lines); `useChatStream` turn lifecycle (accumulation,
  cancel, error).
- **Playwright** smoke: boot the FastAPI mock, load the SPA, send a prompt, assert streamed
  text appears, latency panel populates, and at least one GPU tile activates.
- The mock server doubles as the integration fixture.

## 9. Real-engine recommendation (informational, out of scope for the port)

Behind this contract, run **SGLang or vLLM** on the 8×H100 — both OpenAI-compatible, both
support MoE expert-parallelism and speculative decoding (the latency levers). Emitting
per-token routing into `x_telemetry` requires a light hook/patch on the chosen engine, or the
team's own engine. Until then, chat + latency work out of the box and the viz runs on the mock.

## 10. Build order (for the plan)

1. `server/` mock (contract authority + fixture).
2. `ui/` scaffold + `apiClient` + `useChatStream` (chat working against mock).
3. Latency panel.
4. GPU/expert viz + `/v1/topology`.
5. Tests + READMEs.
6. adapter stub + `ssh -L` docs.
