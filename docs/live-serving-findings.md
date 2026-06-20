# Live serving (real text) — findings & status (2026-06-20)

For the team + agents wiring the demo. The real-text path, the honest perf, the vLLM launch fix, and the
one live blocker (box contention).

## Architecture — the real-text path (already on the box)
- **Team Rust `server`** (`/alloc/data/InferenceHackathon/target/release/server`) listens on **:8000**
  (OpenAI-compatible, model `qwen3-235b-fp8`) + **:9000**. It **proxies to vLLM at `localhost:8001`**.
- **vLLM** serves real Qwen3-235B-A22B (bf16, `/alloc/data/Qwen3-235B-A22B`, 118 safetensors shards,
  **54.9 GiB/GPU** loaded) on **:8001**.
- `tools/start_vllm_nvls.py` patches vLLM with **NVLS comms** + a **routing hook**
  (`Qwen3MoeSparseMoeBlock.forward` → expert IDs to `/tmp/vllm_routing.sock`, which the Rust server reads).
- **This IS "substitute our decode into vLLM"**: vLLM does weight-load / tokenize / **prefill** / sample /
  serve; our **NVLS comms** is the transferable decode optimization. (Substituting *into* vLLM gives its
  prefill for free — no need to extract a prefill kernel.)

## Perf reality (honest — read before quoting numbers)
- **vanilla vLLM** real text: **~85.7 tok/s** (graphed). `--enforce-eager` is slower (~75–85) but more robust.
- **+ our NVLS comms patch**: **~95–97 tok/s** (NCCL ~3.2 ms → NVLS ~1.77 ms comms).
- Our custom-kernel **107.5 / 108.5 tok/s** is measured vs **our own** baseline, on the **proxy (dummy
  weights, synthetic text)**. It does **not** transfer fully to vLLM — vLLM's base kernels (cuBLASLt GEMM,
  tuned fused-MoE) are already strong; **only the NVLS comms win transfers**.
- **Real text @ 107.5 (custom kernels) = a from-scratch engine: weeks.** Blockers: bf16→fp8 quant of 470 GB,
  **no prefill kernel** (decode_step_tp8 is decode-only), tokenizer, sampling, serving loop.

## vLLM 0.10.1 launch fix (saves you the debug)
- `start_vllm.py`'s `from vllm.scripts import serve` is **broken in vLLM 0.10.1** (no `serve`/`cli`; no
  `vllm.__main__`).
- **FIX:** `from vllm.entrypoints.cli.main import main; main()` with `sys.argv=["vllm","serve",MODEL,...]`.
  Applied to `tools/start_vllm_nvls.py`. (Plain `vllm serve ...` as a shell command also works.)

## The live blocker — box contention (NOT a code bug)
- vLLM **loads fully** (54.9 GiB/GPU, weights in 44 s) then gets **hard-killed right after** weight-load —
  **even vanilla** (no patches). No OOM, no traceback, instant GPU→1 MiB = an **external SIGKILL**.
- **Cause:** shared box — competing GPU benches (`dstp8lat`, decode_step, etc.) + the team deployment all
  want all 8 GPUs; vLLM gets **reaped** the moment it grabs them.
- **FIX:** GPU exclusivity — **pause competing GPU benches**, then vLLM holds the 8 GPUs and the Rust server
  :8000 serves real text. Once `localhost:8001` is healthy, the whole stack is live with no code change.

## Display wiring (done)
- `server/mock_engine.py`: Conifer profile = the measured engine (107.5; team now 108.5), with the NVLS
  comms 1.77 ms **floor breakdown**. `VLLMBackend` measures **real** ttft/tok/s **live** from the vLLM stream
  (honest — no painted numbers).
- `ui/src/components/LatencyPanel.tsx`: tuned to show `summary.decode_tok_per_s` (fixed a constant-`t_ms`
  gap bug that showed "—"), a per-forward **floor-breakdown bar** (weight | comms·NVLS | kv | overhead),
  regime, and MBU.

## TL;DR for whoever runs the demo
1. Free the 8 GPUs (pause benches). 2. `tools/start_vllm_nvls.py` (or plain `vllm serve … --port 8001`) →
vLLM on :8001. 3. The Rust server :8000 (or the FastAPI `VLLMBackend`) serves real text immediately.
4. Real text runs ~85 (vanilla) → ~95–97 (NVLS patch). 107.5 is the custom-kernel proxy number, separate.
