"""native_backend.py — wires the native TP=8 engine (kernels/prefill_step_tp8.cu's --serve mode) into
the same Backend interface MockBackend/VLLMBackend already implement, so the FastAPI app in main.py
can serve real requests off the team's own kernels instead of one-off CLI invocations (tools/run_e2e.py).

This is the missing piece between "the engine can generate real text once, from the command line" and
"live serving": NativeEngineProcess starts the --serve binary ONCE (loading all 94 real layers takes
minutes) and keeps it alive for the life of the FastAPI process, sending it one line per request over
stdin and reading back one token per line over stdout -- the wire protocol --serve speaks, documented
in prefill_step_tp8.cu's run_serve_loop comment.

KNOWN LIMITS (stated, not hidden):
  - Single in-flight request at a time. The engine has no internal request queue/batching; a second
    request sent before the first's DONE will interleave nonsensically on the wire. NativeBackend takes
    a process-wide lock so concurrent HTTP requests queue up instead of corrupting the stream.
  - No streaming cancellation. If a client disconnects mid-generation, the engine still runs the
    request to completion (there is no cancel message in the protocol).
  - Prompt is capped at max_prompt tokens (16 by default, matching the GEMM panels' proven width) --
    longer prompts are truncated, same caveat run_e2e.py already states.
  - If the engine process dies (OOM, crash), NativeBackend does NOT auto-restart it (an unsupervised
    restart would silently eat the minutes-long reload on the next request with no warning) -- it
    raises and the caller learns the backend is down. Restarting the engine is a deploy-time action.
"""
import os
import subprocess
import threading
import time
from typing import Iterator

from server.schemas import ChatRequest

NATIVE_BINARY = os.environ.get("NATIVE_BINARY", "/tmp/prefill_real")
NATIVE_WEIGHTS = os.environ.get("NATIVE_WEIGHTS", "/alloc/data/real_weights")
NATIVE_CHECKPOINT = os.environ.get("NATIVE_CHECKPOINT", "/alloc/data/Qwen3-235B-A22B")
NATIVE_MAX_PROMPT = int(os.environ.get("NATIVE_MAX_PROMPT", "16"))
NATIVE_MAX_DECODE = int(os.environ.get("NATIVE_MAX_DECODE", "256"))
NATIVE_READY_TIMEOUT_S = float(os.environ.get("NATIVE_READY_TIMEOUT_S", "1800"))  # weight load is slow


class NativeEngineProcess:
    """Owns the long-lived --serve subprocess. One instance per FastAPI process (module-level
    singleton via get_native_engine() below) -- starting two would both try to load the full real
    weight set and almost certainly OOM the box."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._ready = False

    def start(self):
        if self._proc is not None:
            return
        cmd = [NATIVE_BINARY, "--serve", NATIVE_WEIGHTS, str(NATIVE_MAX_PROMPT), str(NATIVE_MAX_DECODE)]
        print(f"[native_backend] starting engine: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,   # line-buffered, required for the stream-token-as-produced contract
        )
        self._wait_for_ready()

    def _wait_for_ready(self):
        """Blocks until the engine prints SERVE_READY (all 94 layers x 8 ranks loaded) or dies trying.
        This is the minutes-long step -- callers should run start() at app startup, not per-request."""
        deadline = time.time() + NATIVE_READY_TIMEOUT_S
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if line == "":   # process exited without ever printing SERVE_READY
                raise RuntimeError("native engine exited during startup -- check its stdout/stderr "
                                    "(likely OOM or a missing/corrupt real_weights file)")
            print(f"[native_backend] {line.rstrip()}")
            if line.strip() == "SERVE_READY":
                self._ready = True
                return
        raise TimeoutError(f"native engine did not report SERVE_READY within {NATIVE_READY_TIMEOUT_S}s")

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> Iterator[int]:
        """Sends one request, yields generated token ids ONE AT A TIME as the engine produces them
        (true streaming, not buffer-then-return) by reading stdout line-by-line. Holds the process
        lock for the whole generation -- see the single-in-flight-request limit in the module docstring."""
        if not self._ready:
            raise RuntimeError("native engine not ready -- call start() and wait for it at app startup")
        with self._lock:
            req_line = f"{max_new_tokens} " + " ".join(str(t) for t in prompt_ids) + "\n"
            self._proc.stdin.write(req_line)
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if line == "":
                    raise RuntimeError("native engine process died mid-generation")
                line = line.strip()
                if line == "DONE":
                    return
                if line.startswith("ERROR"):
                    raise RuntimeError(f"native engine reported: {line}")
                if line.startswith("TOK "):
                    yield int(line.split()[1])
                # any other line (e.g. stray rank-loading logs) is ignored -- only TOK/DONE/ERROR matter

    def shutdown(self):
        if self._proc is None:
            return
        try:
            self._proc.stdin.write("QUIT\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
        self._proc = None
        self._ready = False


_engine: NativeEngineProcess | None = None


def get_native_engine() -> NativeEngineProcess:
    global _engine
    if _engine is None:
        _engine = NativeEngineProcess()
    return _engine


class NativeBackend:
    """Backend implementation routing to the native engine. Matches MockBackend/VLLMBackend's
    Backend.stream(req, topo) -> Iterator[dict] contract from server/backend.py exactly, so main.py
    needs no changes beyond get_backend() choosing this for engine="conifer" / BACKEND=native."""

    def __init__(self):
        self.engine = get_native_engine()
        self.engine.start()   # no-op if already started; blocks here on the FIRST construction
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(NATIVE_CHECKPOINT)

    def stream(self, req: ChatRequest, topo: dict) -> Iterator[dict]:
        prompt_text = "\n".join(m.content for m in req.messages)
        prompt_ids = self.tokenizer.encode(prompt_text)
        if len(prompt_ids) > NATIVE_MAX_PROMPT:
            prompt_ids = prompt_ids[-NATIVE_MAX_PROMPT:]   # keep the most recent context, like a sliding window

        t_start = time.time()
        t_first = None
        generated_ids: list[int] = []
        prev_text = ""
        n_tokens = 0

        for tok_id in self.engine.generate(prompt_ids, min(req.max_tokens, NATIVE_MAX_DECODE)):
            if t_first is None:
                t_first = time.time()
            generated_ids.append(tok_id)
            # Decode the WHOLE sequence each time and diff against what was already emitted, rather
            # than decoding one token in isolation -- a single BPE token can be half of a multi-byte
            # UTF-8 character, which would otherwise produce mangled/garbage output mid-stream.
            full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            delta_text = full_text[len(prev_text):]
            prev_text = full_text
            n_tokens += 1
            if delta_text:
                yield {"choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}]}

        elapsed = time.time() - t_start
        ttft_ms = round((t_first - t_start) * 1000, 1) if t_first else 0.0
        decode_elapsed = (elapsed - (t_first - t_start)) if t_first else elapsed
        yield {
            "x_summary": {
                "ttft_ms": ttft_ms,
                "decode_tok_per_s": round(max(n_tokens - 1, 0) / max(decode_elapsed, 1e-3), 1),
                "prefill_tokens": len(prompt_ids),
                "completion_tokens": n_tokens,
                "spec_accept_rate": 0.0,
            }
        }
