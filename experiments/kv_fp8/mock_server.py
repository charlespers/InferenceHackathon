#!/usr/bin/env python3
"""Tiny stdlib OpenAI-compatible /v1/completions server for HARNESS TESTING ONLY.

Lets us validate tools/kv_measure.py and tools/kv_quality.py end-to-end (SSE
parsing, usage.prompt_tokens read-back, needle recall logic) without burning a
GPU slot. NOT a model — it echoes a deterministic canned answer and reports a
prompt_tokens count derived from prompt length.

    python3 experiments/kv_fp8/mock_server.py --port 8099 &
    python3 tools/kv_measure.py --base http://localhost:8099 --ctx 2048 --repeat 2
"""
import argparse, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def approx_tokens(s: str) -> int:
    # crude: ~1 token per whitespace-split word; good enough to exercise read-back
    return max(1, len(s.split()))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            body = json.dumps({"data": [{"id": "qwen3", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        prompt = req.get("prompt", "")
        max_tokens = int(req.get("max_tokens", 16))
        stream = bool(req.get("stream", False))
        ptok = approx_tokens(prompt)

        # Deterministic canned answer; if a vault-code needle is present, "recall" it
        # so the quality harness's recall detection is exercised on the happy path.
        if "access code for vault 7" in prompt and "ZULU-4471-OMEGA" in prompt:
            answer = "ZULU-4471-OMEGA"
        else:
            answer = "The answer is forty-two and here is a short explanation."
        words = answer.split()[:max_tokens]

        if not stream:
            body = json.dumps({
                "id": "cmpl-mock", "object": "text_completion",
                "choices": [{"index": 0, "text": " ".join(words),
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": ptok, "completion_tokens": len(words),
                          "total_tokens": ptok + len(words)},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # streaming SSE: one chunk per token, tiny inter-token sleep to give TPOT signal
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for w in words:
            chunk = {"id": "cmpl-mock", "object": "text_completion",
                     "choices": [{"index": 0, "text": w + " ", "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.002)
        usage = {"id": "cmpl-mock", "choices": [],
                 "usage": {"prompt_tokens": ptok, "completion_tokens": len(words),
                           "total_tokens": ptok + len(words)}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
