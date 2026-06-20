"""
Start vLLM with the routing hook applied.

Patches Qwen3MoeSparseMoeBlock.forward() BEFORE vLLM serves any requests,
so every decode step streams expert selections to the Rust server at
/tmp/vllm_routing.sock.

Usage:
    python3 tools/start_vllm.py

Equivalent to:
    python3 tools/vllm_routing_hook.py &
    vllm serve /alloc/data/Qwen3-235B-A22B --tensor-parallel-size 8 --port 8001 ...
"""

import sys
import os
import threading
import socket
from pathlib import Path

SOCK_PATH = "/tmp/vllm_routing.sock"
MODEL_PATH = "/alloc/data/Qwen3-235B-A22B"

# ---------------------------------------------------------------------------
# Socket server (same as vllm_routing_hook.py, inlined for single-file launch)
# ---------------------------------------------------------------------------

_client_sock = None
_client_lock = threading.Lock()
_step_layer = 0
_step_lock = threading.Lock()

import json


def _socket_server():
    global _client_sock
    if Path(SOCK_PATH).exists():
        Path(SOCK_PATH).unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH)
    srv.listen(1)
    print(f"[routing] socket listening on {SOCK_PATH}", flush=True)
    while True:
        conn, _ = srv.accept()
        print("[routing] Rust server connected", flush=True)
        with _client_lock:
            if _client_sock:
                try:
                    _client_sock.close()
                except Exception:
                    pass
            _client_sock = conn


def _send(record: dict):
    with _client_lock:
        if _client_sock is None:
            return
        try:
            _client_sock.sendall((json.dumps(record) + "\n").encode())
        except Exception:
            pass


def _send_with_layer(experts_list):
    global _step_layer
    with _step_lock:
        layer = _step_layer
        _step_layer = (_step_layer + 1) % 94
    _send({"layer": layer, "experts": experts_list})


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

def _apply_hook():
    import torch

    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    except ImportError as e:
        print(f"[routing] hook failed: {e}", flush=True)
        return

    original_forward = Qwen3MoeSparseMoeBlock.forward

    def hooked_forward(self, hidden_states):
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits, _ = self.gate(flat)

        try:
            import torch.distributed as dist
            rank0 = not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            rank0 = True

        if rank0:
            with torch.no_grad():
                top_k = torch.topk(router_logits, k=8, dim=-1).indices
                _send_with_layer(top_k[0].tolist())

        final = self.experts(hidden_states=flat, router_logits=router_logits)
        if self.tp_size > 1:
            final = self.experts.maybe_all_reduce_tensor_model_parallel(final)
        return final.view(orig_shape)

    Qwen3MoeSparseMoeBlock.forward = hooked_forward
    print("[routing] Qwen3MoeSparseMoeBlock patched", flush=True)


# ---------------------------------------------------------------------------
# Main: start socket server, apply hook, hand off to vLLM CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    threading.Thread(target=_socket_server, daemon=True).start()
    _apply_hook()

    # Hand off to vLLM's serve command
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.scripts import serve

    sys.argv = [
        "vllm",
        "serve", MODEL_PATH,
        "--tensor-parallel-size", "8",
        "--port", "8001",
        "--disable-log-requests",
        "--chat-template-content-format", "string",
    ]

    print(f"[routing] starting vLLM: {' '.join(sys.argv[1:])}", flush=True)
    serve()
