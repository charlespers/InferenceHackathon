"""
start_vllm_ipc.py — vLLM launcher with CUDA IPC all-reduce patch.
Drop-in for tools/start_vllm.py.  No coordinator needed; the IPC
handles are exchanged via /tmp/ipc_ar_<rank>.bin files.

Run: python3 tools/start_vllm_ipc.py
"""
import sys, os, threading, socket, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SOCK_PATH  = "/tmp/vllm_routing.sock"
MODEL_PATH = "/alloc/data/Qwen3-235B-A22B"

# ---------------------------------------------------------------------------
# Routing socket (identical to start_vllm.py)
# ---------------------------------------------------------------------------
_client_sock = None; _client_lock = threading.Lock()
_step_layer = 0;    _step_lock  = threading.Lock()

def _socket_server():
    global _client_sock
    if Path(SOCK_PATH).exists(): Path(SOCK_PATH).unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH); srv.listen(1)
    while True:
        conn, _ = srv.accept()
        with _client_lock:
            if _client_sock:
                try: _client_sock.close()
                except Exception: pass
            _client_sock = conn

def _send(record):
    with _client_lock:
        if _client_sock is None: return
        try: _client_sock.sendall((json.dumps(record) + "\n").encode())
        except Exception: pass

def _send_with_layer(experts_list):
    global _step_layer
    with _step_lock:
        layer = _step_layer; _step_layer = (_step_layer + 1) % 94
    _send({"layer": layer, "experts": experts_list})

def _apply_routing_hook():
    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    except ImportError as e:
        print(f"[routing] hook failed: {e}"); return
    orig = Qwen3MoeSparseMoeBlock.forward
    def hooked(self, hidden_states):
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits, _ = self.gate(flat)
        try:
            import torch.distributed as dist
            rank0 = not dist.is_initialized() or dist.get_rank() == 0
        except Exception: rank0 = True
        if rank0:
            import torch
            with torch.no_grad():
                _send_with_layer(torch.topk(router_logits, k=8, dim=-1).indices[0].tolist())
        final = self.experts(hidden_states=flat, router_logits=router_logits)
        if self.tp_size > 1:
            final = self.experts.maybe_all_reduce_tensor_model_parallel(final)
        return final.view(orig_shape)
    Qwen3MoeSparseMoeBlock.forward = hooked
    print("[routing] patched", flush=True)

# ---------------------------------------------------------------------------
# IPC patch hook — applied inside each worker
# ---------------------------------------------------------------------------
def _apply_ipc_hook():
    try:
        from vllm.worker.worker import Worker
    except ImportError:
        try:
            from vllm.worker.worker_base import WorkerBase as Worker
        except ImportError:
            print("[ipc] cannot find vLLM Worker class"); return

    original_init = Worker.__init__
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            sys.path.insert(0, str(REPO))
            import tools.patch_vllm_ipc as ipc_patch
            ipc_patch.patch()
        except Exception as e:
            print(f"[ipc] worker hook failed: {e}")
    Worker.__init__ = patched_init
    print("[ipc] vLLM Worker.__init__ patched ✓", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=_socket_server, daemon=True).start()
    _apply_routing_hook()
    _apply_ipc_hook()

    sys.argv = [
        "vllm", "serve", MODEL_PATH,
        "--tensor-parallel-size", "8",
        "--port", "8001",
        "--disable-log-requests",
        "--chat-template-content-format", "string",
        "--max-model-len", "8192",
    ]

    print(f"[main] starting vLLM+IPC: {' '.join(sys.argv[1:])}", flush=True)
    from vllm.scripts import serve
    serve()
