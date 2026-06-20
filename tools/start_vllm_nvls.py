"""
start_vllm_nvls.py — Drop-in replacement for tools/start_vllm.py that applies
the NVLS cross-process all-reduce patch before starting vLLM.

Prerequisites (run once before this script):
    bash tools/apply_nvls_patch.sh   # builds + starts the coordinator

Usage:
    python3 tools/start_vllm_nvls.py
"""
import sys, os, threading, socket, json
from pathlib import Path

# Repo root so imports work regardless of cwd.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SOCK_PATH  = "/tmp/vllm_routing.sock"
MODEL_PATH = "/alloc/data/Qwen3-235B-A22B"

# ---------------------------------------------------------------------------
# Routing socket (same as start_vllm.py)
# ---------------------------------------------------------------------------
_client_sock = None
_client_lock = threading.Lock()
_step_layer  = 0
_step_lock   = threading.Lock()


def _socket_server():
    global _client_sock
    if Path(SOCK_PATH).exists():
        Path(SOCK_PATH).unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH); srv.listen(1)
    print(f"[routing] socket listening on {SOCK_PATH}", flush=True)
    while True:
        conn, _ = srv.accept()
        print("[routing] Rust server connected", flush=True)
        with _client_lock:
            if _client_sock:
                try: _client_sock.close()
                except Exception: pass
            _client_sock = conn


def _send(record: dict):
    with _client_lock:
        if _client_sock is None: return
        try: _client_sock.sendall((json.dumps(record) + "\n").encode())
        except Exception: pass


def _send_with_layer(experts_list):
    global _step_layer
    with _step_lock:
        layer = _step_layer
        _step_layer = (_step_layer + 1) % 94
    _send({"layer": layer, "experts": experts_list})


# ---------------------------------------------------------------------------
# Routing hook (same as start_vllm.py)
# ---------------------------------------------------------------------------
def _apply_routing_hook():
    import torch
    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    except ImportError as e:
        print(f"[routing] hook failed: {e}", flush=True); return

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
# NVLS patch — applied inside each worker process via env var hook
# ---------------------------------------------------------------------------
def _apply_nvls_hook():
    """
    Register an environment-variable-based hook so that each vLLM worker
    process calls patch_vllm_nvls.patch() after initializing torch.distributed.

    We use vLLM's VLLM_WORKER_MULTIPROC_METHOD environment and monkey-patch
    the Worker class to call our init after it sets up distributed.
    """
    coord_json = "/tmp/nvls_mc.json"
    if not os.path.exists(coord_json):
        print("[nvls] coordinator not running (/tmp/nvls_mc.json missing) — skipping NVLS patch")
        print("[nvls] Run: bash tools/apply_nvls_patch.sh   to start the coordinator")
        return

    try:
        with open(coord_json) as f:
            meta = json.load(f)
        assert meta.get("ready"), "coordinator JSON not marked ready"
    except Exception as e:
        print(f"[nvls] coordinator JSON invalid: {e} — skipping")
        return

    print("[nvls] coordinator ready — hooking vLLM Worker.__init__...", flush=True)

    try:
        from vllm.worker.worker import Worker
    except ImportError:
        try:
            from vllm.worker.worker_base import WorkerBase as Worker
        except ImportError:
            print("[nvls] cannot find vLLM Worker class — skipping")
            return

    original_init = Worker.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # After vLLM's init (which sets up distributed), apply our NVLS patch.
        try:
            import tools.patch_vllm_nvls as nvls
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            world = int(os.environ.get("WORLD_SIZE", "8"))
            nvls._worker_init(rank, world)
            if nvls._nvls_ready:
                nvls.patch()
        except Exception as e:
            print(f"[nvls] worker init hook failed: {e}")

    Worker.__init__ = patched_init
    print("[nvls] vLLM Worker.__init__ patched ✓", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=_socket_server, daemon=True).start()
    _apply_routing_hook()
    _apply_nvls_hook()

    sys.argv = [
        "vllm",
        "serve", MODEL_PATH,
        "--tensor-parallel-size", "8",
        "--port", "8001",
        "--disable-log-requests",
        "--chat-template-content-format", "string",
        "--max-model-len", "8192",
    ]

    print(f"[main] starting vLLM with NVLS patch: {' '.join(sys.argv[1:])}", flush=True)

    launched = False
    try:
        from vllm.scripts import serve
        serve(); launched = True
    except (ImportError, AttributeError):
        pass
    if not launched:
        try:
            from vllm.scripts import cli
            cli(); launched = True
        except (ImportError, AttributeError):
            pass
    if not launched:
        import runpy
        runpy.run_module("vllm", run_name="__main__", alter_sys=True)
