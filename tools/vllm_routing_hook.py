"""
vLLM routing hook for Qwen3-235B-A22B.

Patches Qwen3MoeSparseMoeBlock.forward() to capture per-token expert
selections and stream them to the Rust server via a Unix socket.

The Rust server reads from /tmp/vllm_routing.sock and feeds real routing
data into PredictionPipeline::on_layer_done() instead of RoutingSimulator.

Protocol (newline-delimited JSON over Unix socket):
  {"token": 42, "layer": 5, "experts": [3, 7, 12, 18, 45, 67, 89, 102]}

Usage:
  python3 tools/vllm_routing_hook.py &
  vllm serve /alloc/data/Qwen3-235B-A22B --tensor-parallel-size 8 --port 8001

Or use the combined launcher:
  python3 tools/start_vllm.py
"""

import json
import os
import socket
import threading
import torch
from pathlib import Path

SOCK_PATH = "/tmp/vllm_routing.sock"
TOP_K = 8

# ---------------------------------------------------------------------------
# Socket server: accepts one Rust client, queues routing records
# ---------------------------------------------------------------------------

_client_sock = None
_client_lock = threading.Lock()
_token_counter = 0
_token_lock = threading.Lock()


def _socket_server():
    global _client_sock
    if Path(SOCK_PATH).exists():
        Path(SOCK_PATH).unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH)
    srv.listen(1)
    print(f"[routing_hook] listening on {SOCK_PATH}", flush=True)
    while True:
        conn, _ = srv.accept()
        print("[routing_hook] Rust server connected", flush=True)
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
            pass  # client disconnected; Rust server will reconnect


# ---------------------------------------------------------------------------
# Patch vLLM's Qwen3 MoE block
# ---------------------------------------------------------------------------

def apply_hook():
    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    except ImportError:
        print("[routing_hook] Could not import Qwen3MoeSparseMoeBlock — hook not applied", flush=True)
        return

    original_forward = Qwen3MoeSparseMoeBlock.forward

    def hooked_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states_flat)

        # Only capture on rank 0 to avoid duplicate writes from TP workers
        try:
            import torch.distributed as dist
            is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            is_rank0 = True

        if is_rank0:
            with torch.no_grad():
                top_experts = torch.topk(router_logits, k=TOP_K, dim=-1).indices
                # top_experts: (num_tokens, TOP_K) — during decode, num_tokens=1
                experts_list = top_experts[0].tolist()

            global _token_counter
            layer_idx = getattr(self, "_hook_layer_idx", -1)
            _send({"layer": layer_idx, "experts": experts_list})

            # Increment token counter after the last layer
            if layer_idx == getattr(self, "_hook_last_layer", -1):
                with _token_lock:
                    _token_counter += 1

        # Run original logic
        final_hidden_states = self.experts(
            hidden_states=hidden_states_flat,
            router_logits=router_logits,
        )
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )
        return final_hidden_states.view(orig_shape)

    Qwen3MoeSparseMoeBlock.forward = hooked_forward
    print("[routing_hook] Qwen3MoeSparseMoeBlock.forward patched", flush=True)


def _tag_layers():
    """
    After model is loaded, walk the module tree and tag each MoE block with
    its layer index so hooked_forward can include it in the record.
    """
    try:
        import vllm
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
        # Find all MoE blocks and tag them
        moe_blocks = []

        def _walk(module, prefix=""):
            for name, child in module.named_children():
                full = f"{prefix}.{name}" if prefix else name
                if isinstance(child, Qwen3MoeSparseMoeBlock):
                    moe_blocks.append((full, child))
                _walk(child, full)

        # Can't walk before model loads; tag lazily on first call instead.
        # We use a counter approach: the Nth call to hooked_forward in a
        # single decode step corresponds to layer N.
        print("[routing_hook] layer tagging deferred to runtime", flush=True)
    except Exception as e:
        print(f"[routing_hook] tag_layers error: {e}", flush=True)


# Runtime layer counter (resets each decode step via token boundary detection)
_step_layer = 0
_step_lock = threading.Lock()


def _send_with_layer_counter(experts_list: list[int]):
    """Stateful sender: tracks which layer we're on within a decode step."""
    global _step_layer
    with _step_lock:
        layer = _step_layer
        _step_layer = (_step_layer + 1) % 94  # 94 MoE layers in Qwen3-235B
    _send({"layer": layer, "experts": experts_list})


# Override: use layer counter instead of self._hook_layer_idx
def apply_hook_v2():
    """Simpler hook that uses a global layer counter per decode step."""
    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    except ImportError:
        print("[routing_hook] Could not import Qwen3MoeSparseMoeBlock", flush=True)
        return

    original_forward = Qwen3MoeSparseMoeBlock.forward

    def hooked_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(flat)

        try:
            import torch.distributed as dist
            is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            is_rank0 = True

        if is_rank0:
            with torch.no_grad():
                top_experts = torch.topk(router_logits, k=TOP_K, dim=-1).indices
                _send_with_layer_counter(top_experts[0].tolist())

        final = self.experts(hidden_states=flat, router_logits=router_logits)
        if self.tp_size > 1:
            final = self.experts.maybe_all_reduce_tensor_model_parallel(final)
        return final.view(orig_shape)

    Qwen3MoeSparseMoeBlock.forward = hooked_forward
    print("[routing_hook] hook applied (layer-counter mode)", flush=True)


if __name__ == "__main__":
    # Start socket server in background thread
    t = threading.Thread(target=_socket_server, daemon=True)
    t.start()
    # Apply hook
    apply_hook_v2()
    print("[routing_hook] ready — start vLLM now", flush=True)
    # Keep alive
    import time
    while True:
        time.sleep(60)
