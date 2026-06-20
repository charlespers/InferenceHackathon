from typing import Iterator
from server.schemas import ChatRequest

_WORDS = ("Routing across eight H100s with expert parallelism keeps "
          "per-token latency low at batch size one. ").split()


def _experts_for(token_index: int, topo: dict, top_k: int = 8) -> list[dict]:
    """Pick `top_k` activated experts for this decode step, deterministically.

    Qwen3-235B-A22B activates 8 experts per token; with round-robin placement this
    typically lights up several of the 8 GPUs each step."""
    num_layers = topo["num_layers"]
    experts_per_layer = topo["experts_per_layer"]
    placement = topo["placement"]
    layer = token_index % num_layers
    chosen = []
    for k in range(min(top_k, experts_per_layer)):
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
