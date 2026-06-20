from typing import Iterator
from server.schemas import ChatRequest
from server.optimization_telemetry import summary_fields

_WORDS = ("Routing across eight H100s with expert parallelism keeps "
          "per-token latency low at batch size one. ").split()

# Serving profiles the UI can race head-to-head. Numbers are illustrative of B=1
# (single-request, no batching to hide behind) latency-oriented serving:
#   - conifer: expert-parallel MoE + speculative decoding -> low TTFT, high tok/s.
#   - vllm:    a competent OpenAI-compatible baseline at B=1, no speculative decode.
# The contract (8 experts/token, accepted<=proposed, ttft>0) holds for every profile.
ENGINE_PROFILES = {
    # MEASURED on 8xH100 this session: Conifer forward = 9.81ms (101.9 tok/s plain; NVLS single-barrier
    # comms 1.77ms); spec (measured-flat GEMM verify x EAGLE3-tau) -> ~290 tok/s emitted. vLLM = 85.7
    # tok/s measured (bf16 TP=8, no effective B=1 spec). per_tok_ms = emitted rate; forward_tpot_ms +
    # collective_us drive the floor breakdown (where one forward's time goes).
    "conifer": {"label": "Conifer", "ttft_ms": 34.0, "per_tok_ms": 9.30,
                "spec_enabled": True, "proposed": 8,
                "forward_tpot_ms": 9.30, "collective_us": 9.35, "weight_dtype": "fp8"},
    "vllm": {"label": "vLLM", "ttft_ms": 110.0, "per_tok_ms": 11.7,
             "spec_enabled": False, "proposed": 1,
             "forward_tpot_ms": 11.7, "collective_us": 17.0, "weight_dtype": "bf16"},
}


def engine_profile(name: str) -> dict:
    """Resolve a serving profile by name, defaulting to conifer."""
    return ENGINE_PROFILES.get((name or "").lower(), ENGINE_PROFILES["conifer"])


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
    """Pure, deterministic token stream for `req.engine`. No sleeping here — wall-clock
    pacing happens in the SSE layer (main.py) so this stays fast and testable."""
    prof = engine_profile(req.engine)
    n = max(1, req.max_tokens)
    ttft_ms = prof["ttft_ms"]
    per_tok_ms = prof["per_tok_ms"]
    spec_on = prof["spec_enabled"]
    proposed_each = prof["proposed"]
    accepted_total = 0
    proposed_total = 0
    for i in range(n):
        word = _WORDS[i % len(_WORDS)]
        piece = (word if i == 0 else " " + word)
        proposed = proposed_each
        # Deterministic accept pattern when speculative decoding is on; otherwise the
        # single token is always "accepted" (no speculation).
        accepted = proposed if not spec_on else (3 if (i % 4 != 0) else 2)
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
    # Make the optimization legible (docs/console-telemetry-spec.md): derive the floor breakdown / regime /
    # ceiling-% from this engine's per-token latency so the UI can show WHERE the time goes (overhead-dominated
    # for vLLM; the floor cut for conifer). weight_dtype optional per profile (default bf16).
    # Floor breakdown describes ONE forward (where the time goes), not the spec-amortized emitted token:
    # use forward_tpot_ms (measured 9.81ms conifer) + the measured per-collective comms (NVLS 9.35us).
    fwd_tpot = prof.get("forward_tpot_ms", per_tok_ms)
    opt = summary_fields(fwd_tpot, round(1000.0 / per_tok_ms, 1),
                         weight_dtype=prof.get("weight_dtype", "bf16"),
                         collective_us=prof.get("collective_us", 16.0))
    yield {
        "x_summary": {
            "engine": prof["label"],
            "ttft_ms": ttft_ms,
            "decode_tok_per_s": round(1000.0 / per_tok_ms, 1),
            "prefill_tokens": sum(len(m.content.split()) for m in req.messages),
            "completion_tokens": n,
            "spec_enabled": spec_on,
            "spec_accept_rate": round(accepted_total / proposed_total, 3),
            **opt,
        }
    }
