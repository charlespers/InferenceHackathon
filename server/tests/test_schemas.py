from server.schemas import ChatRequest, ChatMessage, sse, sse_done


def test_chat_request_defaults():
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="hi")])
    assert req.temperature == 0.7
    assert req.max_tokens == 256
    assert req.stream is True


def test_default_model_is_qwen():
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    assert req.model == "qwen3-235b-a22b"


def test_sse_framing():
    assert sse({"a": 1}) == 'data: {"a": 1}\n\n'
    assert sse_done() == "data: [DONE]\n\n"
