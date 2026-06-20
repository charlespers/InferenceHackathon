import json
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    model: str = "qwen3-235b-a22b"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = True


def sse(data: dict) -> str:
    return "data: " + json.dumps(data) + "\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"
