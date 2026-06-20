from __future__ import annotations

import hashlib
from dataclasses import dataclass, astuple


@dataclass(frozen=True)
class BenchConfig:
    name: str
    plan: str                      # "tp" | "ep" | "hybrid"
    dtype_bytes: int               # 2=bf16, 1=fp8
    kv_dtype_bytes: int
    tp: int
    ep: int
    prompt_tokens: int = 512       # fixed window (playbook §G)
    decode_tokens: int = 128
    seed: int = 0
    warmup_steps: int = 8
    repeats: int = 5               # full-run repeats for variance/CIs (override via --repeats)

    @property
    def seq_len(self) -> int:
        """Representative decode context: prompt + generated tokens."""
        return self.prompt_tokens + self.decode_tokens


def config_id(config: BenchConfig) -> str:
    """Stable 12-hex-char id from all config fields → results lineage key."""
    raw = "|".join(str(x) for x in astuple(config))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
