"""MoE model architecture description + parameter / memory accounting.

Default config is Qwen3-235B-A22B, validated against the published
config.json (transformers 4.51.0):

    hidden_size            4096
    intermediate_size      12288   (unused: all 94 layers are MoE)
    moe_intermediate_size  1536    (per-expert SwiGLU inner dim)
    num_hidden_layers      94
    num_attention_heads    64      (head_dim 128 -> q dim 8192)
    num_key_value_heads    4       (GQA, kv dim 512)
    num_experts            128
    num_experts_per_tok    8       (top-8, no shared expert)
    vocab_size             151936
    tie_word_embeddings    False

Param-count check below reproduces ~235B total / ~22B active.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    name: str = "Qwen3-235B-A22B"
    hidden: int = 4096
    n_layers: int = 94
    n_heads: int = 64
    n_kv_heads: int = 4
    head_dim: int = 128
    n_experts: int = 128
    top_k: int = 8
    moe_inter: int = 1536       # per-expert SwiGLU intermediate
    n_shared_experts: int = 0   # Qwen3 MoE has none
    vocab: int = 151936
    tie_embeddings: bool = False

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    # ---- per-component parameter counts ----
    @property
    def attn_params(self) -> int:
        # q,k,v,o projections (no bias in Qwen3). QK-norm params negligible.
        q = self.hidden * self.q_dim
        k = self.hidden * self.kv_dim
        v = self.hidden * self.kv_dim
        o = self.q_dim * self.hidden
        return q + k + v + o

    @property
    def one_expert_params(self) -> int:
        # SwiGLU: gate_proj + up_proj (hidden->inter) + down_proj (inter->hidden)
        return 3 * self.hidden * self.moe_inter

    @property
    def router_params(self) -> int:
        return self.hidden * self.n_experts

    @property
    def moe_params_per_layer(self) -> int:
        experts = (self.n_experts + self.n_shared_experts) * self.one_expert_params
        return experts + self.router_params

    @property
    def layer_params(self) -> int:
        return self.attn_params + self.moe_params_per_layer

    @property
    def embed_params(self) -> int:
        e = self.vocab * self.hidden
        return e if self.tie_embeddings else 2 * e

    @property
    def total_params(self) -> int:
        return self.n_layers * self.layer_params + self.embed_params

    # ---- per-token *active* parameters (what B=1 decode actually touches) ----
    @property
    def active_attn_params(self) -> int:
        return self.attn_params

    @property
    def active_moe_params_per_layer(self) -> int:
        active_experts = (self.top_k + self.n_shared_experts) * self.one_expert_params
        return active_experts + self.router_params

    @property
    def active_layer_params(self) -> int:
        return self.active_attn_params + self.active_moe_params_per_layer

    @property
    def active_params(self) -> int:
        lm_head = self.vocab * self.hidden  # read every token to produce logits
        return self.n_layers * self.active_layer_params + lm_head

    # ---- KV cache ----
    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        # k and v, across layers, GQA kv heads only.
        return 2 * self.n_layers * self.kv_dim * dtype_bytes


QWEN3_235B = MoEConfig()
