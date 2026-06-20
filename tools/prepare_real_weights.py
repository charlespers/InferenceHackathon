#!/usr/bin/env python3
"""prepare_real_weights.py — bf16 safetensors -> per-rank fp8 e4m3 binaries for the native TP=8 engine.

Reads the REAL Qwen3-235B-A22B checkpoint (HF safetensors, bf16) and produces, per (layer, rank), the
exact flat binary layout kernels/decode_step_tp8.cu / prefill_step_tp8.cu already expect for Wqkv/Wo/
Wgate/Wgu/Wd (+ per-channel fp32 scales), plus replicated norms and the embedding/lm_head/final-norm
tensors. The C++ side just fread()s these into the SAME device buffers it currently fills with
fill_fp8()/fill_f32() dummy data -- this script is the "real data" substitute for that, nothing else
changes in the engine's memory layout.

READING: the per-shard read loop below (safe_open + get_tensor per name) is copied from the SHAPE of
vLLM's own safetensors_weights_iterator (vllm/model_executor/model_loader/weight_utils.py) -- same
"safe_open one file, pull every needed tensor out of it" structure -- but written directly against the
standalone `safetensors` library, with no import of the vllm package itself. It turns out vLLM's
reader isn't algorithmically different from a plain safe_open loop -- the real reason vLLM loads this
checkpoint in ~44s while an earlier version of this script took 3.5 min/layer (~5.5h for 94) was NOT
the reading, it was (a) re-reading every expert tensor 8x -- once per rank, inside the rank loop --
instead of once, and (b) doing the fp8 cast on CPU tensors, where torch's e4m3 conversion is far slower
than on a CUDA tensor. Both are fixed below: each layer's needed tensors are read ONCE into a per-layer
cache, sliced per rank from that cache, and quantization runs on GPU before being copied back to host.

QUANTIZATION MATH: per-row amax/448 symmetric e4m3 scaling, the same formula vLLM's CUDA fp8-quant op
uses (verified numerically: matches vllm._custom_ops.scaled_fp8_quant's per-token dynamic output to
within bf16 rounding) -- implemented directly in torch here rather than importing vllm for one op.

TENSOR NAMING / FUSION ORDER -- taken directly from vLLM's qwen3_moe.py load_weights() / stacked_params_
mapping (not guessed): Q,K,V fuse in that order into qkv_proj; gate_proj,up_proj fuse in that order into
gate_up_proj; norm applied BEFORE RoPE. All three match this engine's existing internal assumptions
(QKV_OUT = Q_DIM + 2*KV_DIM with Q first; SiLU(gate)*up with gate columns first) -- confirmed, not
coincidental, since this engine's shapes were designed to match the real Qwen3 architecture from the
start; only the WEIGHT VALUES were dummy until now.

SHARDING (matches decode_step_tp8.cu's RankState layout exactly):
  - Wqkv: Q rows are TP-sharded (this rank's Q_DIM_RANK=1024 of 8192 rows); K,V rows are REPLICATED in
    full on every rank (decode's own convention -- "KV is the replicated full cache").
  - Wo: column-sharded by this rank's Q_DIM_RANK input columns (the GEMM's K dimension).
  - Wgate (router): replicated in full on every rank (every rank computes the full router locally).
  - Wgu (gate+up) / Wd (down), per expert: TP-sharded along the intermediate dimension (this rank's
    MOE_INTER_RANK=192 of 1536) -- the SAME mechanism vLLM uses for --tensor-parallel-size with no EP
    flag (confirmed: 227B MoE params * 2B (bf16) / 8 ranks ~= 56.7GB/GPU, matching vLLM's measured
    54.9 GiB/GPU on this exact box). At fp8 this engine's slice is half that, ~28.4GB/GPU -- fits.
  - embed_tokens / lm_head: VOCAB-sharded the way decode_step_tp8.cu's lm_head already shards (v_rows/
    v_off per rank); embed_tokens is small enough (622M params) to just replicate in full instead.

QUANTIZATION: per-output-channel (per-row) symmetric e4m3, scale = amax(row)/448 -- the SAME scheme
already implemented in every quant kernel in decode_step_tp8.cu/gemm_engine.cuh (gemm_quant, etc.), so
the C++ side needs no new dequant math, only new (real) bytes.

USAGE (layer-by-layer; start with --layers 0 to validate before the full 94):
  python3 tools/prepare_real_weights.py --checkpoint /alloc/data/Qwen3-235B-A22B \
      --out /alloc/data/real_weights --layers 0 [--layers 1 ...] [--tp 8]
  python3 tools/prepare_real_weights.py --checkpoint /alloc/data/Qwen3-235B-A22B \
      --out /alloc/data/real_weights --embeddings   # embed_tokens + final norm + lm_head, once
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open

HIDDEN = 4096
N_Q_HEADS = 64
N_KV_HEADS = 4
HEAD_DIM = 128
Q_DIM = N_Q_HEADS * HEAD_DIM       # 8192
KV_DIM = N_KV_HEADS * HEAD_DIM     # 512
N_EXPERTS = 128
MOE_INTER = 1536
VOCAB = 151936

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def quantize_e4m3_per_row(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """w: [rows, cols] (any device/dtype) -> (fp8 e4m3 bytes [rows,cols] uint8 on host, scale [rows]
    fp32 on host). Per-row dynamic amax/448 scaling -- the same formula vLLM's fp8 quant op uses
    (verified numerically), computed directly here with no vllm import. Runs on DEVICE (cuda if
    available): torch's CPU e4m3 cast is far slower than the same op on a CUDA tensor."""
    w = w.to(DEVICE, dtype=torch.float32)
    amax = w.abs().amax(dim=1).clamp(min=1e-12)
    scale = amax / 448.0
    q = (w / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).cpu().numpy(), scale.to(torch.float32).cpu().numpy()


class Checkpoint:
    """Random-access-by-name reader over the checkpoint's safetensors shards. Same shape as vLLM's own
    safetensors_weights_iterator (safe_open one shard file, pull tensors out of it) but written
    directly against the standalone `safetensors` library -- no vllm import. Reads each shard file
    ONCE and caches its tensors; the index tells us which shard a name lives in."""
    def __init__(self, ckpt_dir: str):
        self.dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    def _load_shard(self, shard: str) -> dict[str, torch.Tensor]:
        if shard in self._shard_cache:
            return self._shard_cache[shard]
        path = os.path.join(self.dir, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            tensors = {name: f.get_tensor(name) for name in f.keys()}
        self._shard_cache = {shard: tensors}   # keep only the most-recent shard resident (memory)
        return tensors

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        return self._load_shard(shard)[name]

    def get_many(self, names: list[str]) -> dict[str, torch.Tensor]:
        """Fetch several names at once, grouping by shard so each shard is only loaded once even if
        names span multiple shards (the expert tensors for one layer are typically split 1-2 shards)."""
        out: dict[str, torch.Tensor] = {}
        by_shard: dict[str, list[str]] = {}
        for n in names:
            by_shard.setdefault(self.weight_map[n], []).append(n)
        for shard, ns in by_shard.items():
            tensors = self._load_shard(shard)
            for n in ns:
                out[n] = tensors[n]
        return out


def write_bin(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)


def process_layer(ck: Checkpoint, out_dir: str, L: int, tp: int):
    Q_DIM_RANK = Q_DIM // tp
    MOE_INTER_RANK = MOE_INTER // tp
    p = f"model.layers.{L}."

    q = ck.get(p + "self_attn.q_proj.weight").to(DEVICE)   # [Q_DIM, HIDDEN] bf16
    k = ck.get(p + "self_attn.k_proj.weight").to(DEVICE)   # [KV_DIM, HIDDEN]
    v = ck.get(p + "self_attn.v_proj.weight").to(DEVICE)   # [KV_DIM, HIDDEN]
    o = ck.get(p + "self_attn.o_proj.weight").to(DEVICE)   # [HIDDEN, Q_DIM]
    q_norm = ck.get(p + "self_attn.q_norm.weight").to(torch.float32).numpy()   # [HEAD_DIM]
    k_norm = ck.get(p + "self_attn.k_norm.weight").to(torch.float32).numpy()
    in_norm = ck.get(p + "input_layernorm.weight").to(torch.float32).numpy()           # [HIDDEN]
    post_norm = ck.get(p + "post_attention_layernorm.weight").to(torch.float32).numpy()
    gate = ck.get(p + "mlp.gate.weight").to(DEVICE)        # [N_EXPERTS, HIDDEN]
    gate_q, gate_s = quantize_e4m3_per_row(gate)

    # ---- read + slice EVERY expert ONCE (not once per rank -- the earlier 8x-redundant-read bug) ----
    names = []
    for e in range(N_EXPERTS):
        names += [p + f"mlp.experts.{e}.gate_proj.weight",
                  p + f"mlp.experts.{e}.up_proj.weight",
                  p + f"mlp.experts.{e}.down_proj.weight"]
    experts = ck.get_many(names)
    # per-rank slices, built once, reused when writing each rank's file below
    gu_by_rank = [[] for _ in range(tp)]   # gu_by_rank[r] = list of (fp8_bytes, scale) per expert
    d_by_rank  = [[] for _ in range(tp)]
    for e in range(N_EXPERTS):
        gate_proj = experts[p + f"mlp.experts.{e}.gate_proj.weight"].to(DEVICE)   # [MOE_INTER, HIDDEN]
        up_proj   = experts[p + f"mlp.experts.{e}.up_proj.weight"].to(DEVICE)     # [MOE_INTER, HIDDEN]
        down_proj = experts[p + f"mlp.experts.{e}.down_proj.weight"].to(DEVICE)   # [HIDDEN, MOE_INTER]
        for r in range(tp):
            gslice = gate_proj[r*MOE_INTER_RANK:(r+1)*MOE_INTER_RANK]
            uslice = up_proj[r*MOE_INTER_RANK:(r+1)*MOE_INTER_RANK]
            gu = torch.cat([gslice, uslice], dim=0)                      # [2*MOE_INTER_RANK, HIDDEN]
            gu_by_rank[r].append(quantize_e4m3_per_row(gu))
            dslice = down_proj[:, r*MOE_INTER_RANK:(r+1)*MOE_INTER_RANK].contiguous()
            d_by_rank[r].append(quantize_e4m3_per_row(dslice))

    for r in range(tp):
        d = os.path.join(out_dir, f"layer{L}", f"rank{r}")
        qkv = torch.cat([q[r*Q_DIM_RANK:(r+1)*Q_DIM_RANK], k, v], dim=0)   # [QKV_OUT_RANK, HIDDEN]
        qkv_q, qkv_s = quantize_e4m3_per_row(qkv)
        write_bin(os.path.join(d, "Wqkv.fp8"), qkv_q)
        write_bin(os.path.join(d, "Wqkv_scale.f32"), qkv_s)

        o_slice = o[:, r*Q_DIM_RANK:(r+1)*Q_DIM_RANK].contiguous()        # [HIDDEN, Q_DIM_RANK]
        o_q, o_s = quantize_e4m3_per_row(o_slice)
        write_bin(os.path.join(d, "Wo.fp8"), o_q)
        write_bin(os.path.join(d, "Wo_scale.f32"), o_s)

        write_bin(os.path.join(d, "q_norm.f32"), q_norm)
        write_bin(os.path.join(d, "k_norm.f32"), k_norm)
        write_bin(os.path.join(d, "in_norm.f32"), in_norm)
        write_bin(os.path.join(d, "post_norm.f32"), post_norm)
        write_bin(os.path.join(d, "Wgate.fp8"), gate_q)
        write_bin(os.path.join(d, "Wgate_scale.f32"), gate_s)

        gu_q = np.concatenate([c[0] for c in gu_by_rank[r]], axis=0)
        gu_s = np.concatenate([c[1] for c in gu_by_rank[r]], axis=0)
        d_q  = np.concatenate([c[0] for c in d_by_rank[r]], axis=0)
        d_s  = np.concatenate([c[1] for c in d_by_rank[r]], axis=0)
        write_bin(os.path.join(d, "Wgu_all.fp8"), gu_q)
        write_bin(os.path.join(d, "Wgu_scale_all.f32"), gu_s)
        write_bin(os.path.join(d, "Wd_all.fp8"), d_q)
        write_bin(os.path.join(d, "Wd_scale_all.f32"), d_s)
    print(f"layer {L}: done ({tp} ranks)", flush=True)


def process_embeddings(ck: Checkpoint, out_dir: str, tp: int):
    v_rows = [VOCAB // tp + (1 if r < VOCAB % tp else 0) for r in range(tp)]
    v_off = [sum(v_rows[:r]) for r in range(tp)]

    embed = ck.get("model.embed_tokens.weight").to(DEVICE)     # [VOCAB, HIDDEN] -- replicate, every rank
    embed_q, embed_s = quantize_e4m3_per_row(embed)
    final_norm = ck.get("model.norm.weight").to(torch.float32).numpy()
    lm_head = ck.get("lm_head.weight").to(DEVICE)               # [VOCAB, HIDDEN] -- vocab-sharded per rank

    for r in range(tp):
        d = os.path.join(out_dir, "embeddings", f"rank{r}")
        write_bin(os.path.join(d, "embed_tokens.fp8"), embed_q)
        write_bin(os.path.join(d, "embed_tokens_scale.f32"), embed_s)
        write_bin(os.path.join(d, "final_norm.f32"), final_norm)
        lm_slice = lm_head[v_off[r]:v_off[r] + v_rows[r]]
        lm_q, lm_s = quantize_e4m3_per_row(lm_slice)
        write_bin(os.path.join(d, "lm_head.fp8"), lm_q)
        write_bin(os.path.join(d, "lm_head_scale.f32"), lm_s)
        with open(os.path.join(d, "lm_head_meta.json"), "w") as f:
            json.dump({"v_rows": v_rows[r], "v_off": v_off[r]}, f)
    print(f"embeddings: done ({tp} ranks), v_rows={v_rows}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=[])
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--tp", type=int, default=8)
    args = ap.parse_args()

    ck = Checkpoint(args.checkpoint)
    if args.embeddings:
        process_embeddings(ck, args.out, args.tp)
    for L in args.layers:
        process_layer(ck, args.out, L, args.tp)
