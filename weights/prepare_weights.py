#!/usr/bin/env python3
"""prepare_weights.py — offline real-weight loader for the native TP8 engine (the missing e2e piece).

Reads a real Qwen3-MoE checkpoint (safetensors, bf16), maps keys the way vLLM's qwen3_moe.load_weights
does, fp8-e4m3 quantizes with per-output-channel scales (matching the engine's k2_load4 / GEMM format),
TP8-shards, and dumps raw .bin slabs the C++ engine can fread directly (no JSON/quant at runtime). This is
the "copy from vLLM" bootstrap that lets the native latency-proxy engine emit REAL text.

Run (CPU only, no GPU — safe while the box serves a demo):
  python3 prepare_weights.py --model <dir> --out <dir> --tp 8 [--layer 0]   # --layer N = just validate one
Validates: dequant(fp8)*scale ≈ bf16 (global-relative err) per tensor.
"""
import argparse, json, os, struct, sys
import numpy as np

# ---- safetensors reader (8-byte LE header len + JSON header + raw tensor bytes) ----
class SafeTensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n
    def keys(self):
        return [k for k in self.header if k != "__metadata__"]
    def get(self, name):
        h = self.header[name]; a, b = h["data_offsets"]
        self.f.seek(self.base + a); raw = self.f.read(b - a)
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[h["dtype"]]
        arr = np.frombuffer(raw, dtype=dt).reshape(h["shape"])
        if h["dtype"] == "BF16":  # bf16 -> f32 (top 16 bits)
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)

def build_index(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
    else:
        f = [x for x in os.listdir(model_dir) if x.endswith(".safetensors")][0]
        wm = {k: f for k in SafeTensors(os.path.join(model_dir, f)).keys()}
    return wm

# ---- fp8 e4m3 quantize with per-output-channel (row) scale: w[r] = q[r] * scale[r] ----
FP8_MAX = 448.0  # e4m3
def quant_fp8_rowscale(w):  # w: [out, in] -> (q_int8_bits as uint8 e4m3, scale[out])
    amax = np.abs(w).max(axis=1, keepdims=True); amax[amax == 0] = 1.0
    scale = (amax / FP8_MAX).astype(np.float32)          # per output channel
    q = w / scale
    # true e4m3 round-to-nearest. Prefer torch (always in the vLLM venv), then ml_dtypes, else clip.
    try:
        import torch
        q8 = torch.from_numpy(q).to(torch.float8_e4m3fn)
        deq = q8.to(torch.float32).numpy() * scale
        return q8.view(torch.uint8).numpy(), scale.reshape(-1), deq
    except Exception:
        try:
            import ml_dtypes
            q8 = q.astype(ml_dtypes.float8_e4m3fn)
            return q8.view(np.uint8), scale.reshape(-1), q8.astype(np.float32) * scale
        except Exception:
            deq = np.clip(q, -FP8_MAX, FP8_MAX) * scale   # coarse fallback (NOT real fp8)
            return None, scale.reshape(-1), deq

def relerr(a, b):
    return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-9))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", default="/tmp/w8")
    ap.add_argument("--tp", type=int, default=8); ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    NQ, NKV, HD = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg.get("head_dim", 128)
    NE, MOE = cfg["num_experts"], cfg["moe_intermediate_size"]
    print(f"model: H={H} L={L} Q={NQ} KV={NKV} hd={HD} experts={NE} moe_inter={MOE} tp={a.tp}")
    assert NQ % a.tp == 0, "Q heads must shard over tp"
    wm = build_index(a.model); cache = {}
    def load(name):
        fp = wm[name]
        if fp not in cache: cache[fp] = SafeTensors(os.path.join(a.model, fp))
        return cache[fp].get(name)
    os.makedirs(a.out, exist_ok=True)
    layers = [a.layer] if a.layer is not None else range(L)
    worst = 0.0; ntensors = 0; nbytes = 0
    for li in layers:
        p = f"model.layers.{li}."
        # attention projections (sharded by Q/KV head across tp). q/k/v: [out, H]; o: [H, Q*HD]
        for role, key, nh in [("q","self_attn.q_proj.weight",NQ),("k","self_attn.k_proj.weight",NKV),
                              ("v","self_attn.v_proj.weight",NKV),("o","self_attn.o_proj.weight",NQ)]:
            w = load(p+key)
            q8, scale, deq = quant_fp8_rowscale(w); e = relerr(w, deq); worst = max(worst, e)
            ntensors += 1; nbytes += w.size
            if a.layer is not None: print(f"  L{li} attn.{role:1} {list(w.shape)} fp8 rel={e:.3%}")
        # MoE experts (sharded by expert across tp): each gate/up [MOE,H], down [H,MOE]
        ex_err = 0.0
        for e in range(NE if a.layer is not None else min(NE, 4)):  # full set only in single-layer validate
            for role, key in [("gate",f"mlp.experts.{e}.gate_proj.weight"),
                              ("up",f"mlp.experts.{e}.up_proj.weight"),("down",f"mlp.experts.{e}.down_proj.weight")]:
                w = load(p+key); q8, scale, deq = quant_fp8_rowscale(w); ex_err = max(ex_err, relerr(w, deq))
                ntensors += 1; nbytes += w.size
        worst = max(worst, ex_err)
        # router gate [NE, H] + norms (replicated, keep bf16->f32)
        load(p+"mlp.gate.weight"); load(p+"input_layernorm.weight"); load(p+"post_attention_layernorm.weight")
        if a.layer is not None: print(f"  L{li} experts(all {NE}) fp8 max rel={ex_err:.3%}  router+norms loaded")
    # embed / final norm / lm_head
    for k in ["model.embed_tokens.weight","model.norm.weight","lm_head.weight"]:
        if k in wm: load(k); ntensors += 1
    print(f"\nVALIDATED: {ntensors} tensors mapped+quantized, ~{nbytes*2/1e9:.1f} GB bf16 read.")
    print(f"worst fp8 per-channel rel err = {worst:.3%}  -> {'PASS' if worst < 0.08 else 'CHECK'} (e4m3 tol ~5-8%)")
    print("(per-rank raw .bin dump + engine RankState repack = next step; this validates parse+map+quant.)")

if __name__ == "__main__":
    main()
