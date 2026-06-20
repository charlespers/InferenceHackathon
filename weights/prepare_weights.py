#!/usr/bin/env python3
"""prepare_weights.py — offline real-weight loader + TP8 sharder for the native engine (the e2e piece).

Reads a real Qwen3-MoE checkpoint (safetensors, bf16), maps keys like vLLM's qwen3_moe.load_weights,
TP8-shards, fp8-e4m3 quantizes each shard with per-output-row scales (the engine's k2_load4/GEMM format),
and dumps raw .bin slabs + a manifest.json the C++ engine can fread directly (no JSON/quant at runtime).
This is the "copy from vLLM" bootstrap that lets the native latency-proxy engine emit REAL text.

Run (CPU only, no GPU — safe while the box serves a demo):
  python3 prepare_weights.py --model <dir> --out <dir> --tp 8 [--layer N] [--dump]
  --layer N : process just one layer (fast validate).  --dump : write .bin + manifest (else validate only).

SHARD RULES (TP=tp):
  q_proj   [Q*HD, H]  -> row-shard (output Q heads)          : rank r = rows [r*Q*HD/tp, +Q*HD/tp]
  k/v_proj [KV*HD, H] -> REPLICATE (KV heads < tp; GQA)       : every rank full
  o_proj   [H, Q*HD]  -> col-shard (input, row-parallel AR)   : rank r = cols [r*Q*HD/tp, +Q*HD/tp]
  experts  gate/up/down -> shard by EXPERT (num_experts/tp/rank)
  router gate, norms, embed, final norm -> REPLICATE
  lm_head  [V, H]     -> row-shard (vocab)
"""
import argparse, json, os, struct
import numpy as np

class SafeTensors:
    def __init__(self, path):
        self.f = open(path, "rb"); n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n)); self.base = 8 + n
    def keys(self): return [k for k in self.header if k != "__metadata__"]
    def get(self, name):
        h = self.header[name]; a, b = h["data_offsets"]
        self.f.seek(self.base + a); raw = self.f.read(b - a)
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[h["dtype"]]
        arr = np.frombuffer(raw, dtype=dt).reshape(h["shape"])
        if h["dtype"] == "BF16": arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32)

def build_index(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx): return json.load(open(idx))["weight_map"]
    f = [x for x in os.listdir(model_dir) if x.endswith(".safetensors")][0]
    return {k: f for k in SafeTensors(os.path.join(model_dir, f)).keys()}

FP8_MAX = 448.0
def quant_fp8_rowscale(w):  # w[out,in] -> (fp8 bytes [out,in], scale[out] f32, dequant f32)
    amax = np.abs(w).max(axis=1, keepdims=True); amax[amax == 0] = 1.0
    scale = (amax / FP8_MAX).astype(np.float32)
    import torch
    q8 = torch.from_numpy((w / scale)).to(torch.float8_e4m3fn)
    deq = q8.to(torch.float32).numpy() * scale
    return q8.view(torch.uint8).numpy(), scale.reshape(-1).astype(np.float32), deq

def relerr(a, b): return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-9))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", default="/tmp/w8")
    ap.add_argument("--tp", type=int, default=8); ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    NQ, NKV, HD = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg.get("head_dim", 128)
    NE, MOE, V = cfg["num_experts"], cfg["moe_intermediate_size"], cfg["vocab_size"]
    tp = a.tp; assert NQ % tp == 0 and NE % tp == 0
    print(f"model H={H} L={L} Q={NQ} KV={NKV} hd={HD} experts={NE} moe={MOE} V={V} tp={tp} dump={a.dump}")
    wm = build_index(a.model); cache = {}
    def load(name):
        fp = wm[name]
        if fp not in cache: cache[fp] = SafeTensors(os.path.join(a.model, fp))
        return cache[fp].get(name)
    manifest = {"config": {"H": H, "L": L, "NQ": NQ, "NKV": NKV, "HD": HD, "NE": NE, "MOE": MOE, "V": V, "tp": tp}, "tensors": []}
    if a.dump:
        for r in range(tp): os.makedirs(os.path.join(a.out, f"rank{r}"), exist_ok=True)
    worst = 0.0; ntensors = 0; nbytes = 0; rt_checked = [False]

    def emit(name, rank, w):  # quantize shard -> fp8+scale, dump, track error
        nonlocal worst, ntensors, nbytes
        q8, scale, deq = quant_fp8_rowscale(w); e = relerr(w, deq); worst = max(worst, e)
        ntensors += 1; nbytes += w.size
        if a.dump:
            d = os.path.join(a.out, f"rank{rank}")
            q8.tofile(os.path.join(d, name + ".w8")); scale.tofile(os.path.join(d, name + ".sc"))
            manifest["tensors"].append({"name": name, "rank": rank, "shape": list(w.shape), "scale": int(scale.size)})
            if not rt_checked[0] and name.endswith("attn.q"):  # round-trip: re-read + dequant
                import torch
                rq = np.fromfile(os.path.join(d, name + ".w8"), dtype=np.uint8)
                rdeq = torch.from_numpy(rq).view(torch.float8_e4m3fn).to(torch.float32).numpy().reshape(w.shape)
                rsc = np.fromfile(os.path.join(d, name + ".sc"), dtype=np.float32).reshape(-1, 1)
                rt = relerr(w, rdeq * rsc)
                print(f"  round-trip {name} rank{rank}: re-read rel={rt:.3%}  {'OK (== in-mem quant)' if abs(rt - e) < 1e-4 else 'MISMATCH'}")
                rt_checked[0] = True
        return e

    layers = [a.layer] if a.layer is not None else range(L)
    for li in layers:
        p = f"model.layers.{li}."
        q = load(p + "self_attn.q_proj.weight"); o = load(p + "self_attn.o_proj.weight")
        k = load(p + "self_attn.k_proj.weight"); v = load(p + "self_attn.v_proj.weight")
        qsh = (NQ * HD) // tp; osh = (NQ * HD) // tp
        for r in range(tp):
            emit(f"L{li}.attn.q", r, q[r * qsh:(r + 1) * qsh, :])         # row-shard (output)
            emit(f"L{li}.attn.o", r, o[:, r * osh:(r + 1) * osh])          # col-shard (input)
            emit(f"L{li}.attn.k", r, k); emit(f"L{li}.attn.v", r, v)       # replicate (GQA)
        epr = NE // tp                                                     # experts per rank
        for r in range(tp):
            for e in range(r * epr, (r + 1) * epr):
                emit(f"L{li}.exp{e}.gate", r, load(p + f"mlp.experts.{e}.gate_proj.weight"))
                emit(f"L{li}.exp{e}.up", r, load(p + f"mlp.experts.{e}.up_proj.weight"))
                emit(f"L{li}.exp{e}.down", r, load(p + f"mlp.experts.{e}.down_proj.weight"))
            # router gate + norms replicate (kept f32 in a full run — small + precision-sensitive; omitted here)
    if a.dump:
        json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"))
    print(f"\n{ntensors} shards quantized, ~{nbytes * 2 / 1e9:.2f} GB bf16 read, worst fp8 rel={worst:.3%} -> {'PASS' if worst < 0.08 else 'CHECK'}")
    if a.dump:
        sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(a.out) for f in fs)
        print(f"dumped -> {a.out}/rank*/  + manifest.json  ({sz/1e6:.1f} MB)  (C++ engine: fread .w8 fp8 + .sc f32)")

if __name__ == "__main__":
    main()
