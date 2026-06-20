#!/usr/bin/env python3
"""conifer_weight_convert.py — Qwen3-235B-A22B safetensors (bf16) -> per-rank fp8 (e4m3) weight files
in decode_step_tp8.cu's exact sharded layout. M1 of real-text-on-our-kernels.

Per-ROW fp8 scaling (matches the kernels' per-output-row scale arrays): for a weight row w,
  scale = amax(|w|) / 448.0  (e4m3 max ~448);  q = clip(round(w/scale), e4m3)  ;  dequant = q*scale.

Layout written per rank r in [0,8) (TP=8), under <out>/rank{r}/:
  embed.bf16                      [VOCAB, HIDDEN]            (replicated; gather, no GEMM -> keep bf16)
  norm_final.f32                  [HIDDEN]
  lm_head.fp8 / .scale            [VOCAB/8, HIDDEN]         (vocab-row shard; 151936/8=18992 exact)
  layer{L}/
    in_ln.f32 post_ln.f32        [HIDDEN]
    q_norm.f32 k_norm.f32        [HEAD_DIM=128]
    wqkv.fp8 / .scale            [2048, HIDDEN]  (8 Q-heads[1024] + K[512] + V[512]; KV replicated)
    wo.fp8 / .scale              [HIDDEN, 1024]  (o_proj column-shard for this rank's 8 heads)
    wgate.fp8 / .scale           [128, HIDDEN]   (router gate, replicated)
    e{E}_wgu.fp8 / .scale        [384, HIDDEN]   (gate[192]+up[192] intermediate-shard) for E in 0..127
    e{E}_wd.fp8  / .scale        [HIDDEN, 192]   (down intermediate-shard)               for E in 0..127

Run on the box (has torch + the weights):  python3 conifer_weight_convert.py /alloc/data/Qwen3-235B-A22B /alloc/data/qwen3_fp8_tp8 [rank]
(rank optional -> convert just one rank for a fast single-rank validation pass.)
"""
import sys, os, json, struct
import numpy as np

HIDDEN=4096; N_LAYERS=94; N_EXPERTS=128; MOE_INTER=1536; TP=8
Q_HEADS=64; KV_HEADS=4; HEAD_DIM=128
Q_DIM=Q_HEADS*HEAD_DIM            # 8192
KV_DIM=KV_HEADS*HEAD_DIM          # 512
MI_R=MOE_INTER//TP               # 192
QH_R=Q_HEADS//TP                 # 8 q-heads/rank
QDIM_R=QH_R*HEAD_DIM             # 1024
VOCAB=151936; VOC_R=VOCAB//TP    # 18992 (exact)
E4M3_MAX=448.0

def _torch():
    import torch; from safetensors import safe_open; return torch, safe_open

def fp8_rowwise(w_bf16):
    """w: [rows, cols] torch bf16 -> (q int8-bits of e4m3 [rows,cols] uint8, scale[rows] f32)."""
    import torch
    w = w_bf16.float()
    amax = w.abs().amax(dim=1).clamp_min(1e-8)
    scale = (amax / E4M3_MAX)
    q = (w / scale[:,None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).cpu().numpy(), scale.cpu().numpy().astype(np.float32)

def wr(path, arr): arr.tofile(path)

def main():
    if len(sys.argv) < 3: sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    only_rank = int(sys.argv[3]) if len(sys.argv) > 3 else None
    torch, safe_open = _torch()
    idx = json.load(open(os.path.join(src,"model.safetensors.index.json")))["weight_map"]
    handles = {}
    def get(name):
        shard = idx[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(src,shard), framework="pt", device="cpu")
        return handles[shard].get_tensor(name)

    ranks = [only_rank] if only_rank is not None else range(TP)
    for r in ranks:
        rd = os.path.join(out, f"rank{r}"); os.makedirs(rd, exist_ok=True)
        if r == 0 or only_rank is not None:
            get("model.embed_tokens.weight").to(torch.bfloat16).view(torch.uint8).cpu().numpy().tofile(os.path.join(rd,"embed.bf16"))
        get("model.norm.weight").float().cpu().numpy().astype(np.float32).tofile(os.path.join(rd,"norm_final.f32"))
        lm = get("lm_head.weight")[r*VOC_R:(r+1)*VOC_R].contiguous()
        q,s = fp8_rowwise(lm); wr(os.path.join(rd,"lm_head.fp8"),q); wr(os.path.join(rd,"lm_head.scale"),s)
        for L in range(N_LAYERS):
            ld = os.path.join(rd, f"layer{L}"); os.makedirs(ld, exist_ok=True)
            p = f"model.layers.{L}."
            for nm,fn in [("input_layernorm","in_ln"),("post_attention_layernorm","post_ln"),
                          ("self_attn.q_norm","q_norm"),("self_attn.k_norm","k_norm")]:
                get(p+nm+".weight").float().cpu().numpy().astype(np.float32).tofile(os.path.join(ld,fn+".f32"))
            # Wqkv: this rank's 8 Q rows + full K + full V (KV replicated)
            qp = get(p+"self_attn.q_proj.weight")[r*QDIM_R:(r+1)*QDIM_R]
            kp = get(p+"self_attn.k_proj.weight"); vp = get(p+"self_attn.v_proj.weight")
            wqkv = torch.cat([qp,kp,vp], dim=0).contiguous()
            q,s = fp8_rowwise(wqkv); wr(os.path.join(ld,"wqkv.fp8"),q); wr(os.path.join(ld,"wqkv.scale"),s)
            # Wo column-shard: o_proj [HIDDEN, Q_DIM] -> [:, r*1024:(r+1)*1024]; per-output-row(HIDDEN) scale
            wo = get(p+"self_attn.o_proj.weight")[:, r*QDIM_R:(r+1)*QDIM_R].contiguous()
            q,s = fp8_rowwise(wo); wr(os.path.join(ld,"wo.fp8"),q); wr(os.path.join(ld,"wo.scale"),s)
            # router gate (replicated)
            wg = get(p+"mlp.gate.weight").contiguous()
            q,s = fp8_rowwise(wg); wr(os.path.join(ld,"wgate.fp8"),q); wr(os.path.join(ld,"wgate.scale"),s)
            # experts: per-expert gate+up intermediate-shard [384,HIDDEN], down [HIDDEN,192]
            for E in range(N_EXPERTS):
                ep = p+f"mlp.experts.{E}."
                g = get(ep+"gate_proj.weight")[r*MI_R:(r+1)*MI_R]
                u = get(ep+"up_proj.weight")[r*MI_R:(r+1)*MI_R]
                wgu = torch.cat([g,u],dim=0).contiguous()
                q,s = fp8_rowwise(wgu); wr(os.path.join(ld,f"e{E}_wgu.fp8"),q); wr(os.path.join(ld,f"e{E}_wgu.scale"),s)
                d = get(ep+"down_proj.weight")[:, r*MI_R:(r+1)*MI_R].contiguous()
                q,s = fp8_rowwise(d); wr(os.path.join(ld,f"e{E}_wd.fp8"),q); wr(os.path.join(ld,f"e{E}_wd.scale"),s)
            print(f"rank{r} layer{L} done", flush=True)
    print("CONVERT DONE")

if __name__ == "__main__":
    main()
