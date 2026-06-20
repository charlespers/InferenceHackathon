# Parallelism, Expert Placement & All-to-All Schedule — B=1, 8×H100

Public-model + standard-technique only. Qwen3-235B-A22B: 94 MoE layers, 64 Q / 4 KV heads
(head_dim 128), 128 experts top-8 (no shared), expert 4096→1536→4096.

## 1. The B=1 framing (why this differs from throughput EP)

At B=1 there is exactly **one token** per step, so each active expert processes **one row**.
Two consequences drive the whole design:

1. **GQA caps clean tensor-parallel at TP ≤ 4** — there are only 4 KV heads. TP=8 must
   replicate each KV head across 2 ranks (2× KV-cache bytes/GPU). 64 Q heads shard cleanly
   to any TP≤8; the binding constraint is the 4 KV heads.
2. **Collective *count*, not volume, dominates comms.** Payloads are tiny (a 4096-d fp8
   activation = 4 KB). Latency per collective on NVSwitch is ~1–5 µs (NCCL) or sub-µs
   (NVSHMEM/DeepEP low-latency). Per token:
   - **TP**: ~2 all-reduces/layer × 94 ≈ **188 all-reduces** (but each tiny, fixed, balanced).
   - **Pure EP**: dispatch + combine all-to-all/layer × 94 ≈ **188 all-to-alls**, *plus*
     per-token routing skew (a GPU may get 0 or 3 of the 8 active experts → latency = max
     over GPUs of local-expert count × expert-GEMV time).

   EP's appeal (each GPU touches only its experts' weights) barely helps at B=1 because the
   active-expert bytes (14.2B/token) are the *same* whether split by EP or by TP — only 8
   experts fire regardless. So EP trades balanced TP all-reduces for skew-prone all-to-alls
   with no bandwidth saving. **EP is a throughput win, not a B=1 latency win** — unless a
   low-latency all-to-all kernel + graph capture makes the 188 collectives ~free.

## 2. Recommended layout (start → bench)

**Start: TP=4 × EP=2.**
- TP=4: clean GQA (1 KV head/rank, 16 Q heads/rank), attention + projections sharded /4,
  per-rank all-reduce after o_proj and after the expert down-proj.
- EP=2: 128 experts → 2 groups of 64 (each on a TP=4 quad). A token's 8 active experts split
  ~4/4 between groups → one all-to-all across the 2 groups, each handled by a balanced TP=4.
- Per-GPU weights (fp8): ~21.6 GB / 8 ≈ **2.7 GB/token read**, balanced.

**Bench these alternatives (pick lowest measured TPOT @ 32k):**

| Layout | Collectives/token | Balance | KV cost | When it wins |
|---|---|---|---|---|
| **TP=4 × EP=2** (start) | ~94 all-reduce + ~94 small all-to-all | good | 1× | balanced default |
| **TP=8 (KV ×2 replicated)** | ~188 all-reduce, all tiny | perfect | 2× KV | short ctx, comms-launch-bound |
| **EP=8** + attn TP=4/DP | ~188 all-to-all + skew | skew-prone | 1× | only with DeepEP + graph capture |
| **TP=4 × DP=2** (replicate, 2 streams) | ~94 all-reduce/replica | n/a | 2× weights | if 80 GB allows 2 model copies — it does NOT at 21.6 GB×… (no) |

(Drop TP=4×DP=2: two full copies don't fit the latency goal and waste BW.)

## 3. Expert placement (the balance lever)

With 128 experts over the expert-parallel dimension, **placement = which GPU holds which
expert**. At B=1, latency of a MoE layer = `max_g (#active experts on GPU g) × t_expert + all_to_all`.
So minimize the worst-case per-GPU active count.

- **Cold start: round-robin** `expert e → group (e mod EP)`, then within group by TP. Maximum
  entropy with no routing stats — the safe first guess. (This is what `server/topology.py`
  already models as `e % num_gpus`.)
- **On the box (autoresearch): co-activation-aware placement.** Log the per-token top-8 sets
  over a representative prompt mix, build the expert co-activation matrix, then partition
  experts across the EP groups to **minimize co-location of frequently co-activated experts**
  (balanced graph partition / spectral or greedy). This flattens the worst-case and is a
  measurable win when a domain concentrates routing.
- **Runtime guard:** if a token sends >ceil(8/EP) experts to one GPU, that GPU just runs the
  extra expert GEMVs sequentially — correct, just slower. Track the histogram (`bench/`).

## 4. Per-MoE-layer timeline (the overlap)

```
layer L:  [router K4 on each rank] → ids/gates (on-device)
          → dispatch all-to-all: send activation y(4096,fp8) to the GPU(s) holding the 8 experts
          → local expert GEMVs (fused gate+up+silu, down×gate; K5)
          → combine all-to-all: gather weighted out_e back, sum into residual
   overlap:  while layer L combine is in flight, start layer L+1's input-norm + router (K1/K4)
             — the only B=1 overlap available (no batch to pipeline).
```
Dispatch+combine are the two collectives/layer. Keep them **inside the CUDA graph** (§6) so
their launch cost is paid once at capture, not per token.

## 5. Comms backend & settings (first guess; verify on box)

- **Prefer NVSHMEM / DeepEP low-latency all-to-all** for the EP path — designed for exactly
  these µs-scale, KB-size, B=1 dispatch/combine ops; NCCL all-to-all has higher fixed latency.
- NCCL fallback env (tune with `nccl-tests alltoall_perf` / `all_reduce_perf` first):
  `NCCL_P2P_LEVEL=NVL`, `NCCL_ALGO=Tree` (small msgs), `NCCL_PROTO=LL128`,
  `NCCL_MAX_NCHANNELS=32`, `NCCL_DEBUG=INFO` (confirm it chose NVLink/NVLS).
- Verify the mesh first: `nvidia-smi topo -m` must show `NV#` between all pairs.
- **Expected:** balanced TP all-reduce of 4 KB ≈ 1–3 µs; 188/token ≈ 0.2–0.6 ms/token of comms
  — *co-equal with the ~0.8 ms weight read*. This is why minimizing collective count (TP-heavy)
  and capturing in a graph matter so much at B=1.

## 6. Composition with the CUDA graph

All of §4 — router, dispatch, expert GEMVs, combine, all-reduces — must be **captured in the
whole-step CUDA graph** so the 188 collectives + 94×K1–K5 launch as one replay. Static shapes
make this clean for greedy decode; the speculative path needs the masking scheme in
`spec-decode-cuda-graph.md`. NVSHMEM kernels and NCCL collectives are both capturable; validate
capture compatibility for the chosen backend on the box (NCCL requires graph-safe comm init).

## 7. Per-GPU byte/latency arithmetic (TP=4×EP=2, 32k ctx, fp8)

- Weights: 21.6 GB / 8 = **2.70 GB/GPU** → 2.70/3.35 TB/s ≈ **0.81 ms** (parallel across 8).
- KV @32k fp8: 3.15 GB total → ~0.39 GB/GPU (TP-sharded) → ~0.12 ms.
- Comms: ~94 all-reduce + ~94 all-to-all, captured → target **<0.3 ms** with NVSHMEM.
- **TPOT target ≈ max(0.81+0.12 read, 0.3 comms overlapped) ≈ ~0.95 ms → ~1,050 tok/s** ideal;
  expect 40–70% → **~430–740 tok/s**. Matches the trajectory milestone.

**Open TODOs for the box:** measure all-to-all µs (NCCL vs NVSHMEM); sweep TP4×EP2 / TP8 / EP8;
collect co-activation stats → solve placement; confirm graph-capture of the comms path.
