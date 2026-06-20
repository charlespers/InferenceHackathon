# B=1 Engine Push — shared context (READ FIRST, every agent)

**Mission.** Build/tune *our own* inference engine — the custom CUDA kernels + fused decode
step in `kernels/` — for **batch-size-1 decode latency**, specialized to the **Qwen3-235B-A22B**
architecture and the **8×H100** box, to reach **1000 tok/s single-stream**. We **benchmark
against vLLM** (vLLM is the baseline we beat, NOT something we tune). Metric: drive up tok/s by
maximizing **MBU** (memory-bandwidth utilization, the binding constraint at B=1) and **MFU** for
any multi-token / spec-verify path.

Every agent working this goal MUST read this file first and report measured numbers cited to a
file — never fabricate a tok/s or MBU. If you change a kernel, re-bench it and report the delta.

---

## The hardware
- 8× **H100 80GB HBM3**, full **NVLink mesh** (NV18 between every pair — ideal for TP8).
- HBM peak per GPU = **3350 GB/s** (this is the MBU denominator). 132 SMs. sm_90a.
- CUDA 12.6 toolkit at `/usr/local/cuda` (nvcc builds `.cu` with `-arch=sm_90a -O3 --use_fast_math`).
- SSH: `ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 31025 -i ~/.ssh/id_github root@147.185.41.162 '<CMD>'`
- **GPU coordination:** the box is shared with teammates (branches: Alyssa, djamoils-*, jminding/*).
  Check `nvidia-smi` before grabbing all 8. Kernel microbenches mostly need **1 GPU**; the full
  TP8 step needs all 8. A vLLM 235B load holds ~64 GB/GPU.

## The model: Qwen3-235B-A22B-Instruct-2507-FP8 (exact dims — tune to these)
- **94** decoder layers, **all MoE** (no dense FFN layers).
- hidden_size **4096**; attention heads **64**, KV heads **4** (GQA 16:1), head_dim **128**
  → q_proj 4096→8192, k_proj/v_proj 4096→512 each, o_proj 8192→4096.
- **Qwen3 quirk: per-head QK-RMSNorm** (q_norm/k_norm on each head before RoPE). rope_theta 1e6.
- MoE: **128 experts**, **top-8** routing, moe_intermediate **1536**, no shared expert,
  norm_topk_prob=true. Per expert: gate 4096→1536, up 4096→1536, SiLU, down 1536→4096.
- vocab **151936** (lm_head 4096→151936).
- **Active params/token ≈ 22B** (≈6.7B attention + ≈14.2B for the 8 routed experts + router/embed).
- Weights are **FP8 e4m3, block-scaled** (`weight_block_size` present, dynamic act scale).
- Assets on box (downloaded): FP8 weights `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (221 GB);
  EAGLE3 draft `nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys` (2.3 GB).

---

## Baselines (the numbers we beat)
- **vLLM (our bench target):** 69.5 tok/s bf16-TP8+EP, 65.8 tok/s FP8 — both B=1, ~11% of roofline,
  TPOT 14.4 / 15.2 ms. FP8 *regresses* vs bf16 at B=1 → **the box is overhead/comms-bound, not
  bandwidth-bound** at this efficiency. (`/root/bench_result.txt`, `/root/fp8res.txt`.)
- **Analytical roofline (FP8, TP8, ctx2048):** weight-only floor ≈ **0.81 ms/token** (≈1240 tok/s
  weight-only); with modeled comms, TP8 floor ≈ **568 tok/s** @ e=1.0, ≈261 @ e=0.46. EP8 is ~2.8×
  worse than TP8 (routing imbalance) — **pure TP8 is the layout.** (`docs/predicted-tok-s-matrix.md`.)
- **"400 tok/s" was never measured** — it's an analytical floor (bf16-TP8 floor=387). Measured
  reality is ~70 tok/s. Don't treat predicted numbers as benchmarks.

## Current kernel state — measured MBU (the headroom map)
From `/root/kernel_bench_result.txt` (H100, peak 3350 GB/s):

| Kernel | file | GB/s | **MBU** | notes |
|---|---|---|---|---|
| K5 MoE experts (DOMINANT) | `k5_experts_v3.cu` / `_warp` | 1530 (benched) / **1947 (v3)** | **45.7% / 58.1%** | 98.7 µs/tok → **9.28 ms/tok over 94 layers**; v3 (cp.async R=4 STAGES=2) not in headline bench |
| lm_head (once/token) | `lmhead_k3_bench.cu` | 1855 | **55.4%** | on-device argmax |
| K3 o_proj+residual | `k3_attn_epilogue.cu` | 1302 | **38.9%** | fused residual |
| K1 QKV proj | `k1_attn_prologue.cu` / `k1k2_mbu_v2.cu` | 77–904 | **2.3–27%** | 🔴 biggest relative headroom |
| K2 flash-decode | `k2_flash_decode.cu` | 17–112 | **0.5–3.3%** | 🔴 tiny bytes, terrible occupancy at short ctx |
| K4 router | `k4_router.cu` | — | cheap | softmax+top8 on-device |
| TP8 all-reduce | `nvshmem_comms.cu` | — | — | 2/layer×94; 5–16µs/coll = **0.94–3 ms/tok** = THE WALL |

## Known blockers
- `decode_step.cu` **crashes** (illegal memory access, line 458) → no end-to-end fused-step number.
- `nvshmem_comms.cu` **won't link** (cu13 device lib vs cu12.6 nvcc) → no device-side all-reduce yet.
- K5 **v3 (58%)** improvement is NOT wired into the headline bench (still shows 45.7% warp config).
- `prefill_attn.cu`, int4 MoE path: validation FAIL — not usable as-is.
- The Rust `engine/` is an **offline simulator/optimizer** (routing + spec-accept modeling), **not**
  the GPU runtime. The GPU runtime is the kernels + a host loop (to be built/measured).

---

## Theory of victory (why 1000 is reachable, and what it requires)
`effective tok/s = (1 / single_forward_ms) × τ`, where τ = mean accepted tokens/forward (spec).
- Single-forward floor: weight 0.81 ms + comms (0.14–0.94 ms) + attention + overhead.
  At current kernel MBU + stock comms it's ~10–14 ms (≈70–100 tok/s). Realistic *tuned* target:
  **~1.7–2.3 ms (≈430–590 tok/s)** with K5→75%+, attention fixed, comms paid down, step graphed.
- 1000 tok/s ⇒ **0.8–1.0 ms/token effective**. The weight-only roofline is 0.81 ms, so a *single*
  forward per token basically cannot reach 1000 once you add comms+attention. **Spec decode (τ≈2)
  is the multiplier that gets us there** — and/or FP4-mixed experts to cut the weight floor.

## Tuning priorities (ranked by tok/s ROI)
1. **K5 MoE MBU 58→75%+** — cp.async double/triple-buffering (HBM→smem pipelining), the single
   biggest compute lever; ~14.2B of the 22B active mass. (`docs/k5-tuning-roadmap.md`.)
2. **Attention kernels K1/K2 from 2–3%→40%+** — largest *relative* headroom; warp-per-head
   restructure, fuse QK-norm + RoPE, coalesced FP8 loads, better split-K/occupancy for flash-decode.
3. **Fix `decode_step.cu`:458 crash** → get a real fused single-forward ms/token to bench vs vLLM.
4. **TP8 comms** — get a working fast all-reduce (custom one-shot NVLink / CUDA-IPC peer, since
   NVSHMEM won't build) + route-prefetch overlap (`engine/routing/scheduler.rs`).
5. **CUDA-graph the whole step** (`k6_graph_capture.cu`) — delete per-kernel launch + host overhead.
6. **Speculative decode** (EAGLE3 draft, on-device tree verify) — the τ multiplier to break 1000.

## Methodology
- MBU = achieved GB/s / 3350. For B=1 GEMV kernels (K1/K3/K5/lm_head) MBU is the target. For
  flash-decode, watch occupancy (it's latency/occupancy-bound, not BW-bound, at short ctx).
- MFU only matters on multi-token paths (prefill, spec-verify) where GEMMs go to tensor cores.
- Always verify correctness (vs CPU fp32 ref, threshold ~1e-2) before trusting a speedup.
- Bench on the box; compare every result to vLLM's 69.5 tok/s and the roofline floor.

## Agent protocol
- Read this doc first. State the goal back in one line. Cite every number to a file.
- One kernel / one concern per agent to avoid context falloff. Report MBU before→after + the change.
- Don't duplicate: check `docs/k5-tuning-roadmap.md`, `docs/megakernel-b1.md`, `kernels/README.md`.
- Coordinate GPUs: `nvidia-smi` first; don't kill a teammate's process.
