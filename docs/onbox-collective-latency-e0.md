# E0 — real collective latency on-box (8×H100 NVSwitch) — 2026-06-20

Measured with `nccl-tests` (`/workspace/nccl-tests`) via the SSH window. Settles the comms-vs-weight
question the team's model left open (it used a *guessed* `collective_latency_s = 5 µs`).

## E0 — small-message latency (the per-layer collective cost at B=1)
| op | 8 KB | 16 KB | vs modeled 5 µs |
|----|-----:|------:|----:|
| **all-reduce** (TP, 2×/layer) | **~35 µs** | ~35 µs | **7× higher** |
| all-to-all (EP dispatch/combine) | ~125 µs | ~128 µs | ~25× higher |

**Decode comms budget (stock NCCL):** 188 all-reduces/token × ~35 µs = **~6.6 ms/token of pure
all-reduce latency** — that alone exceeds the measured ~8.6 ms TPOT. Two consequences:
1. The model's `collective_latency_s = 5 µs` **under-counts comms ~7×**; with stock NCCL, comms is the
   **dominant** floor term (not the ~26% the 5µs model implied).
2. The real engine's TPOT (8.6 ms) is *below* the stock-NCCL comms budget → it must already use a
   **custom/fused all-reduce** faster than stock ring. Get its effective value via Nsight (E-attr); the
   stock ceiling is ~35 µs.
- all-to-all ~125 µs reconfirms **EP is bad at B=1** (its dispatch/combine is 3.5× the TP all-reduce).

## E0b — comms-tuning sweep (does env tuning cut the 35 µs?) → NO. env-comms is DEAD.
all-reduce @ 8–16 KB, identical within noise across every variant:

| variant | 8 KB | 16 KB |
|---------|-----:|------:|
| stock NCCL | 33.3 | 33.1 |
| `NCCL_PROTO=LL` | 33.0 | 32.6 |
| `NCCL_NVLS_ENABLE=1` (NVLink-SHARP) | 32.8 | 32.7 |
| LL + NVLS + `MAX_NCHANNELS=2` | 33.8 | 32.6 |

**Env tuning does not move the ~33 µs small-message floor.** So the comms lever is **structural, not
environmental** — in priority order (blueprint §4):
1. **Fuse all-reduce + residual + RMSNorm** (TRT-LLM `RESIDUAL_RMS_NORM` / userbuffers) — kill the HBM
   round-trip between collective and norm.
2. **One-shot custom all-reduce** (vLLM `--disable-custom-all-reduce` off / TRT-LLM) instead of NCCL ring.
3. **Overlap the collective with the next GEMM** (Flux-style) / in-kernel NVSHMEM put-barrier (the team's
   `nvshmem_inkernel_bench` measured ~17 µs A2A, 3.5× better than NCCL — structural beats env).

## Recommendation
- Treat `src/inferutil/hardware.py: collective_latency_s` as **engine-dependent**: stock NCCL ≈ 35 µs
  (use for a stock-NCCL model); the real engine is lower (custom AR) — pin it from Nsight (E-attr), not
  from the 5 µs ballpark. Either way comms is a first-order floor term → the structural-comms work
  (fused AR+norm, one-shot, in-kernel) is a top lever, and `NCCL env tuning is confirmed dead`.
