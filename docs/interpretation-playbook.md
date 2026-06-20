# Results interpretation playbook — measured value → next action

So the GPU agent's data drives the right lever immediately (and my reactions stay consistent). Each row:
**if you measure X → do Y.** Fill the Results Log in `gpu-agent-experiments.md`; this says what each result means.

## E0 — real all-reduce latency (`nccl-tests`) — ✅ RESOLVED: ~16µs → **comms-bound**
Measured: all-reduce@8 ≈**16µs**, all-to-all@8 ≈10µs, all-reduce@2 ≈6.5µs (`config-sweep.md`). That's the
**≥4–5µs comms-bound branch** below → **NCCL comms tuning is now the #1 lever** (target 16→4–8µs), then
fp8+TP8 (block-64 requant), then the kernel/overhead gap, then spec. See `results-reaction-01.md`.
| 8–16 KB all-reduce latency | meaning | do next |
|---|---|---|
| ≤ ~2 µs | weight-bound | int4 + kernel first |
| ~3–4 µs | mixed | both levers pay |
| **≥ ~4–5 µs (← measured 16µs)** | **comms-bound** | **NCCL tuning (LL/one-shot) #1** → route-prefetch + spec; int4 helps less |
→ set `src/inferutil/hardware.py: collective_latency_s = 16e-6` and re-run `predict_matrix.py 0.46 H100-SXM-80GB 16`.

## E1 — FP8+EP engine baseline (real TTFT/TPOT/tok-s)
Back out the true terms: `weight_ms ≈ ideal_weight/e`, `comms_ms ≈ TPOT − weight_ms − kv_ms`. Then:
| observation | meaning | do next |
|---|---|---|
| tok/s ≪ predicted EP ~94 | CPU/scheduler/launch overhead dominates | **E3 CUDA-graph** + serving fast-path first |
| tok/s ≈ predicted EP ~94, comms big | comms-bound, EP all-to-all hurting | push to **TP-heavier** (block-64 requant → TP8) + prefetch |
| weight term dominates | weight-bound | **int4 (E7)** is the lever; confirm kernel `e` (E4) |
| one GPU hot in `dmon` | EP routing imbalance | **co-activation placement** (`ep-placement-for-b1.md`) or go TP8 |
**The headline:** is real e2e closer to TP8 (~261) or EP (~94)? The 192-crash forces EP today, so the gap is
the prize for unblocking TP8 (block-64 requant).

## E4 — kernel microbench + Nsight
| result | do next |
|---|---|
| down-proj v2 `e` > winner | fold v2 into `k5_experts_warp.cu` |
| down-proj no better, Nsight = occupancy-bound | the per-slot fix missed; try sub-warp split-K (I'll design from the Nsight numbers) |
| Nsight = DRAM-bound at e≈0.46 | kernel is near its ceiling; further gains come from **int4**, not tuning |
| **int4 speedup ≈ 2×** | unpack is free → int4 experts are a real ~1.3–1.5× e2e lever → push E7 |
| **int4 speedup < ~1.5×** | nibble unpack is issue-bound → int4 not worth it on H100 (revisit on Blackwell FP4) |

## E6 — n-gram spec / E9 — self-spec
| result | do next |
|---|---|
| n-gram acceptance > ~25%, realized tok/s up at k=2–3 | ship n-gram for structured prompts |
| n-gram low (general prose) | check **E9**: if shallow-pass top-1 agreement clears break-even at small L_d → self-spec; else → trained MTP head |
| any spec: realized tok/s **down** despite high acceptance | the MoE verify-tax bit — shrink the tree (`spec-decode-moe-tax.md`) |

## E7 — int4 engine / E8 — route prediction
| result | do next |
|---|---|
| int4 ckpt exists + quality gate passes + tok/s ↑ | adopt int4 experts (gate on the kernel result from E4 first) |
| E8 DirectProxy accuracy ≫ random | wire `scheduler.rs` prefetch; if comms-bound (E0) this is a top lever |
| E8 ≈ random | route prediction won't help; drop prefetch, focus byte/kernel levers |

## The single decision spine
**E0 decides comms-vs-weight → that decides the whole lever order.** Everything else refines within that
branch. Run E0 first (sub-minute, no model load); it's the highest information-per-second measurement we have.
