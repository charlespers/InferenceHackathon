# TTFT is 777 ms — a 20–300× gap nobody's looked at (the other half of single-user latency)

The work so far chased decode tok/s (TPOT). But the measured **TTFT = 776.7 ms (bf16-TP8) / 631 ms (fp8-EP8)**
(`config-sweep.md`, and `ab_baseline.json` ttft 759 ms) is the *first* thing a single user experiences — they
wait the full TTFT before token one. And it's **wildly above the physics**, so it's a big, untouched lever.

## The gap
TTFT = prefill of the prompt + the first decode step. For the bench's ~512-token prompt, prefill is
**compute-bound GEMM** (M=512), not the bandwidth-bound GEMV decode is:
| prompt | prefill compute (60% MFU) | + prefill comms (188 AR, batched) | physics TTFT | measured |
|---|---|---|---|---|
| 512 | ~2.3 ms | ~3 ms | **~5–10 ms** | **777 ms** (~80–150×) |
| 8192 | ~37 ms | ~3 ms | ~40 ms | (untested) |

**~770 ms of the 777 ms is overhead, not prefill physics** — the same floor story as decode, but even more
extreme (the prefill is a *single* forward, so per-request fixed costs aren't amortized at all).

## What the 770 ms likely is (ranked)
1. **Prefill not CUDA-graph-captured / runs eager.** Decode graphs capture the B=1 shape; the prefill shape
   (M=prompt_len) is dynamic, so vLLM often runs prefill **eager** → 94 layers × ~15 kernels × Python dispatch
   + per-kernel launch, uncaptured. At ~hundreds of µs/layer of host overhead × 94 → hundreds of ms. Prime suspect.
2. **First-request / cold-path costs** even after `--warmup`: per-request KV-block allocation, the prefill
   attention kernel's first dynamic-shape compile (torch.compile guard miss), sampler setup.
3. **Prefill kernel inefficiency** — the prefill MoE/attn kernels at low MFU (the prefill analogue of the
   decode-kernel inefficiency; cf. `kernels/prefill_attn.cu`, `prefill_moe.cu` the team is building).
4. **TTFT includes the first decode step** (~11.67 ms) — negligible vs the 770 ms.

## The levers (biggest first)
1. **Prefix caching (`--enable-prefix-caching`)** — the single biggest TTFT win, and *free* for the bench's
   highly-repetitive prompt and for any chat with a shared system prompt: a cache hit skips the prefill of the
   shared span entirely → TTFT → ~the first decode step (~10 ms). **For repeated/structured prompts this is a
   ~50–100× TTFT cut.** Test it first.
2. **Graph/compile the prefill path** (or chunked prefill with a captured chunk shape) so prefill isn't eager
   — attacks suspect #1 directly.
3. **Efficient prefill kernels** (the team's `prefill_*.cu`) — FlashAttention-3 prefill + a tuned MoE GEMM;
   prefill is compute-bound so this is a different kernel from the decode GEMV.
4. **Chunked prefill tuning** — at B=1 single-shot is fine for short prompts; for long prompts chunk to keep
   attention working sets in SRAM (the crossover is ~8–16K, see `b1-latency-architecture.md` §TTFT).

## Attribution experiment → `E-ttft`
```bash
# A) prefix caching on/off (the headline lever; repeat the SAME prompt so turn-2 is a cache hit):
vllm serve /alloc/data/Qwen3-235B-A22B --tensor-parallel-size 8 --dtype bfloat16 --enable-prefix-caching ...
#    measure TTFT on a fresh prompt vs an immediate repeat -> the cache-hit TTFT floor.
# B) TTFT vs prompt length (separates fixed cost from per-token prefill):
for P in 16 128 512 2048 8192; do python3 bench/measure.py --base ... --ctx $P --decode 1 ; done
#    intercept = the fixed ~per-request/eager overhead; slope = real prefill per-token. If the intercept is
#    huge and flat, it's #1/#2 (eager/cold path), not prefill compute.
# C) --enforce-eager vs graphs on TTFT (does graph capture even touch prefill?).
```
**Decision:** flat huge intercept → prefill is eager/cold-path bound → graph/compile the prefill + prefix
caching. Slope dominant → prefill kernel efficiency (the team's `prefill_*.cu`). Either way **prefix caching
is the immediate ~50–100× win for repeated/structured prompts** and should ship now.

## Why this matters
Single-user perceived latency = TTFT + N·TPOT. At N=128 decode tokens: TTFT 777 ms + 128×11.67 ms = 777 +
1494 = **2271 ms, of which TTFT is 34%.** Halving TPOT (hard) saves 747 ms; **prefix-caching TTFT to ~10 ms
saves 767 ms — bigger, and easy.** TTFT has been the blind spot; it's the cheapest big single-user-latency win.
