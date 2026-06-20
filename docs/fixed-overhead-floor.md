# The fixed (non-layer-scaling) overhead — the floor that bounds the aggressive target

`depth_reduction.md` correctly notes the ceiling is "bounded by the non-layer-scaling overhead (sampling,
launch, embedding/final-head) that depth reduction cannot remove." This quantifies that floor and adds the
piece that list misses — **the per-step host/scheduler Python** — which is likely the dominant fixed cost,
and is removable by a B=1 fast-path.

## Why it matters now
Spec (~3×) and depth reduction both **divide the per-layer work** (the 94 layers' comms + weight). But a
chunk of the ~11.67 ms step is **per-step, not per-layer** — it doesn't shrink with spec/depth, so as those
levers land it becomes the limiter. If spec gets the per-emitted-token to ~4 ms and a fixed floor of ~1–3 ms
remains, that fixed floor caps the achievable.

## The components (per token, B=1, estimated — `E-attr`/Nsight measures the real split)
| component | estimate | scales with layers? | removable by |
|---|---|---|---|
| **per-step host / scheduler / Python** | **~1–3 ms** | no | **B=1 fast-path** (pinned single-request loop, no admission/block-mgr/continuous-batch scheduler) |
| LM head GEMV (152k×4096, TP8) | ~0.02–0.05 ms | no | (tiny) |
| sampling over 152k vocab | ~0.1 ms greedy / more top-k/p | no | greedy fast-path (argmax, skip the sort/filter pipeline) |
| detokenize (1 token) | ~0.05–0.2 ms (Python) | no | async detok (one token behind the GPU) |
| host↔device sync of the sampled id | ~0.01–0.1 ms | no | on-device sampled-token self-feedback (no PCIe round-trip) |
| embedding lookup | ~0 | no | — |

**The big one is the per-step host/scheduler.** vLLM's continuous-batching scheduler, block manager, and
output processor run **every step even at B=1**, where there's no batch to amortize them — its own low-batch
profile is ~62% CPU (33% API-server, 29% scheduling). That's pure overhead a single user shouldn't pay.

## The lever: a B=1 serving fast-path
- **Scheduler-free pinned loop:** one request, no admission/queue/batch-assembly, no paged-KV block manager
  (contiguous KV — it's one sequence). Removes most of the ~1–3 ms.
- **On-device sampled-token self-feedback:** the sampled id stays on device and feeds the next step — no
  host round-trip per token.
- **Greedy/argmax fast-path + async detok:** skip the full sampler pipeline for greedy; detok one token behind.
- This is what the **cudarc engine** can do natively (a tight decode loop), and what `--enforce-eager` vs
  graphs (in `E-attr`) partially probes on vLLM.

## How to measure it (folds into `E-attr`)
The Nsight `E-attr` trace already separates kernels vs NCCL vs **idle gaps**. The idle gaps between kernels =
this host/launch/sampling floor. Additionally: **TTFT-vs-length intercept** (`E-ttft` B) and a
`--decode 1` vs `--decode 128` comparison isolate per-request vs per-step fixed costs. If the inter-kernel
gaps sum to ~1–3 ms/step, the fast-path is a real lever; if ~0 (graphs hide it well), the floor is comms+kernel.

## Placement in the plan
- **While floor-bound, this is part of the "overhead" the K5 kernels + graphs + fast-path attack** (it's in
  the 60% overhead, distinct from the 26% comms and 14% weight).
- **After spec/depth/comms land**, this fixed ~1–3 ms is the *new* dominant term → the B=1 fast-path (or the
  cudarc engine's tight loop) is what gets the last factor. It's the reason a custom B=1 engine eventually
  beats vLLM even with identical kernels: vLLM pays the per-step scheduler tax a single user shouldn't.
