# B=1 fast-path decode loop — the design that removes the per-step scheduler tax (for the cudarc engine)

`fixed-overhead-floor.md` showed the floor that survives spec/depth/comms is the **per-step host/scheduler
Python (~1–3 ms)** — vLLM runs its continuous-batching scheduler, block manager, and output processor every
step even at B=1, where there's nothing to amortize them against. This is the concrete decode-loop design
that removes it — the reason a custom B=1 engine (the team's `engine/` cudarc crate) eventually beats vLLM
*with identical kernels*. It is the *last* factor (run it after spec/comms/kernels), but it's the part vLLM
structurally can't give a single user.

## The principle
At B=1 there is exactly one sequence, forever. So: **no scheduler, no admission, no batch assembly, no paged
block manager, no per-step host round-trip.** The whole decode step is one captured CUDA graph; the host loop
does nothing but replay it and, one step behind, detokenize.

## The loop
```
# one-time setup (per request):
KV = contiguous_kv_buffer(max_len)          # not paged — it's one sequence
graph = capture_decode_step()               # ONE CUDA graph for the whole 94-layer step + sample
tok_dev = device_scalar()                   # the current token id, stays on device

# steady state — host does ~nothing per token:
loop:
    graph.replay(stream)                     # K1→K2→K3(+AR)→K4→K5(+AR) ×94 → norm → lm_head → argmax → tok_dev
    # tok_dev is written by the in-graph sampler; the graph's NEXT replay reads it as input
    # -> NO host->device feedback of the sampled id (the on-device self-feedback)
    async_copy(tok_dev -> host_ring)         # async, non-blocking
    detok_thread.consume(host_ring)          # detokenize ONE step behind, off the critical path
    if eos_on_device(tok_dev): break         # EOS check in-graph or on the async copy
```
The critical path per token = **one graph replay** (the kernels + the 2×94 in-graph all-reduces). Host cost
≈ the replay launch (~µs) + an async copy. No scheduler, no sync, no Python per layer.

## The four things that kill the fixed floor
1. **Everything in one CUDA graph**, including the sampler — so there's no per-kernel/per-layer launch, no
   Python between layers. (vLLM graphs the layers but still runs host scheduler/output code around the step.)
2. **On-device sampled-token self-feedback** — the argmax writes `tok_dev` on device and the next replay reads
   it; no `D→H→D` round-trip of the token id per step (that round-trip is ~10–100 µs of exposed latency).
3. **Greedy/argmax fast-path** — for greedy B=1, the sampler is a single argmax over the 152k logits, not the
   top-k/top-p sort/filter pipeline. (Sampling temp>0: a fused top-k in-graph.)
4. **Async detokenize, contiguous KV** — detok runs one token behind on a host thread; KV is a flat per-seq
   buffer (no block-table indirection, which also helps the K2 flash-decode coalescing).

## In-graph all-reduce (ties to `comms_floor.md`)
The 2/layer all-reduces must be **inside** the graph as device-initiated NVLS/multimem one-shot kernels (the
`comms_floor.md` lever), not host-launched NCCL calls — otherwise the graph breaks at every collective and the
launch/sync overhead returns. The fast-path and the comms-latency lever are the **same capture**: one graph,
device-side collectives, on-device feedback.

## Expected payoff + placement
- Removes the ~1–3 ms/step host floor → at a post-spec ~4 ms/token that's a further **~1.3–2×** on the residual.
- **Order:** last. While floor-bound by comms+kernels, this is part of the 60% overhead the K5 kernels + graph
  capture already attack; its *standalone* value shows up once comms (E0b), kernels (K5), and spec have landed
  and the per-step host cost is the residual. `E-attr`'s idle-gap measurement sizes it; `--enforce-eager` vs
  graphs brackets it.
- **This is the cudarc engine's reason to exist:** identical kernels, but a tight single-request loop with
  on-device feedback and in-graph collectives — the per-step tax a single user shouldn't pay, which vLLM's
  general-purpose scheduler can't avoid.

## Validation without the engine
`E-attr` (Nsight) → sum of inter-kernel idle gaps per step = this floor. `--decode 1` vs `--decode 128`
(per-request vs per-step fixed cost). If the gaps are ~1–3 ms, build the loop; if ~0, vLLM's graphs already
capture enough and the floor is comms+kernels (then this lever is moot and the cudarc engine's win is only the
kernels).
