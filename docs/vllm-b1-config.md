# vLLM B=1 config checklist — flip the throughput defaults for single-user latency

vLLM's defaults are tuned for **throughput** (many concurrent requests). At B=1 several are wrong, and a few
config knobs attack the fixed-overhead floor *without* a custom engine (refining `fixed-overhead-floor.md`,
where I said the per-step host tax "needs the cudarc engine" — some of it is config-reachable). This is the
consolidated launch, each flag with the *why* and the lever it serves.

## The launch (bf16 pure-TP8 + the cheap wins)
```bash
VLLM_USE_V1=1 vllm serve /alloc/data/Qwen3-235B-A22B \
  --served-model-name q --tensor-parallel-size 8 --dtype bfloat16 \
  --enable-prefix-caching \
  --speculative-config '{"method":"eagle3","model":"nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys","num_speculative_tokens":5,"draft_tensor_parallel_size":8}' \
  --max-num-seqs 1 --max-model-len 8192 --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.90 --disable-log-requests --trust-remote-code
```

## Flag-by-flag (why, and the lever)
| flag | default (throughput) | B=1 (latency) | why / lever |
|---|---|---|---|
| `--tensor-parallel-size 8` (no `--enable-expert-parallel`) | EP common | **pure TP8** | EP busiest-rank 2.53× at B=1 (measured); TP8 avoids it (`b1-tp8-moe-rearchitecture-h200.md`) |
| `--quantization fp8` (dynamic) **or** bf16 | — | fp8 prize / bf16 safe | fp8 ½-weight, dynamic dodges the 192-crash (`E2b`); weight is only 14% while floor-bound |
| `--speculative-config …` | none | **EAGLE3/n-gram** | the #1 lever — amortizes the floor over τ (`why-spec-wins.md`); `draft_tp=8` (`eagle3-draft-tp.md`) |
| `--enable-prefix-caching` | off | **on** | ~50–100× TTFT on chat/repeat (`ttft-analysis.md`) — the cheapest big win |
| `--max-num-seqs 1` | 256 | **1** | the scheduler/block-manager does batch machinery sized for 256 even with 1 seq → cut it to 1 (host floor) |
| `--no-enable-chunked-prefill` | on (recent) | **off for short prompts** | chunking re-launches prefill chunks → adds TTFT latency at B=1 short-prompt; single-shot is faster (`ttft-analysis.md`). (Keep ON for long >16K prompts.) |
| `VLLM_USE_V1=1` | varies | **V1** | the V1 engine has lower per-step Python/scheduler overhead — attacks the fixed floor directly |
| `--disable-log-requests` (+ minimal logging) | logs | **off** | per-request/token logging is host work a single stream shouldn't pay |
| CUDA graphs | on | **keep on** (NOT `--enforce-eager`) | graphs remove per-layer launch; only use `--enforce-eager` to *diagnose* the floor (`E-attr`) or if spec+graphs is unstable (then re-enable) |
| `--gpu-memory-utilization` | 0.9 | **0.85–0.90** | at B=1 the KV reserve can be smaller (one sequence); leaves room for the draft head + prefix cache (no latency effect, just headroom) |

## The deprecated/uncertain knob (verify on the box)
- **`--num-scheduler-steps N`** (multi-step scheduling, V0): runs N decode steps per scheduler call →
  amortizes the per-step host/scheduler tax by N. **At B=1 this is the config-level fix for the fixed floor.**
  *Caveats:* it's a V0 feature (may not exist/behave in V1), and **may not compose with spec decode** (spec
  needs the scheduler each round). Test: V0 + `--num-scheduler-steps 4` *without* spec → does the host floor
  drop? If yes and you can't get both, it's a spec-vs-multistep tradeoff (spec usually wins — it amortizes more).

## How to verify each is helping (folds into `E-attr`)
- `--max-num-seqs 1`, `VLLM_USE_V1=1`, `--num-scheduler-steps`: watch the **inter-kernel idle gaps** in the
  Nsight trace (the host floor) shrink. `--enforce-eager` vs graphs brackets the launch component.
- `--enable-prefix-caching`: cold vs same-prompt-repeat TTFT (`run_bench6.sh`).
- `--no-enable-chunked-prefill`: TTFT on a short (512) vs long (16K) prompt — single-shot wins short, loses long.

## Net
The cheap wins are **config flags, not code**: TP8 + fp8 + spec(draft_tp=8) + prefix-cache + max-num-seqs=1 +
V1 + no-chunked-prefill(short) + no-logging. That's the `run_bench_best.sh` config plus the host-floor knobs.
What remains for the **cudarc engine** is only the residual the flags can't reach: e→1 kernels, in-graph
device-initiated comms, and the truly scheduler-free loop (`b1-fast-path-design.md`) — the last 750→2000 of
`absolute-ceiling.md`.
