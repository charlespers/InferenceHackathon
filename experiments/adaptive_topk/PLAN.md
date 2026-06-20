# Confidence-Adaptive Top-K Expert Routing — Qwen3-235B-A22B on vLLM

B=1 decode latency optimization. Read **fewer expert weights from HBM** on
tokens whose router softmax is already concentrated, instead of always loading
top-8 of 128 experts per layer. Expert-weight HBM reads are ~66% of B=1 decode
latency, so this is a direct byte saving — the dropped experts are never loaded.

---

## 1. Integration point (file + function)

**Hook:** `FusedMoE.custom_routing_function`
(`vllm/model_executor/layers/fused_moe/layer.py`)

vLLM threads this attribute straight through the whole MoE path, unchanged:

```
Qwen3MoeSparseMoeBlock.forward
  -> self.experts(hidden_states, router_logits)             # FusedMoE.forward
  -> torch.ops.vllm.moe_forward -> FusedMoE.forward_impl
  -> quant_method.apply(..., custom_routing_function=self.custom_routing_function,
                        top_k=self.top_k, ...)
  -> UnquantizedFusedMoEMethod.forward_cuda
  -> FusedMoE.select_experts(... custom_routing_function ...)
        if custom_routing_function is not None:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states, router_logits, topk=top_k, renormalize=...)
```

So a custom routing function fully owns `(topk_weights, topk_ids)`. We keep
`top_k = 8` (the kernel requires a rectangular `[M, 8]` `topk_ids`) but stamp the
low-confidence trailing columns with a **DROP sentinel** so the kernel skips
those experts.

**Installer:** we monkeypatch `FusedMoE.__init__` to set
`custom_routing_function` on every Qwen3 MoE layer at construction time
(`install()` / `register()` in `vllm_adaptive_moe.py`). This runs in the engine
**and worker** processes, which is where the model is actually built under TP/EP.

Draft code: `experiments/adaptive_topk/vllm_adaptive_moe.py`

---

## 2. Mechanism — why the HBM saving is real (not zero-weighting)

vLLM's fused MoE kernel does work **per distinct expert id that appears in
`topk_ids`**. `moe_align_block_size` buckets tokens by expert; for each
(expert, block) the Triton kernel loads that expert's gate/up/down weights from
HBM and runs the GEMM. If an expert id never appears, no block is created and its
weights are never read.

With `--enable-expert-parallel` (the target config), vLLM remaps `topk_ids`
through `expert_map`; non-local experts become `-1`, and the Triton kernel does:

```python
off_experts = tl.load(expert_ids_ptr + pid_m)
if off_experts == -1:
    write_zeros_to_output(...)   # skip the GEMM AND the weight load
    return
```

**Therefore:** set the dropped columns to a global expert id that maps to `-1` on
this rank → the kernel early-returns for that block → that expert's weights are
**not loaded from HBM**. At B=1 (M=1) a token routed to 4 real experts loads
exactly 4/8 of the per-layer expert bytes. This is the entire trick, achieved
without touching any kernel.

### Per-token vs fixed k — the hard constraint (honest)

The fused kernel needs a **fixed rectangular** `topk_ids` of shape `[M, top_k]`;
true ragged per-token k is **not** supported. We work *with* that: shape stays
`[M, 8]`, and we make some columns map to `-1` (skipped). At **B=1 (M=1)** there
is exactly one token, so "per-token variable k" and "per-step k" are identical —
this is the regime where the approach is both simplest and most effective.

### Two regimes (auto-detected per layer)

- **A — `expert_map` present (EP on, the user's config):** drop columns get a
  global expert id that maps to `-1` on this rank → kernel skips the load.
  **Clean, full saving.** This is the intended path.
- **B — no `expert_map` (TP-only / single shard):** the CUDA
  `moe_align_block_size` does **not** tolerate raw `-1` (out-of-bounds index), so
  we instead **duplicate the row's top expert id** into the dropped columns with
  weight 0. No *new* distinct expert is introduced, so no extra weights load; the
  saving is real but slightly less clean (the duplicated expert gets a marginally
  larger aligned block). Documented as the fallback.

---

## 3. Launch command

### Build the model with `--enable-expert-parallel` (regime A, full saving):

Preload the patch, then start the OpenAI server:

```bash
ADAPTIVE_TOPK_ENABLE=1 \
ADAPTIVE_TOPK_K=4 \
ADAPTIVE_TOPK_THRESH=0.9 \
ADAPTIVE_TOPK_DEBUG=1 \
python -c "import experiments.adaptive_topk.vllm_adaptive_moe as m; m.install(); \
           import runpy; \
           runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')" \
  -- \
  --model /alloc/data/Qwen3-235B-A22B \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-seqs 1 \
  --dtype bfloat16 \
  --port 8000
```

### Preferred: vLLM general plugin (patches engine + every worker cleanly)

Add to `experiments/adaptive_topk/pyproject.toml`:

```toml
[project.entry-points."vllm.general_plugins"]
adaptive_topk = "experiments.adaptive_topk.vllm_adaptive_moe:register"
```

`pip install -e experiments/adaptive_topk`, then:

```bash
ADAPTIVE_TOPK_ENABLE=1 ADAPTIVE_TOPK_K=4 ADAPTIVE_TOPK_THRESH=0.9 \
VLLM_PLUGINS=adaptive_topk \
vllm serve /alloc/data/Qwen3-235B-A22B \
  --tensor-parallel-size 8 --enable-expert-parallel --max-num-seqs 1
```

The plugin route is more robust than `-c` preload because vLLM imports plugins
in worker subprocesses too (where the model is actually built under TP/EP).

---

## 4. Configurable env vars

| Var | Default | Meaning |
|-----|---------|---------|
| `ADAPTIVE_TOPK_ENABLE` | `0` | Master on/off. Off ⇒ exact baseline top-8. |
| `ADAPTIVE_TOPK_K` | `4` | Reduced k used when confident (1..8). |
| `ADAPTIVE_TOPK_THRESH` | `0.9` | Reduce only if top-K softmax mass > this. |
| `ADAPTIVE_TOPK_MIN_LAYER` | `0` | Only adapt at/after this layer index. |
| `ADAPTIVE_TOPK_DEBUG` | `0` | Print patch info + collect drop-rate stats. |

Policy per token: softmax the 128 router logits; if the top-`K` mass exceeds
`THRESH`, keep `k=K` and drop the other `8-K`; otherwise keep full `k=8`.

---

## 5. Expected byte savings as a function of k

Let experts be ~66% of B=1 decode bytes; attention + non-expert linears the
rest. For a reduced token using k of 8 experts, expert bytes scale by k/8.

| k | expert bytes | e2e decode bytes vs baseline | e2e speedup (mem-bound, ceiling) |
|---|--------------|------------------------------|----------------------------------|
| 8 | 1.00× | 1.00× | 1.00× |
| 6 | 0.75× | 1 − 0.66·0.25 = **0.835×** | ~1.20× |
| 4 | 0.50× | 1 − 0.66·0.50 = **0.67×**  | ~1.49× |
| 2 | 0.25× | 1 − 0.66·0.75 = **0.505×** | ~1.98× |

Those are **ceilings** that apply only to tokens actually reduced. Realized gain:

```
e2e_factor = 1 − 0.66 · (1 − k/8) · reduced_fraction
```

So k=4 on 60% of tokens ⇒ 1 − 0.66·0.5·0.6 = **0.80×** bytes ≈ **~1.25× decode
tok/s** ceiling. `stats_snapshot()` reports the realized `reduced_fraction` and
`avg_k` to plug in here. Real speedup will be below the ceiling: fixed kernel
launch/align overhead and EM padding don't shrink, and attention is unaffected.

---

## 6. Risks / unknowns

1. **Biggest risk — does the EP kernel actually skip the dropped block, or is
   block/EM padding fixed?** The `off_experts == -1` early-return skips the GEMM
   and the weight load, but `moe_align_block_size` may still pad `EM` to a fixed
   size, so the *grid* doesn't shrink even though per-block work does. Net byte
   saving is real (the load is the early-returned part), but **the realized
   speedup must be measured**, not assumed from the byte math. This is the #1
   thing the A/B test exists to settle.
2. **Quality drift.** Dropping experts changes outputs on borderline tokens.
   Mitigated by a high `THRESH` (0.9) so only already-concentrated tokens reduce;
   measured by output-equality in the A/B.
3. **Regime B weakness.** Without `expert_map`, the duplicate-id fallback yields a
   smaller, messier saving. Use regime A (EP on) for the real result.
4. **CUDA graph capture.** vLLM may capture decode in a CUDA graph; a Python
   `custom_routing_function` runs in eager fallback or is traced once. Verify the
   routing fn is on the captured path (or run `--enforce-eager` for the A/B to
   isolate the effect first).
5. **Sentinel correctness.** If a rank has *no* non-local expert (small EP world),
   regime A degrades to regime B automatically — handled, but worth asserting in
   logs via `ADAPTIVE_TOPK_DEBUG=1`.
6. **renormalize semantics.** We renormalize the surviving k weights so the MoE
   output magnitude matches a genuine top-k (config `norm_topk_prob`). If the
   model was trained without renorm, match that instead.

---

## 7. A/B test plan

Two servers (or two runs), identical except `ADAPTIVE_TOPK_ENABLE`.

**Arm 0 (baseline):** `ADAPTIVE_TOPK_ENABLE=0` → exact top-8.
**Arm 1 (adaptive):** `ADAPTIVE_TOPK_ENABLE=1 K=4 THRESH=0.9`.

For each arm, on ~8 fixed prompts (reuse `tools/routing_predict.py::PROMPTS`),
`temperature=0`, `max_tokens=128`:

1. **Decode throughput:** median **decode tok/s** (exclude prefill: use the
   server's per-request decode timing or measure inter-token latency). Report
   median + p90 over prompts. This is the headline metric.
2. **Output equality:** greedy-decode token-id equality vs baseline per prompt:
   exact-match rate, first-divergence token index, and a semantic check
   (embedding cosine or a quick LLM-judge) on any prompt that diverges. Reuse the
   token-level parity check already in this repo's bench harness.
3. **Realized reduction:** `vllm_adaptive_moe.stats_snapshot()` →
   `reduced_fraction`, `avg_k`. Plug into the §5 formula to predict expected
   speedup and compare to measured — if measured ≪ predicted, risk #1 (EM
   padding / CUDA graph) is the cause.
4. **Threshold sweep:** repeat arm 1 at `THRESH ∈ {0.85, 0.9, 0.95}` and
   `K ∈ {2,4,6}` to plot the speed-vs-equality Pareto front.

**Pass criteria (suggested):** ≥10% decode tok/s improvement at K=4/THRESH=0.9
with ≥95% greedy token-exact match and no semantic regression on the 8 prompts.

---

## 8. Honest bottom line: fused path vs custom B=1 expert loop

**The fused path CAN realize the saving** — but only via the EP `-1` /
`expert_map` early-return (regime A), and the realized *speedup* depends on
whether block/EM padding and CUDA-graph capture let the skipped block translate
into wall-clock time (risk #1). The **byte read** genuinely decreases; the
**latency** decrease is the empirical question.

**If measurement shows the fused kernel's fixed overhead/padding eats the gain,**
the fallback is a **custom non-fused B=1 expert loop**: override
`Qwen3MoeSparseMoeBlock.forward` to, at M==1, gather the (variable) k selected
expert ids and run a plain Python loop of `k` dense MLPs (gate_up → SiLU → down),
skipping all unselected experts entirely. Tradeoff:

- **Pro:** loads *exactly* k experts' weights, honors true per-token variable k,
  zero kernel/padding overhead for skipped experts — the cleanest possible
  saving.
- **Con:** loses the fused kernel's launch-coalescing; only worth it at very low
  batch (B=1) where compute has slack and the loop is memory-bound anyway; must
  re-implement EP all-to-all / weight gather by hand; bypasses vLLM's quantized
  expert kernels. Higher engineering + correctness risk.

**Recommendation:** ship the `custom_routing_function` (regime A) first — lowest
risk, no kernel changes — measure with the A/B, and only build the custom B=1
loop if the fused path's padding/graph overhead is shown to swallow the saving.

---

## CUDA-graph viability

**Verdict (the make-or-break question): the routing fn DOES run per decode step
under CUDA graphs, but the fused EP `-1` skip is NOT guaranteed to convert into a
wall-clock win at B=1 — the byte read drops, the grid does not.** Use the fused
plugin for the FP8 + graphs A/B first; if the measured speedup is swallowed by
fixed kernel overhead, switch to the `custom_forward_fallback.py` B=1 expert loop.
Reasoning below (all from vLLM ~v0.10.x source).

### (a) Does our Python `custom_routing_function` run per decode step under graphs? — YES

`FusedMoE.forward` does not call `forward_impl` directly. It dispatches through a
**registered custom op**:

```python
# vllm/model_executor/layers/fused_moe/layer.py
return torch.ops.vllm.moe_forward(hidden_states, router_logits, self.layer_name)[...]

def moe_forward(hidden_states, router_logits, layer_name):
    self = get_forward_context().no_compile_layers[layer_name]
    return self.forward_impl(hidden_states, router_logits)   # -> select_experts -> custom_routing_function

direct_register_custom_op(
    op_name="moe_forward", op_func=moe_forward,
    fake_impl=moe_forward_fake, ...)              # fake_impl => OPAQUE to Dynamo
```

Two consequences that together answer (a):

1. **Dynamo treats `moe_forward` as opaque.** Because it is a custom op with a
   `fake_impl`, `torch.compile`/Dynamo does not trace into `forward_impl` /
   `select_experts` / `custom_routing_function`. The FX graph keeps a single
   `moe_forward` node; at runtime the *real* Python body runs eagerly **every
   call**. So our `torch.softmax` / `torch.topk` / boolean-mask / advanced-index
   logic is never frozen or constant-folded — it executes each decode step,
   reading fresh router logits. Data-dependent `.nonzero()` / masked assignment do
   NOT crash compilation either, precisely because they sit behind the opaque op.

2. **Routing is OUTSIDE the captured graph region.** vLLM's default piecewise
   mode splits the graph only at attention ops:
   `splitting_ops` defaults to `_attention_ops =
   ["vllm.unified_attention", "vllm.unified_attention_with_output",
   "vllm.mamba_mixer2"]` (`vllm/config/compilation.py`,
   `set_splitting_ops_for_v1`). `moe_forward` is **not** a splitting op, but it is
   a custom op the compiler cannot see through, so the MoE body (router + kernel)
   executes via the eager custom-op call regardless of piecewise vs full capture.
   The routing function is effectively *outside* the replay-frozen region.

   → **The router runs live every step and emits fewer distinct experts as
   intended. Graph capture does not bypass it.** (a) is a clean PASS.

### (b) Does `moe_align_block_size` pad the grid so skipped experts don't reduce wall-clock? — LARGELY YES (this is the real limiter)

```python
# vllm/model_executor/layers/fused_moe/moe_align_block_size.py
max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
max_num_m_blocks = cdiv(max_num_tokens_padded, block_size)
```

`EM` / the grid is sized from a **fixed** capacity, not from the count of distinct
experts that actually appear. Dropping experts to `-1` does **not** shrink the
grid. The Triton kernel still launches over the padded block set; for a `-1`
block it hits `if off_experts == -1: write_zeros; return` — it **skips the HBM
weight load and the GEMM** (the expensive part, the whole point), but it does NOT
remove the block iteration / launch / align scaffolding. So:

- **Byte read: genuinely lower** (skipped experts' gate/up/down never loaded).
- **Grid / launch / align overhead: unchanged.**

At **B=1 (M=1)** the grid is already tiny (numel = top_k = 8), so the absolute
overhead is small — but so is the absolute saving, and there is no guarantee the
saved bytes dominate the fixed per-layer kernel floor. This is exactly why the
weight-bound (FP8 + graphs) regime is required: only there are expert-weight
bytes a large enough fraction of per-layer time for the skipped load to outrun
the fixed overhead.

### (c) Does capture succeed with data-dependent `topk_ids` (the -1 positions vary per token)? — YES

The varying `-1` positions live **inside** the opaque `moe_forward` op and the
non-captured eager region, not in the FX graph. CUDA-graph capture records fixed
*tensor addresses and kernel launch sequence*, not data values; the Triton MoE
launch sequence at B=1 is shape-stable (always `[1, 8]` `topk_ids`), and the
per-step `-1` pattern only changes tensor *contents*, which graph replay re-reads
from the same persistent buffers. **No graph-break, no re-capture per token.**
Capture succeeds.

### Net verdict

| Question | Answer |
|----------|--------|
| Router runs per step under graphs? | **Yes** (opaque custom op → eager body each call) |
| Routing inside frozen captured region? | **No** — outside; not bypassed |
| Skipped `-1` experts reduce HBM bytes? | **Yes** — load + GEMM early-returned |
| Skipped experts shrink the grid / fixed overhead? | **No** — `EM` padded to fixed capacity |
| Capture succeeds with varying `-1` positions? | **Yes** — data lives in eager region |
| **Fused-skip realizes a real latency win under graphs?** | **Plausible but NOT guaranteed — measure.** Byte saving is real; wall-clock win depends on byte-time dominating fixed kernel/align floor, which only holds in the weight-bound FP8 regime. |

**Decision for the next A/B:** run the **fused-skip plugin** (`vllm_adaptive_moe`)
in the weight-bound config below — it is lower risk and supports quantized
experts. Use `stats_snapshot()` to confirm `reduced_fraction`/`avg_k`, then
compare measured decode tok/s to the §5 ceiling. **If measured ≪ predicted**
(i.e. EM padding + fixed kernel floor ate the saving), switch arm 1 to the
**custom-forward fallback** (`custom_forward_fallback.py`), which bypasses the
fused kernel and loads *exactly* k experts at B=1 — note that reference loop
currently handles **unquantized** expert weights, so for an FP8 layer either run
it on a bf16 build or extend `_expert_mlp` to dequant first.

### Recommended FP8 + CUDA-graphs launch command (next A/B)

CUDA graphs ON = simply **omit `--enforce-eager`** (graphs are the default).

```bash
# Arm 1 (adaptive, fused-skip) — weight-bound: FP8 + CUDA graphs ON
ADAPTIVE_TOPK_ENABLE=1 ADAPTIVE_TOPK_K=4 ADAPTIVE_TOPK_THRESH=0.9 \
ADAPTIVE_TOPK_DEBUG=1 VLLM_PLUGINS=adaptive_topk \
vllm serve /alloc/data/Qwen3-235B-A22B \
  --quantization fp8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-seqs 1 \
  --no-enable-prefix-caching \
  --port 8000
# (Do NOT pass --enforce-eager. Baseline arm 0: same line with ADAPTIVE_TOPK_ENABLE=0.)
```

If/when falling back to the custom B=1 loop, swap the plugin only:
`VLLM_PLUGINS=adaptive_topk_fallback` (same env knobs).
