# Why spec decode is the convergent answer — it escapes the B=1 regime

The team converged on EAGLE3 tree-spec; this is the unified *why*, in one principle: **spec converts the
inefficient serial B=1 decode into one efficient batched verify.** Three things make B=1 decode slow — an
exposed per-step floor, an inefficient GEMV, and no batch to amortize anything — and a single spec round
attacks all three at once.

## The three B=1 pathologies and how the batched verify dodges each
| B=1 decode pathology | why it's slow | what the spec verify does |
|---|---|---|
| **Exposed floor** (188 all-reduces + launch + host, ~10 ms, 60–86% of the step) | paid **per forward**; nothing to overlap at B=1 | verify is **one** forward for τ accepted tokens → floor paid **once** → ÷τ per token |
| **Inefficient expert GEMV** (e≈0.16; tensor cores idle, ~1 FLOP/byte) | matrix×**vector**, no reuse | verify is matrix×**matrix** over W×D positions → a **grouped GEMM** on the tensor cores (higher MFU) |
| **KV re-read** (grows with context) | each token re-reads the whole KV | the W×D queries attend the **same shared KV** → read **once** for the batch → ÷τ |

So one verify replaces τ serial, floor-paying, tensor-core-idle, KV-re-reading B=1 GEMV steps with one
batched, floor-paid-once, GEMM-efficient, KV-read-once forward. **That's the whole game at B=1.**

## The one cost: the expert union (and why it's cheap *now*)
The price is that the W×D verify positions route to the **union** of experts (up to all 128), and the verify
reads that union's weights once. In *weight-units* that union is the tax (`spec-decode-moe-tax.md`). But it
lands on the **weight term**, which is only **14%** of the floor-bound step — so in real time
`verify_cost = F + (1−F)·(0.34 + 0.66·union/8)` stays small while F is high (`spec_floor_model.py`). The high
E[accepted] of a wide tree then dominates it → ~3.4× at the measured F=0.86 (`tree_spec_optimizer.py`). As the
floor is fixed (F→0), the union tax re-asserts and the tree must shrink + go route-aware
(`route-aware-drafting-design.md`).

## Why this beats the structural levers (the team's own finding)
- **Comms count-reduction** (DP+EP, async-TP) is a B=1 wash-to-loss and needs retraining (`comms_floor.md`,
  `seriality_breaking.md`).
- **Depth reduction / self-spec** loses to off-the-shelf EAGLE3 on Qwen3 (pruning-fragile) (`depth_reduction.md`).
- **Quant/kernels** shrink the 14% weight (invisible while floor-bound; `ab_adaptive` regressed).
- **Spec amortizes the 86% floor AND uses efficient kernels** — it's the only lever that attacks the dominant
  term *and* converts the regime. That's why every research thread converged on it.

## The composition (what the levers do together)
- **Spec (÷τ)** sits on top of everything: it amortizes whatever the per-step cost is.
- **Comms tuning + K5 kernels** reduce the *absolute* per-step cost the verify pays once → spec's τ× rides a
  smaller base. They compose multiplicatively: `(reduced floor) × (÷τ spec) × (prefix-cache TTFT)`.
- **Endpoint** (`latency_budget.py`): cheap + EAGLE3 → ~508 tok/s; + K5 + tuned comms → ~754 tok/s (~13×).

## The corollary for the engine
Because the verify is a batched GEMM, the **verify path wants the *batched* MoE kernel, not the B=1 GEMV K5** —
they're different kernels. K5 (B=1 GEMV) optimizes the *draft* and the no-spec fallback; the *verify* uses the
efficient grouped-GEMM path (vLLM's batched `fused_moe` is already reasonable at batch W×D). So the kernel
effort splits: K5 for the B=1 paths, a tuned grouped-GEMM for the verify — and the cudarc engine should host
both behind the spec loop.
