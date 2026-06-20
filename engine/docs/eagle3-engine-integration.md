# EAGLE3 native integration into the Rust/cudarc inference engine

**Status:** design / interface sketch. No `.rs` changed by this doc.
**Scope:** integrate EAGLE3 speculative decoding *natively* into our custom Rust + cudarc engine
(Qwen3-235B-A22B target, B=1 decode, 8×H100, FP8, TP=8+EP). vLLM is **only** our validation
reference — nothing here calls into vLLM.

This builds on the existing spec scaffold:
- `engine/src/spec/model.rs` — `ModelRunner` (target) + `DrafterPool` (drafters) traits.
- `engine/src/spec/types.rs` — `DraftProposal`, `DraftTree`, `TargetLogits`, `AcceptedRun`, `RngCore`.
- `engine/src/spec/engine.rs` — `SpecEngine::step` (draft → verify → accept).
- `engine/src/spec/accept.rs` — `accept_multi_drafter` lossless acceptance (**math frozen — do not touch**).
- `engine/src/spec/route_aware.rs` — `RouteAwarePolicy` (shrink the expert union the verify reads).
- `engine/src/spec/adaptive_verify.rs` — `adaptive_verify_depth` (truncate the verify chain).
- `engine/src/routing/predictor.rs` — `RoutePredictor` / `DirectProxy` (predict a token's experts).

---

## 0. Validated facts this design is built against

Measured today on vLLM (our reference), to be reproduced natively:

- **Correct head:** `RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3`, architecture
  `Eagle3LlamaForCausalLM`. A **single-layer ~0.6B draft transformer**, `draft_vocab = 64000`,
  that reads **3 auxiliary hidden states** from the target at layers **(1, 46, 90)** and carries a
  **t2d / d2t** map between the 64k draft vocab and the 151936 full vocab.
- **Quality:** lossless accept-length ≈ **2.7**, first-position acceptance ≈ **75%** on Qwen3-235B.
- **Failure mode to avoid:** the broken `nm-testing` conversion gave accept-length **1.4**. The
  draft-vocab (t2d/d2t) mapping **and** the 3-aux-state wiring are load-bearing — getting either
  wrong silently halves the win. Our native impl must assert the mapping against the reference.
- **Target:** Qwen3-235B-A22B — 94 layers, 128 experts top-8, GQA group 4, hidden 4096, FP8.
- **Regime:** B=1 decode is floor/launch-overhead-bound. The spec speedup **only materializes under
  CUDA graphs / the megakernel decode loop** (`docs/megakernel-b1.md`); eager hides it behind launch
  overhead. **This design assumes graph/megakernel decode is the runtime.** Eager is used only for
  the first correctness milestone against vLLM.

EAGLE3 is structurally a **single-drafter, deep chain/tree** drafter — *not* the N-independent-drafter
pool the current `accept.rs` was written for. The acceptance math handles both (a single drafter is
just `N = 1`); the work is in (1) the head, (2) plumbing the 3 aux hidden states out of the target, and
(3) the batched tree verify. See §1–§4.

---

## 1. Data flow

### 1.1 One spec round (chain form, the v0 milestone)

```
 confirmed context (KV warm) ─────────────────────────────────────────────┐
                                                                           │
 ┌──────────────────────── TARGET (Qwen3-235B, FP8, TP8+EP) ──────────────┴────────┐
 │  decode forward for last confirmed token t0                                       │
 │    layer 1   ──► aux h^(1)   ┐                                                     │
 │    ...                        │  capture hook (k_aux): copy 3 rows on-device       │
 │    layer 46  ──► aux h^(46)  ─┼─► aux_states = [h^(1); h^(46); h^(90)]  (3×4096)   │
 │    ...                        │                                                     │
 │    layer 90  ──► aux h^(90)  ┘                                                     │
 │    layer 94  ──► logits(t0)  ──► sample/argmax ──► t1 (this round's first real tok)│
 └──────────────────────────────────────────────────────────────────────────────────┘
                                   │ aux_states (3×4096), emb(t1)
                                   ▼
 ┌──────────────────── EAGLE3 HEAD (1-layer, ~0.6B, own GPU/shard) ───────────────────┐
 │  project [aux_states ⊕ emb(token)]  ──► 1 transformer layer ──► draft hidden        │
 │  draft_lm_head ──► logits over draft_vocab (64000)                                  │
 │  argmax/sample in DRAFT space ──► d2t map ──► full-vocab token  d_1                 │
 │  feed emb(d_1) + reuse aux ──► autoregress ──► d_2 ... d_k   (k = num_spec_tokens)  │
 └─────────────────────────────────────────────────────────────────────────────────┘
                                   │ draft tokens [d_1..d_k] (full-vocab ids) + draft logprobs
                                   ▼
 ┌──────────────────── TARGET batched VERIFY (one forward over k positions) ──────────┐
 │  forward_batch_eagle3(context+[t1], [d_1..d_k]) ──► TargetLogits [k, full_vocab]    │
 │  (bandwidth-bound: weights streamed ONCE, k rows share the sweep — spec_verify_bench)│
 │  ALSO re-capture aux_states for the next round from this same pass (free)           │
 └─────────────────────────────────────────────────────────────────────────────────┘
                                   │ TargetLogits
                                   ▼
 ┌──────────────────── accept.rs::accept_multi_drafter (N=1) — FROZEN ────────────────┐
 │  walk positions, lossless speculative-sampling accept/reject, emit bonus token      │
 └─────────────────────────────────────────────────────────────────────────────────┘
                                   │ AcceptedRun { accepted[0..=k], bonus_token }
                                   ▼
                       push to context, loop
```

### 1.2 The pieces, concretely

**Aux state extraction.** EAGLE3 does not read the target's *final* hidden — it reads the residual
stream **after layers 1, 46, 90** (low/mid/high). In the megakernel/graph decode loop the residual
lives on-chip across layers (`docs/megakernel-b1.md`); the capture is a 3-point side-tap that copies
`h^(1), h^(46), h^(90)` (each `[4096]` at B=1) into a small persistent device buffer
`aux_states: CudaSlice<half>` of shape `[3, 4096]`. This is the **only** new behavior required of the
target forward (see §3). Layer indices are config, not hardcoded — they come from the head's config
(`eagle_aux_hidden_state_layer_ids`), and we assert `[1, 46, 90]` at load.

**Head input.** The head consumes `concat(aux_states.flatten(), embed(token))` → an input projection
`fc: [3*4096 + 4096] → [head_hidden]` (per the Eagle3Llama architecture: aux states are concatenated
and projected, the token embedding is from the **target's** embedding table, shared, not re-trained).
Note EAGLE3 reuses the target embedding — we do **not** ship a second embedding matrix; the head reads
`target.embed(token)`.

**Autoregressive draft (chain v0).** For `i in 0..k`:
1. run the 1-layer head → draft hidden → `draft_lm_head` → logits over `draft_vocab=64000`;
2. argmax/sample in draft space → `d_idx`; map to full vocab via `d2t[d_idx]` → `tok`;
3. record `draft_logprob[i] = log_softmax(draft_logits)[d_idx]` (kept in draft space — consistent units;
   acceptance only needs draft vs target *for the same token*, and the target ratio is computed in full
   space, so we must map: see "logprob units" caveat below);
4. feed `embed(tok)` back as the next step's token; aux states are held fixed for the chain (EAGLE3
   conditions the whole chain on the one set of aux states captured at the chain root).

**Tree form (v2).** Instead of a single chain, the head expands `width` candidates per level (top-`w`
in draft space), forming a small tree of `≤ W×D` nodes with a tree attention mask. This is where
`RouteAwarePolicy` selects which candidates to keep (§2.4). The flattened tree is exactly the layout
`DraftTree` / `TargetLogits` already assume (`engine/src/spec/types.rs`: `[N*k]` row-major). For a
chain, `N=1`.

**Verify.** `forward_batch` over the `k` (or tree-flattened) draft positions returns
`TargetLogits { data:[k*full_vocab], n_positions:k, vocab_size:151936 }`. The verify runs in **full**
vocab (the target has no draft vocab). The accept step compares per-token.

**Close the loop.** `accept_multi_drafter(&tree, &target_logits, rng)` is called **unchanged**. It
already handles the single-drafter case and produces `accepted + bonus_token`.

**logprob units caveat (must get right — this is the nm-testing trap).** `accept.rs` computes the
acceptance ratio as `exp(target_logprob(tok) − draft_logprob(tok))`. `target_logprob` is full-vocab
log-softmax of the *target* row. `draft_logprob` must be the EAGLE3 head's log-prob **for that same
full-vocab token**, i.e. log-softmax over the **64000 draft logits at the d2t-mapped index**, *not*
over full vocab. Tokens with no draft-vocab preimage (`t2d` undefined) can never be proposed by the
head, so they never reach the ratio — consistent. We will unit-test that the proposed `draft_logprob`
fed into `DraftProposal.logprobs` is the draft-space log-softmax value, matching what vLLM's EAGLE3
sampler uses. Getting this wrong is exactly what produced the 1.4 accept-length.

---

## 2. Rust interfaces

### 2.1 Why a new trait (not `DrafterPool`)

`DrafterPool::draft(context, draft_len)` (`engine/src/spec/model.rs:64`) takes **only the token
context**. EAGLE3 cannot draft from tokens alone — it **requires the target's 3 aux hidden states**
for the current position. So `DrafterPool` is the wrong shape: its signature has nowhere to pass
`aux_states`, and EAGLE3 is single-drafter (the "pool" abstraction doesn't fit).

We add a **new** trait `AuxDrafter` alongside (not replacing) `DrafterPool`. The existing N-drafter
pool path stays valid for the multi-small-model experiments; EAGLE3 takes the new path.

```rust
// NEW — engine/src/spec/model.rs (proposed addition; flagged NEW)

/// Auxiliary hidden states captured from the target at the EAGLE3 aux layers.
/// Shape: [n_aux, hidden] flattened row-major (n_aux=3, hidden=4096 for Qwen3).
/// Lives on-device in the real impl; this host view is only used at the eager
/// correctness milestone. The GPU path passes an opaque device handle (see §3).
pub struct AuxHiddenStates {
    pub data: Vec<f32>,     // eager path: [n_aux * hidden]
    pub n_aux: usize,
    pub hidden: usize,
    // GPU path: replace `data` with a `DeviceAux` handle (CudaSlice<half>) — §3.
}

/// A draft head that conditions on the target's aux hidden states (EAGLE3-style).
/// Single drafter that emits a chain or tree in one call.
pub trait AuxDrafter: Send + Sync {
    /// Draft up to `draft_len` tokens (chain) given the confirmed context, the
    /// root token whose forward produced `aux`, and the aux hidden states.
    ///
    /// Returns ONE DraftProposal (drafter_id = 0). Tokens are FULL-vocab ids
    /// (already d2t-mapped); logprobs are DRAFT-space log-softmax values for the
    /// chosen tokens (see "logprob units caveat" in §1.2).
    fn draft_chain(
        &self,
        context: &[crate::spec::types::TokenId],
        root_token: crate::spec::types::TokenId,
        aux: &AuxHiddenStates,
        draft_len: usize,
    ) -> crate::error::Result<crate::spec::types::DraftProposal>;

    /// Tree variant (v2): expand `width` candidates per level, route-aware-pruned.
    /// Returns a DraftTree whose flat layout matches TargetLogits row order.
    /// `policy` shapes which candidates survive (shrinks the expert union).
    fn draft_tree(
        &self,
        context: &[crate::spec::types::TokenId],
        root_token: crate::spec::types::TokenId,
        aux: &AuxHiddenStates,
        plan: &TreePlan,
        policy: &crate::spec::route_aware::RouteAwarePolicy,
        predictor: &dyn crate::routing::predictor::RoutePredictor,
    ) -> crate::error::Result<crate::spec::types::DraftTree>;
}

/// Tree shape knobs (v2).
pub struct TreePlan {
    pub depth: usize,   // D
    pub width: usize,   // W (candidates kept per level after route-aware prune)
    pub fanout: usize,  // top-f draft candidates considered per node before prune
}
```

### 2.2 The `Eagle3Drafter` struct

```rust
// NEW — engine/src/spec/eagle3.rs (proposed new module)

pub struct Eagle3Drafter<H: Eagle3Head, R: AuxEmbedder> {
    head: H,            // the 1-layer draft transformer + draft_lm_head (cudarc)
    embed: R,           // access to the TARGET embedding table (shared, not owned)
    d2t: Vec<i32>,      // draft-vocab -> full-vocab (len 64000)
    t2d: Vec<i32>,      // full-vocab  -> draft-vocab (len 151936, -1 if absent)
    aux_layers: [usize; 3], // asserted == [1, 46, 90] at load
    draft_vocab: usize, // 64000
    cfg: Eagle3Config,
}

/// The on-device head: input projection, 1 transformer layer, draft lm_head.
/// Implemented over cudarc (§4). Trait so the eager reference impl is swappable.
pub trait Eagle3Head: Send + Sync {
    /// One head step: given fused [aux ⊕ emb(token)] hidden, produce draft-vocab logits.
    fn step(&self, fused_input: &DeviceTensor, kv: &mut HeadKv)
        -> crate::error::Result<DeviceTensor /* [draft_vocab] */>;
    fn head_hidden(&self) -> usize;
}

/// Read-only access to the target's (shared) embedding rows.
pub trait AuxEmbedder: Send + Sync {
    fn embed(&self, token: crate::spec::types::TokenId) -> crate::error::Result<DeviceTensor>;
}
```

`Eagle3Drafter` implements `AuxDrafter`. `draft_chain` is the §1.2 loop; `draft_tree` is the
route-aware expansion (§2.4). `DeviceTensor` / `HeadKv` are thin cudarc wrappers (NEW, §4).

### 2.3 Modified `SpecEngine::step`

The current `SpecEngine` (`engine/src/spec/engine.rs:48`) is generic over `D: DrafterPool, T:
ModelRunner`. EAGLE3 needs the target to also be an aux-emitting model and the drafter to be an
`AuxDrafter`. Rather than mutate the existing generic engine (keep it for the pool path), add a
**parallel** entry point — a new `Eagle3Engine` (or a second `impl` block gated on the aux traits).
Sketch:

```rust
// NEW — engine/src/spec/engine.rs (additional struct; existing SpecEngine untouched)

pub struct Eagle3Engine<Dr: AuxDrafter, T: AuxModelRunner> {
    pub drafter: Dr,
    pub target: T,                 // AuxModelRunner: ModelRunner + aux capture (§3)
    pub config: SpecConfig,
    pub route_policy: RouteAwarePolicy,        // λ-knob (route_aware.rs)
    pub floor_fraction: f32,                   // F for adaptive_verify (regime)
    pub adaptive: bool,                        // enable verify-depth truncation
}

impl<Dr: AuxDrafter, T: AuxModelRunner> Eagle3Engine<Dr, T> {
    pub fn step(
        &self,
        context: &[TokenId],
        rng: &mut impl RngCore,
    ) -> Result<(AcceptedRun, RoundStats)> {
        // (A) target forward for the last confirmed token: logits + 3 aux states,
        //     in ONE pass (the aux capture is a side-tap, §3).
        let (root_logits, aux) =
            self.target.forward_single_with_aux(context)?;       // NEW API, §3
        let root_token = argmax(&root_logits);                   // this round's first real token

        // (B) DRAFT — EAGLE3 head, conditioned on aux. Chain (v0) or tree (v2).
        let tree: DraftTree = if self.config_tree() {
            let plan = self.tree_plan();
            // route_aware::RouteAwarePolicy hooks in HERE — inside draft_tree it
            // scores candidates by draft_logprob + λ·expert-overlap and prunes,
            // using `predictor` (DirectProxy on the head's draft hidden) for experts.
            self.drafter.draft_tree(
                context, root_token, &aux, &plan,
                &self.route_policy, self.predictor(),
            )?
        } else {
            let prop = self.drafter.draft_chain(
                context, root_token, &aux, self.config.draft_len,
            )?;
            DraftTree { proposals: vec![prop], draft_len: self.config.draft_len }
        };

        // (C) ADAPTIVE VERIFY — adaptive_verify::adaptive_verify_depth hooks in HERE.
        //     Truncate the chain/tree to the depth that maximizes emitted/verify_cost
        //     for the current floor regime, BEFORE paying for the target verify pass.
        let verify_tree = if self.adaptive {
            let accept_probs = estimate_accept_probs(&tree);         // from draft logprobs
            let experts_per_pos = predict_experts_per_pos(&tree, self.predictor());
            let plan = adaptive_verify_depth(
                &accept_probs, &experts_per_pos, self.floor_fraction,
            );
            truncate_tree(tree, plan.map(|p| p.depth))               // pure CPU reshape
        } else {
            tree
        };

        // (D) VERIFY — batched target forward over the (possibly truncated) tree.
        //     EAGLE3 needs the root token prepended (it was sampled in step A).
        let flat = verify_tree.flat_tokens();
        let target_logits =
            self.target.forward_batch(&with_root(context, root_token),
                                      &flat, self.config.vocab_size)?;

        // (E) ACCEPT — FROZEN. accept.rs handles N=1 (chain) and N>1 (tree) already.
        let run = accept_multi_drafter(&verify_tree, &target_logits, rng);
        let stats = RoundStats {
            n_accepted: run.n_accepted(),
            n_proposed: verify_tree.n_drafters() * verify_tree.draft_len,
            winning_drafter: run.winning_drafter,
        };
        // NOTE: the root_token (step A) is emitted FIRST, then run.all_tokens().
        Ok((run, stats))
    }
}
```

Helpers `estimate_accept_probs`, `predict_experts_per_pos`, `truncate_tree`, `with_root`,
`argmax` are **NEW**, small, pure-CPU, and unit-testable without a GPU (matching the existing
testing discipline in `route_aware.rs` / `adaptive_verify.rs`).

### 2.4 Where the two union-aware levers plug in (explicit)

- **`route_aware::RouteAwarePolicy`** — inside `Eagle3Drafter::draft_tree`, at **draft-candidate
  selection**. When the head emits its top-`fanout` candidates per node, we build a
  `Vec<route_aware::Candidate>` (`token`, `draft_logprob`, predicted `experts` from
  `RoutePredictor::predict` on the head's draft hidden / the target residual proxy) and call
  `policy.select_width(&candidates, width, &mut union)` (`route_aware.rs:128`). This keeps the tree's
  expert union small → the verify reads fewer MoE experts. **Lossless** (`accept.rs` corrects any
  draft distribution). λ is regime-adaptive: ≈0 while floor-bound, >0 as the floor falls / the tree
  grows (per the module docs).

- **`adaptive_verify::adaptive_verify_depth`** — in `Eagle3Engine::step` step **(C)**, *between* draft
  and verify. It picks the verify depth `d ∈ 1..=k` maximizing `emitted/verify_cost` for the current
  `floor_fraction`, then we `truncate_tree` to `d`. Truncating is lossless: an unverified position is
  just decoded normally next round. This is a pure throughput knob and runs entirely on CPU before any
  GPU verify cost is paid.

Both consume `routing::predictor::RoutePredictor` (e.g. `DirectProxy`) for the per-candidate /
per-position expert sets — already the intended consumer per those modules' own docs.

---

## 3. What the target (`ModelRunner`) must additionally expose

Today `ModelRunner` (`engine/src/spec/model.rs:12`) returns **only logits** from `forward_single` /
`forward_batch`. EAGLE3 needs the 3 aux hidden states out of the *same* forward (capturing them in a
separate pass would double the most expensive op). Minimal extension — a new sub-trait, so existing
`ModelRunner` impls (and all current tests) are untouched:

```rust
// NEW — engine/src/spec/model.rs (extension trait; ModelRunner unchanged)

pub trait AuxModelRunner: ModelRunner {
    /// Which residual-stream layers to tap (config; == [1,46,90] for the validated head).
    fn aux_layers(&self) -> &[usize];

    /// Decode forward for the last confirmed token, returning BOTH the full-vocab
    /// logits AND the aux hidden states captured at `aux_layers`, from ONE pass.
    /// (At B=1 the aux capture is a 3-row on-device copy of the residual — §4.)
    fn forward_single_with_aux(
        &self,
        context: &[crate::spec::types::TokenId],
    ) -> crate::error::Result<(crate::spec::types::Logits, super::model::AuxHiddenStates)>;

    /// Optional: capture aux for the NEXT round from the verify pass too (free re-use),
    /// so a round needs only ONE extra-aux forward (the root), not two.
    fn forward_batch_with_aux(
        &self,
        context: &[crate::spec::types::TokenId],
        draft_tokens: &[crate::spec::types::TokenId],
        vocab_size: usize,
    ) -> crate::error::Result<(crate::spec::types::TargetLogits, super::model::AuxHiddenStates)> {
        // default: call forward_batch + forward_single_with_aux (correctness-only fallback)
        unimplemented!("override with a fused capture in the cudarc target")
    }
}
```

Design notes:
- `AuxModelRunner: ModelRunner` so the same target object serves both verify (`forward_batch`) and aux
  capture. `Eagle3Engine<_, T: AuxModelRunner>` requires this bound.
- The aux capture must be **device-side** in the graph/megakernel path: `AuxHiddenStates` carries an
  opaque `DeviceAux` handle (a `CudaSlice<half>` view), not a host `Vec`, to avoid a D→H copy of 3×4096
  per round. The host `Vec` form in §2.1 is the **eager-only** correctness path.
- This is the *only* change required of the target. Everything else (the head, the levers) is additive.

---

## 4. cudarc / kernel work

Coordinate with the existing kernels in `kernels/` (k1 prologue, k2 flash-decode, k3 epilogue, k4
router, k5 experts, k6 graph capture, `megakernel_decode.cu`, and the already-present
`spec_verify_bench.cu` / `spec_decode_bench.cu`). Reuse vs new:

| Component | Reuse | New work |
|---|---|---|
| **Aux capture hook** in target forward at layers (1,46,90) | the megakernel already keeps the residual on-chip across layers (`docs/megakernel-b1.md`); k3 epilogue writes the residual | a 3-point side-tap that `memcpy`s the residual row (`[4096] half`) into a persistent `aux_states` device buffer at the 3 layer boundaries. ~0 cost at B=1 (8 KB×3). Inside the megakernel it's a branch on layer index. |
| **EAGLE3 head: attention** (1 layer, head_hidden, its own small KV) | k1 (RMSNorm→QKV→QK-norm→RoPE→KV write) and k2 (flash-decode GQA) idioms transfer directly — the head is just a *small* Llama-style attention | new instantiation at head dims (much smaller than 4096/94L); a tiny per-chain KV cache (`HeadKv`). Likely a single fused kernel given the size. |
| **EAGLE3 head: input projection `fc`** `[3*4096+4096]→[head_hidden]` | a GEMV — same warp-per-output-row idiom as k5/lm_head | new weight + one GEMV launch per head step. |
| **EAGLE3 head: FFN** (1 layer, dense — the head is **not** MoE) | k5's gate+up+silu+down fused GEMV idiom, but **dense** (no expert gather, no router) — simpler than k5 | new dense-FFN instantiation at head dims. |
| **`draft_lm_head`** → 64000 draft logits + argmax/top-w | lm_head idiom from `lmhead_k3_bench.cu` (warp-per-vocab-row, fused argmax) | new at draft_vocab=64000 (smaller than 151936 full vocab → cheaper). top-`w` for tree. |
| **d2t / t2d mapping** | — | trivial device gather (`d2t[idx]`); load + assert against reference at startup. CPU is fine. |
| **Batched (tree) verify** in target | **`spec_verify_bench.cu` already proves** a (γ+1)-row verify costs ~one single-token read on the real k5 v3 fused-expert forward + real lm_head; the batched idiom (add the M axis, weight tile staged once, dotted against all M rows) is written | wire it into the real decode path: `forward_batch` over k (or W×D tree) positions with a **tree attention mask** (causal within each path). The mask is new; the GEMV batching is reused. |
| **Tree attention mask** | k2 flash-decode online-softmax | new mask plumbing so each tree node attends only to its ancestors (chain v0 needs no mask — it's plain causal). |
| **Graph / megakernel composition** | k6 graph capture; `megakernel_decode.cu` already notes "spec verify is the same megakernel over W×D positions" | capture the head forward + the batched verify as graph regions; ultimately fold the head into the persistent megakernel so a round = one resident launch (endgame). |
| **Acceptance** | `accept.rs` — pure CPU, frozen; logit rows copied D→H (≈5 MB, negligible) | none. |

Net: the **head is small and reuses every kernel idiom we already have** (attention=k1/k2,
proj/ffn/lmhead=k5/lmhead, dense not MoE → simpler). The genuinely new GPU work is (a) the 3-point aux
side-tap, (b) the tree attention mask for multi-path verify, (c) the d2t/t2d gather. The batched-verify
amortization is already proven by `spec_verify_bench.cu`.

---

## 5. Risks / unknowns + staged build order

### Risks / unknowns
1. **Aux-state correctness (the #1 trap).** Wrong layer ids, wrong concat order, or wrong normalization
   of the captured residual silently collapses accept-length toward the broken-conversion 1.4. **Mitigation:**
   bit-compare our captured `aux_states` and head logits against vLLM's EAGLE3 at the eager milestone,
   per-token, before trusting any speed number.
2. **d2t/t2d + logprob units.** The acceptance ratio mixes draft-space draft_logprob with full-space
   target_logprob (§1.2). A units mismatch is a lossless-but-low-acceptance bug — hard to catch by
   correctness tests (output is still target-distributed), only visible as accept-length. **Mitigation:**
   assert accept-length ≈ 2.7 / first-pos ≈ 75% against the reference; unit-test the mapping round-trip.
3. **Graph/megakernel is a hard prerequisite for any win.** Eager will show *correct* output but *no*
   speedup (the floor hides it). Don't measure speedup until graphs are in. Risk: the head adds launches
   that must also be captured.
4. **Tree verify mask complexity.** Chain (v0) is plain causal and easy; the tree mask (v2) is where
   bugs hide. Keep v0 (chain) lossless-equivalent and only move to tree once chain is solid.
5. **λ / floor_fraction tuning.** The two levers can *lose* throughput if mis-set in the wrong regime
   (route_aware/adaptive_verify docs both warn: union tax is second-order while floor-bound). Keep
   λ≈0 and `adaptive` off until floor-bound regime is left behind; gate both behind config.
6. **Device aux handle lifetime.** `AuxHiddenStates` must hold the device buffer alive across the
   draft loop without a host copy; cudarc ownership/lifetime needs care (the head reads it k times).

### Staged build order (smallest first, each independently bankable)

**S0 — frozen-math sanity.** No new code: confirm `accept_multi_drafter` already handles `N=1` via a
chain-shaped `DraftTree` in a unit test. (Tests live per `CLAUDE.md`; run `cargo test --package engine`
before/after.)

**S1 — eager correctness vs vLLM (CPU/eager target, no graphs).**
Implement `AuxHiddenStates`, `AuxModelRunner::forward_single_with_aux` (eager, host Vec), the eager
`Eagle3Head` + `Eagle3Drafter::draft_chain`, d2t/t2d load+assert. Wire `Eagle3Engine::step` chain path.
**Gate:** output is byte-identical-distribution to plain target decode (lossless) AND accept-length ≈
2.7 / first-pos ≈ 75% matches vLLM. This catches risks 1 & 2. **No speedup expected yet.**

**S2 — GPU head + device aux (still chain, eager verify).**
Port the head to cudarc (reusing k1/k2/k5/lmhead idioms, §4); aux capture becomes a device side-tap
returning a `DeviceAux` handle. **Gate:** same accept-length, head step time measured.

**S3 — CUDA-graph capture of the round (chain).**
Capture target forward + head draft loop + batched verify as graph regions (k6). **Gate:** *first
real speedup* — wall-clock tok/s vs no-spec, B=1, on box. This is where the floor lifts.

**S4 — batched/tree verify.**
Add the tree attention mask and `draft_tree`; flatten to the existing `TargetLogits` layout; reuse the
`spec_verify_bench.cu` batched idiom in the real path. **Gate:** accept-length rises with tree width at
flat verify cost (the amortization).

**S5 — route-aware + adaptive-verify levers on top.**
Turn on `RouteAwarePolicy` (λ>0) in `draft_tree` and `adaptive_verify_depth` in step (C), driven by
`DirectProxy`. **Gate:** `emitted/verify_cost` improves in the (now lower-floor) regime; verify reads a
smaller expert union; output still lossless. Keep both behind config and regime-gated (risk 5).

**S6 — megakernel fusion (endgame).**
Fold the head and the batched verify into the persistent `megakernel_decode.cu` loop so a full spec
round is one resident launch with device-side collectives. This is the 750→~2000 closer
(`docs/megakernel-b1.md`); last and hardest.

---

## Appendix: file-by-file change map

| File | Change |
|---|---|
| `engine/src/spec/model.rs` | **NEW** `AuxHiddenStates` struct, `AuxDrafter` trait, `AuxModelRunner: ModelRunner` extension trait. Existing `ModelRunner`/`DrafterPool` **unchanged**. |
| `engine/src/spec/eagle3.rs` | **NEW** module: `Eagle3Drafter`, `Eagle3Head`/`AuxEmbedder` traits, `DeviceTensor`/`HeadKv` cudarc wrappers, d2t/t2d, `draft_chain`/`draft_tree`. |
| `engine/src/spec/engine.rs` | **NEW** `Eagle3Engine` struct + `step` (chain & tree). Existing `SpecEngine` **unchanged**. |
| `engine/src/spec/accept.rs` | **NONE** (frozen). |
| `engine/src/spec/route_aware.rs` | **NONE** — consumed by `draft_tree`. |
| `engine/src/spec/adaptive_verify.rs` | **NONE** — consumed by `Eagle3Engine::step`. |
| `engine/src/routing/predictor.rs` | **NONE** — `DirectProxy` consumed for per-candidate experts. |
| `engine/src/spec/mod.rs` | add `pub mod eagle3;` + re-exports. |
| `kernels/` | head kernels (reuse k1/k2/k5/lmhead idioms), aux side-tap, tree mask, wire `spec_verify_bench.cu` batched idiom into real verify, eventual megakernel fold. |
</content>
</invoke>
