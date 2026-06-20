# Spec acceptance — two correctness notes for the NATIVE verify path

Found while wiring `Eagle3Engine::decode` (LOOP-A). Both are about the contract between
`accept_multi_drafter` (`spec/accept.rs`) and the target's `forward_batch` (`spec/model.rs`). They are
**latent for the CPU tests** (the `EchoTarget`/`OverlapHead` mocks happen to satisfy them by
construction) but the **native cudarc verify must get them right** or decoding will be subtly wrong /
not lossless. Raising, not fixing — these are shared foundations (Charles's verify lane).

## Note 1 — `forward_batch`'s default impl is OFF BY ONE vs what `accept` reads

`accept_multi_drafter` scores draft token `pos` via `target.logprob_at(d*k+pos, draft[pos])`. So its
contract is:

> `target.data[d*k + pos]` must be the target distribution **predicting `draft[pos]`**, i.e.
> `P(· | context + draft_d[0..pos])`  (prefix of length `pos`, NOT including `draft[pos]`).

But the **default** `forward_batch` (model.rs) computes `data[i] = forward_single(context+flat[0..i], flat[i])`
— that is `P(· | context + draft[0..i] + draft[i])` = **predicting `draft[i+1]`** (the token AFTER
`draft[i]`). That's shifted one position from what `accept` reads. Even `data[0]` is wrong: `accept`
wants `P(·|context)` (predicting `draft[0]`); the default produces `P(·|context, draft[0])` (predicting
`draft[1]`).

Why the tests still pass: `EchoTarget.forward_single(ctx, tok)` peaks the logit at **`tok` itself**, so
`data[pos]` peaks at `draft[pos]` — exactly what `accept` checks — masking the shift. A real target does
NOT echo its input, so the shift would bite.

**Native verify contract:** produce `data[d*k+pos] = P(· | context + draft_d[0..pos])`. Concretely, run
one forward over `[ctx_last, draft[0], …, draft[k-1]]` and take the logits emitted **at the position of
`draft[pos-1]`** (and at `ctx_last` for `pos=0`) — i.e. the "predict-next" output one slot to the LEFT of
`draft[pos]`. Do not read the output sitting at `draft[pos]`.

## Note 2 — the bonus token on a FULL-accept round is a stand-in (can duplicate)

When all `k` positions accept, `accept.rs` (lines ~104–114) sets
`bonus = target.greedy_at(last_winner*k + (k-1))` — the argmax of the **last draft position's** row —
with the in-code comment that the caller "must run one more target forward pass for the bonus token, or
pre-compute it by passing k+1 draft positions. We use greedy … as a stand-in." Consequences:

- It's **greedy**, not a sample — wrong for temperature > 0.
- Given Note 1's layout, that row predicts `draft[k]` only if the layout is corrected; with the current
  stand-in it tends to **re-emit the last accepted token** (a duplicate), which I observed would corrupt
  a deterministic-ramp losslessness check on full-accept rounds.

**Native verify contract:** verify `k+1` positions (append one extra slot) OR do one more target forward
so the bonus is a genuine sample from `P(· | context + accepted_run)`. Then full-accept rounds emit a
true (k+1)-th token, and losslessness holds on every round, not just rounds with a rejection.

## Impact / why it matters for our lever
The route-aware pitch is "trades throughput, **never** correctness (exact lossless)." That guarantee
rests on the verify producing exact target distributions at each emitted position — including the bonus.
Both notes are about preserving that exactness in the native path; neither affects the route-aware
*policy* (λ / verify-depth) which only changes which tokens are drafted. An end-to-end losslessness test
(decode output invariant to λ and verify-depth under a deterministic target) is the right regression once
the native verify lands — it needs a target that overrides `forward_batch` with the Note-1 layout and
verifies `k+1` (Note 2), which the default mock does not.
