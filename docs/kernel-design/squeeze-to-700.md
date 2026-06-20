# Squeeze to 700 (and 1000?) tok/s — the B=1 arithmetic

**Target:** Qwen3-235B-A22B, B=1 decode, 8x H100 (HBM3 3.35 TB/s, NVLink/NVSwitch).
**Claim under test:** with (1) in-kernel collectives, (2) the megakernel removing per-layer
launch/sync, (3) int4-v3 expert weights, and (4) speculative decoding, is 700 tok/s reachable?
Is 1000?

This file recomputes the realistic per-token time from the **four levers' kernels in this repo**,
shows the arithmetic, and separates *measured* from *projected*. It does **not** restate the kernel
internals (see the source files).

Source kernels this doc draws on:
- `kernels/nvshmem_inkernel_bench.cu` — in-kernel vs host-relaunched all-reduce (the comms lever).
- `kernels/megakernel_decode.cu` — persistent grid-resident decode step (the launch/sync lever).
- `kernels/k5_experts_int4_v3.cu` — int4 W4A16 fused MoE expert GEMV (the bytes lever).
- `kernels/spec_verify_bench.cu` — verify-in-one-pass amortization + spec multiplier (the spec lever).

---

## 1. The two terms: per-GPU compute + per-token comms

A B=1 sharded decode step is `t_token = t_compute(per-GPU weight read) + t_comms(188 collectives)`.

### 1a. Per-GPU compute (weight-read-bound)

At B=1 every emitted token streams the active weight set from HBM once; the floor is bytes/GPU ÷
bandwidth. Active set is the TP=8/EP=8 shard each GPU actually reads (fp8 = 1 byte/param):

| component (per layer)        | full      | per-GPU (TP=8 / EP=8)          |
|------------------------------|-----------|-------------------------------|
| fused QKV + O-proj (attn)    | 71.3 MB   | /8 = 8.9 MB                   |
| router gate (replicated)     | 0.52 MB   | 0.52 MB (tiny, not sharded)   |
| MoE: 8 active experts        | 151.0 MB  | ~1 expert/GPU = 18.9 MB       |
| **per-layer per-GPU**        |           | **~28.3 MB**                  |

- 94 layers x 28.3 MB = **2.66 GB**; + lm_head 622 MB /8 = 78 MB -> **2.74 GB/GPU/token**.
- BW floor: 2.74 GB ÷ 3.35 TB/s = **0.818 ms** -> **1223 tok/s** (compute-only, 100% MBU).
- At the **measured** K5 MBU of 58.1%: 2.74 GB ÷ (3.35 TB/s x 0.58) = **1.41 ms** -> **709 tok/s**
  (compute-only).

> So the per-GPU **compute alone**, at today's measured MBU, already sits right at ~700 tok/s. The
> reason real systems are far slower is the second term.

### 1b. Per-token comms (188 collectives)

188 collectives/token = 2 all-reduces/layer x 94. Each moves a 16 KB ([HIDDEN]=4096 fp32) payload;
the NVLink transfer of 16 KB is ~0.1 us, so the per-collective cost is essentially **launch + barrier
overhead**, not bandwidth (env/LL128 tuning confirmed dead).

| per-collective latency        | comms ms/token (x188) | comms-only cap |
|-------------------------------|-----------------------|----------------|
| NCCL all-reduce (measured)    | 35 us -> 6.58 ms      | 152 tok/s      |
| NVSHMEM host put+barrier (meas)| 17 us -> 3.20 ms     | 313 tok/s      |
| **in-kernel target ~6 us**    | **1.13 ms**           | **887 tok/s**  |
| **in-kernel target ~5 us**    | **0.94 ms**           | **1064 tok/s** |
| in-kernel stretch ~3 us       | 0.56 ms               | 1773 tok/s     |

The comms term is the entire game: at the host-launched 17 us floor it caps the system at ~310 tok/s
**before compute is even added**. Driving the per-collective latency into single digits is what the
megakernel / in-kernel-collective work exists to do.

`nvshmem_inkernel_bench.cu` measures the persistent in-kernel all-reduce us/round directly; that
number replaces "in-kernel target" below.

---

## 2. Total ms/token = compute + comms (pre-spec)

`t_token = t_compute + 188 x t_coll`. Pre-spec tok/s:

| compute              | + comms@17us (host floor) | + comms@6us | + comms@5us | + comms@3us |
|----------------------|---------------------------|-------------|-------------|-------------|
| 0.818 ms (BW floor)  | 4.01 ms -> **249**        | 1.95 ms -> **514** | 1.76 ms -> **569** | 1.38 ms -> **724** |
| 1.41 ms (58% MBU)    | 4.61 ms -> **217**        | 2.54 ms -> **394** | 2.35 ms -> **426** | 1.97 ms -> **507** |

**Verdict, pre-spec:** in-kernel comms alone gets the system from ~217-249 tok/s (host-launched) to
~400-570 tok/s. Reaching ~700 tok/s pre-spec needs BOTH single-digit-us comms AND compute near the
BW floor (i.e. MBU pushed above ~58%). 1000 pre-spec is not reachable on this compute floor. That
is what spec is for.

---

## 3. The int4-v3 lever (compute term)

`k5_experts_int4_v3.cu` halves the MoE expert weight bytes (int4 W4A16 = 0.5 byte/param vs fp8's 1).
The MoE is 18.9 of the 28.3 MB/layer per-GPU read, i.e. **~67%** of the compute term. If int4 hits its
bandwidth-bound target, the MoE half of the read drops ~2x:

- per-layer per-GPU read: 8.9 (attn, still fp8) + 0.52 (gate) + ~9.4 (int4 MoE) = **~18.8 MB**.
- 94 x 18.8 MB + 78 MB lm = **1.84 GB/GPU/token**.
- at 58% MBU: 1.84 GB ÷ (3.35 TB/s x 0.58) = **0.95 ms** compute -> with comms@5us (0.94 ms) =
  1.89 ms -> **~530 tok/s pre-spec**; at the int4 BW floor (0.55 ms) + comms@5us -> **~670 tok/s**.

**Caveat (load-bearing):** int4-v3 was just fixed for **correctness** — the original used a half-
precision (`__hfma2`) contraction that failed its own `<1e-2` bar by ~100x. v3 now does the fast LOP3
int4->half2 **dequant** (the real win over v2, which removes v2's scalar integer->float convert) but
**contracts in fp32**. Whether it actually beats fp8's 98 us is now a hypothesis the on-box bench must
confirm: the LOP3 unpack ALU (~4 `__hsub2`/word) is still significant at int4's 2x weight density, so
the pure-byte-count 2x is an upper bound, not a guarantee. The bench now **aborts** (returns 1) if
`max_abs >= 1e-2`, so a wrong kernel can never be read as fast.

---

## 4. The spec lever (the multiplier)

Spec reads the weights ONCE per verify pass for `gamma+1` candidate tokens, so a verify pass costs
~one single-token forward (the amortization `spec_verify_bench.cu` measures: forward-time-vs-M is
flat because the weight tile is re-dotted against M rows in the cycles the B=1 GEMV would otherwise
stall). The comms is ALSO once per verify pass: the all-reduce payload grows from 16 KB to
`(gamma+1) x 16 KB` (<=128 KB at gamma=7), still ~0.8 us on NVLink — i.e. still launch/barrier-floored
and flat. So **both** terms are flat in `gamma`, and:

```
eff tok/s  =  E[accepted] / t_pass        where t_pass ~= t_token (single-token, from sec 2)
E[accepted] = (1 - a^(gamma+1)) / (1 - a)  (linear/chain accept, +1 bonus; conservative proxy)
```

`spec_verify_bench.cu` anchors the multiplier to the **measured** M=1 forward time and uses the
measured M=(gamma+1) time for `t_pass`, so a measured slowdown>1 (flatness degrading at large M, as
arithmetic/L2 approach saturation) correctly deflates the result. The `a` values {0.7, 0.8} are a
chain proxy; the team's EAGLE3 big-tree generally accepts MORE for the same per-node `a`, so this is
conservative. Team measures ~3.8x end-to-end.

### Stacked result (compute@58% MBU + in-kernel comms, x E[accepted])

base = compute(58% MBU) + comms; eff = base_tok/s x E[accepted]:

| base (compute+comms)          | base tok/s | a=0.7 g=4 (E=2.77) | a=0.8 g=4 (E=3.36) | a=0.8 g=7 (E=4.16) |
|-------------------------------|------------|--------------------|--------------------|--------------------|
| 2.35 ms (58% MBU + 5us comms) | 426        | 1180               | 1431               | 1771               |
| 1.97 ms (58% MBU + 3us comms) | 507        | 1405               | 1703               | 2108               |

Even the conservative chain `E[accepted]` x the measured-compute base **clears 700 tok/s with a lot
of headroom** and pushes through 1000 at realistic accept rates. With int4 lowering the compute term
(sec 3), the base rises further.

---

## 5. Bottom line

- **700 tok/s: reachable, and it is primarily a COMMS result.** The compute floor at today's 58% MBU
  is already ~700 tok/s per-GPU; the host-launched 17 us collective floor is what drags the system to
  ~217-310 tok/s. In-kernel collectives at single-digit us are the unlock: ~400-570 tok/s **pre-spec**,
  and any realistic spec multiplier (>=1.7x) takes that past 700.
- **1000 tok/s: reachable WITH spec, not without.** Pre-spec, 1000 needs comms ~3 us AND compute at
  the BW floor (724 tok/s) — tight. With spec's ~2.8-4.2x on a ~400-570 base, 1000 is comfortably
  in range; the question is sustained accept rate, not kernel speed.
- **int4-v3** lowers the compute term ~30% (MoE is 2/3 of the per-GPU read) IF it is bandwidth-bound
  on-box — newly corrected, now a fp32-accumulate kernel; speed is a hypothesis to confirm.

### Measured vs projected

| quantity                              | status     | source / value                          |
|---------------------------------------|------------|-----------------------------------------|
| NCCL AR 35us, NVSHMEM put+barrier 17us| MEASURED   | this session                            |
| K5 fp8 MoE 58.1% MBU, lm_head 55.4%   | MEASURED   | k5_experts_v3.cu / lmhead bench         |
| per-GPU compute 1.41 ms (58% MBU)     | DERIVED    | bytes/GPU ÷ (3.35 TB/s x 0.58)          |
| in-kernel per-collective us           | PROJECTED  | run `nvshmem_inkernel_bench.cu`         |
| megakernel us/layer-step              | PROJECTED  | run `megakernel_decode.cu` (proxy weights)|
| int4-v3 us vs fp8 98us                | PROJECTED  | run `k5_experts_int4_v3.cu` (now fp32-acc)|
| spec E[accepted] / flatness           | PARTIAL    | flatness MEASURED in bench; accept-rate from EAGLE3 team |

---

## 6. What the orchestrator must still bench (on the 8x H100 box)

1. **`nvshmem_inkernel_bench.cu`** — the actual persistent in-kernel all-reduce us/round. This is the
   single most load-bearing number: it replaces the "in-kernel target" rows in sec 1b/2. Confirm it is
   single-digit us and that the same-harness `us_host/us_ink` speedup (not the hardcoded 17/35 priors)
   is the honest figure.
2. **`megakernel_decode.cu`** — confirm (a) the **cooperative-launch smoke test** passes (the newly-
   added `mk_coop_smoke`: collective_launch must enable a multi-block `grid.sync()`, else the
   persistent kernel hangs — this is the design's single point of failure), (b) the in-kernel
   all-reduce correctness check passes, (c) us/layer-step, scaling `N_LAYERS_TEST` toward 94. This
   gives the REAL combined compute+comms us/layer (the proxy reuses one layer's weights, so the read
   volume is representative but the absolute throughput is a projection).
3. **`k5_experts_int4_v3.cu`** — must PASS `max_abs < 1e-2` (now fp32-accumulate; it aborts otherwise)
   AND beat fp8's 98 us. If it passes correctness but is still slower than fp8, the int4 compute-term
   win in sec 3 does not apply and the next lever is unpack-ALU/occupancy, not bytes.
4. **`spec_verify_bench.cu`** — confirm exp/M=1 and lm/M=1 ratios stay ~1.0 (flatness) out to M=8; the
   multiplier deflates honestly if they don't. Cross-check `E[accepted]` against the EAGLE3 team's
   measured big-tree accept rate (their ~3.8x), since this bench uses a conservative chain proxy.
5. **Not modeled here:** attention flash-decode time, RMSNorm/router compute, and the KV-cache read
   (grows with context). The compute term above is weight-read-only; the megakernel run folds these in
   and is the authoritative per-layer number.
