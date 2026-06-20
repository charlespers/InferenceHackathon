# Results reaction 04 — the squeeze round: comms is BARRIER-bound, int4 ruled out, spec is the lever

Real measured data (`docs/kernel-design/benched-results.md`, kernels benched on-box). Three findings, two of
which **revise `path-to-1000.md`**. I'm reacting honestly — one of my assumptions (NVLS → ~2 µs) is now in
doubt, and my int4 cushion is dead.

## The data
1. **In-kernel NVSHMEM all-reduce = 1.06× (51.8 µs vs 55.1 µs host).** Comms is **barrier-bound, not
   launch-bound**: the floor is **one 8-GPU NVLink `barrier_all` ≈ 17 µs**, and removing the kernel launch
   doesn't help.
2. **int4 (v3, LOP3 half2 unpack) = 0.58× fp8 (SLOWER).** The int4→half unpack is ALU-bound at B=1, so the
   half-the-bytes win never materializes. **int4 is ruled out at B=1.**
3. **spec is the lever** (the dominant ÷2.77–3.8 multiplier; team's EAGLE3 ≈3.8×). The squeeze-round
   spec-verify bench was mis-modeled (didn't batch the weight read) — a bench bug, not a refutation.

## What it changes in `path-to-1000.md`
**(a) The NVLS rung was optimistic — but not dead.** Crucial nuance: the team measured **NVSHMEM
recursive-doubling** AR = **3 barriers = 51 µs**, which is *worse* than NCCL's measured **~16 µs (≈1 barrier)**.
So "in-kernel doesn't break the wall" means *recursive-doubling* doesn't. The **multimem in-switch reduce**
(SHARP — one switch op, `kernels/nvls_allreduce.cu`) is **a different mechanism and still unmeasured**: it may
do the reduce *inside* the switch in ~one barrier or less. **So `measure_collective.sh` (the NVLS arm) is now
even more the make-or-break:** if multimem in-switch ≈ the same ~16 µs barrier → my low-C ladder is wrong and
the comms floor is ~16 µs; if it beats it → the ~1100 path holds. **The megakernel's in-kernel comms must be the
in-switch reduce, NOT recursive-doubling** (which is what the squeeze round proved is barrier-bound).

**(b) int4 is OFF the cushion.** The weight floor is **fp8 (0.78 ms)**; `int4-experts-quality-gate.md` is moot
at B=1 (the unpack ALU eats the byte saving). Remove it from the path. The remaining cushion is comms-side
(stale-TP / fewer collectives), not weight-side.

**(c) At the 16 µs barrier, 1000 still has a path — via the COLLECTIVE COUNT, not per-collective latency.**
My ladder at C=16 µs (188 TP collectives) → only ~334 tok/s (the comms wall). But the team's arithmetic is
right: **halve the collectives (188→94, 1/layer via EP all-to-all = 1 barrier vs TP recursive-doubling = 3) +
batched spec (÷2.77–3.8)** → `94×17µs = 1.6 ms ÷ 2.77 + ~0.5 ms ≈ ~960 tok/s` (÷3.8 → ~1300). **So with the
barrier fundamental, the lever shifts from "make each collective faster" to "do FEWER collectives + amortize
them with spec."**

## The revised make-or-break (three gates, ranked)
1. **Collective COUNT** — get from 188 (TP, 2/layer × recursive-doubling) toward **~94 one-barrier collectives**
   (EP all-to-all is 1 barrier; TP all-reduce is 3). This is now the #1 comms lever (the per-collective is
   barrier-floored). **The team's EP-decode path is the right attack.**
2. **multimem in-switch reduce** (`measure_collective.sh` NVLS arm) — IF it beats the ~16 µs barrier, it stacks
   with #1 (fewer AND faster). If not, #1 + spec carries it.
3. **batched spec** (÷2.77–3.8) — the dominant multiplier, confirmed. Needs a correctly-batched verify kernel
   (the squeeze bench's flat-forward bug must be fixed — the real verify reads the union ONCE).

## Updated ladder (replacing the optimistic NVLS@2µs rung)
- **Pessimistic-but-real:** 188 coll @ 16 µs barrier → comms 3.0 ms → ~334 with spec. *Not enough.*
- **Team's measured path:** 94 coll @ 16 µs (EP, 1 barrier) + fp8 + batched spec ÷2.77 → **~960**; ÷3.8 → ~1300.
- **My in-switch path (if it pans out):** multimem < 16 µs → stacks on the above.
- **int4: removed.** **stale-TP (LOOP-C): still the upside** (hides comms → roofline) if its quality gate passes.

## Net
The squeeze round is decisive and I'm updating to it: **the comms per-collective latency is barrier-floored
(~16 µs) — so the comms attack is COUNT (EP, fewer collectives) + AMORTIZATION (batched spec), with multimem
in-switch as the only remaining "make-it-faster" hope (measure it). int4 and in-kernel-launch-elision are dead
ends. 1000 is still reachable (~960–1300 via EP + batched spec), but through the count, not the latency.**
I'm fixing `ladder_to_1000.py`, `path-to-1000.md`, and retiring the int4 cushion accordingly.
