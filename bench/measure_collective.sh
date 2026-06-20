#!/bin/bash
# THE make-or-break measurement for 1000 tok/s (docs/path-to-1000.md): the per-collective all-reduce latency C
# at the B=1 payload (8KB bf16 / 4KB fp8 hidden=4096), across NCCL algorithms. No model load -> seconds, not a
# slot. Decides everything: C<=~4us (fp8 hide threshold) => 1000 is on (NVLS and/or stale-TP); C~16us => the
# comms wall caps us ~250 regardless of weights/kernels.
#
# Run anytime (no GPU contention beyond ~10s of nccl-tests; still respect the gpu.lock if a model is loaded).
NT=/workspace/nccl-tests   # box has this prebuilt (Box facts). adjust if elsewhere.
OUT=/root/collective_result.txt; : > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }

[ -x "$NT/build/all_reduce_perf" ] || { log "nccl-tests not at $NT — find/build it (make MPI=0)"; exit 0; }

log "=== per-collective all-reduce latency C at B=1 payloads (8 GPUs) — the 1000-tok/s gate ==="
log "    target: C <= ~4us (fp8 per-collective hide threshold). 188 colls x C = the comms term."
log "    1000 needs comms small (NVLS) OR hidden (stale-TP); both need C low. 16us => stuck ~250."

run() {  # $1=label  $2..=env
  local label="$1"; shift
  log ""; log "--- $label ---"
  # -b 4K -e 16K covers fp8 (4K) and bf16 (8K) hidden=4096; -n many iters for a stable small-msg latency
  env "$@" "$NT/build/all_reduce_perf" -b 4096 -e 16384 -f 2 -g 8 -n 200 -w 50 2>&1 \
    | awk '/^ *[0-9]/ {printf "    size %6s B  busbw %6s GB/s  lat %8s us\n",$1,$(NF-1),$(NF-3)}' | tee -a "$OUT"
}

run "baseline (NCCL default algo)"
run "RING"            NCCL_ALGO=Ring
run "TREE"            NCCL_ALGO=Tree
run "NVLS (in-switch SHARP)"      NCCL_ALGO=NVLS NCCL_NVLS_ENABLE=1
run "NVLS + LL proto"            NCCL_ALGO=NVLS NCCL_NVLS_ENABLE=1 NCCL_PROTO=LL
run "CollnetDirect"   NCCL_ALGO=CollnetDirect NCCL_COLLNET_ENABLE=1

log ""
log "=== READOUT (the 1000-tok/s decision) ==="
log "  * lowest C at 4-8KB across algos = the achievable per-collective latency."
log "  * C<=4us  -> 1000 is ON: 188xC<=0.75ms; fp8 weight 0.78 + small-spec -> ~1170 (NVLS) or ~1280 (stale-TP hides it)."
log "  * C~6-8us -> tight: lossless ~750-900; needs stale-TP (LOOP-C) to hide, or int4 experts (gate)."
log "  * C~16us  -> comms wall: stuck ~250 regardless of weights/kernels -> a CUSTOM multimem AR kernel is REQUIRED"
log "              (NCCL can't get there at 8KB; this is megakernel-build-plan.md Stage 3, the pivot)."
log "  Also note all-to-all (EP dispatch) for comparison:"
"$NT/build/alltoall_perf" -b 4096 -e 16384 -f 2 -g 8 -n 200 -w 50 2>&1 | awk '/^ *[0-9]/{printf "    a2a size %6s B  lat %8s us\n",$1,$(NF-3)}' | tee -a "$OUT"
log "=== done $(date -u +%H:%M:%S)UTC ==="