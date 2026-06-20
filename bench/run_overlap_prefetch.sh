#!/bin/bash
# Build + run kernels/overlap_prefetch.cu — measures the lossless weight-prefetch overlap (hide AR(L)'s
# NVLink latency behind a TOUCH of layer L+1's Wqkv weight, an independent HBM/L2 path with no data
# dependency on AR's result). This is the smaller-scope alternative to the parked persistent megakernel
# (see acfaf05's "comms-overlap measured but needs persistent megakernel -> parked").
#
# Needs all TP=8 GPUs (real NCCL all-reduce across ranks) and ~the same per-rank memory as
# decode_step_tp8.cu (one layer's dummy fp8 weights/KV reused x94, real per-rank byte volume).
#
# SAFETY: this acquires the team's gpu.lock (danielAgentScheduling.md protocol) before touching any
# GPU, and refuses to run if free memory on any of the 8 GPUs is too low (e.g. the live vLLM demo is
# holding them) — see MIN_FREE_MB below. It releases the lock on exit (success, failure, or Ctrl-C).
#
#   bash bench/run_overlap_prefetch.sh [ctx_len] [iters]
set -u
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
KDIR=${KDIR:-kernels}
CTX=${1:-4096}
IT=${2:-200}
BIN=${BIN:-/tmp/overlap_prefetch.bin}
OUT=${OUT:-/root/overlap_prefetch_result.txt}
LOCK=${LOCK:-/alloc/data/gpu.lock}
MIN_FREE_MB=${MIN_FREE_MB:-8000}   # decode_step_tp8.cu-class proxy needs a few GB/rank; refuse below this

NCCL_INC=$(python3 -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'include'))" 2>/dev/null)
NCCL_LIB=$(python3 -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'lib'))" 2>/dev/null)
if [ -z "$NCCL_INC" ] || [ -z "$NCCL_LIB" ]; then
  echo "Could not resolve NCCL include/lib paths via 'python3 -c import nvidia.nccl' — is the nvidia-nccl-cu12 wheel installed?" | tee "$OUT"
  exit 1
fi

echo "=== compile overlap_prefetch.cu (sm_90a, O3, fast-math) $(date -u +%H:%M:%S)UTC ===" | tee "$OUT"
if ! "$NVCC" -arch=sm_90a -O3 --use_fast_math -I "$KDIR" -I "$NCCL_INC" \
     "$KDIR/overlap_prefetch.cu" -L "$NCCL_LIB" -lnccl -o "$BIN" 2>>"$OUT"; then
  echo "COMPILE FAILED — see $OUT" | tee -a "$OUT"
  exit 1
fi
echo "compile OK -> $BIN" | tee -a "$OUT"

# ---- need all 8 GPUs with real headroom (this is a TP=8 job, not a single-GPU microbench) ----
ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$ngpu" -lt 8 ]; then
  echo "need 8 GPUs, found $ngpu — compiled only, not running." | tee -a "$OUT"
  exit 0
fi
min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
echo "min free across 8 GPUs: ${min_free} MiB (need >= ${MIN_FREE_MB} MiB)" | tee -a "$OUT"
if [ "${min_free:-0}" -lt "$MIN_FREE_MB" ]; then
  echo "REFUSING to run: a GPU has < ${MIN_FREE_MB} MiB free (the live demo or another job may be holding the box)." | tee -a "$OUT"
  echo "compiled only; rerun once the box is free (check ${LOCK}/holder and 'nvidia-smi')." | tee -a "$OUT"
  exit 0
fi

# ---- acquire the team's gpu.lock (danielAgentScheduling.md protocol); never steal a fresh lock ----
acquired=0
if mkdir "$LOCK" 2>/dev/null; then
  acquired=1
  echo "overlap_prefetch $(date -u +%FT%TZ)" > "$LOCK/holder"
  echo "acquired $LOCK" | tee -a "$OUT"
else
  holder_age=$(( $(date +%s) - $(stat -c %Y "$LOCK/holder" 2>/dev/null || echo 0) ))
  if [ "$holder_age" -gt 1200 ]; then
    echo "lock held but holder file is ${holder_age}s old (>20min) -- treating as stale, taking over." | tee -a "$OUT"
    echo "overlap_prefetch $(date -u +%FT%TZ)" > "$LOCK/holder"
    acquired=1
  else
    echo "REFUSING to run: $LOCK is held by: $(cat "$LOCK/holder" 2>/dev/null) (${holder_age}s old)." | tee -a "$OUT"
    echo "compiled only; rerun once the lock is free." | tee -a "$OUT"
    exit 0
  fi
fi
release_lock() { if [ "$acquired" -eq 1 ]; then rm -f "$LOCK/holder"; if rmdir "$LOCK" 2>/dev/null; then echo "released $LOCK" | tee -a "$OUT"; else echo "WARNING: failed to release $LOCK (rmdir failed)" | tee -a "$OUT"; fi; fi; }
trap release_lock EXIT

echo "=== run (ctx=$CTX iters=$IT) $(date -u +%H:%M:%S)UTC ===" | tee -a "$OUT"
LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}" "$BIN" "$CTX" "$IT" 2>&1 | tee -a "$OUT"
echo "=== done $(date -u +%H:%M:%S)UTC -> see $OUT ===" | tee -a "$OUT"
