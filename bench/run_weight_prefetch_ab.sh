#!/bin/bash
# A/B the weight-prefetch overlap DIRECTLY on the real engine (kernels/decode_step_tp8.cu) -- the same
# binary that produced the team's measured 108.5 tok/s headline -- rather than an isolated microbench.
# Builds it twice (USE_WEIGHT_PREFETCH=0, the unmodified baseline, vs =1, with the touch added
# concurrently with each AR) and reports BOTH runs' own self-measured tok/s for a direct comparison.
#
# The prefetch code is fully gated behind USE_WEIGHT_PREFETCH (default 0 if undefined), so the =0 build
# is byte-for-byte the same code path that produced every prior measured number on this file.
#
# SAFETY: same gpu.lock + free-memory protocol as run_overlap_prefetch.sh -- refuses to run if the box
# is held (e.g. by the live vLLM demo) and releases the lock on exit.
#
#   bash bench/run_weight_prefetch_ab.sh [ctx_len] [iters]
set -u
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
KDIR=${KDIR:-kernels}
CTX=${1:-4096}
IT=${2:-200}
BIN_OFF=${BIN_OFF:-/tmp/dstp8_prefetch_off}
BIN_ON=${BIN_ON:-/tmp/dstp8_prefetch_on}
OUT=${OUT:-/root/weight_prefetch_ab_result.txt}
LOCK=${LOCK:-/alloc/data/gpu.lock}
MIN_FREE_MB=${MIN_FREE_MB:-8000}

NCCL_INC=$(python3 -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'include'))" 2>/dev/null)
NCCL_LIB=$(python3 -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'lib'))" 2>/dev/null)
if [ -z "$NCCL_INC" ] || [ -z "$NCCL_LIB" ]; then
  echo "Could not resolve NCCL include/lib paths via 'python3 -c import nvidia.nccl'." | tee "$OUT"
  exit 1
fi

echo "=== compile decode_step_tp8.cu BASELINE (USE_WEIGHT_PREFETCH=0) $(date -u +%H:%M:%S)UTC ===" | tee "$OUT"
if ! "$NVCC" -arch=sm_90a -O3 --use_fast_math -DUSE_WEIGHT_PREFETCH=0 -I "$KDIR" -I "$NCCL_INC" \
     "$KDIR/decode_step_tp8.cu" -L "$NCCL_LIB" -lnccl -lcublasLt -lcublas -lcuda -o "$BIN_OFF" 2>>"$OUT"; then
  echo "BASELINE COMPILE FAILED — see $OUT" | tee -a "$OUT"
  exit 1
fi
echo "baseline compile OK -> $BIN_OFF" | tee -a "$OUT"

echo "=== compile decode_step_tp8.cu PREFETCH (USE_WEIGHT_PREFETCH=1) $(date -u +%H:%M:%S)UTC ===" | tee -a "$OUT"
if ! "$NVCC" -arch=sm_90a -O3 --use_fast_math -DUSE_WEIGHT_PREFETCH=1 -I "$KDIR" -I "$NCCL_INC" \
     "$KDIR/decode_step_tp8.cu" -L "$NCCL_LIB" -lnccl -lcublasLt -lcublas -lcuda -o "$BIN_ON" 2>>"$OUT"; then
  echo "PREFETCH COMPILE FAILED — see $OUT" | tee -a "$OUT"
  exit 1
fi
echo "prefetch compile OK -> $BIN_ON" | tee -a "$OUT"

ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$ngpu" -lt 8 ]; then
  echo "need 8 GPUs, found $ngpu — compiled only, not running." | tee -a "$OUT"
  exit 0
fi
min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
echo "min free across 8 GPUs: ${min_free} MiB (need >= ${MIN_FREE_MB} MiB)" | tee -a "$OUT"
if [ "${min_free:-0}" -lt "$MIN_FREE_MB" ]; then
  echo "REFUSING to run: a GPU has < ${MIN_FREE_MB} MiB free (live demo or another job may be holding the box)." | tee -a "$OUT"
  echo "compiled only; rerun once the box is free." | tee -a "$OUT"
  exit 0
fi

acquired=0
if mkdir "$LOCK" 2>/dev/null; then
  acquired=1
  echo "weight_prefetch_ab $(date -u +%FT%TZ)" > "$LOCK/holder"
  echo "acquired $LOCK" | tee -a "$OUT"
else
  holder_age=$(( $(date +%s) - $(stat -c %Y "$LOCK/holder" 2>/dev/null || echo 0) ))
  if [ "$holder_age" -gt 1200 ]; then
    echo "lock held but holder file is ${holder_age}s old (>20min) -- treating as stale, taking over." | tee -a "$OUT"
    echo "weight_prefetch_ab $(date -u +%FT%TZ)" > "$LOCK/holder"
    acquired=1
  else
    echo "REFUSING to run: $LOCK is held by: $(cat "$LOCK/holder" 2>/dev/null) (${holder_age}s old)." | tee -a "$OUT"
    echo "compiled only; rerun once the lock is free." | tee -a "$OUT"
    exit 0
  fi
fi
release_lock() { if [ "$acquired" -eq 1 ]; then rm -f "$LOCK/holder"; if rmdir "$LOCK" 2>/dev/null; then echo "released $LOCK" | tee -a "$OUT"; else echo "WARNING: failed to release $LOCK (rmdir failed)" | tee -a "$OUT"; fi; fi; }
trap release_lock EXIT

echo "=== run BASELINE (ctx=$CTX iters=$IT) $(date -u +%H:%M:%S)UTC ===" | tee -a "$OUT"
LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}" "$BIN_OFF" "$CTX" "$IT" 1 2>&1 | tee -a "$OUT"

echo "=== run PREFETCH (ctx=$CTX iters=$IT) $(date -u +%H:%M:%S)UTC ===" | tee -a "$OUT"
LD_LIBRARY_PATH="$NCCL_LIB:${LD_LIBRARY_PATH:-}" "$BIN_ON" "$CTX" "$IT" 1 2>&1 | tee -a "$OUT"

echo "=== A/B done $(date -u +%H:%M:%S)UTC -> see $OUT for both runs' tok/s ===" | tee -a "$OUT"
echo "compare the '>>> BEST graphed path' lines from each run above." | tee -a "$OUT"
