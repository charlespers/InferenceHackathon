#!/usr/bin/env bash
# run_e2e_bench.sh — reproducible end-to-end TP=8 decode benchmark for decode_step_tp8.cu with the
# NVLS in-switch all-reduce integrated.  scps the kernels to the 8xH100 box, builds, waits for a clean
# GPU window, runs with the correctness gate ON, and prints the tok/s + comms + PASS/FAIL.
#
# This is THE benchmark for the comms-optimization loop.  Run it after every change to the AR path.
#
# Usage:  bench/run_e2e_bench.sh [CTX_LEN] [ITERS] [HBM_GBs] [RUN_CHECK]
#   defaults: 4096 200 3350 1     (RUN_CHECK=1 -> correctness gate; 0 -> timing only)
# Env knobs:
#   NVLS=0           -> build with -DUSE_NVLS=0 (NCCL baseline, for A/B)
#   NO_WAIT=1        -> skip the clean-GPU-window wait (run immediately)
#   REMOTE_DIR=...   -> remote build dir (default /root/e2e)
set -uo pipefail

# ---- box + paths ----
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=5 -o ServerAliveCountMax=6"
PORT=31025
KEY=~/.ssh/id_github
HOST=root@147.185.41.162
REMOTE_DIR="${REMOTE_DIR:-/root/e2e}"
RESULTS_DIR="/root/charles_results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
KERN="$REPO_DIR/kernels"

# ---- args ----
CTX_LEN="${1:-4096}"
ITERS="${2:-200}"
HBM_GBS="${3:-3350}"
RUN_CHECK="${4:-1}"
USE_NVLS="${NVLS:-1}"
USE_GEMM="${GEMM:-1}"     # 1 -> cuBLASLt fp8 tensor-core GEMM forward; 0 -> hand-rolled GEMV baseline

SSH() { ssh $SSH_OPTS -p "$PORT" -i "$KEY" "$HOST" "$@"; }
SCP() { scp $SSH_OPTS -P "$PORT" -i "$KEY" "$@"; }

echo "== run_e2e_bench: ctx=$CTX_LEN iters=$ITERS hbm=$HBM_GBS check=$RUN_CHECK USE_NVLS=$USE_NVLS USE_GEMM=$USE_GEMM =="

# ---- (a) ship the kernels (engine + NVLS header + all included sub-kernels) ----
echo "-- scp kernels -> $HOST:$REMOTE_DIR/"
SSH "mkdir -p $REMOTE_DIR $RESULTS_DIR"
SCP \
  "$KERN/decode_step_tp8.cu" \
  "$KERN/nvls_engine.cuh" \
  "$KERN/common.cuh" \
  "$KERN/k1_attn_prologue.cu" \
  "$KERN/k2_flash_decode.cu" \
  "$KERN/k3_attn_epilogue.cu" \
  "$KERN/k4_router.cu" \
  "$KERN/k5_experts.cu" \
  "$KERN/gemm_engine.cuh" \
  "$HOST:$REMOTE_DIR/" || { echo "SCP FAILED"; exit 1; }

# ---- (b) build dstp8nvls on the box (driver API -lcuda for NVLS multicast) ----
echo "-- build dstp8nvls"
BUILD_LOG=$(SSH bash -lc "'
  set -e
  cd $REMOTE_DIR
  NCCL_BASE=\$(python3 -c \"import nvidia.nccl,os;print(os.path.dirname(nvidia.nccl.__file__))\")
  NCCL_INC=\$NCCL_BASE/include
  NCCL_LIB=\$NCCL_BASE/lib
  /usr/local/cuda/bin/nvcc -arch=sm_90a -O3 --use_fast_math -DUSE_NVLS=$USE_NVLS -DUSE_GEMM=$USE_GEMM \
    -I $REMOTE_DIR -I \"\$NCCL_INC\" $REMOTE_DIR/decode_step_tp8.cu \
    -L \"\$NCCL_LIB\" -lnccl -lcuda -lcublas -lcublasLt -o /tmp/dstp8nvls 2>&1
  echo BUILD_RC=\$?
'" 2>&1)
echo "$BUILD_LOG"
if ! echo "$BUILD_LOG" | grep -q "BUILD_RC=0"; then echo "BUILD FAILED"; exit 1; fi
echo "-- build OK"

# ---- (c) wait for a clean GPU window (no heavy foreign procs pinning the GPUs) ----
if [ "${NO_WAIT:-0}" != "1" ]; then
  echo "-- waiting for a clean GPU window (<=10 min)..."
  for i in $(seq 1 60); do
    USED=$(SSH "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | sort -n | tail -1)
    USED="${USED:-100}"
    if [ "$USED" -lt 20 ] 2>/dev/null; then echo "   clean (max util ${USED}%)"; break; fi
    echo "   busy (max util ${USED}%), retry $i/60..."; sleep 10
  done
fi

# ---- (d) run with the correctness gate, tee to results ----
OUT="$RESULTS_DIR/nvls_integrated.txt"
echo "-- run /tmp/dstp8nvls $CTX_LEN $ITERS $HBM_GBS $RUN_CHECK   (tee $OUT)"
RUN_LOG=$(SSH bash -lc "'
  NCCL_LIB=\$(python3 -c \"import nvidia.nccl,os;print(os.path.dirname(nvidia.nccl.__file__))\")/lib
  LD_LIBRARY_PATH=\"\$NCCL_LIB:\$LD_LIBRARY_PATH\" /tmp/dstp8nvls $CTX_LEN $ITERS $HBM_GBS $RUN_CHECK 2>&1 | tee $OUT
'" 2>&1)
echo "$RUN_LOG"

# ---- (e) extract + print the headline numbers ----
echo ""
echo "=================== BENCH SUMMARY ==================="
echo "$RUN_LOG" | grep -E "NVLS: multicast|NVLS unavailable" || true
echo "-- correctness gate:"
echo "$RUN_LOG" | grep -E "post-attention residual|post-MoE|TOL=|CORRECTNESS" || echo "   (no correctness lines — run_check=$RUN_CHECK)"
echo "-- throughput:"
echo "$RUN_LOG" | grep -E "EAGER step \(baseline\)|kernels-graph \+ eager AR|full NCCL-in-graph|BEST graphed path|SPEEDUP" || true
echo "-- comms:"
echo "$RUN_LOG" | grep -E "all-reduces only|per-all-reduce|AR overhead / token|AR-share" || true

if echo "$RUN_LOG" | grep -q "PASS"; then echo ">>> CORRECTNESS: PASS"; CORR=0
elif echo "$RUN_LOG" | grep -q "FAIL"; then echo ">>> CORRECTNESS: FAIL"; CORR=1
else echo ">>> CORRECTNESS: (skipped)"; CORR=0; fi
echo "===================================================="
exit ${CORR}
