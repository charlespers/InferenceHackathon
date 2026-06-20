#!/usr/bin/env bash
# apply_ipc_patch.sh — Build and wire the CUDA IPC all-reduce fallback.
# No IMEX / no CAP_SYS_ADMIN required.
#
# Usage:
#   bash tools/apply_ipc_patch.sh
#   python3 tools/start_vllm_ipc.py

set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SO=/tmp/ipc_ar.so

echo "=== CUDA IPC all-reduce patch (no IMEX fallback) ==="

# Build the .so
echo "[1/2] Building ipc_allreduce.so..."
nvcc -arch=sm_90a -O3 --use_fast_math \
     --shared -Xcompiler -fPIC \
     "$REPO/kernels/ipc_allreduce.cu" \
     -lcuda -lcudart -o "$SO" 2>&1
if [ $? -ne 0 ]; then echo "FAIL: compile error"; exit 1; fi
echo "      built OK -> $SO"

# Clean up stale IPC handle files from any previous run.
echo "[2/2] Cleaning stale IPC handles..."
rm -f /tmp/ipc_ar_*.bin
echo "      done"

echo ""
echo "=== IPC patch ready ==="
echo "    Launch vLLM with:"
echo ""
echo "        python3 $REPO/tools/start_vllm_ipc.py"
echo ""
echo "    Each TP worker will use the IPC all-reduce kernel instead of NCCL."
echo "    Expected gain: ~1-2 ms/token saved on 188 × 16 KB all-reduces."
