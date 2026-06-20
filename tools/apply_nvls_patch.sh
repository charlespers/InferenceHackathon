#!/usr/bin/env bash
# apply_nvls_patch.sh — Build, test, and apply the NVLS cross-process patch.
#
# Steps:
#   1. Build the viability test (nvls_xproc_test.cu)
#   2. Run it: if PASS → multicast FD export works without IMEX → proceed
#             if FAIL → print fallback instructions
#   3. Build the coordinator binary (nvls_coordinator.cu)
#   4. Start the coordinator in the background
#   5. Patch tools/start_vllm.py to call the Python patch at startup
#
# Usage:
#   cd /alloc/data/InferenceHackathon
#   bash tools/apply_nvls_patch.sh
#   python3 tools/start_vllm_nvls.py   # the patched launcher

set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== NVLS cross-process patch (no IMEX / no CAP_SYS_ADMIN) ==="
echo "    repo: $REPO"
echo ""

# ---- Step 1: build viability test ----
echo "[1/4] Building nvls_xproc_test..."
nvcc -arch=sm_90a -O3 -I "$REPO/kernels" \
     "$REPO/kernels/nvls_xproc_test.cu" \
     -lcuda -o /tmp/nvls_xproc_test 2>&1
if [ $? -ne 0 ]; then echo "FAIL: nvcc compile error"; exit 1; fi
echo "      built OK -> /tmp/nvls_xproc_test"

# ---- Step 2: run viability test ----
echo ""
echo "[2/4] Running viability test (POSIX-FD multicast export)..."
/tmp/nvls_xproc_test
RC=$?
if [ $RC -ne 0 ]; then
    echo ""
    echo "FAIL: cuMemExportToShareableHandle blocked (rc=$RC)."
    echo "  => IMEX is required for cross-process multicast on this container."
    echo "  => Options:"
    echo "     A) Relaunch container with --privileged (ask Prime Intellect support)"
    echo "     B) Use the CUDA IPC fallback: bash tools/apply_ipc_patch.sh"
    exit 1
fi

echo ""
echo "PASS: POSIX-FD multicast works without IMEX!"
echo "  => Proceeding to build and install the patch..."

# ---- Step 3: build coordinator ----
echo ""
echo "[3/4] Building NVLS coordinator..."
nvcc -arch=sm_90a -O3 -I "$REPO/kernels" \
     "$REPO/kernels/nvls_coordinator.cu" \
     -lcuda -lcudart -o /tmp/nvls_coord 2>&1
if [ $? -ne 0 ]; then echo "FAIL: coordinator compile error"; exit 1; fi
echo "      built OK -> /tmp/nvls_coord"

# ---- Step 4: start coordinator ----
echo ""
echo "[4/4] Starting coordinator in background..."
pkill -f nvls_coord 2>/dev/null; sleep 0.5
rm -f /tmp/nvls_mc.json
/tmp/nvls_coord &
COORD_PID=$!
echo "      coordinator pid=$COORD_PID"

# Wait for coordinator to write JSON.
for i in $(seq 1 10); do
    if [ -f /tmp/nvls_mc.json ]; then
        if python3 -c "import json; d=json.load(open('/tmp/nvls_mc.json')); assert d.get('ready')" 2>/dev/null; then
            break
        fi
    fi
    sleep 0.5
done
if [ ! -f /tmp/nvls_mc.json ]; then
    echo "FAIL: coordinator did not write /tmp/nvls_mc.json"
    kill $COORD_PID 2>/dev/null
    exit 1
fi
echo "      /tmp/nvls_mc.json written OK"

# ---- Done ----
echo ""
echo "=== NVLS patch ready ==="
echo "    Coordinator running (pid=$COORD_PID)"
echo "    Launch vLLM with the patched starter:"
echo ""
echo "        python3 $REPO/tools/start_vllm_nvls.py"
echo ""
echo "    The patched launcher imports tools/patch_vllm_nvls.py in each worker"
echo "    and replaces tensor_model_parallel_all_reduce with our multimem kernel."
echo ""
echo "    To stop the coordinator: kill $COORD_PID"
