#!/usr/bin/env bash
# run_vllm_bench.sh — self-contained B=1 vLLM baseline for one testing slot.
#   launch (enforce-eager) -> wait ready -> B=1 bench -> teardown (free GPUs).
#
# Designed to be invoked through the slot guard so it can never overrun:
#   gpu-slot run djamoils -- tools/run_vllm_bench.sh
#
# A SIGTERM (from the slot auto-stop) triggers the teardown trap, so the GPUs
# are always released before the next person's slot.
set -uo pipefail

REPO="${REPO:-/alloc/data/InferenceHackathon}"
MODEL="${MODEL:-/alloc/data/Qwen3-235B-A22B}"
PORT="${PORT:-8001}"
SERVED="${SERVED:-qwen3-235b-bf16}"
LOG="${LOG:-/alloc/data/vllm_serve_slot.log}"
OUT="${OUT:-/alloc/data/vllm_b1_bench.json}"
export VLLM_NCCL_SO_PATH="${VLLM_NCCL_SO_PATH:-/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib/libnccl.so.2}"

SERVER_PID=""

teardown() {
  echo ">> teardown: stopping vLLM on :$PORT, freeing GPUs"
  if [ -n "$SERVER_PID" ]; then kill -INT "$SERVER_PID" 2>/dev/null || true; fi
  sleep 5
  pkill -INT  -f "vllm serve .*--port $PORT" 2>/dev/null || true
  sleep 4
  pkill -KILL -f "vllm serve .*--port $PORT" 2>/dev/null || true
  sleep 3
  echo -n ">> GPU mem after teardown: "
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s" MiB total"}'
}
trap teardown EXIT TERM INT

echo ">> launching vLLM bf16 TP=8 --enforce-eager on :$PORT"
vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!

echo ">> waiting for readiness (<= ~6 min)"
ready=0
for i in $(seq 1 36); do
  if curl -s --max-time 4 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo ">> READY after ~$((i*10))s"; ready=1; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ">> server process died during startup:"; tail -20 "$LOG"; exit 1
  fi
  sleep 10
done
[ "$ready" = 1 ] || { echo ">> not ready in time:"; tail -20 "$LOG"; exit 1; }

echo ">> running B=1 benchmark (512-in / 128-out, 10 reps, 2 warmup)"
python3 "$REPO/tools/bench_b1_client.py" \
  --base "http://localhost:$PORT" \
  --model "$SERVED" \
  --prompt-tokens 512 --max-tokens 128 \
  --repeats 10 --warmup 2 \
  --out "$OUT"

echo ">> done; results at $OUT"
# teardown runs via the EXIT trap
