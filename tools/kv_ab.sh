#!/usr/bin/env bash
# KV-cache FP8 A/B — runs ONE kv-cache-dtype's full ctx sweep + quality capture,
# sized to fit a single 15-min GPU slot (model load ~135s + sweep). Run twice
# across slots: `kv_ab.sh auto` then `kv_ab.sh fp8`, then compare offline.
#
#   bash tools/kv_ab.sh {auto|fp8} [PORT]
#
# Non-interference: serializes on GPU free memory (waits if MIN free < 65 GB),
# uses port 8088, never touches the adaptive-topk loop's port 8077 / paths.
set -u
KV="${1:?usage: kv_ab.sh {auto|fp8} [port]}"
PORT="${2:-8088}"
REPO="${REPO:-/alloc/data/InferenceHackathon}"   # override to an isolated worktree
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"   # FP8 weights (HF-cached)
SERVED=qwen3
OUT="$REPO/results/kv_fp8/$KV"
LOG=/root/vllm_kv_${KV}.log
mkdir -p "$OUT"

# --- GPU gate: do not launch if the box is in use by the other loop ---
minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
echo "=== kv=$KV : MIN GPU free=${minfree} MiB $(date -u +%H:%M:%S)UTC ==="
if [ "${minfree:-0}" -lt 65000 ]; then
  echo "GPUs busy (min free ${minfree} < 65000 MiB) — NOT launching. Retry next slot." ; exit 3
fi

# --- launch vLLM: FP8 weights + EP + CUDA graphs (no --enforce-eager) ---
# baseline kv=auto (fp16/bf16 KV) vs kv=fp8 (e4m3, default scales). B=1.
pkill -9 -f "served-model-name $SERVED" 2>/dev/null; sleep 2
setsid python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED" \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --max-num-seqs 1 --max-model-len 36864 \
  --gpu-memory-utilization 0.90 --trust-remote-code \
  --kv-cache-dtype "$KV" --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
VPID=$!

ready=0
for i in $(seq 1 96); do   # up to 8 min for load
  curl -s -m 4 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED" \
    && { ready=1; echo "=== READY ~$((i*5))s $(date -u +%H:%M:%S)UTC ==="; break; }
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "=== kv=$KV did NOT come up — tail log: ==="; tail -25 "$LOG"
  kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name $SERVED" 2>/dev/null; exit 1
fi

# --- ctx sweep: the FP8-KV win grows with context length ---
for ctx in 128 2048 8192 32768; do
  echo "--- kv=$KV ctx=$ctx ---"
  python3 "$REPO/tools/kv_measure.py" --base "http://localhost:$PORT" --model "$SERVED" \
    --ctx "$ctx" --decode 128 --warmup 2 --repeat 3 --label "kv=$KV ctx=$ctx" \
    --json-out "$OUT/ctx_${ctx}.json"
done

# --- quality capture (greedy; compared offline vs the other dtype) ---
echo "--- kv=$KV quality capture ---"
python3 "$REPO/tools/kv_quality.py" capture --base "http://localhost:$PORT" \
  --model "$SERVED" --out "$OUT/quality.json"

echo "=== kv=$KV DONE $(date -u +%H:%M:%S)UTC — results in $OUT ==="
kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name $SERVED" 2>/dev/null
