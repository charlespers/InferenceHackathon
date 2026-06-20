#!/usr/bin/env bash
# KV-cache FP8 A/B — runs ONE kv-cache-dtype's full ctx sweep + quality capture,
# sized to fit a single 15-min GPU slot. Run twice across slots:
# `kv_ab.sh auto` then `kv_ab.sh fp8`, then compare offline.
#
#   [REPO=/alloc/data/kvquant] bash tools/kv_ab.sh {auto|fp8} [PORT]
#
# Coordination (see danielAgentScheduling.md — shared with the sibling LOOP-A):
#  - djamoils owns :45-:00 UTC. Refuses to launch outside that window.
#  - Mutual-exclusion vs LOOP-A via the AGREED atomic lock `mkdir /alloc/data/gpu.lock`
#    + holder file (stale after 20 min) AND the >65GB-free gate. Port 8088 (LOOP-A: 8077).
#  - Hard slot-end deadline: tears the server down ~30s before :00 so it can never
#    overrun into Jaymin's :00 slot. Skips remaining ctx points if time is short
#    (logged, never silently dropped).
set -u
KV="${1:?usage: kv_ab.sh {auto|fp8} [port]}"
PORT="${2:-8088}"
REPO="${REPO:-/alloc/data/InferenceHackathon}"   # override to an isolated worktree
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"   # FP8 weights (HF-cached)
SERVED=qwen3
OUT="$REPO/results/kv_fp8/$KV"
LOG=/root/vllm_kv_${KV}.log
LOCKDIR=/alloc/data/gpu.lock     # shared atomic lock (mkdir) — agreed with LOOP-A
HOLDER="$LOCKDIR/holder"
LOOP="LOOP-B-kvfp8"
GUARD_MARGIN=30   # stop this many seconds before the slot boundary
mkdir -p "$OUT"

now_min() { date -u +%-M; }
now_sec() { date -u +%-S; }
# seconds until the end of djamoils' slot (:00). If we're at :45-:59, that's the
# top of the next hour; the deadline subtracts GUARD_MARGIN.
secs_to_slot_end() { local m s; m=$(now_min); s=$(now_sec); echo $(( (60 - m) * 60 - s )); }

cleanup() {
  kill -9 -"${VPID:-0}" 2>/dev/null
  pkill -9 -f "served-model-name $SERVED" 2>/dev/null
  # release the shared lock only if WE hold it
  if [ -f "$HOLDER" ] && grep -q "$LOOP" "$HOLDER" 2>/dev/null; then
    rm -f "$HOLDER"; rmdir "$LOCKDIR" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

# --- slot ownership: only run in djamoils' :45-:00 window ---
M=$(now_min)
if [ "$M" -lt 45 ]; then
  echo "DENY: not djamoils' slot (UTC :$M; slot is :45-:00). Not launching." ; exit 2
fi
BUDGET=$(( $(secs_to_slot_end) - GUARD_MARGIN ))
echo "=== kv=$KV slot OK, ${BUDGET}s budget to deadline $(date -u +%H:%M:%S)UTC ==="
if [ "$BUDGET" -lt 180 ]; then
  echo "DENY: only ${BUDGET}s left in slot — not enough to load+measure. Wait for next slot." ; exit 2
fi

# --- atomic shared lock (agreed convention with LOOP-A) ---
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  if [ -f "$HOLDER" ] && [ -n "$(find "$HOLDER" -mmin +20 2>/dev/null)" ]; then
    echo "Lock STALE (>20min): $(cat "$HOLDER" 2>/dev/null) — taking over."
    rm -f "$HOLDER"; rmdir "$LOCKDIR" 2>/dev/null
    mkdir "$LOCKDIR" 2>/dev/null || { echo "DENY: lost race for lock — back off." ; exit 3; }
  else
    echo "DENY: GPU lock held by $(cat "$HOLDER" 2>/dev/null). Back off to next slot." ; exit 3
  fi
fi
echo "$LOOP kv=$KV pid=$$ port=$PORT $(date -u +%H:%M:%S)UTC" > "$HOLDER"

# --- GPU gate: do not launch if the box is in use (TOCTOU-narrowed by the lock) ---
minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
echo "=== kv=$KV : MIN GPU free=${minfree} MiB ==="
if [ "${minfree:-0}" -lt 65000 ]; then
  echo "GPUs busy (min free ${minfree} < 65000 MiB) — NOT launching. Retry next slot." ; exit 3
fi

# --- launch vLLM: FP8 weights + EP + CUDA graphs (no --enforce-eager) ---
pkill -9 -f "served-model-name $SERVED" 2>/dev/null; sleep 2
setsid python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED" \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --max-num-seqs 1 --max-model-len 36864 \
  --gpu-memory-utilization 0.90 --trust-remote-code \
  --kv-cache-dtype "$KV" --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
VPID=$!

# --- watchdog: hard-kill at the slot deadline no matter what ---
( sleep "$BUDGET"; echo "=== DEADLINE reached — killing vLLM to vacate slot ===" >> "$LOG"; cleanup ) &
WPID=$!

ready=0
for i in $(seq 1 96); do   # up to 8 min for load (watchdog bounds it to slot end)
  curl -s -m 4 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED" \
    && { ready=1; echo "=== READY ~$((i*5))s $(date -u +%H:%M:%S)UTC ==="; break; }
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "=== kv=$KV did NOT come up — tail log: ==="; tail -25 "$LOG"; exit 1
fi
grep -iE "flashattention|FlashAttn|FA3|attention backend|backend" "$LOG" | tail -3 || true

# --- ctx sweep: the FP8-KV win grows with context length. Deadline-aware. ---
for ctx in 128 2048 8192 16384 32768; do   # 16k brackets the roofline crossover
  left=$(( $(secs_to_slot_end) - GUARD_MARGIN ))
  need=$(( ctx >= 32768 ? 100 : 45 ))   # rough per-point budget
  if [ "$left" -lt "$need" ]; then
    echo "SKIP ctx=$ctx — only ${left}s left (<${need}s). Logged, not silently dropped." | tee -a "$OUT/SKIPPED.txt"
    continue
  fi
  echo "--- kv=$KV ctx=$ctx (${left}s left) ---"
  python3 "$REPO/tools/kv_measure.py" --base "http://localhost:$PORT" --model "$SERVED" \
    --ctx "$ctx" --decode 128 --warmup 2 --repeat 3 --label "kv=$KV ctx=$ctx" \
    --json-out "$OUT/ctx_${ctx}.json"
done

# --- quality capture (greedy; compared offline vs the other dtype) ---
left=$(( $(secs_to_slot_end) - GUARD_MARGIN ))
if [ "$left" -gt 120 ]; then
  echo "--- kv=$KV quality capture (${left}s left) ---"
  python3 "$REPO/tools/kv_quality.py" capture --base "http://localhost:$PORT" \
    --model "$SERVED" --out "$OUT/quality.json"
else
  echo "SKIP quality capture — only ${left}s left. Run in a later slot." | tee -a "$OUT/SKIPPED.txt"
fi

echo "=== kv=$KV DONE $(date -u +%H:%M:%S)UTC — results in $OUT ==="
kill "$WPID" 2>/dev/null   # cancel watchdog; trap cleanup tears down the server
