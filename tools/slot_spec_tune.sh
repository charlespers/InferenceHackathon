#!/usr/bin/env bash
# LOOP-A (djamoils) SPEC TUNING slot — improve the spec speedup by sweeping the two levers Charles's
# flat-in-k verify points to: tree size k (bigger = more E[accepted], nearly free verify) and
# draft_tensor_parallel_size (the 1.7B head is bandwidth-bound at draft_tp=1).
# Arms (each = one eagle3 launch, capture sizes fixed to multiples of k+1, measure tau + decode tok/s):
#   A) k=8, draft_tp=1   — does a bigger tree raise tau / S_spec? (vs the 14:45 k=3 baseline)
#   B) k=3, draft_tp=8   — does sharding the head speed the draft? (may FAIL to shard -> logged)
# Compare tau and decode tok/s across arms + the 14:45 eagle3_graphs(k=3,dtp1). Same venv/target.
set -u
VENV=/alloc/data/eagle3-venv
VLLM="$VENV/bin/vllm"
PY=/usr/bin/python3
TOOLS=/alloc/data/eagle3_tools
OUT=/alloc/data/eagle3_tune; mkdir -p "$OUT"
LOG=/alloc/data/slot_tune.log
SCHED=/alloc/data/danielAgentScheduling.md
MODEL=${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}
HEAD=${HEAD:-RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3}
DECODE=${DECODE:-64}
spec_cfg () { echo "{\"method\":\"eagle3\",\"model\":\"$HEAD\",\"num_speculative_tokens\":$1,\"draft_tensor_parallel_size\":$2}"; }
comp_cfg () { local kp1=$(( $1 + 1 )); echo "{\"cudagraph_capture_sizes\":[$kp1,$((2*kp1)),$((4*kp1))],\"max_cudagraph_capture_size\":$((4*kp1))}"; }
PORT=8077

echo "armed $(date -u) tune — waiting for a FRESH :45 slot" > "$LOG"
while [ $((10#$(date +%M))) -ge 45 ]; do sleep 15; done
while [ $((10#$(date +%M))) -lt 45 ]; do sleep 10; done
echo "slot start $(date -u)" >> "$LOG"

mins () { echo $((10#$(date +%M))); }
freemin () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1; }

launch_measure () {
  local label="$1" extra="$2" gmu="$3" vpid ok=0
  echo "=== launch $label $(date -u): $extra ===" >> "$LOG"
  $VLLM serve "$MODEL" --served-model-name qwen3 \
      --tensor-parallel-size 8 --enable-expert-parallel \
      --max-num-seqs 1 --max-model-len 8192 --no-enable-prefix-caching \
      --gpu-memory-utilization "$gmu" --port $PORT $extra \
      > "$OUT/vllm_$label.log" 2>&1 &
  vpid=$!
  for i in $(seq 1 180); do
    curl -sf -m3 http://localhost:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $vpid 2>/dev/null || { echo "$label vLLM exited early — see vllm_$label.log" >> "$LOG"; break; }
    [ "$(mins)" -ge 59 ] && { echo "TIME GUARD :59 — abort $label readiness wait" >> "$LOG"; break; }
    sleep 5
  done
  if [ $ok -eq 1 ]; then
    echo "$label serving — measuring decode tok/s (decode=$DECODE x3) + accept-len" >> "$LOG"
    $PY "$TOOLS/measure_baseline.py" --base http://localhost:$PORT --model qwen3 \
        --decode $DECODE --repeats 3 --out "$OUT/m_$label.json" >> "$LOG" 2>&1
    curl -s -m5 http://localhost:$PORT/metrics 2>/dev/null \
      | grep -iE 'spec_decode|accept|draft|num_emitted' > "$OUT/metrics_$label.txt" 2>/dev/null
    grep -i "acceptance length" "$OUT/vllm_$label.log" | tail -1 >> "$LOG"
    echo "  m_$label.json: $(grep -o '\"decode_tok_s\"[^,}]*' "$OUT/m_$label.json" 2>/dev/null | head -1)" >> "$LOG"
  fi
  kill $vpid 2>/dev/null; sleep 20
  LAST_OK=$ok
}

[ -d /alloc/data/gpu.lock ] && [ -n "$(find /alloc/data/gpu.lock -mmin +20 2>/dev/null)" ] && { rm -f /alloc/data/gpu.lock/holder; rmdir /alloc/data/gpu.lock 2>/dev/null; }
FREE=$(freemin); echo "min GPU free ${FREE}MB" >> "$LOG"
if [ "$FREE" -gt 65000 ] && mkdir /alloc/data/gpu.lock 2>/dev/null; then
  echo "LOOP-A(tune) $(date -u)" > /alloc/data/gpu.lock/holder
  echo "- $(date -u) LOOP-A: acquired gpu.lock -> SPEC TUNE (k=8/dtp1 ; k=3/dtp8)" >> "$SCHED"

  # A) bigger tree: k=8, draft_tp=1
  launch_measure "eagle3_k8_dtp1" "--speculative-config $(spec_cfg 8 1) --compilation-config $(comp_cfg 8)" 0.85

  # B) sharded head: k=3, draft_tp=8 (may fail to shard -> falls to the log; harmless)
  if [ "$(mins)" -lt 55 ]; then
    launch_measure "eagle3_k3_dtp8" "--speculative-config $(spec_cfg 3 8) --compilation-config $(comp_cfg 3)" 0.85
  else
    echo "skip k3_dtp8 (out of slot time)" >> "$LOG"
  fi

  rm -f /alloc/data/gpu.lock/holder; rmdir /alloc/data/gpu.lock 2>/dev/null
  echo "- $(date -u) LOOP-A: released gpu.lock (tune results /alloc/data/eagle3_tune/)" >> "$SCHED"
else
  echo "GPUs busy (${FREE}MB) or gpu.lock held -> NOT my window, skipping" >> "$LOG"
fi
echo "tune done $(date -u)" >> "$LOG"
touch /alloc/data/slot_tune.DONE
