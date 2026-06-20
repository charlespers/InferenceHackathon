#!/usr/bin/env bash
# LOOP-A (djamoils) SPEC measurement slot — get the deferred eagle3_graphs arm + a matched baseline.
# Runs eagle3_graphs FIRST (so the :53 guard can't skip it again), then baseline_graphs for the ratio.
#   S_spec = tok/s(eagle3_graphs)/tok/s(baseline_graphs) ; tau from /metrics ; V=tau/S_spec.
# eagle3 uses the cudagraph capture-size FIX (sizes = multiples of NSPEC+1). Fresh OUT dir so the
# 13:45 baseline results are preserved. Keeps the proven guards (mem>65000 + atomic lock + release).
set -u
VENV=/alloc/data/eagle3-venv
VLLM="$VENV/bin/vllm"
PY=/usr/bin/python3
TOOLS=/alloc/data/eagle3_tools
OUT=/alloc/data/eagle3_diag2; mkdir -p "$OUT"
LOG=/alloc/data/slot_spec.log
SCHED=/alloc/data/danielAgentScheduling.md
MODEL=${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}
HEAD=${HEAD:-RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3}
DRAFT_TP=${DRAFT_TP:-1}
NSPEC=${NSPEC:-3}
DECODE=${DECODE:-64}
KP1=$(( NSPEC + 1 ))
COMPCFG="{\"cudagraph_capture_sizes\":[$KP1,$((2*KP1)),$((4*KP1))],\"max_cudagraph_capture_size\":$((4*KP1))}"
spec_cfg () { echo "{\"method\":\"eagle3\",\"model\":\"$HEAD\",\"num_speculative_tokens\":$1,\"draft_tensor_parallel_size\":$DRAFT_TP}"; }
PORT=8077

echo "armed $(date -u) spec — waiting for a FRESH :45 slot" > "$LOG"
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
    echo "$label serving — measuring decode tok/s (decode=$DECODE x3) + greedy" >> "$LOG"
    $PY "$TOOLS/measure_baseline.py" --base http://localhost:$PORT --model qwen3 \
        --decode $DECODE --repeats 3 --out "$OUT/m_$label.json" >> "$LOG" 2>&1
    $PY "$TOOLS/quality_probe.py" --base http://localhost:$PORT --model qwen3 \
        --tokens 64 --out "$OUT/q_$label.json" >> "$LOG" 2>&1
    curl -s -m5 http://localhost:$PORT/metrics 2>/dev/null \
      | grep -iE 'spec_decode|accept|draft|num_emitted' > "$OUT/metrics_$label.txt" 2>/dev/null
    echo "  m_$label.json: $(grep -o '\"decode_tok_s\"[^,}]*' "$OUT/m_$label.json" 2>/dev/null | head -1)" >> "$LOG"
  fi
  kill $vpid 2>/dev/null; sleep 20
  LAST_OK=$ok
}

[ -d /alloc/data/gpu.lock ] && [ -n "$(find /alloc/data/gpu.lock -mmin +20 2>/dev/null)" ] && { rm -f /alloc/data/gpu.lock/holder; rmdir /alloc/data/gpu.lock 2>/dev/null; }
FREE=$(freemin); echo "min GPU free ${FREE}MB" >> "$LOG"
if [ "$FREE" -gt 65000 ] && mkdir /alloc/data/gpu.lock 2>/dev/null; then
  echo "LOOP-A(spec) $(date -u)" > /alloc/data/gpu.lock/holder
  echo "- $(date -u) LOOP-A: acquired gpu.lock -> SPEC (eagle3_graphs FIRST + baseline_graphs; decode $DECODE)" >> "$SCHED"

  # 1) eagle3_graphs FIRST (guaranteed) — spec speedup + accept-length tau, with the capture-size fix
  launch_measure "eagle3_graphs" "--speculative-config $(spec_cfg $NSPEC) --compilation-config $COMPCFG" 0.85

  # 2) baseline_graphs — matched denominator for S_spec (if a full launch fits)
  if [ "$(mins)" -lt 55 ]; then
    launch_measure "baseline_graphs" "" 0.9
  else
    echo "skip baseline_graphs (out of slot time) — reuse 13:45 baseline_graphs=29.92" >> "$LOG"
  fi

  rm -f /alloc/data/gpu.lock/holder; rmdir /alloc/data/gpu.lock 2>/dev/null
  echo "- $(date -u) LOOP-A: released gpu.lock (spec results /alloc/data/eagle3_diag2/)" >> "$SCHED"
else
  echo "GPUs busy (${FREE}MB) or gpu.lock held -> NOT my window, skipping" >> "$LOG"
fi

if [ -f "$OUT/q_baseline_graphs.json" ] && [ -f "$OUT/q_eagle3_graphs.json" ]; then
  $PY "$TOOLS/quality_compare.py" "$OUT/q_baseline_graphs.json" "$OUT/q_eagle3_graphs.json" \
      --out "$OUT/parity_gate_graphs.json" >> "$LOG" 2>&1
fi
echo "spec done $(date -u)" >> "$LOG"
touch /alloc/data/slot_spec.DONE
