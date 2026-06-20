#!/usr/bin/env bash
# LOOP-A (djamoils) EAGLE3 spec-decode slot runner.
#
# Runs in the :45-:00 UTC slot. Uses the ISOLATED venv /alloc/data/eagle3-venv
# (vLLM 0.11.0) so the team's system vLLM 0.10.1 is untouched. Two launches,
# priority order (each gated by a time guard so a partial slot still yields value):
#   1. EAGLE3 eager  (--enforce-eager + spec-config) — DE-RISK: proves the MoE+EP
#      EAGLE3 path loads & is lossless; captures accept-length + eager decode tok/s.
#   2. baseline FP8  (graphs, no spec)               — the speedup denominator +
#      the greedy reference for the lossless parity gate.
# Computes the parity gate whenever both greedy captures exist (persisted across
# slots in /alloc/data/eagle3/), then pushes results to origin/djamoils-results.
#
# The CUDA-graph EAGLE3 headline (drop --enforce-eager) is a SEPARATE slot — only
# trust it after eager parity passes (INTEGRATION.md §4: graphs may crash on MoE+EP).
set -u
VENV=/alloc/data/eagle3-venv
VLLM="$VENV/bin/vllm"
PY=/usr/bin/python3                       # measurement clients are stdlib-only
TOOLS=/alloc/data/eagle3_tools            # standalone copies (avoid jminding's checkout)
OUT=/alloc/data/eagle3; mkdir -p "$OUT"
LOG=/alloc/data/slot_eagle3.log
SCHED=/alloc/data/danielAgentScheduling.md
MODEL=${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}
HEAD=nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys
SPEC="{\"method\":\"eagle3\",\"model\":\"$HEAD\",\"num_speculative_tokens\":3,\"draft_tensor_parallel_size\":1}"
PORT=8077
NSPEC=${NSPEC:-3}

echo "armed $(date -u) — waiting for a FRESH :45 slot (EAGLE3)" > "$LOG"
# If armed mid-slot (:45-:59), first wait OUT to :00 so we catch a full window.
while [ $((10#$(date +%M))) -ge 45 ]; do sleep 15; done
while [ $((10#$(date +%M))) -lt 45 ]; do sleep 10; done
echo "slot start $(date -u)" >> "$LOG"

mins () { echo $((10#$(date +%M))); }     # current UTC minute, base-10 safe
freemin () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1; }

# $1=label  $2=extra vllm args (spec/eager)  $3=gpu-mem-util
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
    echo "$label serving — measuring decode tok/s + capturing greedy" >> "$LOG"
    $PY "$TOOLS/measure_baseline.py" --base http://localhost:$PORT --model qwen3 \
        --decode 64 --repeats 3 --out "$OUT/m_$label.json" >> "$LOG" 2>&1
    $PY "$TOOLS/quality_probe.py" --base http://localhost:$PORT --model qwen3 \
        --tokens 96 --out "$OUT/q_$label.json" >> "$LOG" 2>&1
    echo "--- $label spec/accept metrics (server log) ---" >> "$LOG"
    grep -iE 'accept|draft|spec|efficien' "$OUT/vllm_$label.log" | tail -8 >> "$LOG"
    # vLLM v1 reports spec counters via Prometheus, not always stdout — scrape both.
    curl -s -m5 http://localhost:$PORT/metrics 2>/dev/null \
      | grep -iE 'spec_decode|accept|draft|num_emitted' > "$OUT/metrics_$label.txt" 2>/dev/null
    echo "  scraped $(wc -l < "$OUT/metrics_$label.txt" 2>/dev/null || echo 0) spec metric lines -> metrics_$label.txt" >> "$LOG"
  fi
  kill $vpid 2>/dev/null; sleep 20    # free HBM before the next launch
}

# stale-lock cleanup (>20 min) then atomic acquire
[ -d /alloc/data/gpu.lock ] && [ -n "$(find /alloc/data/gpu.lock -mmin +20 2>/dev/null)" ] && rmdir /alloc/data/gpu.lock 2>/dev/null
FREE=$(freemin); echo "min GPU free ${FREE}MB" >> "$LOG"
if [ "$FREE" -gt 65000 ] && mkdir /alloc/data/gpu.lock 2>/dev/null; then
  echo "LOOP-A(eagle3) $(date -u)" > /alloc/data/gpu.lock/holder
  echo "- $(date -u) LOOP-A: acquired gpu.lock -> EAGLE3 eager+baseline" >> "$SCHED"

  # 1) EAGLE3 eager — the de-risk + accept-length + lossless capture (most important)
  launch_measure eagle3_eager "--enforce-eager --speculative-config $SPEC" 0.85

  # 2) baseline FP8 graphs — speedup denominator + parity reference (if time remains)
  if [ "$(mins)" -lt 56 ]; then
    launch_measure baseline_fp8 "" 0.9
  else
    echo "skip baseline (out of slot time)" >> "$LOG"
  fi

  rmdir /alloc/data/gpu.lock 2>/dev/null
  echo "- $(date -u) LOOP-A: released gpu.lock (results /alloc/data/eagle3/)" >> "$SCHED"
else
  echo "GPUs busy (${FREE}MB) or gpu.lock held -> NOT my window, skipping" >> "$LOG"
fi

# Lossless parity gate: EAGLE3 greedy must EXACTLY match baseline greedy.
if [ -f "$OUT/q_baseline_fp8.json" ] && [ -f "$OUT/q_eagle3_eager.json" ]; then
  echo "=== EAGLE3 lossless parity gate (baseline vs eagle3 greedy) ===" >> "$LOG"
  $PY "$TOOLS/quality_compare.py" "$OUT/q_baseline_fp8.json" "$OUT/q_eagle3_eager.json" \
      --out "$OUT/parity_gate.json" >> "$LOG" 2>&1
fi

# Share results to origin/djamoils-results (box can't push main; results/* gitignored)
RES=$(ls "$OUT"/m_*.json "$OUT"/parity_gate.json "$OUT"/q_*.json "$OUT"/metrics_*.txt 2>/dev/null)
if [ -n "$RES" ]; then
  REPO=/alloc/data/InferenceHackathon
  git -C "$REPO" fetch origin -q 2>/dev/null
  rm -rf /tmp/eag_reswt
  if git -C "$REPO" worktree add -f /tmp/eag_reswt origin/main >/dev/null 2>&1; then
    mkdir -p /tmp/eag_reswt/results/eagle3
    cp $RES /tmp/eag_reswt/results/eagle3/ 2>/dev/null
    cp "$LOG" /tmp/eag_reswt/results/eagle3/ 2>/dev/null
    ( cd /tmp/eag_reswt; git add -f results/eagle3 2>/dev/null
      git -c user.email=djamoils25@gmail.com -c user.name="djamoils-box" \
        commit -q -m "results: EAGLE3 slot $(date -u)" 2>/dev/null \
        && git push -q -f origin HEAD:djamoils-results 2>>"$LOG" \
        && echo "results pushed -> origin/djamoils-results" >> "$LOG" )
    git -C "$REPO" worktree remove -f /tmp/eag_reswt 2>/dev/null
  fi
fi
echo "slot done $(date -u)" >> "$LOG"
touch /alloc/data/slot_eagle3.DONE
