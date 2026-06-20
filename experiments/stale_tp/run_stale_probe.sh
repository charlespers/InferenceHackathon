#!/usr/bin/env bash
# LOOP-C stale-TP QUALITY PROBE slot runner.  (port 8099)
#
# Answers ONE question: how much TP all-reduce staleness does Qwen3-235B tolerate
# before greedy decode diverges from exact? (research/n4_speculative_stale_tp.md §4)
# This is a QUALITY probe — it does NOT speed up decode; it measures the ceiling so
# we know whether a real stale-TP kernel is worth building.
#
# ONE vLLM launch, sweep K/mode/policy via the control file (STALE_TP_CTL) edited
# between quality_probe runs — so we do NOT reload the 235B per config (slot-cheap).
# Runs on bf16-TP8 (the measured 85.7 tok/s baseline regime; FP8 is ~25% slower at
# B=1 per Alyssa). Uses SYSTEM vLLM (no EAGLE3 → 0.10.x is fine); no isolated venv.
#
# Guards mirror tools/slot_eagle3.sh: stale-lock cleanup, atomic mkdir lock, min-free
# >65GB, time-window wait, holder-then-rmdir release. NEVER launches outside guards.
set -u
REPO=${REPO:-/alloc/data/InferenceHackathon}
TOOLSRC=/alloc/data/stale_tp_tools          # standalone copy of stale_tp.py (avoid jminding's checkout)
PY=/usr/bin/python3                          # clients are stdlib-only
PYV=${PYV:-python3}                          # interpreter that has vLLM importable
MODEL=${MODEL:-/alloc/data/Qwen3-235B-A22B}  # bf16
OUT=/alloc/data/stale_tp; mkdir -p "$OUT"
LOG="$OUT/slot_stale.log"
SCHED=/alloc/data/danielAgentScheduling.md
CTL=/alloc/data/stale_tp.ctl
PORT=8099
TOK=${TOK:-96}                               # greedy tokens per prompt (parity probe)

# Sweep points: "mode policy K label". layer/proxy is the N4 hypothesis; layer/local
# is the sanity FLOOR (must degrade → proves the probe detects breakage); higher K and
# temporal are opportunistic. Ordered by priority (slot may end early).
SWEEP=(
  "layer proxy 2 lyr_proxy_k2"
  "layer proxy 4 lyr_proxy_k4"
  "layer local 2 lyr_local_k2"
  "layer proxy 8 lyr_proxy_k8"
  "temporal proxy 2 tmp_proxy_k2"
)

mins () { echo $((10#$(date +%M))); }
freemin () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1; }
write_ctl () { # $1=enable $2=mode $3=policy $4=K
  printf '{"enable": %s, "mode": "%s", "policy": "%s", "K": %s, "decode_only": true}\n' \
    "$1" "$2" "$3" "$4" > "$CTL"; sleep 1;  # mtime bump; scheduler reloads next pass
}
probe () { # $1=label  -> greedy capture under the CURRENT ctl
  echo "  probe $1 ($(date -u)) ctl=$(cat $CTL 2>/dev/null)" >> "$LOG"
  $PY "$TOOLSRC/quality_probe.py" --base http://localhost:$PORT --model qwen3 \
      --tokens "$TOK" --out "$OUT/q_$1.json" >> "$LOG" 2>&1
}

echo "armed $(date -u) — stale-TP probe waiting for the FIRST genuinely-idle window" > "$LOG"
# DEFER to any running job, then grab the FIRST idle window (the probe is short ~7min
# and releases the lock immediately, so borrowing one idle window is low-impact). Fire
# when ALL hold: lock-free, NO active vLLM serve (EAGLE3/other running -> wait), >65GB
# free. This catches the moment EAGLE3 releases (~:00) instead of waiting a whole slot.
# Require a 2-cycle confirm so we don't race a job that's mid-teardown. Capped wait.
WAITED=0; idle_streak=0
while :; do
  free=$(freemin); busy=0
  pgrep -f 'vllm.entrypoints|vllm serve' >/dev/null 2>&1 && busy=1
  if [ "$busy" -eq 0 ] && [ "$free" -gt 65000 ] && [ ! -d /alloc/data/gpu.lock ]; then
    idle_streak=$((idle_streak + 1))
  else
    idle_streak=0
  fi
  if [ "$idle_streak" -ge 2 ]; then
    echo "IDLE window confirmed $(date -u) (free=${free}MB, no active vLLM, lock-free)" >> "$LOG"; break
  fi
  WAITED=$((WAITED + 20))
  if [ "$WAITED" -ge 5700 ]; then echo "no idle window in 95min — giving up $(date -u)" >> "$LOG"; exit 0; fi
  sleep 20
done

# stale-lock cleanup (>20 min) then atomic acquire
[ -d /alloc/data/gpu.lock ] && [ -n "$(find /alloc/data/gpu.lock -mmin +20 2>/dev/null)" ] && { rm -f /alloc/data/gpu.lock/holder; rmdir /alloc/data/gpu.lock 2>/dev/null; }
FREE=$(freemin); echo "min GPU free ${FREE}MB" >> "$LOG"
if [ "$FREE" -gt 65000 ] && mkdir /alloc/data/gpu.lock 2>/dev/null; then
  echo "LOOP-C(stale-tp) $(date -u)" > /alloc/data/gpu.lock/holder
  echo "- $(date -u) LOOP-C: acquired gpu.lock -> stale-TP quality probe (port $PORT)" >> "$SCHED"

  # launch ONCE with stale_tp installed; ctl starts as exact (enable=false)
  write_ctl false layer proxy 1
  echo "=== launch bf16-TP8 + stale_tp ($(date -u)) ===" >> "$LOG"
  STALE_TP_CTL=$CTL STALE_TP_ENABLE=0 STALE_TP_PERIOD=188 STALE_TP_DEBUG=1 \
  $PYV -c "import sys; sys.path.insert(0,'$TOOLSRC'); import stale_tp; stale_tp.install(); import runpy; runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')" -- \
      --model "$MODEL" --served-model-name qwen3 \
      --tensor-parallel-size 8 --max-num-seqs 1 --max-model-len 8192 \
      --no-enable-prefix-caching --enforce-eager --gpu-memory-utilization 0.9 \
      --port $PORT > "$OUT/vllm_stale.log" 2>&1 &
  VPID=$!
  ok=0
  for i in $(seq 1 240); do
    curl -sf -m3 http://localhost:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $VPID 2>/dev/null || { echo "vLLM exited early — see vllm_stale.log" >> "$LOG"; break; }
    [ "$(mins)" -ge 58 ] && { echo "TIME GUARD — abort readiness wait" >> "$LOG"; break; }
    sleep 5
  done

  if [ $ok -eq 1 ]; then
    # 0) EXACT baseline greedy (the reference) + a tok/s sanity number
    probe exact
    $PY "$TOOLSRC/measure_baseline.py" --base http://localhost:$PORT --model qwen3 \
        --decode 64 --repeats 3 --out "$OUT/m_exact.json" >> "$LOG" 2>&1
    # confirm the all-reduce period the scheduler observed (calibrates STALE_TP_PERIOD)
    grep -i "observed" "$OUT/vllm_stale.log" | tail -2 >> "$LOG"

    # 1) sweep — each point: rewrite ctl, probe greedy. Stop if slot is ending.
    for spec in "${SWEEP[@]}"; do
      [ "$(mins)" -ge 58 ] && { echo "out of slot time — stop sweep" >> "$LOG"; break; }
      set -- $spec; mode=$1; policy=$2; K=$3; label=$4
      write_ctl true "$mode" "$policy" "$K"
      probe "$label"
    done
  fi
  kill $VPID 2>/dev/null; sleep 15
  rm -f /alloc/data/gpu.lock/holder; rmdir /alloc/data/gpu.lock 2>/dev/null
  echo "- $(date -u) LOOP-C: released gpu.lock (results /alloc/data/stale_tp/)" >> "$SCHED"
else
  echo "GPUs busy (${FREE}MB) or gpu.lock held -> not launching" >> "$LOG"
fi

# Parity gates: each sweep point's greedy vs EXACT. >~99% prefix agreement at K>=2 (proxy)
# => stale-TP is quality-tolerable without retraining (GO). local should degrade (floor check).
if [ -f "$OUT/q_exact.json" ]; then
  for spec in "${SWEEP[@]}"; do
    set -- $spec; label=$4
    [ -f "$OUT/q_$label.json" ] || continue
    echo "=== parity gate: exact vs $label ===" >> "$LOG"
    $PY "$TOOLSRC/quality_compare.py" "$OUT/q_exact.json" "$OUT/q_$label.json" \
        --out "$OUT/parity_$label.json" >> "$LOG" 2>&1
  done
fi

# Share results (box can't push main; results/* gitignored -> add -f to a results branch)
RES=$(ls "$OUT"/q_*.json "$OUT"/parity_*.json "$OUT"/m_*.json 2>/dev/null)
if [ -n "$RES" ]; then
  git -C "$REPO" fetch origin -q 2>/dev/null; rm -rf /tmp/stale_reswt
  if git -C "$REPO" worktree add -f /tmp/stale_reswt origin/main >/dev/null 2>&1; then
    mkdir -p /tmp/stale_reswt/results/stale_tp
    cp $RES "$LOG" /tmp/stale_reswt/results/stale_tp/ 2>/dev/null
    ( cd /tmp/stale_reswt; git add -f results/stale_tp 2>/dev/null
      git -c user.email=djamoils25@gmail.com -c user.name="loopc-box" \
        commit -q -m "results: stale-TP quality probe $(date -u)" 2>/dev/null \
        && git push -q -f origin HEAD:loopc-results 2>>"$LOG" \
        && echo "results pushed -> origin/loopc-results" >> "$LOG" )
    git -C "$REPO" worktree remove -f /tmp/stale_reswt 2>/dev/null
  fi
fi
echo "slot done $(date -u)" >> "$LOG"; touch "$OUT/slot_stale.DONE"
