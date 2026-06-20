#!/usr/bin/env bash
# djamoils slot A/B: in the :45-:00 slot, if GPUs are free, launch vLLM twice on
# the bf16 Qwen3-235B + --enable-expert-parallel (regime A) and measure B=1 decode:
#   mode baseline (ADAPTIVE_TOPK_ENABLE=0) = plain top-8  -> also the E1 baseline
#   mode adaptive (ENABLE=1 K=4 THRESH=0.9) = confidence-adaptive top-k
# Compares decode tok/s + greedy output + the patch's realized drop-rate. Pushes
# results to origin/djamoils-results. Hard time-guard at :58 protects Jaymin's slot.
set -u
LOG=/alloc/data/slot_run.log
REPO=/alloc/data/InferenceHackathon
MODEL=/alloc/data/Qwen3-235B-A22B
PORT=8077
echo "armed $(date -u), waiting for :45 slot (adaptive top-k A/B)" > "$LOG"
while :; do m=$((10#$(date +%M))); [ "$m" -ge 45 ] && break; sleep 10; done
echo "slot start $(date -u)" >> "$LOG"
cd "$REPO" || exit 1
freemb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
echo "min GPU free ${freemb}MB" >> "$LOG"
RESULTS=()

run_mode () {  # $1=label  $2=ENABLE
  local mode=$1 en=$2 plug=""
  [ "$en" = "1" ] && plug="adaptive_topk"   # load the plugin only for adaptive
  echo "=== launch vLLM mode=$mode ENABLE=$en plugin='$plug' $(date -u) ===" >> "$LOG"
  VLLM_PLUGINS=$plug ADAPTIVE_TOPK_ENABLE=$en ADAPTIVE_TOPK_K=4 ADAPTIVE_TOPK_THRESH=0.9 \
  ADAPTIVE_TOPK_DEBUG=1 \
    python3 -m vllm.entrypoints.openai.api_server \
       --model "$MODEL" --served-model-name qwen3 --tensor-parallel-size 8 \
       --enable-expert-parallel --max-num-seqs 1 --dtype bfloat16 --max-model-len 8192 \
       --enforce-eager --gpu-memory-utilization 0.9 --port $PORT \
    > /alloc/data/vllm_$mode.log 2>&1 &
  local vpid=$! ok=0
  for i in $(seq 1 100); do
    curl -sf -m3 http://localhost:$PORT/v1/models >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $vpid 2>/dev/null || { echo "vLLM($mode) exited early — see vllm_$mode.log" >> "$LOG"; break; }
    [ $((10#$(date +%M))) -ge 58 ] && { echo "TIME GUARD :58 — abort $mode" >> "$LOG"; break; }
    sleep 5
  done
  if [ $ok -eq 1 ]; then
    echo "vLLM($mode) serving — measuring" >> "$LOG"
    PYTHONPATH=src python3 tools/measure_baseline.py --base http://localhost:$PORT \
      --model qwen3 --decode 64 --repeats 2 --out /alloc/data/ab_$mode.json >> "$LOG" 2>&1
    RESULTS+=(/alloc/data/ab_$mode.json)
    python3 tools/quality_probe.py --base http://localhost:$PORT --model qwen3 \
      --tokens 96 --out /alloc/data/q_$mode.json >> "$LOG" 2>&1   # for the quality gate
    RESULTS+=(/alloc/data/q_$mode.json)
    echo "--- $mode adaptive_topk debug ---" >> "$LOG"
    grep -i 'adaptive_topk' /alloc/data/vllm_$mode.log | tail -4 >> "$LOG"
  fi
  kill $vpid 2>/dev/null; sleep 20  # release HBM before next launch
}

if [ "$freemb" -gt 65000 ]; then
  run_mode baseline 0
  echo "pip install -e adaptive_topk plugin ..." >> "$LOG"
  pip install -e experiments/adaptive_topk -q >> "$LOG" 2>&1
  [ $((10#$(date +%M))) -lt 56 ] && run_mode adaptive 1 \
    || echo "skip adaptive (out of slot time)" >> "$LOG"
else
  echo "GPUs busy (${freemb}MB free) -> cannot launch our vLLM; skipping A/B" >> "$LOG"
fi

# Quality gate: does adaptive (k=4) match baseline (k=8) greedy output?
if [ -f /alloc/data/q_baseline.json ] && [ -f /alloc/data/q_adaptive.json ]; then
  echo "=== quality gate (baseline vs adaptive output) ===" >> "$LOG"
  python3 tools/quality_compare.py /alloc/data/q_baseline.json \
    /alloc/data/q_adaptive.json --out /alloc/data/quality_gate.json >> "$LOG" 2>&1
  RESULTS+=(/alloc/data/quality_gate.json)
fi

# Auto-share results to origin/djamoils-results (isolated worktree).
if [ ${#RESULTS[@]} -gt 0 ]; then
  git -C "$REPO" fetch origin -q 2>/dev/null
  rm -rf /tmp/reswt
  if git -C "$REPO" worktree add -f /tmp/reswt origin/main >/dev/null 2>&1; then
    mkdir -p /tmp/reswt/results
    cp "${RESULTS[@]}" /tmp/reswt/results/ 2>/dev/null
    ( cd /tmp/reswt; git add -f results 2>/dev/null
      git -c user.email=djamoils25@gmail.com -c user.name="djamoils-box" \
        commit -q -m "results: adaptive top-k A/B slot $(date -u)" 2>/dev/null \
        && git push -q -f origin HEAD:djamoils-results 2>>"$LOG" \
        && echo "results pushed -> origin/djamoils-results" >> "$LOG" )
    git -C "$REPO" worktree remove -f /tmp/reswt 2>/dev/null
  fi
fi
echo "slot done $(date -u)" >> "$LOG"
touch /alloc/data/slot_run.DONE
