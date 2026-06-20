#!/usr/bin/env bash
# djamoils slot runner: waits for the :45-:00 slot, runs the most useful B=1
# experiment given GPU state, then AUTO-COMMITS+PUSHES the result JSON to the
# origin/djamoils-results branch so the whole team can see it (no human in loop).
# Runs from the box checkout. Best-effort; failures are logged, never fatal.
set -u
LOG=/alloc/data/slot_run.log
REPO=/alloc/data/InferenceHackathon
echo "armed $(date -u), waiting for :45 slot" > "$LOG"
while :; do m=$(date +%M); [ "$((10#$m))" -ge 45 ] && break; sleep 10; done
echo "slot start $(date -u)" >> "$LOG"
cd "$REPO" || exit 1

freemb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
echo "min GPU free ${freemb}MB" >> "$LOG"
RESULTS=()
if [ "$freemb" -gt 65000 ]; then
  echo "GPUs free -> router_mass (adaptive top-k headroom)" >> "$LOG"
  PYTHONPATH=src python3 tools/router_mass.py --n-prompts 16 --gpu-mem-gib 70 \
    --out /alloc/data/router_mass.json >> "$LOG" 2>&1
  RESULTS+=(/alloc/data/router_mass.json)
elif curl -sf -m4 http://localhost:8001/v1/models >/dev/null 2>&1; then
  mid=$(curl -sf -m4 http://localhost:8001/v1/models | python3 -c \
    "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
  echo "vLLM serving ($mid) -> measure baseline" >> "$LOG"
  PYTHONPATH=src python3 tools/measure_baseline.py --base http://localhost:8001 \
    --model "$mid" --out /alloc/data/baseline.json >> "$LOG" 2>&1
  RESULTS+=(/alloc/data/baseline.json)
else
  echo "GPUs busy, no vLLM -> nothing to run" >> "$LOG"
fi

# Auto-share results to origin/djamoils-results (isolated worktree, won't disturb
# the live box checkout; single-writer branch so force-push is safe).
if [ ${#RESULTS[@]} -gt 0 ]; then
  git -C "$REPO" fetch origin -q 2>/dev/null
  rm -rf /tmp/reswt
  if git -C "$REPO" worktree add -f /tmp/reswt origin/main >/dev/null 2>&1; then
    mkdir -p /tmp/reswt/results
    cp "${RESULTS[@]}" /tmp/reswt/results/ 2>/dev/null
    ( cd /tmp/reswt
      git add -f results 2>/dev/null
      git -c user.email=djamoils25@gmail.com -c user.name="djamoils-box" \
        commit -q -m "results: slot run $(date -u)" 2>/dev/null \
        && git push -q -f origin HEAD:djamoils-results 2>>"$LOG" \
        && echo "results pushed -> origin/djamoils-results" >> "$LOG" )
    git -C "$REPO" worktree remove -f /tmp/reswt 2>/dev/null
  fi
fi
echo "slot done $(date -u)" >> "$LOG"
touch /alloc/data/slot_run.DONE
