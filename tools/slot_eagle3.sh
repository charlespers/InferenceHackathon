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
# draft_tensor_parallel_size=8 (Charles, docs/eagle3-draft-tp.md): at B=1 the 1B head is
# BANDWIDTH-bound — draft_tp=1 reads ~2GB on one rank (~0.6ms x k ~3ms/round, confounds the
# F-backout since it scales with k); draft_tp=8 reads 0.25GB/GPU + ~32us AR (~6x faster) and
# keeps the aux hidden states TP8-sharded (no gather). Fallback to 1 if the head won't shard
# (4 KV heads over 8 ranks). Eager-first reveals a load failure fast.
DRAFT_TP=${DRAFT_TP:-8}
spec_cfg () { echo "{\"method\":\"eagle3\",\"model\":\"$HEAD\",\"num_speculative_tokens\":$1,\"draft_tensor_parallel_size\":$DRAFT_TP}"; }
PORT=8077
# Two tree sizes so V at k1 and k2 over-determine the floor F (Charles' backout_floor.py):
# V(k)=F+(1-F)(0.34+0.66*union(k)/8). NSPEC1 = primary (de-risk + parity); NSPEC2 = opportunistic
# 2nd point (only if slot time remains). On EP, go BIG (Charles: verify balances EP at large union).
NSPEC1=${NSPEC1:-3}
NSPEC2=${NSPEC2:-8}
# MODE=eager (slot 1: de-risk + lossless parity + accept-len, matched-eager speedup)
# MODE=graphs (slot 2: drop --enforce-eager for the CUDA-graph headline; only after eager passes)
MODE=${MODE:-eager}
if [ "$MODE" = "graphs" ]; then EAGER=""; else EAGER="--enforce-eager"; fi

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
  LAST_OK=$ok                          # so callers can fall back on failure
}

# stale-lock cleanup (>20 min) then atomic acquire
[ -d /alloc/data/gpu.lock ] && [ -n "$(find /alloc/data/gpu.lock -mmin +20 2>/dev/null)" ] && { rm -f /alloc/data/gpu.lock/holder; rmdir /alloc/data/gpu.lock 2>/dev/null; }
FREE=$(freemin); echo "min GPU free ${FREE}MB" >> "$LOG"
if [ "$FREE" -gt 65000 ] && mkdir /alloc/data/gpu.lock 2>/dev/null; then
  echo "LOOP-A(eagle3) $(date -u)" > /alloc/data/gpu.lock/holder
  echo "- $(date -u) LOOP-A: acquired gpu.lock -> EAGLE3 $MODE + baseline ($MODE)" >> "$SCHED"

  # 1) EAGLE3 @ k=NSPEC1 — de-risk + accept-length + lossless capture (most important)
  launch_measure "eagle3_$MODE" "$EAGER --speculative-config $(spec_cfg $NSPEC1)" 0.85
  # FALLBACK: if draft_tp=8 won't load (4 KV heads / 8 ranks), retry the de-risk run at draft_tp=1
  if [ "${LAST_OK:-0}" -ne 1 ] && [ "$DRAFT_TP" != "1" ] && [ "$(mins)" -lt 54 ]; then
    echo "draft_tp=$DRAFT_TP failed — falling back to draft_tp=1 for the de-risk run" >> "$LOG"
    DRAFT_TP=1
    launch_measure "eagle3_$MODE" "$EAGER --speculative-config $(spec_cfg $NSPEC1)" 0.85
  fi

  # 2) baseline FP8 in the SAME mode — so S=tok/s(eagle3)/tok/s(base) and V=tau/S are valid
  #    (matched floor in both). If time remains.
  if [ "$(mins)" -lt 56 ]; then
    launch_measure "baseline_$MODE" "$EAGER" 0.9
  else
    echo "skip baseline (out of slot time)" >> "$LOG"
  fi

  # 3) OPPORTUNISTIC 2nd tree size @ k=NSPEC2 — gives the 2nd V point to back out F.
  #    Only if there's a full launch's worth of slot left (needs <52 to finish by :00).
  if [ "$(mins)" -lt 52 ]; then
    launch_measure "eagle3_${MODE}_k${NSPEC2}" "$EAGER --speculative-config $(spec_cfg $NSPEC2)" 0.85
  else
    echo "skip 2nd tree size k=$NSPEC2 (out of slot time) — get it next slot" >> "$LOG"
  fi

  rm -f /alloc/data/gpu.lock/holder; rmdir /alloc/data/gpu.lock 2>/dev/null   # holder-inside-dir: rm then rmdir
  echo "- $(date -u) LOOP-A: released gpu.lock (results /alloc/data/eagle3/)" >> "$SCHED"
else
  echo "GPUs busy (${FREE}MB) or gpu.lock held -> NOT my window, skipping" >> "$LOG"
fi

# Lossless parity gate: EAGLE3 greedy must EXACTLY match baseline greedy (same MODE).
if [ -f "$OUT/q_baseline_$MODE.json" ] && [ -f "$OUT/q_eagle3_$MODE.json" ]; then
  echo "=== EAGLE3 lossless parity gate ($MODE: baseline vs eagle3 greedy) ===" >> "$LOG"
  $PY "$TOOLS/quality_compare.py" "$OUT/q_baseline_$MODE.json" "$OUT/q_eagle3_$MODE.json" \
      --out "$OUT/parity_gate_$MODE.json" >> "$LOG" 2>&1
fi

# Share results to origin/djamoils-results (box can't push main; results/* gitignored)
RES=$(ls "$OUT"/m_*.json "$OUT"/parity_gate_*.json "$OUT"/q_*.json "$OUT"/metrics_*.txt 2>/dev/null)
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
