#!/bin/bash
# Charles's :30-slot bench #5 — validate the floor-amortization claim (docs/spec-decode-floor-bound.md):
#   while floor-bound, n-gram spec amortizes the per-step floor (188 all-reduces + launch) over tau
#   accepted tokens -> ~tau x. Test: bf16 pure-TP8 (current best 85.7 tok/s) WITHOUT vs WITH n-gram spec.
#   Prediction: ~1.5-2.5x on the repetitive measure.py prompt (n-gram fires). If realized tok/s does NOT
#   rise, the floor-amortization claim is wrong (or n-gram acceptance is low) -> report acceptance.
# Self-timing, GPU-free-gated, self-cleaning (kills only its own served-model). Public/clean-room only.
RES=/root/bench5_result.txt; DONE=/root/bench5_done
rm -f "$DONE"; : > "$RES"
MODEL=/alloc/data/Qwen3-235B-A22B   # bf16, pure TP8 (no EP -> no 192 constraint, no EP penalty)
SERVED=q5
log(){ echo "$@" >> "$RES"; }
free_min(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1; }

run_cfg(){  # $1=tag  $2=extra vllm flags
  log ""; log "=== [$1] launch $(date -u +%H:%M:%S)UTC (free $(free_min)MiB) ==="
  cd /root
  setsid python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" \
    --served-model-name "$SERVED" --tensor-parallel-size 8 --dtype bfloat16 \
    --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.92 --trust-remote-code $2 \
    > /root/vllm_b5.log 2>&1 < /dev/null &
  local VPID=$! ready=0
  for i in $(seq 1 96); do
    curl -s -m4 http://localhost:8001/v1/models 2>/dev/null | grep -q "$SERVED" && { ready=1; log "  READY ~$((i*5))s"; break; }
    sleep 5
  done
  if [ "$ready" -eq 1 ]; then
    log "  --- measure.py 512/128 (repetitive prompt -> n-gram fires) ---"
    python3 /root/bench/measure.py --base http://localhost:8001 --model "$SERVED" --ctx 512 --decode 128 --warmup 2 >> "$RES" 2>&1
  else
    log "  DID NOT COME UP ([$1]); tail:"; tail -20 /root/vllm_b5.log >> "$RES"
  fi
  kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name $SERVED" 2>/dev/null; sleep 8
}

log "=== armed $(date -u +%H:%M:%S)UTC; waiting for slot :30-:43 ==="
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 43 ] && break; sleep 10; done
mf=0; for i in $(seq 1 18); do mf=$(free_min); [ "${mf:-0}" -gt 70000 ] && break; sleep 10; done
if [ "${mf:-0}" -lt 70000 ]; then log "=== ABORT: GPUs busy (${mf}MiB) ==="; touch "$DONE"; exit 0; fi

# 1) baseline: bf16 pure-TP8, no spec (confirm ~85.7 tok/s / 11.67ms)
run_cfg "baseline bf16-TP8 no-spec" ""
# 2) + n-gram spec, k=4 (floor-bound regime -> moderate k is fine; verify-tax barely bites)
run_cfg "bf16-TP8 + n-gram k=4" "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":3,\"prompt_lookup_min\":1}"

log ""; log "=== compare: spec tok/s vs baseline. floor-amortization predicts ~1.5-2.5x on this repetitive prompt. $(date -u +%H:%M:%S)UTC ==="
touch "$DONE"
