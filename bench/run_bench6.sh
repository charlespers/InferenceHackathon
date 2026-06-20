#!/bin/bash
# Charles's :30-slot bench #6 — TTFT / prefix-caching (docs/ttft-analysis.md):
#   measured TTFT 777ms is ~20-300x the prefill physics. Prefix caching should make a REPEAT of the same
#   prompt a cache hit -> TTFT ~ first decode step (~10ms) = the cheapest big single-user-latency win.
# Tests bf16 pure-TP8 + --enable-prefix-caching:
#   1) cold TTFT (fresh prompt)  2) cached TTFT (same prompt repeated)  3) TTFT-vs-length curve.
# Self-timing, GPU-free-gated, self-cleaning. Public/clean-room only.
RES=/root/bench6_result.txt; DONE=/root/bench6_done
rm -f "$DONE"; : > "$RES"
MODEL=/alloc/data/Qwen3-235B-A22B; SERVED=q6
log(){ echo "$@" >> "$RES"; }
free_min(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1; }

log "=== armed $(date -u +%H:%M:%S)UTC; waiting for slot :30-:43 ==="
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 43 ] && break; sleep 10; done
mf=0; for i in $(seq 1 18); do mf=$(free_min); [ "${mf:-0}" -gt 70000 ] && break; sleep 10; done
if [ "${mf:-0}" -lt 70000 ]; then log "=== ABORT: GPUs busy (${mf}MiB) ==="; touch "$DONE"; exit 0; fi

log "=== launch bf16-TP8 + prefix caching $(date -u +%H:%M:%S)UTC (free ${mf}MiB) ==="
cd /root
setsid python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" \
  --served-model-name "$SERVED" --tensor-parallel-size 8 --dtype bfloat16 --enable-prefix-caching \
  --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.92 --trust-remote-code \
  > /root/vllm_b6.log 2>&1 < /dev/null &
VPID=$!; ready=0
for i in $(seq 1 96); do curl -s -m4 http://localhost:8001/v1/models 2>/dev/null | grep -q "$SERVED" && { ready=1; log "  READY ~$((i*5))s"; break; }; sleep 5; done
if [ "$ready" -ne 1 ]; then log "  DID NOT COME UP"; tail -20 /root/vllm_b6.log >> "$RES"; kill -9 -"$VPID" 2>/dev/null; touch "$DONE"; exit 0; fi

M="python3 /root/bench/measure.py --base http://localhost:8001 --model $SERVED --warmup 0"
log ""; log "--- 1) COLD TTFT (fresh 512-tok prompt) ---";   $M --ctx 512 --decode 4 >> "$RES" 2>&1
log ""; log "--- 2) CACHED TTFT (SAME prompt repeated -> prefix hit) ---"; $M --ctx 512 --decode 4 >> "$RES" 2>&1
log ""; log "--- 3) TTFT vs prompt length (decode 1; intercept=fixed/eager, slope=prefill/token) ---"
for P in 16 128 512 2048; do log "  ctx=$P:"; $M --ctx $P --decode 1 >> "$RES" 2>&1; done

log ""; log "=== EXPECT: cached TTFT << cold (prefix-cache hit ~50-100x); if cold==cached, prefix cache not engaging. $(date -u +%H:%M:%S)UTC ==="
kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name $SERVED" 2>/dev/null
touch "$DONE"
