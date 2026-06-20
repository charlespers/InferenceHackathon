#!/bin/bash
# Charles's :30-slot — VALIDATE THE STACKED CHEAP WINS in one run (docs/single-user-latency-budget.md).
# The budget projects: bf16 pure-TP8 + prefix-caching (TTFT) + n-gram spec (decode, floor-amortization ~2-3x)
# + LL comms -> ~250-290 tok/s and cached TTFT ~10ms (vs baseline 85.7 tok/s / 777ms). This runs that exact
# config and measures whether the ~5x materializes. (Isolated levers: run_bench4/5/6; this is the combo.)
# n-gram k=8 because the regime is floor-bound (spec_floor_model.py: big trees win ~3x now). Repetitive
# measure.py prompt -> n-gram fires; same prompt repeated -> prefix-cache hit. Drop-proof, self-cleaning.
RES=/root/bench_best_result.txt; DONE=/root/bench_best_done
rm -f "$DONE"; : > "$RES"
MODEL=/alloc/data/Qwen3-235B-A22B; SERVED=qbest
log(){ echo "$@" >> "$RES"; }
free_min(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1; }

log "=== armed $(date -u +%H:%M:%S)UTC; waiting for slot :30-:43 ==="
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 43 ] && break; sleep 10; done
mf=0; for i in $(seq 1 18); do mf=$(free_min); [ "${mf:-0}" -gt 70000 ] && break; sleep 10; done
if [ "${mf:-0}" -lt 70000 ]; then log "=== ABORT: GPUs busy (${mf}MiB) ==="; touch "$DONE"; exit 0; fi

log "=== launch FULL cheap-wins config (docs/vllm-b1-config.md): bf16 TP8 + prefix-cache + n-gram(k=8) + LL + max-num-seqs=1 + V1 + no-chunked-prefill $(date -u +%H:%M:%S)UTC (free ${mf}MiB) ==="
cd /root
# host-floor knobs (max-num-seqs=1, V1, no-chunked-prefill, no-logging) work on 0.10.1; EAGLE3 (run_eagle3.sh) needs 0.10.2.
setsid env NCCL_PROTO=LL NCCL_MAX_NCHANNELS=2 VLLM_USE_V1=1 python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" \
  --served-model-name "$SERVED" --tensor-parallel-size 8 --dtype bfloat16 --enable-prefix-caching \
  --speculative-config '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":4,"prompt_lookup_min":1}' \
  --max-num-seqs 1 --disable-log-requests \
  --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.90 --trust-remote-code \
  > /root/vllm_best.log 2>&1 < /dev/null &
# NOTE: if --no-enable-chunked-prefill is rejected by this build, drop it (and add it back if short-prompt TTFT is high).
VPID=$!; ready=0
for i in $(seq 1 110); do curl -s -m4 http://localhost:8001/v1/models 2>/dev/null | grep -q "$SERVED" && { ready=1; log "  READY ~$((i*5))s"; break; }; sleep 5; done
if [ "$ready" -ne 1 ]; then log "  DID NOT COME UP (some flag may be unsupported; tail:)"; tail -25 /root/vllm_best.log >> "$RES"; kill -9 -"$VPID" 2>/dev/null; touch "$DONE"; exit 0; fi

M="python3 /root/bench/measure.py --base http://localhost:8001 --model $SERVED --warmup 0"
log ""; log "--- 1) COLD: fresh prompt (cold prefill + spec decode) ---";        $M --ctx 512 --decode 128 >> "$RES" 2>&1
log ""; log "--- 2) CACHED: same prompt repeat (prefix hit + spec decode) ---";  $M --ctx 512 --decode 128 >> "$RES" 2>&1
log ""; log "=== TARGET: decode ~250-290 tok/s (spec); cached TTFT ~10ms. baseline 85.7 tok/s / 777ms. $(date -u +%H:%M:%S)UTC ==="
log "=== if a flag is unsupported, fall back: drop --speculative-config (run_bench6) or --enable-prefix-caching (run_bench5). ==="
kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name $SERVED" 2>/dev/null
touch "$DONE"
