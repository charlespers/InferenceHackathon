#!/bin/bash
# Charles's :30-slot bench #4 — the data-reaction priorities (docs/results-reaction-01.md):
#   E2b: fp8 + pure TP8 via DYNAMIC quant (--quantization fp8, no block_size -> dodges the 192%128 crash
#        that blocks the released block-128 fp8 ckpt). This is the PRIZE cell (TP8 no-EP-penalty + fp8).
#   E0b: comms tuning (NCCL_PROTO=LL + NVLS in-switch all-reduce + few channels) on that same cell —
#        E0 measured all-reduce@8 ~16us (comms-bound), so this is the #1 lever.
# Two loads in the window: fp8-TP8 default-comms, then fp8-TP8 tuned-comms. Self-timing, drop-proof, self-clean.
RES=/root/bench4_result.txt; DONE=/root/bench4_done
rm -f "$DONE"; : > "$RES"
MODEL=/alloc/data/Qwen3-235B-A22B            # bf16 ckpt; --quantization fp8 dynamically quantizes at load
SERVED=q4

log(){ echo "$@" >> "$RES"; }
free_min(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1; }

run_cfg(){  # $1=tag  $2=extra NCCL env (string)  $3=extra vllm flags
  local tag="$1" env="$2" flags="$3"
  log ""; log "=== [$tag] launch $(date -u +%H:%M:%S)UTC (free $(free_min)MiB) ==="
  cd /root
  setsid env $env python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" \
    --served-model-name "$SERVED" --quantization fp8 --tensor-parallel-size 8 \
    --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.92 --trust-remote-code $flags \
    > /root/vllm_b4.log 2>&1 < /dev/null &
  local VPID=$!
  local ready=0
  for i in $(seq 1 96); do
    curl -s -m4 http://localhost:8001/v1/models 2>/dev/null | grep -q "$SERVED" && { ready=1; log "  READY ~$((i*5))s"; break; }
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then log "  DID NOT COME UP ([$tag]); tail:"; tail -20 /root/vllm_b4.log >> "$RES"; fi
  if [ "$ready" -eq 1 ]; then
    log "  --- measure.py 512/128 ---"
    python3 /root/bench/measure.py --base http://localhost:8001 --model "$SERVED" --ctx 512 --decode 128 --warmup 2 >> "$RES" 2>&1
  fi
  kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name $SERVED" 2>/dev/null
  sleep 8
}

log "=== armed $(date -u +%H:%M:%S)UTC; waiting for slot :30-:43 ==="
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 43 ] && break; sleep 10; done
mf=0; for i in $(seq 1 18); do mf=$(free_min); [ "${mf:-0}" -gt 70000 ] && break; sleep 10; done
if [ "${mf:-0}" -lt 70000 ]; then log "=== ABORT: GPUs busy (${mf}MiB free) ==="; touch "$DONE"; exit 0; fi

# 1) prize cell, default comms (does dynamic fp8 + pure TP8 even launch? vs the block-128 192-crash)
run_cfg "E2b fp8+TP8 dynamic, default comms" "" ""
# 2) same cell, comms-tuned (E0b): LL protocol + NVLS in-switch all-reduce + few channels
run_cfg "E2b+E0b fp8+TP8, NCCL_PROTO=LL NVLS chans=2" "NCCL_PROTO=LL NCCL_NVLS_ENABLE=1 NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=2" ""

log ""; log "=== compare to: bf16+TP8 85.7 tok/s (11.67ms), fp8+EP8 64.5 (15.51ms). predicted fp8-TP8 floor 262(16us)->638(tuned). $(date -u +%H:%M:%S)UTC ==="
touch "$DONE"
