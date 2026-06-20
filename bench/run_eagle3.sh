#!/bin/bash
# Charles's :30-slot — validate the CONVERGENT ANSWER: EAGLE3 tree-spec (the comms-amortized #1 lever).
# Tests the floor-bound OVER-DELIVERY prediction (spec-decode-floor-bound.md): EAGLE3's published ~1.9x is a
# weight-bound number; on THIS floor-bound engine (86% floor, 85.7 tok/s) it should approach its accept length
# (~tau 2.5-3x) because the verify amortizes the larger floor here. Measures realized tok/s + accept length.
#
# PREREQS (do BEFORE the slot — not slot-time): vLLM >= 0.10.2 (0.10.1 lacks qwen3 EAGLE3, PR #24392), and the
# CONVERTED head downloaded: nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys (raw lmsys is
# SGLang-format — see experiments/eagle3/INTEGRATION.md). This script checks the version and skips cleanly if not met.
RES=/root/eagle3_result.txt; DONE=/root/eagle3_done
rm -f "$DONE"; : > "$RES"
MODEL=/alloc/data/Qwen3-235B-A22B; SERVED=qe3
DRAFT=nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys
log(){ echo "$@" >> "$RES"; }
free_min(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1; }

VER=$(python3 -c "import vllm;print(vllm.__version__)" 2>/dev/null)
log "vLLM version: ${VER:-unknown}"
python3 - "$VER" <<'PY' 2>/dev/null || { echo "=== SKIP: vLLM < 0.10.2 (EAGLE3 qwen3 unsupported); upgrade then re-run. ===" >> /root/eagle3_result.txt; touch /root/eagle3_done; exit 0; }
import sys
v=sys.argv[1].split('+')[0].split('.')
maj,minr,pat=(int(v[0]),int(v[1]),int(v[2]) if len(v)>2 else 0)
sys.exit(0 if (maj,minr,pat)>=(0,10,2) else 1)
PY

log "=== armed $(date -u +%H:%M:%S)UTC; waiting for slot :30-:43 ==="
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 43 ] && break; sleep 10; done
mf=0; for i in $(seq 1 18); do mf=$(free_min); [ "${mf:-0}" -gt 70000 ] && break; sleep 10; done
if [ "${mf:-0}" -lt 70000 ]; then log "=== ABORT: GPUs busy (${mf}MiB) ==="; touch "$DONE"; exit 0; fi

run_cfg(){  # $1=tag $2=spec-config-json("" = baseline no-spec) $3=enforce-eager-or-empty
  log ""; log "=== [$1] launch $(date -u +%H:%M:%S)UTC (free $(free_min)MiB) ==="
  local spec=""; [ -n "$2" ] && spec="--speculative-config $2"
  cd /root
  setsid python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED" \
    --tensor-parallel-size 8 --dtype bfloat16 --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.90 \
    --trust-remote-code $3 $spec > /root/vllm_e3.log 2>&1 < /dev/null &
  local VPID=$! ready=0
  for i in $(seq 1 120); do curl -s -m4 http://localhost:8001/v1/models 2>/dev/null | grep -q "$SERVED" && { ready=1; log "  READY ~$((i*5))s"; break; }; sleep 5; done
  if [ "$ready" -eq 1 ]; then
    python3 /root/bench/measure.py --base http://localhost:8001 --model "$SERVED" --ctx 512 --decode 128 --warmup 1 >> "$RES" 2>&1
    grep -iE "accept|spec|draft" /root/vllm_e3.log | tail -3 >> "$RES" 2>&1
  else log "  DID NOT COME UP ([$1]); tail:"; tail -25 /root/vllm_e3.log >> "$RES"; fi
  kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name $SERVED" 2>/dev/null; sleep 8
}

run_cfg "baseline bf16-TP8 no-spec" "" ""
SPEC='{"method":"eagle3","model":"'"$DRAFT"'","num_speculative_tokens":5,"draft_tensor_parallel_size":1}'
run_cfg "EAGLE3 enforce-eager (parity)" "$SPEC" "--enforce-eager"   # INTEGRATION.md: eager-first, then graphs
run_cfg "EAGLE3 graphs" "$SPEC" ""

log ""; log "=== PREDICTION: EAGLE3 tok/s ~2.5-3x baseline (floor-bound over-delivery), > published ~1.9x. accept-len ~3-3.5. $(date -u +%H:%M:%S)UTC ==="
touch "$DONE"
