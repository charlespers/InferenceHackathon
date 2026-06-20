#!/bin/bash
# Charles's :30-slot — validate the CONVERGENT ANSWER: EAGLE3 tree-spec (the comms-amortized #1 lever).
# Tests the floor-bound OVER-DELIVERY prediction (spec-decode-floor-bound.md): EAGLE3's published ~1.9x is a
# weight-bound number; on THIS floor-bound engine (86% floor, 85.7 tok/s) it should approach its accept length
# (~tau 2.5-3x) because the verify amortizes the larger floor here. Measures realized tok/s + accept length.
#
# COORDINATION (danielAgentScheduling.md): LOOP-A solved the vLLM prereq — isolated venv at
# /alloc/data/eagle3-venv (vLLM 0.11.0) + the converted head cached. Per the agreed split, this is MY lane:
# the **bf16 floor-bound over-delivery** test (LOOP-A owns the FP8+EP run on its :45 slot/port 8077). Different
# model + slot + port → both data points useful. This uses LOOP-A's venv interpreter so it runs TODAY.
PY=/alloc/data/eagle3-venv/bin/python   # LOOP-A's vLLM-0.11 venv; falls back to system python3 if absent
[ -x "$PY" ] || PY=python3
RES=/root/eagle3_result.txt; DONE=/root/eagle3_done
rm -f "$DONE"; : > "$RES"
MODEL=/alloc/data/Qwen3-235B-A22B; SERVED=qe3
DRAFT=nm-testing/Qwen3-235B-A22B-EAGLE3-converted-speculators-lmsys
log(){ echo "$@" >> "$RES"; }
free_min(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1; }

VER=$("$PY" -c "import vllm;print(vllm.__version__)" 2>/dev/null)
log "interpreter: $PY  vLLM: ${VER:-unknown}"
"$PY" - "$VER" <<'PYV' 2>/dev/null || { echo "=== SKIP: vLLM < 0.10.2 at $PY (need LOOP-A's /alloc/data/eagle3-venv, vLLM 0.11). ===" >> /root/eagle3_result.txt; touch /root/eagle3_done; exit 0; }
import sys
v=sys.argv[1].split('+')[0].split('.')
maj,minr,pat=(int(v[0]),int(v[1]),int(v[2]) if len(v)>2 else 0)
sys.exit(0 if (maj,minr,pat)>=(0,10,2) else 1)
PYV

log "=== armed $(date -u +%H:%M:%S)UTC; waiting for slot :30-:43 ==="
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 43 ] && break; sleep 10; done
mf=0; for i in $(seq 1 18); do mf=$(free_min); [ "${mf:-0}" -gt 70000 ] && break; sleep 10; done
if [ "${mf:-0}" -lt 70000 ]; then log "=== ABORT: GPUs busy (${mf}MiB) ==="; touch "$DONE"; exit 0; fi

run_cfg(){  # $1=tag $2=spec-config-json("" = baseline no-spec) $3=enforce-eager-or-empty
  log ""; log "=== [$1] launch $(date -u +%H:%M:%S)UTC (free $(free_min)MiB) ==="
  local spec=""; [ -n "$2" ] && spec="--speculative-config $2"
  cd /root
  setsid env VLLM_USE_V1=1 "$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED" \
    --tensor-parallel-size 8 --dtype bfloat16 --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.90 \
    --max-num-seqs 1 --disable-log-requests --enable-prefix-caching \
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
# k-SWEEP for tools/backout_floor.py: V=tau/S at >=2 tree sizes backs out the floor fraction F. graphs-on (the
# real config). draft_tp=8 (NOT 1): B=1 draft is bandwidth-bound; /8 sharding ~6x faster + no aux-hidden gather
# (docs/eagle3-draft-tp.md). Big tree first so we get the most informative point if the slot ends early.
for K in 8 5 2; do
  m=$((10#$(date +%M))); [ "$m" -gt 43 ] && { log "  slot window closing — stop k-sweep before k=$K"; break; }
  SPEC='{"method":"eagle3","model":"'"$DRAFT"'","num_speculative_tokens":'"$K"',"draft_tensor_parallel_size":8}'
  run_cfg "EAGLE3 draft_tp8 graphs k=$K" "$SPEC" ""
done
# If draft_tp=8 errors (head pins draft_tp=1), set it to 1 above and note the ~3ms draft penalty (expect ~2.5x).
# Parity (lossless) + eager-vs-graphs floor-delta are a follow-up slot (eager doubles load count).

log ""; log "=== PREDICTION (my lane, bf16 floor-bound): EAGLE3 draft_tp8 ~2.5-3x baseline (over-delivery vs published ~1.9x). Feed (k,tau,S) per arm into backout_floor.py -> expect F~0.86 (FLOOR-BOUND) -> route-aware NO-GO. accept-len ~3-3.5 GREEDY (lower temp>0). $(date -u +%H:%M:%S)UTC ==="
log "=== compare with LOOP-A's FP8+EP (they post to danielAgentScheduling.md): the bf16-vs-FP8+graphs ΔF = the floor reduction, which decides their route-aware lever. ==="
touch "$DONE"
