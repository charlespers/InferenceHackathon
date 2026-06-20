#!/bin/bash
# Clean, self-timing, self-cleaning B=1 baseline for Charles's :30 slot.
# Uses the KNOWN-GOOD fp8 + EP config (pure-TP fp8 is broken on this vLLM build) and
# the zero-dep measure.py (direct SSE timing) instead of `vllm bench serve` (which hit a
# Bad Request). Launches its server, measures TPOT, computes roofline %, then kills ONLY
# the server it started (scoped to its own process group). Robust to SSH drops.
RES=/root/bench_result.txt; DONE=/root/bench_done
rm -f "$DONE"; : > "$RES"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969

echo "=== armed $(date -u +%H:%M:%S)UTC; waiting for Charles slot :30-:43 ===" >> "$RES"
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 43 ] && break; sleep 10; done

mf=0
for i in $(seq 1 18); do mf=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1); [ "${mf:-0}" -gt 70000 ] && break; sleep 10; done
if [ "${mf:-0}" -lt 70000 ]; then echo "=== ABORT: GPUs busy (${mf}MiB free) at slot open ===" >> "$RES"; touch "$DONE"; exit 0; fi

echo "=== launch fp8+EP $(date -u +%H:%M:%S)UTC (free ${mf}MiB) ===" >> "$RES"
cd /root
setsid python3 -m vllm.entrypoints.openai.api_server --model "$SNAP" \
  --served-model-name qwen3-235b-fp8 --tensor-parallel-size 8 --enable-expert-parallel \
  --port 8001 --max-model-len 8192 --trust-remote-code > /root/vllm_charles.log 2>&1 < /dev/null &
VPID=$!; echo "vllm pgid $VPID" >> "$RES"

ready=0
for i in $(seq 1 84); do
  curl -s -m 4 http://localhost:8001/v1/models 2>/dev/null | grep -q qwen3 && { ready=1; echo "=== READY ~$((i*5))s $(date -u +%H:%M:%S)UTC ===" >> "$RES"; break; }
  sleep 5
done
if [ "$ready" -ne 1 ]; then echo "=== vLLM did NOT come up ===" >> "$RES"; tail -25 /root/vllm_charles.log >> "$RES"; kill -9 -"$VPID" 2>/dev/null; touch "$DONE"; exit 0; fi

echo "=== B=1 measure.py (512 prompt / 128 decode) ===" >> "$RES"
python3 /root/bench/measure.py --base http://localhost:8001 --model qwen3-235b-fp8 --ctx 512 --decode 128 --warmup 2 >> "$RES" 2>&1
TPOT=$(grep -oE "TPOT [0-9.]+" "$RES" | tail -1 | grep -oE "[0-9.]+")
echo "=== roofline (H100 fp8 weights, fp16 KV), measured TPOT=${TPOT}ms ===" >> "$RES"
python3 /root/bench/roofline.py --ctx 4096 --weight-bytes 1 --kv-bytes 2 ${TPOT:+--tpot-ms "$TPOT"} >> "$RES" 2>&1

echo "=== teardown own server $(date -u +%H:%M:%S)UTC ===" >> "$RES"
kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name qwen3-235b-fp8" 2>/dev/null
touch "$DONE"
