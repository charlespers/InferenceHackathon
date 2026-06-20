#!/bin/bash
# Charles's :30-slot bench — targets the OPEN de-confounding cell in docs/config-sweep.md:
#   the team has bf16+TP=8 and fp8+EP=8, but those confound precision with parallelism
#   (pure-TP fp8 is broken on this vLLM build). Measuring **bf16 + EP=8** isolates both:
#     bf16/TP=8 (have) vs bf16/EP=8 (this)  -> TP-vs-EP effect at fixed precision
#     bf16/EP=8 (this) vs fp8/EP=8 (have)   -> precision effect at fixed parallelism
# Self-timing, self-cleaning, drop-proof. Uses the bf16 checkpoint already on the box.
RES=/root/bench_result.txt; DONE=/root/bench_done
rm -f "$DONE"; : > "$RES"
MODEL=/alloc/data/Qwen3-235B-A22B            # bf16 weights (already on box)
SERVED=qwen3-235b-bf16ep

echo "=== armed $(date -u +%H:%M:%S)UTC; waiting for Charles slot :30-:43 ===" >> "$RES"
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 43 ] && break; sleep 10; done

mf=0
for i in $(seq 1 18); do mf=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1); [ "${mf:-0}" -gt 70000 ] && break; sleep 10; done
if [ "${mf:-0}" -lt 70000 ]; then echo "=== ABORT: GPUs busy (${mf}MiB free) at slot open ===" >> "$RES"; touch "$DONE"; exit 0; fi

echo "=== launch bf16 + EP=8 (TP=8) $(date -u +%H:%M:%S)UTC (free ${mf}MiB) ===" >> "$RES"
cd /root
setsid python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" \
  --served-model-name "$SERVED" --tensor-parallel-size 8 --enable-expert-parallel \
  --dtype bfloat16 --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.92 \
  --trust-remote-code > /root/vllm_charles.log 2>&1 < /dev/null &
VPID=$!; echo "vllm pgid $VPID" >> "$RES"

ready=0
for i in $(seq 1 96); do
  curl -s -m 4 http://localhost:8001/v1/models 2>/dev/null | grep -q "$SERVED" && { ready=1; echo "=== READY ~$((i*5))s $(date -u +%H:%M:%S)UTC ===" >> "$RES"; break; }
  sleep 5
done
if [ "$ready" -ne 1 ]; then echo "=== vLLM did NOT come up ===" >> "$RES"; tail -25 /root/vllm_charles.log >> "$RES"; kill -9 -"$VPID" 2>/dev/null; touch "$DONE"; exit 0; fi

echo "=== B=1 measure.py (512 prompt / 128 decode) ===" >> "$RES"
python3 /root/bench/measure.py --base http://localhost:8001 --model "$SERVED" --ctx 512 --decode 128 --warmup 2 >> "$RES" 2>&1
TPOT=$(grep -oE "TPOT [0-9.]+" "$RES" | tail -1 | grep -oE "[0-9.]+")
echo "=== roofline (bf16 weights, bf16 KV), measured TPOT=${TPOT}ms ===" >> "$RES"
python3 /root/bench/roofline.py --ctx 4096 --weight-bytes 2 --kv-bytes 2 ${TPOT:+--tpot-ms "$TPOT"} >> "$RES" 2>&1

echo "=== teardown own server $(date -u +%H:%M:%S)UTC ===" >> "$RES"
kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name $SERVED" 2>/dev/null
touch "$DONE"
