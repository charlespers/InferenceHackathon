#!/bin/bash
# Same-harness fp8+EP=8 measurement to pair with the bf16+EP=8 run (both via measure.py),
# giving a clean precision A/B at fixed parallelism. Fires immediately (already in-slot),
# self-cleans. fp8 checkpoint, known-good EP config (loads ~135s).
RES=/root/fp8res.txt; DONE=/root/fp8_done
rm -f "$DONE"; : > "$RES"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969
SERVED=qwen3-235b-fp8

mf=0
for i in $(seq 1 18); do mf=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1); [ "${mf:-0}" -gt 70000 ] && break; sleep 5; done
if [ "${mf:-0}" -lt 70000 ]; then echo "=== ABORT: GPUs busy (${mf}MiB) ===" >> "$RES"; touch "$DONE"; exit 0; fi
echo "=== launch fp8+EP=8 $(date -u +%H:%M:%S)UTC (free ${mf}MiB) ===" >> "$RES"
cd /root
setsid python3 -m vllm.entrypoints.openai.api_server --model "$SNAP" \
  --served-model-name "$SERVED" --tensor-parallel-size 8 --enable-expert-parallel \
  --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.92 --trust-remote-code \
  > /root/vllm_fp8c.log 2>&1 < /dev/null &
VPID=$!; echo "vllm pgid $VPID" >> "$RES"
ready=0
for i in $(seq 1 84); do curl -s -m 4 http://localhost:8001/v1/models 2>/dev/null | grep -q "$SERVED" && { ready=1; echo "=== READY ~$((i*5))s $(date -u +%H:%M:%S)UTC ===" >> "$RES"; break; }; sleep 5; done
if [ "$ready" -ne 1 ]; then echo "=== did NOT come up ===" >> "$RES"; tail -20 /root/vllm_fp8c.log >> "$RES"; kill -9 -"$VPID" 2>/dev/null; touch "$DONE"; exit 0; fi
echo "=== B=1 measure.py (512/128) ===" >> "$RES"
python3 /root/bench/measure.py --base http://localhost:8001 --model "$SERVED" --ctx 512 --decode 128 --warmup 2 >> "$RES" 2>&1
echo "=== teardown $(date -u +%H:%M:%S)UTC ===" >> "$RES"
kill -9 -"$VPID" 2>/dev/null; pkill -9 -f "served-model-name $SERVED" 2>/dev/null
touch "$DONE"
