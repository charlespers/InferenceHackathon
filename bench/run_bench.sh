#!/bin/bash
# Charles's :30-:45 testing-slot bench: clean B=1 latency baseline for
# Qwen3-235B-A22B-Instruct-2507-FP8 (TP=8) on 8xH100. Self-waits for the window
# and for free GPUs, runs `vllm bench latency`, writes results + a done sentinel.
# GPU-idle until the window opens, so it's safe to launch during others' slots.
RES=/root/bench_result.txt
DONE=/root/bench_done
rm -f "$DONE"
echo "=== staged $(date -u +%H:%M:%S)UTC; waiting for Charles slot :30-:45 ===" > "$RES"

# 1) wait for Charles's testing slot (minute-of-hour in [30,44])
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 44 ] && break; sleep 10; done
echo "=== window open $(date -u +%H:%M:%S)UTC; waiting for GPUs to free ===" >> "$RES"

# 2) wait for GPUs free (min free > 70GB), cap ~3 min; abort rather than OOM
minfree=0
for i in $(seq 1 18); do
  minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
  [ -n "$minfree" ] && [ "$minfree" -gt 70000 ] && break
  sleep 10
done
if [ "${minfree:-0}" -lt 70000 ]; then
  echo "=== ABORT: GPUs still busy (${minfree}MiB free) at window open — did not launch ===" >> "$RES"
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader >> "$RES" 2>&1
  touch "$DONE"; exit 0
fi
echo "=== min free ${minfree}MiB; launching vllm bench latency $(date -u +%H:%M:%S)UTC ===" >> "$RES"

# 3) clean B=1 latency baseline: 512 prompt / 128 decode, fp8 weights, TP=8
cd /root
HF_HUB_OFFLINE=1 vllm bench latency \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --tensor-parallel-size 8 --max-model-len 4096 \
  --input-len 512 --output-len 128 --batch-size 1 \
  --num-iters-warmup 1 --num-iters 3 >> "$RES" 2>&1
echo "=== vllm bench exit $? at $(date -u +%H:%M:%S)UTC ===" >> "$RES"
touch "$DONE"
