#!/bin/bash
# Fully-detached launch+bench for Charles's slot. Uses the known-good fp8+EP config
# (plain TP=8 fp8 is broken on this vLLM build — needs --enable-expert-parallel).
# Launches vLLM, waits until serving, benches via the team harness (vllm_bench.py ->
# % of analytical floor), writes results + a done sentinel. Robust to SSH drops.
RES=/root/bench_result.txt; DONE=/root/bench_done
rm -f "$DONE"; : > "$RES"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8/snapshots/e156cb4efae43fbee1a1ab073f946a1377e6b969

echo "=== launch fp8 + EP=8 (TP=8) $(date -u +%H:%M:%S)UTC ===" >> "$RES"
cd /root
setsid python3 -m vllm.entrypoints.openai.api_server --model "$SNAP" \
  --served-model-name qwen3-235b-fp8 --tensor-parallel-size 8 --enable-expert-parallel \
  --port 8001 --max-model-len 8192 --trust-remote-code \
  > /root/vllm_charles.log 2>&1 < /dev/null &
echo "vllm pid $!" >> "$RES"

ready=0
for i in $(seq 1 72); do
  if curl -s -m 4 http://localhost:8001/v1/models 2>/dev/null | grep -q qwen3; then
    ready=1; echo "=== READY after ~$((i*5))s at $(date -u +%H:%M:%S)UTC ===" >> "$RES"; break; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "=== vLLM did NOT come up (cap reached) — log tail: ===" >> "$RES"
  tail -30 /root/vllm_charles.log >> "$RES"; touch "$DONE"; exit 0
fi

# Real B=1 baseline through the team harness. plan=tp/tp8 floor = ideal balanced roofline
# (actual layout is EP=8); measured TPOT is real, % of floor shows the gap to ideal.
echo "=== vllm_bench.py (fp8, measured layout EP=8, floor=TP8-ideal, 512/128) ===" >> "$RES"
python3 /root/bench/vllm_bench.py --base-url http://localhost:8001 --model qwen3-235b-fp8 \
  --name fp8-ep8 --plan tp --tp 8 --ep 1 --dtype 1 --kv-dtype 2 \
  --prompt 512 --decode 128 --warmup 6 --src /alloc/data/InferenceHackathon/src \
  --results-dir /root/results >> "$RES" 2>&1
echo "=== vllm_bench exit $? at $(date -u +%H:%M:%S)UTC ===" >> "$RES"
touch "$DONE"
