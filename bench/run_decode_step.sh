#!/bin/bash
# Build + run the FULL fused B=1 decode step — kernels/decode_step.cu, which #includes
# K1(attn prologue) -> K2(flash decode) -> K3(O-proj+residual) -> K4(router) -> K5(experts) x94
# layers + final-norm + lm_head + on-device argmax, and captures the whole thing into ONE CUDA
# graph via K6 (kernels/k6_graph_capture.cu). It then times graph-replay vs eager (per-launch) to
# expose the launch-overhead delta the graph collapses.
#
# Compiling needs only nvcc (CPU); running needs one free H100. This is the missing build+run target
# (compile_kernels.sh only compile-checks the individual benches) — i.e. the reason there was no
# "fully working step" to run yet.
#
#   bash bench/run_decode_step.sh [ctx_len] [iters] [hbm_GBps]
# NOTE: latency/launch-overhead PROXY — one layer's dummy weights reused x94 (see decode_step.cu
# header). The kernel CHAIN, launch COUNT, grid/block shapes and per-token read VOLUME are real; the
# produced token id is not (numeric validation vs HF transformers is a separate on-box step).
set -u
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
KDIR=${KDIR:-kernels}
CTX=${1:-4096}
IT=${2:-200}
PEAK=${3:-3350}
BIN=${BIN:-/tmp/decode_step.bin}
OUT=${OUT:-/root/decode_step_result.txt}

echo "=== compile decode_step.cu (sm_90a, O3, fast-math) $(date -u +%H:%M:%S)UTC ===" | tee "$OUT"
if ! "$NVCC" -arch=sm_90a -O3 --use_fast_math -I "$KDIR" "$KDIR/decode_step.cu" -o "$BIN" 2>>"$OUT"; then
  echo "COMPILE FAILED — see $OUT" | tee -a "$OUT"
  exit 1
fi
echo "compile OK -> $BIN" | tee -a "$OUT"

# Pick the GPU with the most free memory; the step needs ~1.5 GB (1 layer of weights + lm_head + KV).
read -r idx free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
  | sort -t, -k2 -n | tail -1 | tr ',' ' ')
if [ "${free:-0}" -lt 2000 ]; then
  echo "no free GPU (max free ${free:-0} MiB) — compiled only; rerun in a GPU window." | tee -a "$OUT"
  exit 0
fi

echo "=== run on GPU $idx (ctx=$CTX iters=$IT peak=${PEAK}GB/s) ===" | tee -a "$OUT"
CUDA_VISIBLE_DEVICES="$idx" "$BIN" "$CTX" "$IT" "$PEAK" 2>&1 | tee -a "$OUT"
echo "=== done -> $OUT ===" | tee -a "$OUT"
