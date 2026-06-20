#!/bin/bash
# Compile-check the fused decode step + autotuner (CPU nvcc, no GPU).
NVCC=/usr/local/cuda/bin/nvcc
cd /root/kernels_check || exit 1
: > /root/c3.log
# K3/K4 are kernel libraries (no main) -> object compile to syntax-check.
for f in k3_attn_epilogue.cu k4_router.cu; do
  echo "===== $f (-c) =====" >> /root/c3.log
  "$NVCC" -arch=sm_90a -O3 --use_fast_math -I . -c "$f" -o "/tmp/c_${f}.o" >"/tmp/ce_$f" 2>&1 \
    && echo OK >> /root/c3.log || { echo FAIL >> /root/c3.log; head -18 "/tmp/ce_$f" >> /root/c3.log; }
done
# decode_step + autotune have main -> full build.
for f in decode_step.cu autotune.cu; do
  echo "===== $f =====" >> /root/c3.log
  "$NVCC" -arch=sm_90a -O3 --use_fast_math -I . "$f" -o "/tmp/c_${f}.bin" >"/tmp/ce_$f" 2>&1 \
    && echo OK >> /root/c3.log || { echo FAIL >> /root/c3.log; head -22 "/tmp/ce_$f" >> /root/c3.log; }
done
touch /root/c3_done
