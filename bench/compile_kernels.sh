#!/bin/bash
# Compile-check the new kernels on the box (CPU only — nvcc compile, no GPU).
NVCC=/usr/local/cuda/bin/nvcc
cd /root/kernels_check || exit 1
: > /root/compile2.log
for f in k12_bench.cu prefill_attn.cu prefill_moe.cu k5_microbench.cu; do
  echo "===== $f =====" >> /root/compile2.log
  if "$NVCC" -arch=sm_90a -O3 --use_fast_math -I . "$f" -o "/tmp/o_${f}.bin" >"/tmp/e_${f}.txt" 2>&1; then
    echo "OK" >> /root/compile2.log
  else
    echo "FAILED" >> /root/compile2.log
    head -22 "/tmp/e_${f}.txt" >> /root/compile2.log
  fi
done
touch /root/compile2_done
