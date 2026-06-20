#!/bin/bash
# Charles's-slot KERNEL test+tune harness: compile + microbench every custom kernel on the H100,
# report GB/s and %-of-HBM-peak (3.35 TB/s) per kernel, and stack them against the vLLM baseline
# (bf16/EP 69.5 tok/s, fp8/EP 65.8) and the bandwidth roofline (bf16 ~610 / fp8 ~1240 tok/s).
# Includes the int4 MoE variant (half the bytes of fp8 -> the biggest lever on the 14.2B-param
# expert read). Self-times to the :30 slot, self-cleans. CPU-compile + short GPU microbenches only.
NVCC=/usr/local/cuda/bin/nvcc
SRC=/root/kernels_check
RES=/root/kernel_bench_result.txt
rm -f /root/kbench_done; : > "$RES"
echo "=== kernel test+tune: armed $(date -u +%H:%M:%S)UTC; waiting for Charles slot :30-:43 ===" >> "$RES"
while :; do m=$((10#$(date +%M))); [ "$m" -ge 30 ] && [ "$m" -le 43 ] && break; sleep 10; done
# microbenches are small (a few GB); want low contention -> wait briefly for a mostly-free GPU
mf=0
for i in $(seq 1 12); do mf=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1); [ "${mf:-0}" -gt 40000 ] && break; sleep 10; done
echo "=== slot open $(date -u +%H:%M:%S)UTC, min GPU free ${mf}MiB ===" >> "$RES"
cd "$SRC" || { echo "no $SRC"; touch /root/kbench_done; exit 1; }

# label:file pairs — the standalone microbench TUs (each prints its own GB/s / %-peak / us)
BENCHES="FUSED-decode-step-graph:decode_step.cu AUTOTUNE-sweep:autotune.cu K5-fp8-MoE-decode:k5_experts.cu K5-int4-MoE-decode:k5_int4_bench.cu K1+K2-attn-decode:k12_bench.cu prefill-attn:prefill_attn.cu prefill-MoE:prefill_moe.cu"
for pair in $BENCHES; do
  label=${pair%%:*}; f=${pair##*:}
  echo "" >> "$RES"; echo "########## $label ($f) ##########" >> "$RES"
  [ -f "$f" ] || { echo "(missing)" >> "$RES"; continue; }
  bin="/tmp/kb_${f}.bin"
  if "$NVCC" -arch=sm_90a -O3 --use_fast_math -I . "$f" -o "$bin" >"/tmp/kc_$f" 2>&1; then
    timeout 240 "$bin" >> "$RES" 2>&1 || echo "(run failed/timeout)" >> "$RES"
  else
    echo "COMPILE FAILED:" >> "$RES"; head -12 "/tmp/kc_$f" >> "$RES"
  fi
done
echo "" >> "$RES"
echo "=== REFERENCE: vLLM bf16/EP 69.5 tok/s (~11%), fp8/EP 65.8 (~11%); roofline bf16 ~610 / fp8 ~1240 tok/s ===" >> "$RES"
echo "=== read each kernel's % of HBM peak: that is the per-kernel headroom vs vLLM's ~11-13% ===" >> "$RES"
touch /root/kbench_done
