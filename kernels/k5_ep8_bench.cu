// k5_ep8_bench.cu — 8-GPU Expert Parallelism benchmark for K5 MoE.
//
// Models real EP=8 decode: 128 experts sharded across 8 H100s (16 per GPU).
// Each decode step:
//   1. Broadcast hidden state (4096 fp32, 16 KB) from GPU 0 → all 8 GPUs
//   2. Each GPU runs K5 for its locally-assigned selected experts (~1 per step)
//   3. AllReduce (sum) the output (4096 fp32) across all GPUs
//
// Expert selection: we use a fixed balanced assignment (1 expert per GPU) to
// model the idealized B=1 decode.  In practice ~65% of GPUs see ≥1 expert
// per token; the balanced case is the upper-bound throughput.
//
// Build:
//   /usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math \
//     k5_ep8_bench.cu -I . -L/usr/lib/x86_64-linux-gnu -lnccl -o k5_ep8_bench
// Run:
//   ./k5_ep8_bench

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <nccl.h>
#include "common.cuh"
using namespace q3;

#define K5_NO_MAIN
#include "k5_experts.cu"
#include "k5_experts_warp.cu"

#define CK(x) do { cudaError_t _e = (x); \
  if (_e != cudaSuccess) { printf("CUDA %s:%d: %s\n", __FILE__, __LINE__, \
    cudaGetErrorString(_e)); exit(1); } } while(0)
#define NK(x) do { ncclResult_t _r = (x); \
  if (_r != ncclSuccess) { printf("NCCL %s:%d: %s\n", __FILE__, __LINE__, \
    ncclGetErrorString(_r)); exit(1); } } while(0)

__global__ void fill_fp8(fp8* w, size_t n, unsigned seed) {
  for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < n;
       i += (size_t)gridDim.x*blockDim.x) {
    unsigned h = (unsigned)(i*2654435761u) + seed*40503u;
    w[i] = fp8((((h % 2000) / 1000.0f) - 1.0f) * 0.25f);
  }
}
__global__ void fill_f32(float* a, size_t n, unsigned seed, float sc) {
  for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < n;
       i += (size_t)gridDim.x*blockDim.x) {
    unsigned h = (unsigned)(i*2246822519u) + seed*40503u;
    a[i] = (((h % 2000) / 1000.0f) - 1.0f) * sc;
  }
}
__global__ void zero_f32(float* a, size_t n) {
  for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < n;
       i += (size_t)gridDim.x*blockDim.x) a[i] = 0.f;
}

int main(int argc, char** argv) {
  const int    NGPU    = 8;
  const int    EXPERTS_PER_GPU = N_EXPERTS / NGPU;   // 16
  const int    CTAS    = (argc > 1) ? atoi(argv[1]) : 264;
  const int    BLK     = (argc > 2) ? atoi(argv[2]) : 1024;
  const double PEAK    = 3350.0;
  const int    WARM    = 10;
  const int    IT      = 200;
  const int    LAYERS  = N_LAYERS;

  printf("\n");
  printf("=================================================================\n");
  printf(" K5 EP=8 Benchmark  (8×H100 SXM5, sm_90a, fp8)\n");
  printf(" Qwen3-235B-A22B · 128 experts · %d/GPU · top-%d · %d layers\n",
         EXPERTS_PER_GPU, TOP_K, LAYERS);
  printf("=================================================================\n\n");

  // Check peer access between all GPU pairs
  for (int i = 0; i < NGPU; i++) {
    for (int j = 0; j < NGPU; j++) {
      if (i == j) continue;
      int canAccess;
      CK(cudaDeviceCanAccessPeer(&canAccess, i, j));
      if (!canAccess) printf("  WARNING: no P2P between GPU %d and %d\n", i, j);
    }
  }
  for (int i = 0; i < NGPU; i++) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < NGPU; j++) {
      if (i == j) continue;
      cudaDeviceEnablePeerAccess(j, 0);  // ignore already-enabled error
    }
  }

  // ── Allocate per-GPU buffers ──────────────────────────────────────────────
  const size_t gu_bytes = (size_t)2 * MOE_INTER * HIDDEN;
  const size_t d_bytes  = (size_t)HIDDEN * MOE_INTER;

  // Expert weights: EXPERTS_PER_GPU per device
  fp8   *Wgu[NGPU][TOP_K], *Wd[NGPU][TOP_K];   // we only need TOP_K slots (1 selected per GPU)
  float *Sgu[NGPU][TOP_K], *Sd[NGPU][TOP_K];

  // Device-side pointer arrays (the kernel takes arrays of per-expert ptrs)
  const fp8   **Wgu_d[NGPU], **Wd_d[NGPU];
  const float **Sgu_d[NGPU], **Sd_d[NGPU];

  // Hidden state + output per GPU
  float *hidden[NGPU], *output[NGPU], *act[NGPU];
  int   *sel_d[NGPU];   float *selw_d[NGPU];

  for (int g = 0; g < NGPU; g++) {
    CK(cudaSetDevice(g));
    // Allocate 1 expert slot (we run 1 expert per GPU in the balanced scenario)
    const int E_local = 1;
    for (int e = 0; e < E_local; e++) {
      CK(cudaMalloc(&Wgu[g][e], gu_bytes * sizeof(fp8)));
      CK(cudaMalloc(&Wd[g][e],  d_bytes  * sizeof(fp8)));
      CK(cudaMalloc(&Sgu[g][e], (size_t)2 * MOE_INTER * sizeof(float)));
      CK(cudaMalloc(&Sd[g][e],  (size_t)HIDDEN * sizeof(float)));
      fill_fp8<<<512,256>>>(Wgu[g][e], gu_bytes, (unsigned)(g*16 + e + 1));
      fill_fp8<<<512,256>>>(Wd[g][e],  d_bytes,  (unsigned)(g*16 + e + 100));
      fill_f32<<<64, 256>>>(Sgu[g][e], 2*MOE_INTER, (unsigned)(g*16+e+7), 0.02f);
      fill_f32<<<64, 256>>>(Sd[g][e],  HIDDEN, (unsigned)(g*16+e+13), 0.02f);
    }
    CK(cudaMalloc(&Wgu_d[g], E_local * sizeof(fp8*)));
    CK(cudaMalloc(&Wd_d[g],  E_local * sizeof(fp8*)));
    CK(cudaMalloc(&Sgu_d[g], E_local * sizeof(float*)));
    CK(cudaMalloc(&Sd_d[g],  E_local * sizeof(float*)));
    CK(cudaMemcpy(Wgu_d[g], Wgu[g], E_local*sizeof(fp8*),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Wd_d[g],  Wd[g],  E_local*sizeof(fp8*),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sgu_d[g], Sgu[g], E_local*sizeof(float*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd_d[g],  Sd[g],  E_local*sizeof(float*), cudaMemcpyHostToDevice));

    // hidden (bcast dest), output (allreduce src/dst), intermediate activations
    CK(cudaMalloc(&hidden[g], HIDDEN * sizeof(float)));
    CK(cudaMalloc(&output[g], HIDDEN * sizeof(float)));
    CK(cudaMalloc(&act[g],    (size_t)E_local * MOE_INTER * sizeof(float)));
    fill_f32<<<16,256>>>(hidden[g], HIDDEN, (unsigned)(99+g), 1.0f);
    CK(cudaMemset(output[g], 0, HIDDEN * sizeof(float)));

    // Expert index (each GPU runs expert 0 from its local bank = expert g*16+0 globally)
    int sel_h[1] = {0};  float selw_h[1] = {0.125f};  // uniform weight (1/8)
    CK(cudaMalloc(&sel_d[g],  1 * sizeof(int)));
    CK(cudaMalloc(&selw_d[g], 1 * sizeof(float)));
    CK(cudaMemcpy(sel_d[g],  sel_h,  sizeof(int),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(selw_d[g], selw_h, sizeof(float), cudaMemcpyHostToDevice));
  }

  // ── NCCL communicator ────────────────────────────────────────────────────
  int devList[NGPU]; for (int g = 0; g < NGPU; g++) devList[g] = g;
  ncclComm_t comms[NGPU];
  NK(ncclCommInitAll(comms, NGPU, devList));

  // ── CUDA streams (one per GPU) ───────────────────────────────────────────
  cudaStream_t streams[NGPU];
  for (int g = 0; g < NGPU; g++) {
    CK(cudaSetDevice(g));
    CK(cudaStreamCreate(&streams[g]));
  }

  // ── Kernel config ─────────────────────────────────────────────────────────
  const size_t smemA = (size_t)HIDDEN * sizeof(float);
  const size_t smemB = (size_t)1 * MOE_INTER * sizeof(float);   // E_local=1
  for (int g = 0; g < NGPU; g++) {
    CK(cudaSetDevice(g));
    CK(cudaFuncSetAttribute(k5a_gateup_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA));
    CK(cudaFuncSetAttribute(k5b_down_warp,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB));
  }
  dim3 gA(CTAS), gB(CTAS);

  // Helper: run one decode step (broadcast → local K5 → allreduce)
  auto step = [&]() {
    // 1. Broadcast hidden state from GPU 0 to all GPUs
    NK(ncclGroupStart());
    for (int g = 0; g < NGPU; g++)
      NK(ncclBroadcast(hidden[0], hidden[g], HIDDEN, ncclFloat, 0, comms[g], streams[g]));
    NK(ncclGroupEnd());

    // 2. Each GPU runs K5 for its 1 local expert (overlaps with broadcast completion)
    for (int g = 0; g < NGPU; g++) {
      CK(cudaSetDevice(g));
      k5a_gateup_warp<<<gA, BLK, smemA, streams[g]>>>(
          hidden[g], sel_d[g], Wgu_d[g], Sgu_d[g], act[g], 1);
      k5b_down_warp  <<<gB, BLK, smemB, streams[g]>>>(
          sel_d[g], selw_d[g], Wd_d[g], Sd_d[g], act[g], output[g], 1);
    }

    // 3. AllReduce output across all GPUs (sum expert contributions)
    NK(ncclGroupStart());
    for (int g = 0; g < NGPU; g++)
      NK(ncclAllReduce(output[g], output[g], HIDDEN, ncclFloat, ncclSum, comms[g], streams[g]));
    NK(ncclGroupEnd());

    // Sync all streams
    for (int g = 0; g < NGPU; g++) {
      CK(cudaSetDevice(g));
      CK(cudaStreamSynchronize(streams[g]));
    }
  };

  // ── Warmup ────────────────────────────────────────────────────────────────
  printf(" Warming up (%d iters)...\n", WARM);
  for (int i = 0; i < WARM; i++) step();
  for (int g = 0; g < NGPU; g++) { CK(cudaSetDevice(g)); CK(cudaDeviceSynchronize()); }

  // ── Measure allcomm-only overhead (no K5) ─────────────────────────────────
  // Isolate NCCL latency: broadcast + allreduce without compute.
  cudaEvent_t t0, t1;
  CK(cudaSetDevice(0));
  CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

  CK(cudaEventRecord(t0, streams[0]));
  for (int i = 0; i < IT; i++) {
    NK(ncclGroupStart());
    for (int g = 0; g < NGPU; g++)
      NK(ncclBroadcast(hidden[0], hidden[g], HIDDEN, ncclFloat, 0, comms[g], streams[g]));
    NK(ncclGroupEnd());
    NK(ncclGroupStart());
    for (int g = 0; g < NGPU; g++)
      NK(ncclAllReduce(output[g], output[g], HIDDEN, ncclFloat, ncclSum, comms[g], streams[g]));
    NK(ncclGroupEnd());
    for (int g = 0; g < NGPU; g++) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(streams[g])); }
  }
  CK(cudaSetDevice(0));
  CK(cudaEventRecord(t1, streams[0]));
  CK(cudaEventSynchronize(t1));
  float ms_comm; CK(cudaEventElapsedTime(&ms_comm, t0, t1));
  float us_comm = ms_comm * 1000.0f / IT;

  // ── Full EP=8 decode step ─────────────────────────────────────────────────
  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  for (int i = 0; i < IT; i++) step();
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  double ms_ep = ((ts1.tv_sec - ts0.tv_sec)*1e9 + (ts1.tv_nsec - ts0.tv_nsec)) / 1e6 / IT;
  float us_ep = (float)(ms_ep * 1000.0);

  // ── 1-GPU reference (from k5_microbench results) ──────────────────────────
  // The single-GPU benchmark ran all 8 experts on one card.  EP=8 runs 1 expert
  // per card in parallel, so compute wall time ≈ 1-expert time.
  const float us_1gpu_seq = 98.1f;   // measured: full 8-expert sequential (us/layer)
  const float us_1expert  = us_1gpu_seq / TOP_K;   // ≈ 12.3 us (1 expert worth of compute)

  // ── Results ───────────────────────────────────────────────────────────────
  printf("\n");
  printf(" NCCL comm overhead (bcast 16KB + allreduce 16KB):\n");
  printf("   %.1f us/step   (%d msg × %.0f us each)\n\n",
         us_comm, 2, us_comm / 2.0f);

  printf(" Decode TPOT breakdown (per layer):\n");
  printf("   1-GPU sequential K5 (8 experts, 151 MB)    %6.1f us\n", us_1gpu_seq);
  printf("   EP=8 local compute  (1 expert,  19 MB)     %6.1f us  (theoretical)\n", us_1expert);
  printf("   EP=8 NCCL comm      (bcast+allreduce)      %6.1f us  (measured)\n", us_comm);
  printf("   EP=8 full step      (measured wall-clock)  %6.1f us\n\n", us_ep);

  float ms_per_token_ep  = us_ep       * LAYERS / 1e3f;
  float ms_per_token_seq = us_1gpu_seq * LAYERS / 1e3f;
  float toks_ep  = 1e6f / (us_ep       * LAYERS);
  float toks_seq = 1e6f / (us_1gpu_seq * LAYERS);

  printf(" Throughput comparison (MoE portion only):\n");
  printf("   1-GPU K5 (8 experts, 1 card)    %5.2f ms/token   %5.0f tok/s\n",
         ms_per_token_seq, toks_seq);
  printf("   EP=8  K5 (1 expert/card×8)      %5.2f ms/token   %5.0f tok/s\n",
         ms_per_token_ep, toks_ep);
  printf("   EP speedup:                      %.2fx\n\n", toks_ep / toks_seq);

  printf(" Roofline check:\n");
  printf("   8-GPU peak HBM:  8 × %.0f GB/s = %.0f GB/s\n", PEAK, 8.0 * PEAK);
  printf("   Expert bytes/step: 1 expert × 19 MB × 8 GPUs = 151 MB → same as 1-GPU\n");
  printf("   Comm bytes:        bcast 16 KB + allreduce 16 KB = 32 KB (negligible)\n");
  printf("   EP HBM utilization: %.0f GB/s on %.0f GB/s peak = %.1f%%\n",
         19.0 * 8 / (us_ep * 1e-6) / 1e9,
         8.0 * PEAK,
         19.0 * 8 / (us_ep * 1e-6) / 1e9 / (8.0 * PEAK) * 100.0);
  printf("=================================================================\n\n");

  // Cleanup
  for (int g = 0; g < NGPU; g++) ncclCommDestroy(comms[g]);
  return 0;
}
