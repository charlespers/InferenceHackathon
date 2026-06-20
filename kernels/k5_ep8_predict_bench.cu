// k5_ep8_predict_bench.cu — EP=8 benchmark with Markov prediction pipeline.
//
// The Markov predictor runs at the END of token t's decode step and predicts
// which expert GPUs will be needed for token t+1.  The scatter (broadcast of
// hidden state to those GPUs) is issued immediately on a background stream —
// it completes during token t's tail / token t+1's attention — so it is NOT
// on the critical path of token t+1's expert compute.
//
// Only the GATHER (allreduce of expert outputs) stays in the critical path,
// because we cannot know the results before computing them.
//
// On a prediction MISS (22.6% of tokens): we fall back to the full sequential
// flow — scatter + compute + gather.
//
// This benchmark measures:
//   A. Baseline: full sequential EP=8 (no prediction)  [scatter + compute + gather]
//   B. Hit path: scatter already done (prediction correct) [compute + gather only]
//   C. Realistic: 77.4% hits B + 22.6% miss A
//
// Build:
//   /usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math \
//     k5_ep8_predict_bench.cu -I . -L/usr/lib/x86_64-linux-gnu -lnccl \
//     -o k5_ep8_predict_bench
// Run:
//   ./k5_ep8_predict_bench

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include "common.cuh"
using namespace q3;

#define K5_NO_MAIN
#include "k5_experts.cu"
#include "k5_experts_warp.cu"

#define CK(x) do { cudaError_t _e = (x); \
  if (_e != cudaSuccess) { printf("CUDA %s:%d: %s\n",__FILE__,__LINE__, \
    cudaGetErrorString(_e)); exit(1); } } while(0)
#define NK(x) do { ncclResult_t _r = (x); \
  if (_r != ncclSuccess) { printf("NCCL %s:%d: %s\n",__FILE__,__LINE__, \
    ncclGetErrorString(_r)); exit(1); } } while(0)

__global__ void fill_fp8(fp8* w, size_t n, unsigned seed) {
  for (size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; i<n;
       i+=(size_t)gridDim.x*blockDim.x) {
    unsigned h=(unsigned)(i*2654435761u)+seed*40503u;
    w[i]=fp8((((h%2000)/1000.0f)-1.0f)*0.25f);
  }
}
__global__ void fill_f32(float* a, size_t n, unsigned seed, float sc) {
  for (size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; i<n;
       i+=(size_t)gridDim.x*blockDim.x) {
    unsigned h=(unsigned)(i*2246822519u)+seed*40503u;
    a[i]=(((h%2000)/1000.0f)-1.0f)*sc;
  }
}

static double wall_ms() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec*1e3 + ts.tv_nsec*1e-6;
}

int main(int argc, char** argv) {
  const int    NGPU    = 8;
  const int    CTAS    = (argc>1)?atoi(argv[1]):264;
  const int    BLK     = (argc>2)?atoi(argv[2]):1024;
  const double PEAK    = 3350.0;
  const float  HIT     = 0.774f;
  const int    WARM    = 10;
  const int    IT      = 300;
  const int    LAYERS  = N_LAYERS;   // 94

  printf("\n=================================================================\n");
  printf(" K5 EP=8 + Prediction Pipeline Benchmark  (8×H100, sm_90a, fp8)\n");
  printf(" Qwen3-235B-A22B · 128 experts · 16/GPU · top-8 · %d layers\n", LAYERS);
  printf(" Markov hit rate: %.1f%%\n", HIT*100.f);
  printf("=================================================================\n\n");

  // ── Per-GPU allocations ───────────────────────────────────────────────────
  const size_t gu_bytes = (size_t)2*MOE_INTER*HIDDEN;
  const size_t d_bytes  = (size_t)HIDDEN*MOE_INTER;

  fp8   *Wgu[NGPU], *Wd[NGPU];
  float *Sgu[NGPU], *Sd[NGPU];
  const fp8   **Wgu_d[NGPU], **Wd_d[NGPU];
  const float **Sgu_d[NGPU], **Sd_d[NGPU];
  float *hidden[NGPU], *output[NGPU], *act[NGPU];
  int   *sel_d[NGPU]; float *selw_d[NGPU];

  for (int g=0; g<NGPU; g++) {
    CK(cudaSetDevice(g));
    // Enable peer access
    for (int j=0; j<NGPU; j++) if (j!=g) cudaDeviceEnablePeerAccess(j,0);

    CK(cudaMalloc(&Wgu[g], gu_bytes*sizeof(fp8)));
    CK(cudaMalloc(&Wd[g],  d_bytes *sizeof(fp8)));
    CK(cudaMalloc(&Sgu[g], (size_t)2*MOE_INTER*sizeof(float)));
    CK(cudaMalloc(&Sd[g],  (size_t)HIDDEN*sizeof(float)));
    fill_fp8<<<512,256>>>(Wgu[g], gu_bytes, (unsigned)(g+1));
    fill_fp8<<<512,256>>>(Wd[g],  d_bytes,  (unsigned)(g+100));
    fill_f32<<<64, 256>>>(Sgu[g], 2*MOE_INTER, (unsigned)(g+7),  0.02f);
    fill_f32<<<64, 256>>>(Sd[g],  HIDDEN,       (unsigned)(g+13), 0.02f);

    CK(cudaMalloc(&Wgu_d[g], sizeof(fp8*)));
    CK(cudaMalloc(&Wd_d[g],  sizeof(fp8*)));
    CK(cudaMalloc(&Sgu_d[g], sizeof(float*)));
    CK(cudaMalloc(&Sd_d[g],  sizeof(float*)));
    CK(cudaMemcpy(Wgu_d[g], &Wgu[g], sizeof(fp8*),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Wd_d[g],  &Wd[g],  sizeof(fp8*),   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sgu_d[g], &Sgu[g], sizeof(float*), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Sd_d[g],  &Sd[g],  sizeof(float*), cudaMemcpyHostToDevice));

    CK(cudaMalloc(&hidden[g], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&output[g], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&act[g],    (size_t)MOE_INTER*sizeof(float)));
    fill_f32<<<16,256>>>(hidden[g], HIDDEN, (unsigned)(99+g), 1.0f);
    CK(cudaMemset(output[g], 0, HIDDEN*sizeof(float)));

    int sh=0; float sw=0.125f;
    CK(cudaMalloc(&sel_d[g],  sizeof(int)));   CK(cudaMemcpy(sel_d[g],  &sh, sizeof(int),   cudaMemcpyHostToDevice));
    CK(cudaMalloc(&selw_d[g], sizeof(float))); CK(cudaMemcpy(selw_d[g], &sw, sizeof(float), cudaMemcpyHostToDevice));
  }

  int devList[NGPU]; for(int g=0;g<NGPU;g++) devList[g]=g;
  ncclComm_t comms[NGPU];
  NK(ncclCommInitAll(comms, NGPU, devList));

  // Two stream sets: compute + background (pre-scatter)
  cudaStream_t s_compute[NGPU], s_pre[NGPU];
  cudaEvent_t  ev_pre_done[NGPU];
  for (int g=0; g<NGPU; g++) {
    CK(cudaSetDevice(g));
    CK(cudaStreamCreate(&s_compute[g]));
    CK(cudaStreamCreate(&s_pre[g]));
    CK(cudaEventCreate(&ev_pre_done[g]));
  }

  const size_t smemA=(size_t)HIDDEN*sizeof(float);
  const size_t smemB=(size_t)MOE_INTER*sizeof(float);
  for (int g=0; g<NGPU; g++) {
    CK(cudaSetDevice(g));
    CK(cudaFuncSetAttribute(k5a_gateup_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA));
    CK(cudaFuncSetAttribute(k5b_down_warp,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB));
  }
  dim3 gA(CTAS), gB(CTAS);

  // ── Benchmark A: Baseline (full sequential, no prediction) ────────────────
  // scatter + compute + gather, all in critical path — same as k5_ep8_bench
  auto run_seq = [&]() {
    // Scatter: broadcast hidden from GPU 0
    NK(ncclGroupStart());
    for (int g=0; g<NGPU; g++) NK(ncclBroadcast(hidden[0],hidden[g],HIDDEN,ncclFloat,0,comms[g],s_compute[g]));
    NK(ncclGroupEnd());
    // K5 compute (each GPU: 1 expert)
    for (int g=0; g<NGPU; g++) {
      CK(cudaSetDevice(g));
      k5a_gateup_warp<<<gA,BLK,smemA,s_compute[g]>>>(hidden[g],sel_d[g],Wgu_d[g],Sgu_d[g],act[g],1);
      k5b_down_warp  <<<gB,BLK,smemB,s_compute[g]>>>(sel_d[g],selw_d[g],Wd_d[g],Sd_d[g],act[g],output[g],1);
    }
    // Gather: allreduce outputs
    NK(ncclGroupStart());
    for (int g=0; g<NGPU; g++) NK(ncclAllReduce(output[g],output[g],HIDDEN,ncclFloat,ncclSum,comms[g],s_compute[g]));
    NK(ncclGroupEnd());
    for (int g=0; g<NGPU; g++) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(s_compute[g])); }
  };

  // ── Benchmark B: Hit path (scatter pre-done, only compute + gather) ───────
  // Models: predictor correctly predicted experts → scatter ran during prev token.
  // We issue a "pre-scatter" on s_pre, wait for it on s_compute, then run K5.
  // This is the inter-token pipeline:
  //   token t:   ... [full decode] → [issue pre-scatter for t+1 on s_pre]
  //   token t+1: [attn (not timed here)] → [s_compute waits on ev_pre_done] → [K5] → [gather]
  auto run_hit = [&]() {
    // PRE-SCATTER: issue on background stream (this simulates being done during prev token's tail)
    NK(ncclGroupStart());
    for (int g=0; g<NGPU; g++) NK(ncclBroadcast(hidden[0],hidden[g],HIDDEN,ncclFloat,0,comms[g],s_pre[g]));
    NK(ncclGroupEnd());
    // Record event when pre-scatter done
    for (int g=0; g<NGPU; g++) { CK(cudaSetDevice(g)); CK(cudaEventRecord(ev_pre_done[g], s_pre[g])); }

    // COMPUTE STREAM: wait for pre-scatter, then immediately run K5 + gather
    // (In the real pipeline this "wait" is replaced by attention compute time,
    //  during which the scatter has already finished — the event wait is a formality.)
    for (int g=0; g<NGPU; g++) { CK(cudaSetDevice(g)); CK(cudaStreamWaitEvent(s_compute[g], ev_pre_done[g])); }

    // K5 compute
    for (int g=0; g<NGPU; g++) {
      CK(cudaSetDevice(g));
      k5a_gateup_warp<<<gA,BLK,smemA,s_compute[g]>>>(hidden[g],sel_d[g],Wgu_d[g],Sgu_d[g],act[g],1);
      k5b_down_warp  <<<gB,BLK,smemB,s_compute[g]>>>(sel_d[g],selw_d[g],Wd_d[g],Sd_d[g],act[g],output[g],1);
    }
    // Gather (always in critical path)
    NK(ncclGroupStart());
    for (int g=0; g<NGPU; g++) NK(ncclAllReduce(output[g],output[g],HIDDEN,ncclFloat,ncclSum,comms[g],s_compute[g]));
    NK(ncclGroupEnd());
    for (int g=0; g<NGPU; g++) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(s_compute[g])); }
  };

  // Warmup
  printf(" Warming up...\n");
  for (int i=0; i<WARM; i++) run_seq();
  for (int i=0; i<WARM; i++) run_hit();
  for (int g=0; g<NGPU; g++) { CK(cudaSetDevice(g)); CK(cudaDeviceSynchronize()); }

  // ── Measure A: sequential baseline ───────────────────────────────────────
  double t0 = wall_ms();
  for (int i=0; i<IT; i++) run_seq();
  double us_seq = (wall_ms()-t0)*1000.0/IT;

  // ── Measure B: hit path (pre-scatter hidden) ──────────────────────────────
  t0 = wall_ms();
  for (int i=0; i<IT; i++) run_hit();
  double us_hit = (wall_ms()-t0)*1000.0/IT;

  // ── Realistic: 77.4% hits, 22.6% miss ────────────────────────────────────
  double us_real = HIT*us_hit + (1.f-HIT)*us_seq;

  // ── Measure allreduce-only (gather cost) ──────────────────────────────────
  t0 = wall_ms();
  for (int i=0; i<IT; i++) {
    NK(ncclGroupStart());
    for (int g=0; g<NGPU; g++) NK(ncclAllReduce(output[g],output[g],HIDDEN,ncclFloat,ncclSum,comms[g],s_compute[g]));
    NK(ncclGroupEnd());
    for (int g=0; g<NGPU; g++) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(s_compute[g])); }
  }
  double us_gather = (wall_ms()-t0)*1000.0/IT;

  // ── Measure K5 compute only (no comm) ────────────────────────────────────
  t0 = wall_ms();
  for (int i=0; i<IT; i++) {
    for (int g=0; g<NGPU; g++) {
      CK(cudaSetDevice(g));
      k5a_gateup_warp<<<gA,BLK,smemA,s_compute[g]>>>(hidden[g],sel_d[g],Wgu_d[g],Sgu_d[g],act[g],1);
      k5b_down_warp  <<<gB,BLK,smemB,s_compute[g]>>>(sel_d[g],selw_d[g],Wd_d[g],Sd_d[g],act[g],output[g],1);
    }
    for (int g=0; g<NGPU; g++) { CK(cudaSetDevice(g)); CK(cudaStreamSynchronize(s_compute[g])); }
  }
  double us_compute = (wall_ms()-t0)*1000.0/IT;

  // ── Results ──────────────────────────────────────────────────────────────
  printf("\n");
  printf(" Per-layer timings (1 expert per GPU, EP=8):\n");
  printf("   K5 compute only  (no comm)         %6.1f us\n", us_compute);
  printf("   gather only      (allreduce)        %6.1f us\n", us_gather);
  printf("   scatter+compute+gather  (baseline)  %6.1f us\n", us_seq);
  printf("   compute+gather   (prediction hit)   %6.1f us\n", us_hit);
  printf("   scatter overhead:                   %6.1f us  (= baseline - hit)\n\n",
         us_seq - us_hit);

  printf(" Decode TPOT (%d layers):\n", LAYERS);
  printf("   %-42s %5.2f ms   %5.0f tok/s\n",
         "Baseline EP=8 (no prediction)",
         us_seq*LAYERS/1e3, 1e6/(us_seq*LAYERS));
  printf("   %-42s %5.2f ms   %5.0f tok/s\n",
         "EP=8 + prediction (100%% hit rate)",
         us_hit*LAYERS/1e3, 1e6/(us_hit*LAYERS));
  printf("   %-42s %5.2f ms   %5.0f tok/s\n",
         "EP=8 + prediction (77.4%% Markov hit)",
         us_real*LAYERS/1e3, 1e6/(us_real*LAYERS));

  printf("\n Speedup from prediction:   ideal=%.2fx   realistic=%.2fx\n",
         us_seq/us_hit, us_seq/us_real);
  printf(" Remaining bottleneck:     gather (%.0f us/layer) — needs K6 CUDA graph\n",
         us_gather);

  printf("\n Projected with K6 (graphed NCCL ~1-2 us/call):\n");
  double us_k6_floor = us_compute + 2.0;   // ~2 us comm per layer once graphed
  printf("   K6 only (no pred):   %.2f ms   %.0f tok/s\n",
         us_k6_floor*LAYERS/1e3, 1e6/(us_k6_floor*LAYERS));
  printf("   K6 + prediction:     %.2f ms   %.0f tok/s  (scatter already hidden)\n\n",
         us_k6_floor*LAYERS/1e3, 1e6/(us_k6_floor*LAYERS));

  printf("=================================================================\n\n");

  for (int g=0; g<NGPU; g++) ncclCommDestroy(comms[g]);
  return 0;
}
