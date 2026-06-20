// k5_predict_bench.cu — benchmarks the prediction-pipelined K5 MoE kernel.
//
// PrefetchSink simulation: during token t's down-projection (K5b), a second
// CUDA stream concurrently runs gate+up (K5a) for the Markov-predicted expert
// set of token t+1.  On a hit (77.4%) the activation buffer is ready before
// K5b completes, shaving the gate+up cost from the critical path.  On a miss
// the buffer is discarded and K5a re-runs on the correct experts, paying the
// full sequential cost for that step.
//
// Build:
//   /usr/local/cuda-12.6/bin/nvcc -arch=sm_90a -O3 --use_fast_math \
//     kernels/k5_predict_bench.cu -I kernels -o k5_predict_bench
// Run:
//   CUDA_VISIBLE_DEVICES=0 ./k5_predict_bench

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

#define K5_NO_MAIN
#include "k5_experts.cu"
#include "k5_experts_warp.cu"

#define CK(x) do { cudaError_t _e = (x); \
  if (_e != cudaSuccess) { printf("CUDA %s:%d: %s\n", __FILE__, __LINE__, \
    cudaGetErrorString(_e)); exit(1); } } while(0)

__global__ void fill_fp8(fp8* w, size_t n, unsigned seed) {
  for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < n;
       i += (size_t)gridDim.x*blockDim.x) {
    unsigned h = (unsigned)(i*2654435761u) + seed*40503u;
    w[i] = fp8((((h % 2000) / 1000.0f) - 1.0f) * 0.25f);
  }
}
__global__ void fill_f32(float* a, size_t n, unsigned seed, float sc, int pos) {
  for (size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x; i < n;
       i += (size_t)gridDim.x*blockDim.x) {
    unsigned h = (unsigned)(i*2246822519u) + seed*40503u;
    float v = (((h % 2000) / 1000.0f) - 1.0f) * sc;
    a[i] = pos ? (fabsf(v) + 1e-3f) : v;
  }
}

int main(int argc, char** argv) {
  const int    E          = TOP_K;          // 8 experts per token
  const int    CTAS       = (argc > 1) ? atoi(argv[1]) : 264;
  const int    BLK        = (argc > 2) ? atoi(argv[2]) : 1024;
  const double PEAK       = (argc > 3) ? atof(argv[3]) : 3350.0;  // H100 HBM GB/s
  const float  HIT_RATE   = 0.774f;   // Markov predictor accuracy from routing_predict.json
  const int    LAYERS     = N_LAYERS;  // 94 for Qwen3-235B-A22B
  const int    WARM       = 20;
  const int    IT         = 400;

  // Expert weight buffers — two identical "expert banks" so the prefetch stream
  // reads different memory addresses than the compute stream (realistic scenario).
  const size_t gu_bytes = (size_t)2 * MOE_INTER * HIDDEN;   // gate+up weights per expert
  const size_t d_bytes  = (size_t)HIDDEN * MOE_INTER;        // down weights per expert

  fp8   *Wgu[E], *Wd[E];
  float *Sgu[E], *Sd[E];
  // Second bank for predicted-next experts
  fp8   *Wgu2[E], *Wd2[E];
  float *Sgu2[E], *Sd2[E];

  for (int e = 0; e < E; e++) {
    CK(cudaMalloc(&Wgu[e],  gu_bytes * sizeof(fp8)));
    CK(cudaMalloc(&Wd[e],   d_bytes  * sizeof(fp8)));
    CK(cudaMalloc(&Sgu[e],  (size_t)2 * MOE_INTER * sizeof(float)));
    CK(cudaMalloc(&Sd[e],   (size_t)HIDDEN * sizeof(float)));
    CK(cudaMalloc(&Wgu2[e], gu_bytes * sizeof(fp8)));
    CK(cudaMalloc(&Wd2[e],  d_bytes  * sizeof(fp8)));
    CK(cudaMalloc(&Sgu2[e], (size_t)2 * MOE_INTER * sizeof(float)));
    CK(cudaMalloc(&Sd2[e],  (size_t)HIDDEN * sizeof(float)));

    fill_fp8<<<512,256>>>(Wgu[e],  gu_bytes, 1u + e);
    fill_fp8<<<512,256>>>(Wd[e],   d_bytes,  100u + e);
    fill_f32<<<64, 256>>>(Sgu[e],  2 * MOE_INTER, 7u + e,  0.02f, 1);
    fill_f32<<<64, 256>>>(Sd[e],   HIDDEN,         13u + e, 0.02f, 1);
    fill_fp8<<<512,256>>>(Wgu2[e], gu_bytes, 200u + e);
    fill_fp8<<<512,256>>>(Wd2[e],  d_bytes,  300u + e);
    fill_f32<<<64, 256>>>(Sgu2[e], 2 * MOE_INTER, 17u + e, 0.02f, 1);
    fill_f32<<<64, 256>>>(Sd2[e],  HIDDEN,         23u + e, 0.02f, 1);
  }

  // Device pointer arrays
  const fp8   **Wgu_d,  **Wd_d,  **Wgu2_d,  **Wd2_d;
  const float **Sgu_d,  **Sd_d,  **Sgu2_d,  **Sd2_d;
  CK(cudaMalloc(&Wgu_d,  E * sizeof(fp8*)));  CK(cudaMemcpy(Wgu_d,  Wgu,  E * sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd_d,   E * sizeof(fp8*)));  CK(cudaMemcpy(Wd_d,   Wd,   E * sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sgu_d,  E * sizeof(float*))); CK(cudaMemcpy(Sgu_d,  Sgu,  E * sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd_d,   E * sizeof(float*))); CK(cudaMemcpy(Sd_d,   Sd,   E * sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wgu2_d, E * sizeof(fp8*)));  CK(cudaMemcpy(Wgu2_d, Wgu2, E * sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd2_d,  E * sizeof(fp8*)));  CK(cudaMemcpy(Wd2_d,  Wd2,  E * sizeof(fp8*),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sgu2_d, E * sizeof(float*))); CK(cudaMemcpy(Sgu2_d, Sgu2, E * sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd2_d,  E * sizeof(float*))); CK(cudaMemcpy(Sd2_d,  Sd2,  E * sizeof(float*), cudaMemcpyHostToDevice));

  // Expert indices + weights
  int   sel_h[E]; float selw_h[E];
  for (int e = 0; e < E; e++) { sel_h[e] = e; selw_h[e] = 0.1f + 0.01f * e; }
  int   *sel_d;   float *selw_d;
  CK(cudaMalloc(&sel_d,  E * sizeof(int)));   CK(cudaMemcpy(sel_d,  sel_h,  E * sizeof(int),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&selw_d, E * sizeof(float))); CK(cudaMemcpy(selw_d, selw_h, E * sizeof(float), cudaMemcpyHostToDevice));

  // Activation buffers — cur/nxt ping-pong
  float *y_d, *h_d, *act_cur, *act_nxt;
  CK(cudaMalloc(&y_d,     HIDDEN * sizeof(float)));
  CK(cudaMalloc(&h_d,     HIDDEN * sizeof(float)));
  CK(cudaMalloc(&act_cur, (size_t)E * MOE_INTER * sizeof(float)));
  CK(cudaMalloc(&act_nxt, (size_t)E * MOE_INTER * sizeof(float)));
  fill_f32<<<16,256>>>(y_d, HIDDEN, 99u, 1.0f, 0);
  CK(cudaDeviceSynchronize());

  // Shared memory sizes (same as k5_microbench)
  const size_t smemA = (size_t)HIDDEN * sizeof(float);
  const size_t smemB = (size_t)E * MOE_INTER * sizeof(float);
  CK(cudaFuncSetAttribute(k5a_gateup_warp, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA));
  CK(cudaFuncSetAttribute(k5b_down_warp,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB));
  dim3 gA(CTAS), gB(CTAS);

  // CUDA streams + timing events
  cudaStream_t s_compute, s_prefetch;
  CK(cudaStreamCreate(&s_compute));
  CK(cudaStreamCreate(&s_prefetch));
  cudaEvent_t ev0, ev1;
  CK(cudaEventCreate(&ev0));
  CK(cudaEventCreate(&ev1));

  // ── Benchmark 1: Sequential (baseline, no prediction) ────────────────────
  for (int i = 0; i < WARM; i++) {
    k5a_gateup_warp<<<gA, BLK, smemA>>>(y_d, sel_d, Wgu_d,  Sgu_d,  act_cur, E);
    k5b_down_warp  <<<gB, BLK, smemB>>>(      sel_d, selw_d, Wd_d,   Sd_d,   act_cur, h_d, E);
  }
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(ev0));
  for (int i = 0; i < IT; i++) {
    k5a_gateup_warp<<<gA, BLK, smemA>>>(y_d, sel_d, Wgu_d,  Sgu_d,  act_cur, E);
    k5b_down_warp  <<<gB, BLK, smemB>>>(      sel_d, selw_d, Wd_d,   Sd_d,   act_cur, h_d, E);
  }
  CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
  float ms_seq; CK(cudaEventElapsedTime(&ms_seq, ev0, ev1));
  float us_seq = ms_seq * 1000.0f / IT;

  // Measure gate+up and down individually so we can model the miss penalty.
  for (int i = 0; i < WARM; i++)
    k5a_gateup_warp<<<gA, BLK, smemA>>>(y_d, sel_d, Wgu_d, Sgu_d, act_cur, E);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(ev0));
  for (int i = 0; i < IT; i++)
    k5a_gateup_warp<<<gA, BLK, smemA>>>(y_d, sel_d, Wgu_d, Sgu_d, act_cur, E);
  CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
  float ms_a; CK(cudaEventElapsedTime(&ms_a, ev0, ev1));
  float us_a = ms_a * 1000.0f / IT;

  for (int i = 0; i < WARM; i++)
    k5b_down_warp<<<gB, BLK, smemB>>>(sel_d, selw_d, Wd_d, Sd_d, act_cur, h_d, E);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(ev0));
  for (int i = 0; i < IT; i++)
    k5b_down_warp<<<gB, BLK, smemB>>>(sel_d, selw_d, Wd_d, Sd_d, act_cur, h_d, E);
  CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
  float ms_b; CK(cudaEventElapsedTime(&ms_b, ev0, ev1));
  float us_b = ms_b * 1000.0f / IT;

  // ── Benchmark 2: Pipelined (ideal 100% prediction hit) ───────────────────
  // Token t: s_compute runs K5b (down), s_prefetch concurrently runs K5a
  // (gate+up) for the predicted token t+1 experts.
  // Both streams use different weight banks to model distinct expert memory.
  for (int i = 0; i < WARM; i++) {
    // Prime: gate+up for first token on compute stream
    k5a_gateup_warp<<<gA, BLK, smemA, s_compute>>>(y_d, sel_d, Wgu_d, Sgu_d, act_cur, E);
    CK(cudaStreamSynchronize(s_compute));
    k5b_down_warp  <<<gB, BLK, smemB, s_compute>>>(     sel_d, selw_d, Wd_d,  Sd_d,  act_cur, h_d, E);
    k5a_gateup_warp<<<gA, BLK, smemA, s_prefetch>>>(y_d, sel_d, Wgu2_d, Sgu2_d, act_nxt, E);
    CK(cudaStreamSynchronize(s_compute));
    CK(cudaStreamSynchronize(s_prefetch));
  }
  CK(cudaDeviceSynchronize());
  // Prime the pipeline: gate+up for token 0
  k5a_gateup_warp<<<gA, BLK, smemA>>>(y_d, sel_d, Wgu_d, Sgu_d, act_cur, E);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(ev0));
  for (int i = 0; i < IT; i++) {
    // s_compute: down projection for current token (uses act_cur from previous gate+up)
    k5b_down_warp  <<<gB, BLK, smemB, s_compute>>>(     sel_d, selw_d, Wd_d,   Sd_d,   act_cur, h_d, E);
    // s_prefetch: gate+up for PREDICTED next token (different weight bank = different HBM addresses)
    k5a_gateup_warp<<<gA, BLK, smemA, s_prefetch>>>(y_d, sel_d, Wgu2_d, Sgu2_d, act_nxt, E);
    CK(cudaStreamSynchronize(s_compute));
    CK(cudaStreamSynchronize(s_prefetch));
    // Swap: act_nxt becomes act_cur for next iteration
    float* tmp = act_cur; act_cur = act_nxt; act_nxt = tmp;
  }
  CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
  float ms_pipe; CK(cudaEventElapsedTime(&ms_pipe, ev0, ev1));
  float us_pipe = ms_pipe * 1000.0f / IT;

  // ── Benchmark 3: Realistic (77.4% hit rate) ──────────────────────────────
  // On a miss: the pre-fetched gate+up result is wrong → re-run sequentially.
  // Pipelined cost per token:
  //   hit  (77.4%): max(us_a, us_b) — the longer of the two overlapped kernels
  //   miss (22.6%): max(us_a, us_b) + us_a  (wasted prefetch + re-run gate+up) + us_b overhead
  // We use the MEASURED us_pipe (which captures hardware overlap reality) for the hit path.
  float us_hit  = us_pipe;
  float us_miss = us_pipe + us_a;   // wasted prefetch + correct gate+up (down already done)
  float us_real = HIT_RATE * us_hit + (1.0f - HIT_RATE) * us_miss;

  // ── Results ───────────────────────────────────────────────────────────────
  const double gb_a   = (double)E * gu_bytes / 1e9;  // gate+up bytes GB
  const double gb_b   = (double)E * d_bytes  / 1e9;  // down bytes GB
  const double gb_tot = gb_a + gb_b;

  printf("\n");
  printf("=================================================================\n");
  printf(" K5 Prediction Pipeline Benchmark  (H100 SXM5, sm_90a, fp8)\n");
  printf(" Qwen3-235B-A22B · %d layers · top-%d experts · peak %.0f GB/s\n",
         LAYERS, E, PEAK);
  printf(" Markov predictor hit rate: %.1f%%  (from routing_predict.json)\n",
         HIT_RATE * 100.0f);
  printf("=================================================================\n\n");

  printf(" Per-kernel HBM bandwidth:\n");
  printf("   gate+up (K5a, %.0f MB)  %7.1f us   %6.0f GB/s   %4.1f%% peak\n",
         gb_a * 1e3, us_a, gb_a / (us_a * 1e-6), gb_a / (us_a * 1e-6) / PEAK * 100.0);
  printf("   down    (K5b, %.0f MB)  %7.1f us   %6.0f GB/s   %4.1f%% peak\n",
         gb_b * 1e3, us_b, gb_b / (us_b * 1e-6), gb_b / (us_b * 1e-6) / PEAK * 100.0);
  printf("   total   (seq, %.0f MB)  %7.1f us   %6.0f GB/s   %4.1f%% peak\n\n",
         gb_tot * 1e3, us_seq, gb_tot / (us_seq * 1e-6), gb_tot / (us_seq * 1e-6) / PEAK * 100.0);

  printf(" Decode TPOT (per layer, 1-GPU proxy):\n");
  printf("   %-38s %7.1f us  %5.2f ms/94L  %6.0f tok/s\n",
         "sequential  (no prediction)",
         us_seq, us_seq * LAYERS / 1e3, 1e6 / (us_seq * LAYERS));
  printf("   %-38s %7.1f us  %5.2f ms/94L  %6.0f tok/s\n",
         "pipelined   (ideal 100% hit)",
         us_pipe, us_pipe * LAYERS / 1e3, 1e6 / (us_pipe * LAYERS));
  printf("   %-38s %7.1f us  %5.2f ms/94L  %6.0f tok/s\n\n",
         "pipelined   (77.4% Markov hit)",
         us_real, us_real * LAYERS / 1e3, 1e6 / (us_real * LAYERS));

  printf(" Speedup vs sequential:  ideal = %.2fx    realistic = %.2fx\n",
         us_seq / us_pipe, us_seq / us_real);
  printf(" MoE-only tok/s:  %d  →  %d (realistic)  →  %d (ideal)\n\n",
         (int)(1e6 / (us_seq  * LAYERS)),
         (int)(1e6 / (us_real * LAYERS)),
         (int)(1e6 / (us_pipe * LAYERS)));

  printf(" Note: these numbers are for the MoE portion only (1 GPU).\n");
  printf(" In 8-GPU EP mode, prediction additionally reduces all-to-all scatter\n");
  printf(" overhead: routing_predict.json shows affinity placement raises\n");
  printf(" local-GPU expert fraction from 12.3%% (round-robin) to 31.7%%,\n");
  printf(" cutting ~68%% of cross-GPU expert transfers on a hit.\n");
  printf("=================================================================\n\n");

  return 0;
}
