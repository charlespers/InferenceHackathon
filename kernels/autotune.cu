// autotune.cu — standalone launch-config autotuner for the two decode bottlenecks
// of Qwen3-235B-A22B (B=1 decode), target sm_90a / H100.
//
//   K5  (kernels/k5_experts.cu):  k5a_gateup + k5b_down  — fused fp8 MoE experts.
//       The decode latency bottleneck (~14.2B of ~21.6B active params/token live in the
//       8 active experts). HBM-bandwidth bound at B=1. We SWEEP {threads/block, grid CTAs}
//       and report the config that maximizes effective HBM bandwidth (% of 3.35 TB/s peak).
//
//   K2  (kernels/k2_flash_decode.cu):  k2_flash_decode_partial + k2_flash_decode_reduce
//       — split-KV flash-decode. We SWEEP {n_splits} at ctx 4096 and 32768 and report the
//       split count that maximizes the effective KV-read bandwidth at each context length.
//
// We DO NOT modify the existing kernel files. To compile as a single translation unit we
// re-declare the needed kernels with their _NO_MAIN guards defined and pull the launchable
// device code in; all host driver / sweep logic lives here. Shapes + fp8 helpers come from
// the read-only common.cuh.
//
// Build (must compile clean):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/autotune.cu -o /tmp/at
//
// Standard CUDA only; no proprietary references.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

// ---------------------------------------------------------------------------------------------
// Pull in the kernels under test as one TU.
//
// k5_experts.cu guards its microbench main() behind K5_NO_MAIN; defining it gives us the two
// expert kernels (k5a_gateup, k5b_down) plus the device dot primitive, with no second main().
// k2_flash_decode.cu has no main() at all; including it gives k2_flash_decode_partial /
// k2_flash_decode_reduce and the K2_VPL / k2_warp_sum device helpers (Q3_K2_DEFS guard). We do
// NOT define Q3_K2_LAUNCH_HELPER (we drive the launches ourselves), so its host helper stays out.
// Both files #include "common.cuh" which is #pragma once, so the second include is a no-op.
// ---------------------------------------------------------------------------------------------
#define K5_NO_MAIN
#include "k5_experts.cu"
#include "k2_flash_decode.cu"

// ---------------------------------------------------------------------------------------------
// Small utilities.
// ---------------------------------------------------------------------------------------------
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                     \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));         \
  exit(1); } } while (0)

// Deterministic seeded host inputs (self-contained; mirrors the style in k5_experts.cu but the
// helpers there are inside its main() guard, so we define our own here).
static inline unsigned at_hash_u(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static inline float at_rnd(unsigned seed, size_t i, float scale, bool positive) {
  unsigned h = at_hash_u((unsigned)(i * 2654435761u) ^ (seed * 40503u));
  float v = (((h % 2001) / 1000.0f) - 1.0f) * scale;       // in [-scale, scale]
  return positive ? (fabsf(v) + 1e-3f) : v;
}

static double g_peak_gbps = 3350.0;                        // H100 HBM3 = 3.35 TB/s
static inline double gbps(double bytes, float ms) { return bytes / 1e6 / (double)ms; } // bytes/ms

// =============================================================================================
// K5 sweep — {threads/block} x {grid CTAs} over k5a_gateup + k5b_down.
// =============================================================================================
struct K5Result {
  int block;            // threads / block
  int ctasA, ctasB;     // grid CTAs for gate+up and down
  float msA, msB, msAB; // per-launch ms (avg)
  double gbpsAB;        // effective fused HBM bandwidth
};

static K5Result run_k5_sweep() {
  printf("\n======================================================================\n");
  printf("K5 autotune: fused fp8 MoE experts  (k5a_gateup + k5b_down)\n");
  printf("  sweep {threads/block} x {grid CTAs}, maximize fused HBM bandwidth\n");
  printf("======================================================================\n");

  const int E = TOP_K;                                     // 8 active experts (slots)
  const size_t gu_n = (size_t)2 * MOE_INTER * HIDDEN;      // gate+up fp8 per expert
  const size_t d_n  = (size_t)HIDDEN * MOE_INTER;          // down fp8 per expert

  // ---- build + upload seeded inputs (same inputs reused for every config) -------------------
  std::vector<fp8*>   Wgu_dp(E), Wd_dp(E);
  std::vector<float*> Sgu_dp(E), Sd_dp(E);
  {
    std::vector<fp8>   wbuf(gu_n), dbuf(d_n);
    std::vector<float> sgubuf(2 * MOE_INTER), sdbuf(HIDDEN);
    for (int e = 0; e < E; ++e) {
      for (size_t i = 0; i < gu_n; ++i) wbuf[i] = (fp8)at_rnd(1u + e, i, 0.25f, false);
      for (size_t i = 0; i < d_n;  ++i) dbuf[i] = (fp8)at_rnd(100u + e, i, 0.25f, false);
      for (int i = 0; i < 2 * MOE_INTER; ++i) sgubuf[i] = at_rnd(7u + e, i, 0.02f, true);
      for (int i = 0; i < HIDDEN; ++i)        sdbuf[i]  = at_rnd(13u + e, i, 0.02f, true);
      CK(cudaMalloc(&Wgu_dp[e], gu_n * sizeof(fp8)));
      CK(cudaMalloc(&Wd_dp[e],  d_n  * sizeof(fp8)));
      CK(cudaMalloc(&Sgu_dp[e], 2 * MOE_INTER * sizeof(float)));
      CK(cudaMalloc(&Sd_dp[e],  HIDDEN * sizeof(float)));
      CK(cudaMemcpy(Wgu_dp[e], wbuf.data(),   gu_n * sizeof(fp8), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(Wd_dp[e],  dbuf.data(),   d_n  * sizeof(fp8), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(Sgu_dp[e], sgubuf.data(), 2 * MOE_INTER * sizeof(float), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(Sd_dp[e],  sdbuf.data(),  HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  const fp8 **Wgu_d, **Wd_d; const float **Sgu_d, **Sd_d;
  CK(cudaMalloc(&Wgu_d, E * sizeof(fp8*)));   CK(cudaMemcpy(Wgu_d, Wgu_dp.data(), E * sizeof(fp8*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd_d,  E * sizeof(fp8*)));   CK(cudaMemcpy(Wd_d,  Wd_dp.data(),  E * sizeof(fp8*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sgu_d, E * sizeof(float*))); CK(cudaMemcpy(Sgu_d, Sgu_dp.data(), E * sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Sd_d,  E * sizeof(float*))); CK(cudaMemcpy(Sd_d,  Sd_dp.data(),  E * sizeof(float*), cudaMemcpyHostToDevice));

  std::vector<int>   sel_host(E);
  std::vector<float> selw_host(E);
  for (int e = 0; e < E; ++e) { sel_host[e] = e; selw_host[e] = 0.1f + 0.01f * e; }
  std::vector<float> y_host(HIDDEN);
  for (int k = 0; k < HIDDEN; ++k) y_host[k] = at_rnd(99u, k, 1.0f, false);

  int *sel_d; float *selw_d, *y_d, *h_d, *a_d;
  CK(cudaMalloc(&sel_d,  E * sizeof(int)));     CK(cudaMemcpy(sel_d,  sel_host.data(),  E * sizeof(int),   cudaMemcpyHostToDevice));
  CK(cudaMalloc(&selw_d, E * sizeof(float)));   CK(cudaMemcpy(selw_d, selw_host.data(), E * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&y_d, HIDDEN * sizeof(float))); CK(cudaMemcpy(y_d, y_host.data(), HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&h_d, HIDDEN * sizeof(float)));
  CK(cudaMalloc(&a_d, (size_t)E * MOE_INTER * sizeof(float)));
  CK(cudaDeviceSynchronize());

  // Dynamic smem: A stages y (HIDDEN floats, independent of block); B stages the full a buffer
  // (E*MOE_INTER floats, also independent of block). Opt in to >48KB once (B = 48KB at E=8).
  const size_t smemA = (size_t)HIDDEN * sizeof(float);
  const size_t smemB = (size_t)E * MOE_INTER * sizeof(float);
  CK(cudaFuncSetAttribute(k5a_gateup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemA));
  CK(cudaFuncSetAttribute(k5b_down,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB));

  // Bytes that MUST come from HBM per token = the fp8 expert weights (the bottleneck).
  const double bytesA = (double)E * gu_n;                  // gate+up weights
  const double bytesB = (double)E * d_n;                   // down weights
  const double bytesT = bytesA + bytesB;

  cudaEvent_t evs, eve; CK(cudaEventCreate(&evs)); CK(cudaEventCreate(&eve));
  const int WARM = 20, IT = 200;
  auto bench = [&](auto launch) -> float {
    for (int i = 0; i < WARM; ++i) launch();
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(evs));
    for (int i = 0; i < IT; ++i) launch();
    CK(cudaEventRecord(eve)); CK(cudaEventSynchronize(eve));
    float ms; CK(cudaEventElapsedTime(&ms, evs, eve)); return ms / IT;
  };

  // ---- knob ranges ---------------------------------------------------------------------------
  // threads/block: warp-per-row kernels, must be a multiple of 32 (we use power-of-two blocks).
  const int blocks[]   = {128, 256, 512, 1024};
  // grid CTAs: number of resident waves over the 132 SMs. The H100 has 132 SMs; we sweep from
  // "one wave" up through heavy oversubscription. Same CTA count used for A and B per config so
  // the knob is a single dimension; the kernels are grid-stride so any count is correct.
  const int cta_mult[] = {1, 2, 4, 8};                     // multiples of the SM count (132)
  const int SMS = 132;

  std::vector<K5Result> results;
  printf("\n  %-7s %-7s %-7s %10s %10s %10s %10s %10s\n",
         "block", "ctas", "warps", "us_A", "us_B", "us_A+B", "GB/s", "%peak");
  printf("  ------------------------------------------------------------------------------------\n");

  for (int bi = 0; bi < (int)(sizeof(blocks)/sizeof(blocks[0])); ++bi) {
    const int block = blocks[bi];
    for (int ci = 0; ci < (int)(sizeof(cta_mult)/sizeof(cta_mult[0])); ++ci) {
      const int ctas = SMS * cta_mult[ci];                 // CTAs in the grid for both A and B
      auto runA  = [&]() { k5a_gateup<<<ctas, block, smemA>>>(y_d, sel_d, Wgu_d, Sgu_d, a_d, E); };
      auto runB  = [&]() { k5b_down  <<<ctas, block, smemB>>>(sel_d, selw_d, Wd_d, Sd_d, a_d, h_d, E); };
      auto runAB = [&]() { runA(); runB(); };

      // sanity launch to catch any per-config launch error (e.g. resource limits) and skip it
      CK(cudaMemset(h_d, 0, HIDDEN * sizeof(float)));
      runAB();
      cudaError_t le = cudaGetLastError();
      if (le != cudaSuccess) {
        printf("  %-7d %-7d %-7d   launch failed: %s (skipped)\n",
               block, ctas, ctas * (block >> 5), cudaGetErrorString(le));
        CK(cudaDeviceSynchronize());
        continue;
      }
      CK(cudaDeviceSynchronize());

      K5Result r;
      r.block = block; r.ctasA = ctas; r.ctasB = ctas;
      r.msA = bench(runA);
      r.msB = bench(runB);
      r.msAB = bench(runAB);
      r.gbpsAB = gbps(bytesT, r.msAB);
      results.push_back(r);

      printf("  %-7d %-7d %-7d %10.2f %10.2f %10.2f %10.1f %9.1f%%\n",
             r.block, r.ctasA, ctas * (block >> 5),
             r.msA * 1e3, r.msB * 1e3, r.msAB * 1e3,
             r.gbpsAB, 100.0 * r.gbpsAB / g_peak_gbps);
    }
  }

  // ---- rank + report -------------------------------------------------------------------------
  std::sort(results.begin(), results.end(),
            [](const K5Result& a, const K5Result& b) { return a.gbpsAB > b.gbpsAB; });

  K5Result k5best;
  k5best.block = 0; k5best.ctasA = k5best.ctasB = 0;
  k5best.msA = k5best.msB = k5best.msAB = 0.f; k5best.gbpsAB = 0.0;

  printf("\n  per-token expert weight read: %.1f MB  (gate+up %.1f MB + down %.1f MB)\n",
         bytesT / 1e6, bytesA / 1e6, bytesB / 1e6);
  printf("  ranked by fused (A+B) bandwidth:\n");
  printf("  %-5s %-7s %-7s %10s %9s\n", "rank", "block", "ctas", "GB/s", "%peak");
  for (int i = 0; i < (int)results.size(); ++i)
    printf("  %-5d %-7d %-7d %10.1f %8.1f%%\n",
           i + 1, results[i].block, results[i].ctasA,
           results[i].gbpsAB, 100.0 * results[i].gbpsAB / g_peak_gbps);

  if (!results.empty()) {
    k5best = results[0];
    const K5Result& b = k5best;
    printf("\n  >>> K5 BEST CONFIG: block=%d  ctas=%d  ->  %.1f GB/s  (%.1f%% of %.0f GB/s peak),"
           "  %.2f us/token (A+B)\n",
           b.block, b.ctasA, b.gbpsAB, 100.0 * b.gbpsAB / g_peak_gbps, g_peak_gbps, b.msAB * 1e3);
    printf("      MoE-expert decode over %d layers at best config: %.2f ms/token\n",
           N_LAYERS, b.msAB * N_LAYERS);
  }

  // ---- cleanup -------------------------------------------------------------------------------
  for (int e = 0; e < E; ++e) { cudaFree(Wgu_dp[e]); cudaFree(Wd_dp[e]); cudaFree(Sgu_dp[e]); cudaFree(Sd_dp[e]); }
  cudaFree(Wgu_d); cudaFree(Wd_d); cudaFree(Sgu_d); cudaFree(Sd_d);
  cudaFree(sel_d); cudaFree(selw_d); cudaFree(y_d); cudaFree(h_d); cudaFree(a_d);
  cudaEventDestroy(evs); cudaEventDestroy(eve);
  return k5best;
}

// =============================================================================================
// K2 sweep — {n_splits} for split-KV flash-decode, at ctx 4096 and 32768.
// =============================================================================================
struct K2Result {
  int ctx;
  int n_splits;
  float msPart, msReduce, msTot;
  double gbpsKV;        // effective KV-read bandwidth (partial pass streams the whole cache)
};

static K2Result run_k2_one_ctx(int ctx_len) {
  // ---- build + upload one seeded KV cache for this ctx (reused across all split counts) ------
  const size_t kv_elems = (size_t)ctx_len * KV_DIM;
  std::vector<fp8>   kbuf(kv_elems), vbuf(kv_elems);
  for (size_t i = 0; i < kv_elems; ++i) {
    kbuf[i] = (fp8)at_rnd(201u, i, 0.25f, false);
    vbuf[i] = (fp8)at_rnd(202u, i, 0.25f, false);
  }
  std::vector<float> ksc(KV_DIM), vsc(KV_DIM), q_host(Q_DIM);
  for (int i = 0; i < KV_DIM; ++i) { ksc[i] = at_rnd(203u, i, 0.02f, true); vsc[i] = at_rnd(204u, i, 0.02f, true); }
  for (int i = 0; i < Q_DIM; ++i)  q_host[i] = at_rnd(205u, i, 1.0f, false);

  fp8 *kv_k_d, *kv_v_d; float *ksc_d, *vsc_d, *q_d, *attn_d;
  CK(cudaMalloc(&kv_k_d, kv_elems * sizeof(fp8))); CK(cudaMemcpy(kv_k_d, kbuf.data(), kv_elems * sizeof(fp8), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&kv_v_d, kv_elems * sizeof(fp8))); CK(cudaMemcpy(kv_v_d, vbuf.data(), kv_elems * sizeof(fp8), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&ksc_d, KV_DIM * sizeof(float)));  CK(cudaMemcpy(ksc_d, ksc.data(), KV_DIM * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&vsc_d, KV_DIM * sizeof(float)));  CK(cudaMemcpy(vsc_d, vsc.data(), KV_DIM * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&q_d, Q_DIM * sizeof(float)));     CK(cudaMemcpy(q_d, q_host.data(), Q_DIM * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&attn_d, Q_DIM * sizeof(float)));
  CK(cudaDeviceSynchronize());

  // KV bytes that MUST be read from HBM = both K and V caches (fp8), the bandwidth bound.
  const double kvBytes = 2.0 * (double)kv_elems;            // 1 byte / fp8 elem

  cudaEvent_t evs, eve; CK(cudaEventCreate(&evs)); CK(cudaEventCreate(&eve));
  const int WARM = 20, IT = 200;
  auto bench = [&](auto launch) -> float {
    for (int i = 0; i < WARM; ++i) launch();
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(evs));
    for (int i = 0; i < IT; ++i) launch();
    CK(cudaEventRecord(eve)); CK(cudaEventSynchronize(eve));
    float ms; CK(cudaEventElapsedTime(&ms, evs, eve)); return ms / IT;
  };

  // knob: number of KV-splits. The partial buffers (part_m/l/acc) are sized per split count, so
  // we allocate to the max we will try and reuse the buffer for every config.
  const int splits[] = {4, 8, 16, 32, 64};
  const int max_splits = 64;
  float *part_m, *part_l, *part_acc;
  CK(cudaMalloc(&part_m,   (size_t)N_Q_HEADS * max_splits * sizeof(float)));
  CK(cudaMalloc(&part_l,   (size_t)N_Q_HEADS * max_splits * sizeof(float)));
  CK(cudaMalloc(&part_acc, (size_t)N_Q_HEADS * max_splits * HEAD_DIM * sizeof(float)));

  const int warps_per_cta = 4;                             // 128 threads/CTA -> 4 head-warps
  const int block = warps_per_cta * 32;

  printf("\n  ctx=%d  (KV read = %.1f MB:  K %.1f MB + V %.1f MB)\n",
         ctx_len, kvBytes / 1e6, kvBytes / 2e6, kvBytes / 2e6);
  printf("    %-9s %12s %12s %12s %10s %9s\n",
         "n_splits", "us_partial", "us_reduce", "us_total", "GB/s", "%peak");
  printf("    -------------------------------------------------------------------------------\n");

  std::vector<K2Result> rs;
  for (int si = 0; si < (int)(sizeof(splits)/sizeof(splits[0])); ++si) {
    const int S = splits[si];
    if (S > ctx_len) continue;                             // need at least 1 timestep / split
    dim3 gP(S, (N_Q_HEADS + warps_per_cta - 1) / warps_per_cta);
    dim3 gR((N_Q_HEADS + warps_per_cta - 1) / warps_per_cta);
    auto runP = [&]() {
      k2_flash_decode_partial<<<gP, block>>>(q_d, kv_k_d, kv_v_d, ksc_d, vsc_d,
                                             ctx_len, S, part_m, part_l, part_acc);
    };
    auto runR = [&]() {
      k2_flash_decode_reduce<<<gR, block>>>(part_m, part_l, part_acc, S, attn_d);
    };
    auto runPR = [&]() { runP(); runR(); };

    runPR();
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
      printf("    %-9d  launch failed: %s (skipped)\n", S, cudaGetErrorString(le));
      CK(cudaDeviceSynchronize());
      continue;
    }
    CK(cudaDeviceSynchronize());

    K2Result r;
    r.ctx = ctx_len; r.n_splits = S;
    r.msPart   = bench(runP);
    r.msReduce = bench(runR);
    r.msTot    = bench(runPR);
    r.gbpsKV   = gbps(kvBytes, r.msPart);                  // partial pass is the KV-read pass
    rs.push_back(r);

    printf("    %-9d %12.2f %12.2f %12.2f %10.1f %8.1f%%\n",
           S, r.msPart * 1e3, r.msReduce * 1e3, r.msTot * 1e3,
           r.gbpsKV, 100.0 * r.gbpsKV / g_peak_gbps);
  }

  // rank by KV-read bandwidth; keep the best for this ctx
  std::sort(rs.begin(), rs.end(),
            [](const K2Result& a, const K2Result& b) { return a.gbpsKV > b.gbpsKV; });
  K2Result best;
  best.ctx = ctx_len; best.n_splits = 0; best.msPart = best.msReduce = best.msTot = 0.f; best.gbpsKV = 0.0;
  if (!rs.empty()) {
    best = rs[0];
    printf("    ranked: ");
    for (int i = 0; i < (int)rs.size(); ++i)
      printf("%s%d(%.0f GB/s)", i ? " > " : "", rs[i].n_splits, rs[i].gbpsKV);
    printf("\n    >>> ctx=%d BEST n_splits=%d  ->  %.1f GB/s (%.1f%% peak),  %.2f us total\n",
           ctx_len, best.n_splits, best.gbpsKV, 100.0 * best.gbpsKV / g_peak_gbps, best.msTot * 1e3);
  }

  cudaFree(kv_k_d); cudaFree(kv_v_d); cudaFree(ksc_d); cudaFree(vsc_d); cudaFree(q_d); cudaFree(attn_d);
  cudaFree(part_m); cudaFree(part_l); cudaFree(part_acc);
  cudaEventDestroy(evs); cudaEventDestroy(eve);
  return best;
}

static void run_k2_sweep(K2Result& best4k, K2Result& best32k) {
  printf("\n======================================================================\n");
  printf("K2 autotune: split-KV flash-decode  (k2_flash_decode_partial + _reduce)\n");
  printf("  sweep {n_splits} at ctx 4096 and 32768, maximize KV-read bandwidth\n");
  printf("======================================================================\n");
  best4k  = run_k2_one_ctx(4096);
  best32k = run_k2_one_ctx(32768);
}

// =============================================================================================
// main — run both sweeps, print a clear BEST CONFIG summary per kernel.
// =============================================================================================
int main(int argc, char** argv) {
  // optional arg: HBM peak GB/s override (default 3350 = H100 HBM3 @ 3.35 TB/s)
  if (argc > 1) g_peak_gbps = atof(argv[1]);

  int ndev = 0, dev = 0; cudaDeviceProp prop;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    printf("No CUDA device found.\n");
    return 1;
  }
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("autotuner: device=%s  SMs=%d  assumed HBM peak=%.0f GB/s  (target sm_90a / H100)\n",
         prop.name, prop.multiProcessorCount, g_peak_gbps);

  K5Result k5best = run_k5_sweep();

  K2Result best4k, best32k;
  run_k2_sweep(best4k, best32k);

  // ---- final BEST CONFIG summary -------------------------------------------------------------
  printf("\n======================================================================\n");
  printf("BEST CONFIG SUMMARY\n");
  printf("======================================================================\n");
  printf("BEST CONFIG  K5 (fused MoE experts):  block=%d  ctas=%d  (%.1f GB/s, %.1f%% peak,"
         "  %.2f us/token)\n",
         k5best.block, k5best.ctasA, k5best.gbpsAB,
         100.0 * k5best.gbpsAB / g_peak_gbps, k5best.msAB * 1e3);
  printf("BEST CONFIG  K2 (flash-decode) ctx=4096 :  n_splits=%d  (%.1f GB/s, %.1f%% peak)\n",
         best4k.n_splits, best4k.gbpsKV, 100.0 * best4k.gbpsKV / g_peak_gbps);
  printf("BEST CONFIG  K2 (flash-decode) ctx=32768:  n_splits=%d  (%.1f GB/s, %.1f%% peak)\n",
         best32k.n_splits, best32k.gbpsKV, 100.0 * best32k.gbpsKV / g_peak_gbps);
  printf("======================================================================\n");
  return 0;
}
