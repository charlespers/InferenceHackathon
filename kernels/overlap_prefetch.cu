// overlap_prefetch.cu — lossless comms/weight-prefetch overlap: hide AR(L)'s NVLink latency behind a
// WEIGHT READ of layer L+1 (an independent HBM/L2 path with NO data dependency on AR's result), instead
// of behind L+1's actual compute (which DOES depend on AR's result and therefore can't be overlapped
// losslessly — see overlap_decode_wide.cu / research/n4_speculative_stale_tp.md's "exact deferred
// overlap" proposal). This is the smaller-scope alternative to the parked persistent megakernel
// (acfaf05's "comms-overlap measured but needs persistent megakernel ... -> parked"): no kernel fusion,
// just a second stream inside the existing discrete graph-captured pipeline.
//
// MECHANISM
//   AR(L) needs to finish before K1(L+1) can correctly read the post-AR residual and produce a result —
//   that dependency is real and cannot be removed without changing the math. But K1(L+1)'s WEIGHT
//   (Wqkv, fp8, read-only, independent of any activation) has no such dependency: it can be streamed
//   from HBM into L2 at any time. So while AR(L) runs on its own stream, a second stream concurrently
//   "touches" (reads, discards) Wqkv into L2. By the time AR(L) finishes and the real K1(L+1) launches,
//   the weight is L2-resident -> K1 pays only its compute+L2-read time, not compute+HBM-read-latency.
//   This is LOSSLESS by construction: the touch kernel writes nothing K1 reads, so K1's numerical output
//   must be bit-identical whether or not the touch ran. We assert that, not just claim it.
//
// WHAT THIS MEASURES
//   (1) correctness: K1's output bit-identical with vs without the prefetch touch.
//   (2) K1's OWN kernel duration, cold (no touch beforehand) vs warm (touch ran concurrently with the
//       preceding AR) — the direct evidence the mechanism works, isolated from AR/K2/K3 noise.
//   (3) full per-layer cycle time, SERIAL (today's real pattern: K1-K3 -> AR -> next K1, no overlap)
//       vs PREFETCH-OVERLAPPED (K1-K3 -> [AR || touch] -> next K1) over N_LAYERS layers.
//
// LATENCY-PROXY DISCLAIMER (same convention as decode_step_tp8.cu / overlap_decode_wide.cu): one reused
// dummy layer's weights/KV, real per-rank byte volume, real NCCL collective on a real stream.
//
// BUILD (same NCCL resolution as decode_step_tp8.cu):
//   NCCL_INC=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'include'))")
//   NCCL_LIB=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'lib'))")
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -I "$NCCL_INC" \
//        kernels/overlap_prefetch.cu -L "$NCCL_LIB" -lnccl -o /tmp/overlap_prefetch
//   LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" /tmp/overlap_prefetch [ctx_len=4096] [iters=200]
//
// =================================================================================================
#define DSTP8_NO_MAIN
#include "decode_step_tp8.cu"   // RankState, alloc_rank, tp8_k1_launch/k2/k3, TP, N_LAYERS, all CK/NK

// ================================================================================================
// Weight-touch (prefetch) kernel: grid-stride read of W[0..n) as uint4 (16B/lane-step), summed into a
// per-block partial that's written to `sink` (one float per block) so the compiler cannot eliminate the
// loads as dead code. No data dependency on any activation -> safe to run concurrently with the AR.
// ================================================================================================
__global__ void touch_weights_kernel(const fp8* __restrict__ W, size_t n_bytes, float* __restrict__ sink) {
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(W);
  const size_t nv = n_bytes >> 4;                             // 16 bytes per uint4
  unsigned acc = 0;
  for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < nv;
       v += (size_t)gridDim.x * blockDim.x) {
    uint4 p = wv[v];
    acc ^= p.x ^ p.y ^ p.z ^ p.w;                              // cheap, can't be hoisted/eliminated
  }
  __shared__ unsigned sh[256];
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
    if (threadIdx.x < o) sh[threadIdx.x] ^= sh[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) sink[blockIdx.x] = (float)sh[0];
}

static void touch_launch(const fp8* W, size_t n_bytes, float* sink, int n_blocks, cudaStream_t s) {
  touch_weights_kernel<<<n_blocks, 256, 0, s>>>(W, n_bytes, sink);
}

// Double-buffer for the AR target, ping-ponged by layer parity (same pattern as overlap_decode_wide.cu).
struct PfBuf {
  float* buf[2];
  cudaEvent_t k3_done[2];
  cudaEvent_t ar_done[2];
  cudaEvent_t touch_done[2];
};
static void alloc_pfbuf(PfBuf& w, cudaStream_t seed_stream) {
  for (int b = 0; b < 2; ++b) {
    CK(cudaMalloc(&w.buf[b], HIDDEN * sizeof(float)));
    CK(cudaEventCreateWithFlags(&w.k3_done[b], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&w.ar_done[b], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&w.touch_done[b], cudaEventDisableTiming));
    CK(cudaEventRecord(w.ar_done[b], seed_stream));
    CK(cudaEventRecord(w.touch_done[b], seed_stream));
  }
}

int main(int argc, char** argv) {
  const int ctx_len = (argc > 1) ? atoi(argv[1]) : 4096;
  const int n_layers = N_LAYERS;
  const int iters    = (argc > 2) ? atoi(argv[2]) : 200;
  const int warmup   = 20;

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < TP) {
    printf("Need >= %d CUDA devices for TP=%d; found %d.\n", TP, TP, ndev); return 1;
  }
  printf("== overlap_prefetch: AR(L) || weight-touch(Wqkv,L+1)  ->  K1(L+1) reads warm cache ==\n");
  printf("   TP=%d, ctx_len=%d, n_layers=%d, iters=%d\n", TP, ctx_len, n_layers, iters);

  for (int i = 0; i < TP; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < TP; ++j) if (i != j) {
      int can = 0; cudaDeviceCanAccessPeer(&can, i, j);
      if (can) cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  std::vector<RankState> R(TP);
  std::vector<ncclComm_t> comms(TP);
  std::vector<int> devs(TP);
  for (int r = 0; r < TP; ++r) devs[r] = r;
  NK(ncclCommInitAll(comms.data(), TP, devs.data()));

  std::vector<cudaStream_t> compute_s(TP), comm_s(TP), prefetch_s(TP);
  std::vector<PfBuf> pb(TP);
  std::vector<float*> sink(TP);                                 // touch-kernel dead-code-elim guard
  const int touch_blocks = 132;                                 // 1 block/SM on H100
  for (int r = 0; r < TP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    CK(cudaSetDevice(r));
    CK(cudaStreamCreate(&compute_s[r]));
    CK(cudaStreamCreate(&comm_s[r]));
    CK(cudaStreamCreate(&prefetch_s[r]));
    R[r].stream = compute_s[r];
    alloc_rank(R[r], ctx_len);
    alloc_pfbuf(pb[r], compute_s[r]);
    CK(cudaMalloc(&sink[r], touch_blocks * sizeof(float)));
  }

  const size_t wqkv_bytes = (size_t)QKV_OUT_RANK * HIDDEN * sizeof(fp8);
  printf("   Wqkv per rank = %.2f MB (the prefetch target; H100 L2 = 50 MB)\n", wqkv_bytes / 1e6);

  auto run_k1k2k3 = [&](int r, float* out_buf, cudaStream_t s) {
#if USE_GEMM
    gemm_k1_launch(R[r], R[r].h_a, s);
    tp8_k2_launch(R[r], s);
    gemm_k3_launch(R[r], s);
#else
    tp8_k1_launch(R[r], R[r].h_a, s);
    tp8_k2_launch(R[r], s);
    tp8_k3_launch(R[r], s);
#endif
    CK(cudaMemcpyAsync(out_buf, R[r].attn_partial, HIDDEN * sizeof(float),
                       cudaMemcpyDeviceToDevice, s));
  };
  auto run_k1_only = [&](int r, cudaStream_t s) {
#if USE_GEMM
    gemm_k1_launch(R[r], R[r].h_a, s);
#else
    tp8_k1_launch(R[r], R[r].h_a, s);
#endif
  };
  auto launch_ar = [&](int r, float* buf, cudaStream_t s) {
    NK(ncclAllReduce(buf, buf, HIDDEN, ncclFloat32, ncclSum, R[r].comm, s));
  };
  // L2 evict: a strided write over a buffer bigger than L2 (50 MB on H100) so each timed condition
  // starts from a genuinely cold cache, not warmed by the previous iteration's traffic.
  std::vector<float*> evictbuf(TP);
  const size_t evict_elems = 64ull * 1024 * 1024;               // 256 MB > 50 MB L2
  for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaMalloc(&evictbuf[r], evict_elems * sizeof(float))); }
  auto evict_l2 = [&](int r, cudaStream_t s) {
    CK(cudaMemsetAsync(evictbuf[r], 0x5a, evict_elems * sizeof(float), s));
  };

  // ---- correctness: K1's output must be BIT-IDENTICAL with vs without the prefetch touch ----
  {
    std::vector<float*> qkv_cold(TP), qkv_warm(TP);
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaMalloc(&qkv_cold[r], QKV_OUT_RANK * sizeof(float)));
                                                          CK(cudaMalloc(&qkv_warm[r], QKV_OUT_RANK * sizeof(float))); }
    bool all_finite = true, all_match = true;
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r));
      evict_l2(r, compute_s[r]); CK(cudaStreamSynchronize(compute_s[r]));
      run_k1_only(r, compute_s[r]);
      CK(cudaMemcpyAsync(qkv_cold[r], R[r].qkv_proj, QKV_OUT_RANK * sizeof(float), cudaMemcpyDeviceToDevice, compute_s[r]));
      CK(cudaStreamSynchronize(compute_s[r]));
    }
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r));
      evict_l2(r, compute_s[r]); CK(cudaStreamSynchronize(compute_s[r]));
      touch_launch(R[r].Wqkv, wqkv_bytes, sink[r], touch_blocks, prefetch_s[r]);
      CK(cudaStreamSynchronize(prefetch_s[r]));
      run_k1_only(r, compute_s[r]);
      CK(cudaMemcpyAsync(qkv_warm[r], R[r].qkv_proj, QKV_OUT_RANK * sizeof(float), cudaMemcpyDeviceToDevice, compute_s[r]));
      CK(cudaStreamSynchronize(compute_s[r]));
    }
    std::vector<float> hc(QKV_OUT_RANK), hw(QKV_OUT_RANK);
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(hc.data(), qkv_cold[0], QKV_OUT_RANK * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hw.data(), qkv_warm[0], QKV_OUT_RANK * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < QKV_OUT_RANK; ++i) {
      if (!std::isfinite(hc[i]) || !std::isfinite(hw[i])) all_finite = false;
      if (hc[i] != hw[i]) all_match = false;                    // exact bit-match expected: same inputs, same math
    }
    printf("  [check] K1 finite: %s | K1(cold) == K1(warm) bit-exact: %s\n",
           all_finite ? "PASS" : "FAIL", all_match ? "PASS" : "FAIL (prefetch changed the answer -- BUG)");
    if (!all_finite || !all_match) { printf("ABORT: correctness failed; not reporting timing.\n"); return 2; }
    for (int r = 0; r < TP; ++r) { cudaFree(qkv_cold[r]); cudaFree(qkv_warm[r]); }
  }

  // ---- (2) isolate K1's OWN duration: cold (AR ran, no touch) vs warm (AR || touch ran concurrently) ----
  cudaEvent_t e0, e1; CK(cudaSetDevice(0)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto time_k1_after = [&](bool with_touch) -> float {
    for (int it = 0; it < warmup; ++it) {
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); evict_l2(r, comm_s[r]); }
      NK(ncclGroupStart());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); launch_ar(r, pb[r].buf[0], comm_s[r]); }
      NK(ncclGroupEnd());
      if (with_touch) for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); touch_launch(R[r].Wqkv, wqkv_bytes, sink[r], touch_blocks, prefetch_s[r]); }
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(comm_s[r])); CK(cudaStreamSynchronize(prefetch_s[r])); }
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); run_k1_only(r, compute_s[r]); CK(cudaStreamSynchronize(compute_s[r])); }
    }
    float total_ms = 0.f;
    for (int it = 0; it < iters; ++it) {
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); evict_l2(r, comm_s[r]); CK(cudaStreamSynchronize(comm_s[r])); }
      NK(ncclGroupStart());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); launch_ar(r, pb[r].buf[0], comm_s[r]); }
      NK(ncclGroupEnd());
      if (with_touch) for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); touch_launch(R[r].Wqkv, wqkv_bytes, sink[r], touch_blocks, prefetch_s[r]); }
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(comm_s[r])); CK(cudaStreamSynchronize(prefetch_s[r])); }
      CK(cudaSetDevice(0)); CK(cudaEventRecord(e0, compute_s[0]));
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); run_k1_only(r, compute_s[r]); }
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(compute_s[r])); }
      CK(cudaSetDevice(0)); CK(cudaEventRecord(e1, compute_s[0])); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); total_ms += ms;
    }
    return total_ms / iters;
  };
  float k1_cold_ms = time_k1_after(false);
  float k1_warm_ms = time_k1_after(true);
  printf("\n  K1 duration, COLD (AR ran, cache untouched):       %.3f ms\n", k1_cold_ms);
  printf("  K1 duration, WARM (AR || weight-touch concurrent): %.3f ms\n", k1_warm_ms);
  printf("  K1 speedup from prefetch: %.1f%%  (this much of the AR's window is recovered, losslessly)\n",
         100.0 * (1.0 - (double)k1_warm_ms / k1_cold_ms));

  // ---- (3) full per-layer cycle: SERIAL vs PREFETCH-OVERLAPPED, over n_layers ----
  auto run_serial = [&]() {
    for (int L = 0; L < n_layers; ++L) {
      const int b = L & 1;
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); run_k1k2k3(r, pb[r].buf[b], compute_s[r]); }
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(compute_s[r])); }
      NK(ncclGroupStart());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); launch_ar(r, pb[r].buf[b], comm_s[r]); }
      NK(ncclGroupEnd());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(comm_s[r])); }
    }
  };
  auto run_prefetch_overlapped = [&]() {
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); run_k1k2k3(r, pb[r].buf[0], compute_s[r]); }
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaEventRecord(pb[r].k3_done[0], compute_s[r])); }
    for (int L = 0; L < n_layers; ++L) {
      const int b = L & 1;
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamWaitEvent(comm_s[r], pb[r].k3_done[b], 0)); }
      NK(ncclGroupStart());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); launch_ar(r, pb[r].buf[b], comm_s[r]); }
      NK(ncclGroupEnd());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaEventRecord(pb[r].ar_done[b], comm_s[r])); }
      // weight-touch runs CONCURRENTLY with the AR above -- no dependency on it, so no WaitEvent needed.
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); touch_launch(R[r].Wqkv, wqkv_bytes, sink[r], touch_blocks, prefetch_s[r]);
                                     CK(cudaEventRecord(pb[r].touch_done[b], prefetch_s[r])); }
      if (L + 1 < n_layers) {
        const int nb = (L + 1) & 1;
        for (int r = 0; r < TP; ++r) {
          CK(cudaSetDevice(r));
          CK(cudaStreamWaitEvent(compute_s[r], pb[r].ar_done[b], 0));     // real dependency: needs AR's result
          CK(cudaStreamWaitEvent(compute_s[r], pb[r].touch_done[b], 0));  // just for cache-warmth ordering in the bench
          run_k1k2k3(r, pb[r].buf[nb], compute_s[r]);
          CK(cudaEventRecord(pb[r].k3_done[nb], compute_s[r]));
        }
      }
    }
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(comm_s[r])); CK(cudaStreamSynchronize(compute_s[r])); CK(cudaStreamSynchronize(prefetch_s[r])); }
  };

  for (int it = 0; it < warmup; ++it) run_serial();
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e0, compute_s[0]));
  for (int it = 0; it < iters; ++it) run_serial();
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e1, compute_s[0])); CK(cudaEventSynchronize(e1));
  float ms_serial; CK(cudaEventElapsedTime(&ms_serial, e0, e1)); ms_serial /= (float)iters;

  for (int it = 0; it < warmup; ++it) run_prefetch_overlapped();
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e0, compute_s[0]));
  for (int it = 0; it < iters; ++it) run_prefetch_overlapped();
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e1, compute_s[0])); CK(cudaEventSynchronize(e1));
  float ms_overlap; CK(cudaEventElapsedTime(&ms_overlap, e0, e1)); ms_overlap /= (float)iters;

  printf("\n  SERIAL    (K1-K3 + AR, no overlap):              %.3f ms/token  -> %.1f tok/s comms+attn cap\n",
         ms_serial, 1000.0f / ms_serial);
  printf("  PREFETCH-OVERLAPPED (AR(L) || touch, then K1(L+1)): %.3f ms/token  -> %.1f tok/s comms+attn cap\n",
         ms_overlap, 1000.0f / ms_overlap);
  printf("  improvement: %.1f%%\n", 100.0 * (1.0 - (double)ms_overlap / ms_serial));
  printf("\nNOTE: compare against overlap_decode_wide.cu's compute-overlap result. That one overlaps AR\n");
  printf("with NEXT-LAYER COMPUTE (a real data dependency it can't satisfy losslessly with dummy inputs).\n");
  printf("This file overlaps AR with a WEIGHT TOUCH (no data dependency) -- the lossless mechanism that\n");
  printf("doesn't need the parked persistent megakernel.\n");

  for (int r = 0; r < TP; ++r) { ncclCommDestroy(comms[r]); cudaFree(evictbuf[r]); cudaFree(sink[r]); }
  return 0;
}
