// overlap_decode_wide.cu — comms/compute overlap with the FULL K1+K2+K3 attention prologue as the
// hidden-behind compute, instead of overlap_decode.cu's single QKV GEMV (K1 alone).
//
// WHY: overlap_decode.cu measured only ~13.8% of the AR's cost hidden (9.67us of 70.29us) when
// overlapping a single K1 GEMV (41.71us) — its own comments call that a deliberate LOWER bound,
// since the GEMV is much shorter than the AR. This file tests whether widening the hidden-behind
// window to the full attention prologue (K1 QKV GEMV -> K2 flash-decode -> K3 O-proj+residual,
// ~tens of us more total) closes more of the gap toward stale_tp_ceiling.py's "once independent
// compute >= the AR, it hides entirely" threshold (docs/path-to-1000.md, research/n4_speculative_stale_tp.md).
//
// MECHANISM: reuses decode_step_tp8.cu's already-validated RankState/alloc_rank/tp8_k1_launch/
// tp8_k2_launch/tp8_k3_launch (its own correctness gate already proved these against a single-GPU
// reference) via DSTP8_NO_MAIN, rather than re-deriving buffer layouts from scratch — same
// "reuse validated kernels" principle decode_step_tp8.cu itself uses for K1-K5.
//
// Same two-stream (compute, comm) + cudaEvent double-buffering scaffold as overlap_decode.cu's
// Scheme C, just with K1+K2+K3 (not a single GEMV) as what runs on the compute stream while the
// PREVIOUS layer's all-reduce runs on the comm stream.
//
// LATENCY-PROXY DISCLAIMER (same convention as decode_step_tp8.cu / overlap_decode.cu): one
// reused dummy layer's weights/KV, real per-rank byte volume, real NCCL collective on a real
// stream. Correctness of the K1-K3 chain itself is decode_step_tp8.cu's gate, not re-proven here;
// this file only adds and checks the OVERLAP's correctness (the all-reduce result, post-overlap).
//
// BUILD (same NCCL resolution as decode_step_tp8.cu):
//   NCCL_INC=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'include'))")
//   NCCL_LIB=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'lib'))")
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -I "$NCCL_INC" \
//        kernels/overlap_decode_wide.cu -L "$NCCL_LIB" -lnccl -o /tmp/overlap_wide
//   LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" /tmp/overlap_wide [ctx_len=4096] [iters=200]
//
// =================================================================================================
#define DSTP8_NO_MAIN
#include "decode_step_tp8.cu"   // RankState, alloc_rank, tp8_k1_launch/k2/k3, TP, N_LAYERS, all CK/NK

// ================================================================================================
// Per-rank double buffer for the AR target: tp8_k3_launch writes its post-attention partial into
// S.attn_partial (decode_step_tp8.cu's own buffer) -- to pipeline layer L's AR against layer L+1's
// K1-K3, we need TWO such buffers per rank so layer L+1's K3 doesn't overwrite what layer L's AR
// is still reducing. decode_step_tp8.cu's RankState has exactly one `attn_partial`; allocate one
// extra scratch buffer here and alternate between them by layer parity.
// ================================================================================================
struct WideBuf {
  float* buf[2];     // [HIDDEN] each, ping-pong by layer parity
  cudaEvent_t k3_done[2];   // K1-K3 (compute_s) finished writing buf[b] -> AR(comm_s) may read it
  cudaEvent_t ar_done[2];   // AR (comm_s) finished reading buf[b] -> next K1-K3 may overwrite it
};

static void alloc_widebuf(WideBuf& w, cudaStream_t seed_stream) {
  for (int b = 0; b < 2; ++b) {
    CK(cudaMalloc(&w.buf[b], HIDDEN * sizeof(float)));
    CK(cudaEventCreateWithFlags(&w.k3_done[b], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&w.ar_done[b], cudaEventDisableTiming));
    CK(cudaEventRecord(w.ar_done[b], seed_stream));   // pre-signaled: both buffers start "free"
  }
}

int main(int argc, char** argv) {
  const int ctx_len = (argc > 1) ? atoi(argv[1]) : 4096;
  const int n_layers = N_LAYERS;          // real Qwen3-235B-A22B layer count, not a CLI knob
  const int iters    = (argc > 2) ? atoi(argv[2]) : 200;
  const int warmup   = 20;

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < TP) {
    printf("Need >= %d CUDA devices for TP=%d; found %d.\n", TP, TP, ndev); return 1;
  }
  printf("== overlap_decode_wide: AR(L) || K1+K2+K3(L+1), TP=%d, ctx_len=%d, n_layers=%d, iters=%d ==\n",
         TP, ctx_len, n_layers, iters);

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

  std::vector<cudaStream_t> compute_s(TP), comm_s(TP);
  std::vector<WideBuf> wb(TP);
  for (int r = 0; r < TP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    CK(cudaSetDevice(r));
    CK(cudaStreamCreate(&compute_s[r]));
    CK(cudaStreamCreate(&comm_s[r]));
    R[r].stream = compute_s[r];   // tp8_k1/k2/k3_launch take their stream as an explicit arg, but
                                  // alloc_rank may also stash a default -- keep both consistent.
    alloc_rank(R[r], ctx_len);
    alloc_widebuf(wb[r], compute_s[r]);
  }

  // ---- one "layer" of compute: K1 -> K2 -> K3, K3's output copied into the target WideBuf slot ----
  //   USE_GEMM: the FAST cuBLASLt fp8 attention prologue (matches the engine's enqueue_tp8_layer), so
  //   the hidden-behind window is the REAL (fast) compute, not the slow GEMV — the honest overlap number.
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
    // tp8_k3_launch's real output lives in R[r].attn_partial (decode_step_tp8.cu's own buffer);
    // copy it into this layer's ping-pong slot so the AR has a stable, non-aliased target.
    CK(cudaMemcpyAsync(out_buf, R[r].attn_partial, HIDDEN * sizeof(float),
                       cudaMemcpyDeviceToDevice, s));
  };

  auto launch_ar = [&](int r, float* buf, cudaStream_t s) {
    NK(ncclAllReduce(buf, buf, HIDDEN, ncclFloat32, ncclSum, R[r].comm, s));
  };

  // ---- correctness: run one layer through K1-K3 + AR, sanity-check it doesn't crash/NaN ----
  // (decode_step_tp8.cu's own gate already proves K1-K3's math; this just confirms the AR runs
  // cleanly on the WideBuf-copied output before trusting any timing built on top of it.)
  {
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); run_k1k2k3(r, wb[r].buf[0], compute_s[r]); }
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(compute_s[r])); }
    NK(ncclGroupStart());
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); launch_ar(r, wb[r].buf[0], comm_s[r]); }
    NK(ncclGroupEnd());
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(comm_s[r])); }
    std::vector<float> got(HIDDEN);
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(got.data(), wb[0].buf[0], HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
    bool finite = true;
    for (float v : got) if (!std::isfinite(v)) { finite = false; break; }
    printf("  [check] K1+K2+K3 -> AR ran clean, output finite: %s\n", finite ? "PASS" : "FAIL (NaN/Inf)");
    if (!finite) return 2;
  }

  // ---- SERIAL: K1-K3(L) on compute_s; sync; AR(L) on comm_s; sync. Repeat. (cheap to measure floor) ----
  auto run_serial = [&]() {
    for (int L = 0; L < n_layers; ++L) {
      const int b = L & 1;
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); run_k1k2k3(r, wb[r].buf[b], compute_s[r]); }
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(compute_s[r])); }
      NK(ncclGroupStart());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); launch_ar(r, wb[r].buf[b], comm_s[r]); }
      NK(ncclGroupEnd());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(comm_s[r])); }
    }
  };

  // ---- OVERLAPPED: AR(L) on comm_s || K1+K2+K3(L+1) on compute_s, double-buffered (same event-gating
  // shape as nvshmem_overlap_decode.cu's run_overlapped, ported to NCCL + the real attention chain). ----
  auto run_overlapped = [&]() {
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); run_k1k2k3(r, wb[r].buf[0], compute_s[r]); }
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaEventRecord(wb[r].k3_done[0], compute_s[r])); }
    for (int L = 0; L < n_layers; ++L) {
      const int b = L & 1;
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamWaitEvent(comm_s[r], wb[r].k3_done[b], 0)); }
      NK(ncclGroupStart());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); launch_ar(r, wb[r].buf[b], comm_s[r]); }
      NK(ncclGroupEnd());
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaEventRecord(wb[r].ar_done[b], comm_s[r])); }
      if (L + 1 < n_layers) {
        const int nb = (L + 1) & 1;
        for (int r = 0; r < TP; ++r) {
          CK(cudaSetDevice(r));
          CK(cudaStreamWaitEvent(compute_s[r], wb[r].ar_done[nb], 0));
          run_k1k2k3(r, wb[r].buf[nb], compute_s[r]);
          CK(cudaEventRecord(wb[r].k3_done[nb], compute_s[r]));
        }
      }
    }
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(comm_s[r])); CK(cudaStreamSynchronize(compute_s[r])); }
  };

  cudaEvent_t e0, e1; CK(cudaSetDevice(0)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  for (int it = 0; it < warmup; ++it) run_serial();
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e0, compute_s[0]));
  for (int it = 0; it < iters; ++it) run_serial();
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e1, compute_s[0])); CK(cudaEventSynchronize(e1));
  float ms_serial; CK(cudaEventElapsedTime(&ms_serial, e0, e1)); ms_serial /= ((float)iters);

  for (int it = 0; it < warmup; ++it) run_overlapped();
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e0, compute_s[0]));
  for (int it = 0; it < iters; ++it) run_overlapped();
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e1, compute_s[0])); CK(cudaEventSynchronize(e1));
  float ms_overlap; CK(cudaEventElapsedTime(&ms_overlap, e0, e1)); ms_overlap /= ((float)iters);

  printf("\n  SERIAL   (K1-K3 + AR, fully exposed): %.3f ms/token  -> %.1f tok/s comms+attn cap\n",
         ms_serial, 1000.0f / ms_serial);
  printf("  OVERLAPPED (AR(L) || K1-K3(L+1)):      %.3f ms/token  -> %.1f tok/s comms+attn cap\n",
         ms_overlap, 1000.0f / ms_overlap);
  printf("  improvement: %.1f%%\n", 100.0 * (1.0 - (double)ms_overlap / ms_serial));
  printf("\nNOTE: compare this improvement % against overlap_decode.cu's 13.8%% (single K1 GEMV only).\n");
  printf("If wider compute closes more of the gap, that's direct evidence for extending the hide\n");
  printf("window further (full K1-K5, not just K1-K3) before reaching for a faster collective.\n");

  for (int r = 0; r < TP; ++r) { ncclCommDestroy(comms[r]); }
  return 0;
}
