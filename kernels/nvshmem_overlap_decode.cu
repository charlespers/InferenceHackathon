// nvshmem_overlap_decode.cu — THE COMBINATION: device-initiated NVSHMEM collectives +
// comms/compute overlap, for the sharded Qwen3-235B-A22B B=1 decode step (8x H100, sm_90a).
//
// This is the merge `overlap_decode.cu` and `nvshmem_comms.cu` each flagged as the missing
// piece in their own closing notes:
//   - overlap_decode.cu hides a *NCCL* all-reduce behind independent compute (Scheme C), but
//     NCCL's host-mediated launch/handshake floor (~16-35us measured) is still paid per call.
//   - nvshmem_comms.cu makes the collective itself cheap (device-initiated put+barrier over
//     NVLink, ~3-5us projected) but only as a standalone microbench — never overlapped with
//     anything.
// Neither alone removes both costs. This file does: the collective is device-initiated (cheap
// per-call) AND overlapped with the next layer's independent compute (mostly hidden), so the
// EXPOSED cost on the critical path is whichever is smaller after both effects, not their sum.
//
// =================================================================================================
// ARCHITECTURE NOTE (read before extending)
// -------------------------------------------------------------------------------------------------
// overlap_decode.cu is SINGLE-PROCESS / multi-GPU: one host thread loops over 8 ranks via
// cudaSetDevice(r), driving 8 NCCL communicators in one process. NVSHMEM's device-initiated model
// is MULTI-PROCESS: one PE = one process = one GPU, bootstrapped like nvshmem_comms.cu's main().
// These are NOT directly compatible — you cannot keep overlap_decode.cu's run_all_ranks() shape and
// just swap in NVSHMEM calls. This file ports the OVERLAP PATTERN (double-buffered pipeline +
// cudaEvent handoffs between a compute stream and a comm stream) into the multi-process NVSHMEM
// shape: every PE independently runs its own 2-stream pipeline over the SAME symmetric buffers
// every other PE uses, so the per-PE local pipelining composes into a real cross-PE overlap (PE i's
// comm-stream AR for layer L touches the same symmetric memory every other PE's comm-stream AR for
// layer L touches, at roughly the same time, because every PE runs the identical instruction
// sequence — same convention nvshmem_comms.cu and overlap_decode.cu both rely on).
//
// DOUBLE BUFFERING (why there are two `acc`/`recv` slots, not one)
// -------------------------------------------------------------------------------------------------
// The GEMV for layer L+1 must NOT write into the same symmetric buffer the AR for layer L is still
// reading/reducing — that's a write-after-read hazard across streams with no implicit ordering.
// Standard fix: ping-pong between two buffer sets, indexed by `layer & 1`. GEMV for layer L+1 writes
// acc[(L+1)&1] (the buffer NOT in use by layer L's in-flight AR), and a cudaEvent on the comm stream
// gates the NEXT cycle's GEMV into that same slot until the AR two layers prior has actually finished
// reading it. With 2 buffers this gates one layer behind; more buffers would deepen the pipeline at
// the cost of more symmetric memory (TODO(on-box): try 3-4 stage if 2 leaves the AR still exposed).
//
// WHAT THIS FILE MEASURES
// -------------------------------------------------------------------------------------------------
//   SERIAL:     GEMV(L) on compute stream; sync; NVSHMEM AR(L) on comm stream; sync. Repeat.
//               (cheap collective, but still fully exposed — isolates the "NVSHMEM alone" win.)
//   OVERLAPPED: GEMV(L+1) on compute stream runs CONCURRENTLY with AR(L) on comm stream (different
//               buffers), joined only by the double-buffer hazard event. (the combination.)
// and reports the EXPOSED collective per layer after overlap, plus the implied per-token comms
// (x188 = 2 collectives/layer x 94 layers) versus both overlap_decode.cu's NCCL-overlap number and
// nvshmem_comms.cu's NVSHMEM-alone number, so the three files form one progression:
//   NCCL serial -> NCCL overlapped -> NVSHMEM serial -> NVSHMEM overlapped (this file, expected best)
//
// CORRECTNESS: every PE fills its GEMV input deterministically; after the AR, every PE's reduced
// buffer is checked against a CPU-computed cross-PE sum (tol 1e-2, matching the other two files'
// convention). A mismatch aborts before any latency number is trusted.
//
// LATENCY-PROXY DISCLAIMER (same convention as decode_step_tp8.cu / overlap_decode.cu): the GEMV
// reads a real per-rank fp8 weight shard at the true TP=8 byte volume (one reused dummy layer, not
// all 94 — a latency proxy), and the collective is the real NVSHMEM all-reduce over the real
// [HIDDEN] message. Only the weight VALUES are fake; the kernel chain, grid shapes, byte volumes,
// and collective calls are real, so the measured us/layer and overlap fraction are representative.
//
// IP: public NVSHMEM/CUDA only; recursive-doubling all-reduce + double-buffered software pipelining
// are standard PGAS/HPC idioms. No proprietary engine internals. Self-contained; edits nothing else.
//
// ================================ BUILD (on the 8xH100 box) =====================================
//   Same NVSHMEM resolution as nvshmem_comms.cu, but note: nvidia-nvshmem-cu13 is a DATA-ONLY
//   namespace package (no __init__.py) -> nvidia.nvshmem.__file__ is None. Use __path__[0], not
//   os.path.dirname(__file__) (the bug the header comment in nvshmem_comms.cu would lead you into):
//     NVSHMEM_HOME=$(python3 -c "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])")
//     NVS_INC="$NVSHMEM_HOME/include"   NVS_LIB="$NVSHMEM_HOME/lib"
//   Device-side NVSHMEM calls require relocatable device code (-rdc=true):
//     /usr/local/cuda/bin/nvcc -arch=sm_90a -O3 --use_fast_math -rdc=true \
//        -I kernels/ -I "$NVS_INC" \
//        kernels/nvshmem_overlap_decode.cu \
//        -L "$NVS_LIB" -lnvshmem_host -lnvshmem_device -lnvidia-ml -lcuda \
//        -o /tmp/nvs_overlap
//
// ================================ RUN (8 PEs, 1 node) ===========================================
//   No nvshrun shipped in this wheel build (checked: no bin/ dir) -> use the MPI bootstrap:
//     export LD_LIBRARY_PATH="$NVS_LIB:$LD_LIBRARY_PATH"
//     export NVSHMEM_BOOTSTRAP=MPI
//     mpirun --allow-run-as-root -np 8 /tmp/nvs_overlap [n_layers=94] [iters=200]
//   Each PE binds to GPU (mype % n_gpus_visible) — one PE per H100 on an 8-GPU node.
// ================================================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#include "common.cuh"
using namespace q3;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                         \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));             \
  exit(1); } } while (0)

constexpr int NPES_EXPECT = 8;
constexpr int AR_N        = HIDDEN;          // 4096 floats = 16 KB all-reduce payload, matches
                                              // nvshmem_comms.cu / overlap_decode.cu's [HIDDEN] msg.
constexpr int N_BUF       = 2;                // double buffer (ping-pong by layer parity)

// =================================================================================================
// GEMV: a per-PE-shard fp8 dot-product producing this PE's PARTIAL contribution to the residual,
// written DIRECTLY into the symmetric AR buffer (no separate local-then-copy step — NVSHMEM puts
// only work on symmetric memory, so the producer writes there from the start).
// Reuses the warp-split-K coalesced-fp8 idiom from overlap_decode.cu (ov_warp_dot_fp8) /
// k5_experts.cu, reproduced locally so this file stays one self-contained translation unit.
// =================================================================================================
static __device__ __forceinline__ float warp_dot_fp8(const fp8* __restrict__ w,
                                                       const float* __restrict__ ys,
                                                       int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;
  for (int v = lane; v < nv; v += 32) {
    uint4 p = wv[v];
    const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
    const float* yy = ys + (v << 4);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
      unsigned wq = wu[q];
      #pragma unroll
      for (int b = 0; b < 4; ++b) {
        fp8 fv; memcpy(&fv, reinterpret_cast<const uint8_t*>(&wq) + b, 1);
        float yv = yy[q * 4 + b];
        (((q * 4 + b) & 1) ? a1 : a0) += float(fv) * yv;
      }
    }
  }
  float part = a0 + a1;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
  return part;   // valid on lane 0
}

// One simplified GEMV "layer": QKV_OUT_RANK/8 rows -> reduce to a [HIDDEN]-shaped partial via a
// fixed fan-in (stand-in for the real K1+K3 chain; the point is realistic byte volume + a nontrivial
// reduction, not architectural fidelity — same proxy convention as overlap_decode.cu's launch_qkv).
__global__ void gemv_partial_kernel(const fp8* __restrict__ w,      // [HIDDEN, HIDDEN/8] shard
                                     const float* __restrict__ act,  // [HIDDEN/8] staged activation
                                     float* __restrict__ out_sym,    // symmetric AR buffer, [HIDDEN]
                                     int n_in) {
  const int row  = blockIdx.x;          // one row per block, HIDDEN blocks
  const int lane = threadIdx.x & 31;
  if (threadIdx.x >= 32) return;        // one warp/block: simple, matches the proxy's intent
  float v = warp_dot_fp8(w + (size_t)row * n_in, act, n_in, lane);
  if (lane == 0) out_sym[row] = v;       // overwrite (this PE's fresh partial for this layer)
}

// =================================================================================================
// Recursive-doubling all-reduce(SUM), reused verbatim from nvshmem_comms.cu (the validated,
// device-initiated, 3-round-for-P=8 small-message collective). Operates in place on `acc`.
// =================================================================================================
__global__ void ar_recdouble_block(float* __restrict__ acc, float* __restrict__ recv, int n) {
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  const int tid  = threadIdx.x, nthr = blockDim.x;
  for (int mask = 1; mask < npes; mask <<= 1) {
    const int peer = mype ^ mask;
    nvshmemx_float_put_block(recv, acc, n, peer);
    nvshmemx_barrier_all_block();
    for (int i = tid; i < n; i += nthr) acc[i] += recv[i];
    __syncthreads();
    nvshmemx_barrier_all_block();
  }
}

// ================================================================================================
// Host helpers: deterministic fill (so the GEMV's effective "weights" and "activation" are fixed,
// reproducible per (pe, layer)) + CPU cross-PE-sum reference for the correctness gate.
// ================================================================================================
static inline float w_val(int pe, int layer, int row, int col) {
  // Bounded synthetic fp8-range value; layer/pe/row/col-dependent so each layer's partial differs.
  return 0.01f * (float)((row * 7 + col * 3 + layer) % 23 - 11);
}
static inline float act_val(int pe, int layer, int col) {
  return 0.02f * (float)((col * 5 + layer + pe) % 17 - 8);
}
// Reference: this PE's row-sum for `layer`, as a function of (pe, layer) — must match what
// gemv_partial_kernel actually computes from w_val/act_val for the SAME (pe, layer, row).
static double ref_partial(int pe, int layer, int row, int n_in) {
  double s = 0.0;
  for (int c = 0; c < n_in; ++c)
    s += (double)w_val(pe, layer, row, c) * (double)act_val(pe, layer, c);
  return s;
}

int main(int argc, char** argv) {
  const int n_layers = (argc > 1) ? atoi(argv[1]) : 94;     // default: real Qwen3-235B layer count
  const int iters     = (argc > 2) ? atoi(argv[2]) : 200;
  const int warmup     = 30;
  const int N_IN       = HIDDEN / 8;     // per-rank shard width (TP=8 column-shard), matches
                                         // decode_step_tp8.cu's "1/8 of the contraction dim" shape.

  nvshmem_init();
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  int n_dev = 0; CK(cudaGetDeviceCount(&n_dev));
  const int dev = (n_dev > 0) ? (mype % n_dev) : 0;
  CK(cudaSetDevice(dev));

  if (mype == 0) {
    printf("nvshmem_overlap_decode: npes=%d visible_gpus=%d n_layers=%d iters=%d "
           "AR_N=%d floats (%.1f KB) N_IN=%d\n",
           npes, n_dev, n_layers, iters, AR_N, AR_N * 4 / 1024.0, N_IN);
    if (npes != NPES_EXPECT) printf("  NOTE: expected %d PEs, running with %d.\n", NPES_EXPECT, npes);
  }

  // ---- streams: one compute, one comm, per PE (this PE's own GPU only). -------------------------
  cudaStream_t compute_s, comm_s;
  CK(cudaStreamCreate(&compute_s));
  CK(cudaStreamCreate(&comm_s));

  // ---- symmetric heap: double-buffered AR accumulator + scratch; one (small) weight+activation
  //      buffer reused across layers (latency-proxy convention — see file header). ----------------
  float* acc[N_BUF];
  float* recv[N_BUF];
  for (int b = 0; b < N_BUF; ++b) {
    acc[b]  = (float*)nvshmem_malloc(sizeof(float) * AR_N);
    recv[b] = (float*)nvshmem_malloc(sizeof(float) * AR_N);
    if (!acc[b] || !recv[b]) { printf("PE %d: nvshmem_malloc failed (buf %d)\n", mype, b); nvshmem_global_exit(2); }
  }
  fp8*   d_w   = nullptr;     // [HIDDEN, N_IN] fp8 shard, reused every layer (proxy)
  float* d_act = nullptr;     // [N_IN] staged activation, reused every layer (proxy)
  CK(cudaMalloc(&d_w, sizeof(fp8) * HIDDEN * N_IN));
  CK(cudaMalloc(&d_act, sizeof(float) * N_IN));

  // Fill weight/activation with this PE's layer-0 values (layer-dependence is folded into the CPU
  // reference instead of re-filling device memory every layer — keeps the device buffers reused,
  // matching the file's own latency-proxy convention; correctness is checked at layer 0 only, which
  // is sufficient since every layer runs the identical kernel on the identical buffer shapes).
  {
    std::vector<fp8> hw(HIDDEN * N_IN);
    std::vector<float> ha(N_IN);
    for (int r = 0; r < HIDDEN; ++r)
      for (int c = 0; c < N_IN; ++c) hw[r * N_IN + c] = fp8(w_val(mype, 0, r, c));
    for (int c = 0; c < N_IN; ++c) ha[c] = act_val(mype, 0, c);
    CK(cudaMemcpy(d_w, hw.data(), sizeof(fp8) * HIDDEN * N_IN, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_act, ha.data(), sizeof(float) * N_IN, cudaMemcpyHostToDevice));
  }

  const dim3 gemv_grid(HIDDEN), gemv_blk(32);
  const dim3 ar_grid1(1), ar_blk(1024);

  // CRITICAL (per nvshmem_comms.cu): kernels calling synchronizing NVSHMEM device APIs
  // (barrier_all_block) MUST go through nvshmemx_collective_launch, not <<<>>>.
  auto launch_ar = [&](float* a, float* r, cudaStream_t s) {
    int n = AR_N;
    void* args[] = { (void*)&a, (void*)&r, (void*)&n };
    int rc = nvshmemx_collective_launch((const void*)ar_recdouble_block, ar_grid1, ar_blk, args, 0, s);
    if (rc != 0) { printf("PE %d: collective_launch(ar) rc=%d\n", mype, rc); nvshmem_global_exit(3); }
  };
  auto launch_gemv = [&](float* out_sym, cudaStream_t s) {
    gemv_partial_kernel<<<gemv_grid, gemv_blk, 0, s>>>(d_w, d_act, out_sym, N_IN);
  };

  // ================================ CORRECTNESS (layer 0, buffer 0) ============================
  launch_gemv(acc[0], compute_s);
  CK(cudaStreamSynchronize(compute_s));
  launch_ar(acc[0], recv[0], comm_s);
  CK(cudaStreamSynchronize(comm_s));
  nvshmem_barrier_all();
  {
    std::vector<float> got(HIDDEN);
    CK(cudaMemcpy(got.data(), acc[0], sizeof(float) * HIDDEN, cudaMemcpyDeviceToHost));
    double maxerr = 0.0; int bad = -1;
    for (int row = 0; row < HIDDEN; ++row) {
      double ref = 0.0;
      for (int p = 0; p < npes; ++p) ref += ref_partial(p, 0, row, N_IN);
      double e = fabs((double)got[row] - ref);
      if (e > maxerr) { maxerr = e; if (e > 1e-2) bad = row; }
    }
    if (bad >= 0) {
      printf("PE %d: GEMV+AR MISMATCH at row=%d got=%g maxerr=%g\n", mype, bad, got[bad], maxerr);
      nvshmem_global_exit(2);
    }
    if (mype == 0) printf("  [check] GEMV -> symmetric buffer -> NVSHMEM all-reduce OK (maxerr=%.2e)\n", maxerr);
  }

  // ================================ TIMING ======================================================
  // Two events PER BUFFER, not one — this is the part easy to get subtly wrong:
  //   gemv_done[b]: GEMV(L) (compute_s) finished WRITING acc[b]  -> gates AR(L) (comm_s) reading it.
  //   ar_done[b]  : AR(L)   (comm_s)    finished READING/REDUCING acc[b] -> gates GEMV(L+N_BUF)
  //                 (compute_s) from overwriting it. (this is the buffer-reuse hazard.)
  // Without gemv_done, AR(L) could start reducing acc[b] on comm_s before GEMV(L) on compute_s has
  // actually finished writing it — a real cross-stream race (no implicit ordering between streams).
  cudaEvent_t e0, e1;
  cudaEvent_t gemv_done[N_BUF];
  cudaEvent_t ar_done[N_BUF];
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int b = 0; b < N_BUF; ++b) {
    CK(cudaEventCreateWithFlags(&gemv_done[b], cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&ar_done[b], cudaEventDisableTiming));
    CK(cudaEventRecord(ar_done[b], comm_s));   // pre-signaled: both buffers start "free"
  }

  // ---- (a) SERIAL: cheap (NVSHMEM) collective, still fully exposed. ----------------------------
  auto run_serial = [&]() {
    for (int L = 0; L < n_layers; ++L) {
      const int b = L & 1;
      launch_gemv(acc[b], compute_s);
      CK(cudaStreamSynchronize(compute_s));
      launch_ar(acc[b], recv[b], comm_s);
      CK(cudaStreamSynchronize(comm_s));
    }
  };
  for (int it = 0; it < warmup; ++it) run_serial();
  nvshmem_barrier_all();
  CK(cudaEventRecord(e0, compute_s));
  for (int it = 0; it < iters; ++it) run_serial();
  CK(cudaEventRecord(e1, compute_s)); CK(cudaEventSynchronize(e1));
  float ms_serial; CK(cudaEventElapsedTime(&ms_serial, e0, e1));
  ms_serial /= ((float)iters * n_layers);   // per-layer-pair (one GEMV + one AR)
  nvshmem_barrier_all();

  // ---- (b) OVERLAPPED: AR(L) on comm stream || GEMV(L+1) on compute stream, double-buffered. ----
  // Pipeline fill: GEMV(0) into buffer 0 (known free), no AR to overlap with yet. Then for each
  // layer L: AR(L) (comm_s) waits only for GEMV(L)'s write (gemv_done[b]), and CONCURRENTLY GEMV
  // (L+1) (compute_s) waits only for buffer nb's previous AR to have finished (ar_done[nb]) before
  // overwriting it. Two independent waits on two independent streams -> AR(L) and GEMV(L+1) actually
  // run side by side; only the buffer-reuse and write-then-read hazards are enforced, nothing else.
  auto run_overlapped = [&]() {
    launch_gemv(acc[0], compute_s);
    CK(cudaEventRecord(gemv_done[0], compute_s));
    for (int L = 0; L < n_layers; ++L) {
      const int b = L & 1;
      CK(cudaStreamWaitEvent(comm_s, gemv_done[b], 0));     // AR(L) waits for GEMV(L)'s write
      launch_ar(acc[b], recv[b], comm_s);
      CK(cudaEventRecord(ar_done[b], comm_s));              // marks buffer b free once AR(L) finishes
      if (L + 1 < n_layers) {
        const int nb = (L + 1) & 1;
        CK(cudaStreamWaitEvent(compute_s, ar_done[nb], 0)); // GEMV(L+1) waits only if AR(L-1) on nb
        launch_gemv(acc[nb], compute_s);                    // is still in flight (else fires immediately)
        CK(cudaEventRecord(gemv_done[nb], compute_s));
      }
    }
    CK(cudaStreamSynchronize(comm_s));
    CK(cudaStreamSynchronize(compute_s));
  };
  for (int it = 0; it < warmup; ++it) run_overlapped();
  nvshmem_barrier_all();
  CK(cudaEventRecord(e0, compute_s));
  for (int it = 0; it < iters; ++it) run_overlapped();
  CK(cudaEventRecord(e1, compute_s)); CK(cudaEventSynchronize(e1));
  float ms_overlap; CK(cudaEventElapsedTime(&ms_overlap, e0, e1));
  ms_overlap /= ((float)iters * n_layers);
  nvshmem_barrier_all();

  // ================================ REPORT (PE 0) ===============================================
  if (mype == 0) {
    const double NCCL_AR_US        = 16.0;   // this team's OWN measured NCCL AR@8 (nccl-tests),
                                              // NOT the 35us figure in nvshmem_comms.cu's header —
                                              // see config-sweep.md / step 0 of the test plan.
    const int    COLL_PER_TOK      = 2 * N_LAYERS;   // 188, real Qwen3-235B-A22B layer count
    const double us_serial  = ms_serial  * 1e3;
    const double us_overlap = ms_overlap * 1e3;
    printf("\n  %-38s %12s\n", "scheme (per GEMV+AR pair)", "us/pair");
    printf("  %-38s %12.2f  (NVSHMEM AR, fully exposed)\n", "SERIAL", us_serial);
    printf("  %-38s %12.2f  (AR(L) || GEMV(L+1), double-buffered)\n", "OVERLAPPED", us_overlap);
    printf("\n  exposed AR after overlap (serial - overlap, floor 0): %.2f us\n",
           std::max(0.0, us_serial - us_overlap));
    printf("\n  -- implied per-token comms (%d collectives, real n_layers=%d) --\n",
           COLL_PER_TOK, n_layers);
    printf("  NCCL serial (measured, no overlap):      %6.2f ms/token -> ~%4.0f tok/s cap\n",
           NCCL_AR_US * COLL_PER_TOK / 1e3, 1000.0 / (NCCL_AR_US * COLL_PER_TOK / 1e3));
    printf("  NVSHMEM serial (this file, no overlap):  %6.2f ms/token -> ~%4.0f tok/s cap\n",
           us_serial * COLL_PER_TOK / 1e3, 1000.0 / (us_serial * COLL_PER_TOK / 1e3));
    printf("  NVSHMEM + overlap (this file, combined): %6.2f ms/token -> ~%4.0f tok/s cap\n",
           us_overlap * COLL_PER_TOK / 1e3, 1000.0 / (us_overlap * COLL_PER_TOK / 1e3));
    printf("\nNOTES:\n");
    printf("  * Per-iter sync brackets SERIAL's timing exactly at the GEMV+AR boundary; OVERLAPPED\n");
    printf("    syncs once per full n_layers pass, so its number reflects STEADY-STATE pipeline\n");
    printf("    throughput, not single-pair latency -- the two are comparable cap-on-tok/s figures,\n");
    printf("    not comparable single-event latencies. This mirrors overlap_decode.cu's own caveat.\n");
    printf("  * N_BUF=2 gates the pipeline 1 layer deep. If OVERLAPPED still shows material exposed\n");
    printf("    AR, try N_BUF=3-4 (TODO(on-box)) before concluding the overlap is maxed out.\n");
    printf("  * This still measures GEMV-vs-AR overlap, not the real K1+K2+K3 prologue's overlap\n");
    printf("    window (a single warp-per-row GEMV is a LOWER bound on available independent work,\n");
    printf("    same convention overlap_decode.cu uses for K1 alone).\n");
    fflush(stdout);
  }

  // ---- teardown ----
  nvshmem_barrier_all();
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  for (int b = 0; b < N_BUF; ++b) {
    CK(cudaEventDestroy(gemv_done[b])); CK(cudaEventDestroy(ar_done[b]));
    nvshmem_free(acc[b]); nvshmem_free(recv[b]);
  }
  CK(cudaFree(d_w)); CK(cudaFree(d_act));
  CK(cudaStreamDestroy(compute_s)); CK(cudaStreamDestroy(comm_s));
  nvshmem_finalize();
  return 0;
}
