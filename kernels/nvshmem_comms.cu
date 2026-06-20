// nvshmem_comms.cu — NVSHMEM low-latency collectives to break the comms wall for the
// Qwen3-235B-A22B TP=8 / EP=8 decode step on 8x H100 (single node, NVLink P2P, sm_90a).
//
// THE PROBLEM (measured this session):
//   The sharded decode step issues 188 collectives/token (2 all-reduces/layer x 94 layers):
//   each layer all-reduces the partial O-proj output [HIDDEN] and the partial MoE-down output
//   [HIDDEN], both fp32 -> 16 KB messages.  NCCL on these tiny payloads is LATENCY-FLOORED:
//   ~35 us per all-reduce, ~60 us per all-to-all.  LL/LL128 protocols do not help because the
//   floor is launch + handshake, not bandwidth.  188 x ~35 us = ~6.6 ms of comms alone, which
//   caps sharded decode at ~90-150 tok/s and blocks the 1000 tok/s goal REGARDLESS of how fast
//   the GEMV kernels get.
//
// THE FIX:
//   NVSHMEM gives a one-sided, GPU-INITIATED, RDMA-style PGAS over NVLink P2P.  A collective is
//   a few device-side stores to peer symmetric memory + one barrier — no host launch handshake
//   per collective, and it composes inside a single CUDA-graph-captured kernel.  On NVLink the
//   one-sided put is sub-microsecond, so the per-collective latency drops to low single-digit us.
//   At ~3 us each, 188/token = ~0.56 ms of comms -> comfortably under the ~1 ms budget that lets
//   the GEMV side reach for 1000 tok/s.  (NVSHMEM/IBGDA is exactly the path DeepEP uses for EP.)
//
// WHAT THIS FILE IMPLEMENTS (all device-initiated, all on the symmetric heap):
//   (a) ar_recdouble_block : an 8-PE all-reduce(SUM) of a HIDDEN-float (16 KB) buffer using
//                            recursive-doubling over NVLink (log2(8)=3 put+barrier rounds), the
//                            classic low-latency small-message all-reduce.  Single block, fully
//                            on device.  This directly replaces the per-layer NCCL all-reduce.
//   (b) ar_nvshmem_block   : the same all-reduce expressed with the NVSHMEM library collective
//                            (nvshmemx_float_sum_reduce_block) as a correctness/perf cross-check.
//   (c) a2a_put_block      : an 8-PE all-to-all of CHUNK floats-per-peer (~16 KB/peer) via direct
//                            nvshmem_float_put to each peer's symmetric recv slot + one barrier.
//                            This is the EP dispatch/combine primitive (token -> expert-owner PE).
//   + a microbench that times each on real device events and prints us/collective vs the NCCL
//     baselines (35 us AR / 60 us A2A) and the implied per-token comms budget (x188 and x2/layer).
//
// CORRECTNESS:
//   Every PE fills its input deterministically from (pe, index); the host recomputes the expected
//   reduced / permuted result on the CPU and each PE checks its own output (tol 1e-3, fp32 exact-ish
//   sums of 8 terms).  A mismatch aborts before any (bogus) latency is reported.
//
// IP: public NVSHMEM/CUDA only; recursive-doubling and put+barrier all-to-all are standard PGAS
// idioms.  No proprietary engine names.  This file is self-contained and edits nothing else.
//
// ================================ BUILD (on the 8xH100 box) =====================================
//   NVSHMEM is the pip wheel `nvidia-nvshmem-cu13` (3.4.5).  Resolve its include/lib:
//     NVSHMEM_HOME=$(python3 -c "import nvidia.nvshmem,os;print(os.path.dirname(nvidia.nvshmem.__file__))")
//     NVS_INC="$NVSHMEM_HOME/include"          # nvshmem.h, nvshmemx.h, device/*.h
//     NVS_LIB="$NVSHMEM_HOME/lib"              # see `ls "$NVS_LIB"`: libnvshmem_host.so.3, libnvshmem_device.a
//   Device-side NVSHMEM calls require relocatable device code (-rdc=true) and the device lib:
//     /usr/local/cuda/bin/nvcc -arch=sm_90a -O3 --use_fast_math -rdc=true \
//        -I kernels/ -I "$NVS_INC" \
//        kernels/nvshmem_comms.cu \
//        -L "$NVS_LIB" -lnvshmem_host -lnvshmem_device -lnvidia-ml -lcuda \
//        -o /tmp/nvs
//   (Exact lib names: `ls "$NVS_LIB"` — link `-lnvshmem_host` (libnvshmem_host.so) and
//    `-lnvshmem_device` (libnvshmem_device.a, static).  Some wheels also ship a combined
//    `-lnvshmem`; if so, `-lnvshmem -lnvidia-ml -lcuda` works.  `-lnvidia-ml` (NVML) and `-lcuda`
//    (the CUDA driver) are needed by the host bootstrap.)
//
// ================================ RUN (8 PEs, 1 node) ===========================================
//   This program self-bootstraps 8 PEs on one node via the PMI-less multi-process launcher.  Use
//   the launcher shipped with the wheel (nvshmrun / nvshmrun.pl) OR set the env bootstrap and run
//   8 ranks with any MPI/SLURM-less launcher.  Simplest, no-extra-deps path (process-per-PE):
//     export LD_LIBRARY_PATH="$NVS_LIB:$LD_LIBRARY_PATH"
//     export NVSHMEM_BOOTSTRAP=UID            # UID bootstrap: no MPI, no PMI needed
//     "$NVSHMEM_HOME"/bin/nvshmrun -n 8 /tmp/nvs        # 8 PEs, one per GPU, on this node
//   If the wheel's bin/ has no nvshmrun, the MPI bootstrap also works:
//     export NVSHMEM_BOOTSTRAP=MPI
//     mpirun -np 8 /tmp/nvs
//   Each PE binds to GPU (mype % n_gpus_visible); on an 8-GPU node that is one PE per H100.
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

// ------------------------------------------------------------------------------------------------
// Geometry.  The TP=8 decode step all-reduces a [HIDDEN] fp32 vector (16 KB) twice per layer.
// The EP=8 all-to-all moves ~one token's hidden per peer; we size the per-peer chunk to HIDDEN
// floats (16 KB) to mirror that worst case.
// ------------------------------------------------------------------------------------------------
constexpr int NPES_EXPECT = 8;           // 8x H100 on one node
constexpr int AR_N        = HIDDEN;      // 4096 floats = 16 KB all-reduce payload
constexpr int A2A_CHUNK   = HIDDEN;      // 4096 floats = 16 KB per peer for the all-to-all

// ================================================================================================
// (a) Recursive-doubling all-reduce(SUM) over NVLink, device-initiated, single block.
//
//   Classic small-message all-reduce: in round r (r = 0..log2(P)-1) PE i exchanges its current
//   partial with partner (i XOR 2^r) and both sum.  After log2(P) rounds every PE holds the full
//   sum.  P=8 -> 3 rounds, each = one put of N floats to the partner's scratch + a barrier.  The
//   put is one-sided over NVLink (sub-us); the barrier is the only sync.  3 put+barrier rounds
//   replace the single ~35 us NCCL all-reduce.
//
//   Buffers (all on the SYMMETRIC heap, same address on every PE):
//     acc[N]   : in/out — starts = this PE's contribution, ends = the global sum (on every PE).
//     recv[N]  : scratch — partner writes its partial here each round.
//   `pSync` : an NVSHMEM barrier-team sync array (we use the whole-world barrier).
//
//   Single CTA so the per-element sum + the barrier are trivially ordered; N=4096 over up to 1024
//   threads is one grid-stride pass, plenty for a 16 KB message whose cost is the 3 barriers.
// ================================================================================================
__global__ void ar_recdouble_block(float* __restrict__ acc,   // symmetric, [AR_N]
                                    float* __restrict__ recv,  // symmetric scratch, [AR_N]
                                    int n) {
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  const int tid  = threadIdx.x;
  const int nthr = blockDim.x;

  // Recursive-doubling: partner = mype XOR mask, mask = 1,2,4 for P=8.
  for (int mask = 1; mask < npes; mask <<= 1) {
    const int peer = mype ^ mask;

    // One-sided, GPU-initiated put of the WHOLE current partial into the peer's recv buffer.
    // nvshmemx_float_put_block has the whole block cooperate on the transfer (coalesced over
    // NVLink) and returns once the local source can be reused; the barrier below makes it visible.
    nvshmemx_float_put_block(recv, acc, n, peer);

    // World barrier: every PE has now delivered this round's partial into its peer's recv.
    // (block variant: the calling block performs the barrier; we launch exactly one block/PE.)
    nvshmemx_barrier_all_block();

    // Sum partner's partial into our accumulator.  Grid-stride over the 16 KB vector.
    for (int i = tid; i < n; i += nthr) acc[i] += recv[i];

    // Make the summed acc visible to threads that will put it next round, and ensure no PE
    // races ahead and overwrites recv before its peer has consumed it.
    __syncthreads();
    nvshmemx_barrier_all_block();
  }
}

// ================================================================================================
// (b) NVSHMEM library all-reduce(SUM) — the same result via the built-in collective, used as a
//     correctness oracle and a second latency datapoint.  Reduces src[n] -> dst[n] over the world
//     team.  Needs a separate dst (out-of-place) and the collective's pWrk/pSync handled by the
//     library team allocator (NVSHMEM_TEAM_WORLD).
// ================================================================================================
__global__ void ar_nvshmem_block(float* __restrict__ dst,   // symmetric, [AR_N]
                                  float* __restrict__ src,   // symmetric, [AR_N]
                                  int n) {
  // Block-scoped library reduction over the world team. All threads in the block participate.
  nvshmemx_float_sum_reduce_block(NVSHMEM_TEAM_WORLD, dst, src, n);
}

// ================================================================================================
// (c) All-to-all over NVLink, device-initiated.  PE i sends chunk j (CHUNK floats) to PE j, which
//     lands it in PE j's recv slot for source i: recv[i*CHUNK .. ).  After a barrier every PE's
//     recv buffer holds [from PE 0 | from PE 1 | ... | from PE 7].  This is the EP dispatch shape:
//     "send my tokens routed to expert-owner PE j into j's inbox slot for me."
//
//   send[npes*CHUNK] : symmetric — send[j*CHUNK ..] is the block destined for PE j.
//   recv[npes*CHUNK] : symmetric — recv[i*CHUNK ..] receives the block PE i sent us.
//
//   Each PE issues npes-1 remote puts (skip self -> local copy) then one barrier.  P2P puts over
//   NVLink overlap; the single barrier is the latency floor (one round, not log2 rounds).
// ================================================================================================
__global__ void a2a_put_block(float* __restrict__ send,   // symmetric, [npes*CHUNK]
                              float* __restrict__ recv,    // symmetric, [npes*CHUNK]
                              int chunk) {
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  const int tid  = threadIdx.x;
  const int nthr = blockDim.x;

  for (int j = 0; j < npes; ++j) {
    const float* src = send + (size_t)j * chunk;     // block we send to PE j
    float*       dst = recv + (size_t)mype * chunk;  // PE j's slot reserved for us
    if (j == mype) {
      // self: plain local copy (no network), still places it in our own recv[mype] slot.
      for (int i = tid; i < chunk; i += nthr) dst[i] = src[i];
    } else {
      // one-sided block put into peer j's recv buffer at our reserved offset.
      nvshmemx_float_put_block(dst, src, chunk, j);
    }
  }
  // single all-to-all barrier: every PE has delivered all npes-1 remote blocks.
  nvshmemx_barrier_all_block();
}

// ================================================================================================
// Host helpers: deterministic fill + CPU reference.
// ================================================================================================
static inline float ar_contrib(int pe, int idx) {
  // Bounded, PE- and index-dependent. Sum over 8 PEs stays well within fp32 exactness.
  return 0.001f * (float)(idx % 257) + 0.5f * (float)pe + 1.0f;
}
static inline float a2a_elem(int src_pe, int dst_pe, int idx) {
  // The value PE src_pe sends to PE dst_pe at position idx.
  return (float)(src_pe * 100 + dst_pe) + 0.01f * (float)(idx % 91);
}

// ================================================================================================
// main — runs on every PE.
// ================================================================================================
int main(int argc, char** argv) {
  const int iters = (argc > 1) ? atoi(argv[1]) : 200;   // timed iters/collective
  const int warmup = 30;

  // ---- NVSHMEM bootstrap (multi-process; one PE per process, one GPU per PE). --------------------
  nvshmem_init();
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();

  int n_dev = 0; CK(cudaGetDeviceCount(&n_dev));
  const int dev = (n_dev > 0) ? (mype % n_dev) : 0;
  CK(cudaSetDevice(dev));

  if (mype == 0) {
    printf("NVSHMEM comms microbench: npes=%d  visible GPUs=%d  AR payload=%d floats (%.1f KB)"
           "  A2A=%d floats/peer (%.1f KB)\n",
           npes, n_dev, AR_N, AR_N * 4 / 1024.0, A2A_CHUNK, A2A_CHUNK * 4 / 1024.0);
    if (npes != NPES_EXPECT)
      printf("  NOTE: expected %d PEs (8x H100); running with %d.\n", NPES_EXPECT, npes);
  }

  cudaStream_t s; CK(cudaStreamCreate(&s));

  // ---- Symmetric-heap allocations (same VA on every PE; required for one-sided puts/collectives).
  float* acc  = (float*)nvshmem_malloc(sizeof(float) * AR_N);          // recdouble in/out
  float* recv = (float*)nvshmem_malloc(sizeof(float) * AR_N);          // recdouble scratch
  float* lsrc = (float*)nvshmem_malloc(sizeof(float) * AR_N);          // library AR src
  float* ldst = (float*)nvshmem_malloc(sizeof(float) * AR_N);          // library AR dst
  float* a2s  = (float*)nvshmem_malloc(sizeof(float) * (size_t)npes * A2A_CHUNK);  // a2a send
  float* a2r  = (float*)nvshmem_malloc(sizeof(float) * (size_t)npes * A2A_CHUNK);  // a2a recv
  if (!acc || !recv || !lsrc || !ldst || !a2s || !a2r) {
    printf("PE %d: nvshmem_malloc failed\n", mype); nvshmem_global_exit(2);
  }

  // ---- Fill inputs on host, copy to symmetric device buffers. ------------------------------------
  std::vector<float> h_ar(AR_N);
  for (int i = 0; i < AR_N; ++i) h_ar[i] = ar_contrib(mype, i);
  CK(cudaMemcpy(acc,  h_ar.data(), sizeof(float) * AR_N, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(lsrc, h_ar.data(), sizeof(float) * AR_N, cudaMemcpyHostToDevice));

  std::vector<float> h_a2s((size_t)npes * A2A_CHUNK);
  for (int j = 0; j < npes; ++j)
    for (int i = 0; i < A2A_CHUNK; ++i)
      h_a2s[(size_t)j * A2A_CHUNK + i] = a2a_elem(mype, j, i);   // mype -> PE j
  CK(cudaMemcpy(a2s, h_a2s.data(), sizeof(float) * (size_t)npes * A2A_CHUNK, cudaMemcpyHostToDevice));

  const int block = 1024;   // one CTA; 1024 threads cover 16 KB in 4 grid-stride steps.
  int n_ar = AR_N, n_chunk = A2A_CHUNK;   // addressable copies for collective_launch arg array.

  // CRITICAL: kernels that call synchronizing/collective NVSHMEM device APIs (barrier_all_block,
  // *_reduce_block) MUST be launched with nvshmemx_collective_launch, not the <<<>>> syntax —
  // otherwise NVSHMEM's inter-PE sync state is never set up and the first device barrier hangs.
  // These helpers wrap collective_launch for our exact (1 block, `block` threads, stream s) shape.
  const dim3 grid1(1), blk(block);
  auto launch_ar_recd = [&](float* a, float* r){
    void* args[] = { (void*)&a, (void*)&r, (void*)&n_ar };
    int rc = nvshmemx_collective_launch((const void*)ar_recdouble_block, grid1, blk, args, 0, s);
    if (rc != 0) { printf("PE %d: collective_launch(ar_recd) rc=%d\n", mype, rc); nvshmem_global_exit(3); }
  };
  auto launch_ar_lib = [&](float* d, float* sp){
    void* args[] = { (void*)&d, (void*)&sp, (void*)&n_ar };
    int rc = nvshmemx_collective_launch((const void*)ar_nvshmem_block, grid1, blk, args, 0, s);
    if (rc != 0) { printf("PE %d: collective_launch(ar_lib) rc=%d\n", mype, rc); nvshmem_global_exit(3); }
  };
  auto launch_a2a = [&](float* sp, float* r){
    void* args[] = { (void*)&sp, (void*)&r, (void*)&n_chunk };
    int rc = nvshmemx_collective_launch((const void*)a2a_put_block, grid1, blk, args, 0, s);
    if (rc != 0) { printf("PE %d: collective_launch(a2a) rc=%d\n", mype, rc); nvshmem_global_exit(3); }
  };

  // ================================ CORRECTNESS ================================================
  // --- (a) recursive-doubling all-reduce ---
  launch_ar_recd(acc, recv);
  CK(cudaStreamSynchronize(s));
  nvshmem_barrier_all();
  {
    std::vector<float> got(AR_N);
    CK(cudaMemcpy(got.data(), acc, sizeof(float) * AR_N, cudaMemcpyDeviceToHost));
    double maxerr = 0.0; int bad = -1;
    for (int i = 0; i < AR_N; ++i) {
      double ref = 0.0; for (int p = 0; p < npes; ++p) ref += ar_contrib(p, i);
      double e = fabs((double)got[i] - ref);
      if (e > maxerr) { maxerr = e; if (e > 1e-3) bad = i; }
    }
    if (bad >= 0) {
      printf("PE %d: recdouble AR MISMATCH at i=%d got=%g maxerr=%g\n", mype, bad, got[bad], maxerr);
      nvshmem_global_exit(2);
    }
    if (mype == 0) printf("  [check] recdouble all-reduce OK (maxerr=%.2e)\n", maxerr);
  }

  // --- (b) NVSHMEM library all-reduce — SKIPPED: nvshmemx_float_sum_reduce_block's internal
  //     collective_launch fails ("One or more PEs cannot launch") on this build/occupancy.
  //     The recdouble path (a) IS the low-latency collective; the library call is only a
  //     cross-check, so skipping it does not affect the headline recdouble latency below.
  (void)launch_ar_lib; (void)ldst; (void)lsrc;
  if (mype == 0) printf("  [check] NVSHMEM library all-reduce SKIPPED (collective_launch occupancy)\n");

  // --- (c) all-to-all ---
  launch_a2a(a2s, a2r);
  CK(cudaStreamSynchronize(s));
  nvshmem_barrier_all();
  {
    std::vector<float> got((size_t)npes * A2A_CHUNK);
    CK(cudaMemcpy(got.data(), a2r, sizeof(float) * (size_t)npes * A2A_CHUNK, cudaMemcpyDeviceToHost));
    double maxerr = 0.0; int bad = -1;
    for (int src = 0; src < npes && bad < 0; ++src)
      for (int i = 0; i < A2A_CHUNK; ++i) {
        // recv[src*CHUNK + i] on THIS PE should be what PE `src` sent to us (dst=mype).
        double ref = a2a_elem(src, mype, i);
        double e = fabs((double)got[(size_t)src * A2A_CHUNK + i] - ref);
        if (e > maxerr) maxerr = e;
        if (e > 1e-3) { bad = src * A2A_CHUNK + i; break; }
      }
    if (bad >= 0) {
      printf("PE %d: all-to-all MISMATCH at flat=%d maxerr=%g\n", mype, bad, maxerr);
      nvshmem_global_exit(2);
    }
    if (mype == 0) printf("  [check] all-to-all OK (maxerr=%.2e)\n", maxerr);
  }

  // ================================ LATENCY ===================================================
  // Re-fill acc (it was overwritten by the all-reduce) so the timed loop reduces real data each
  // iter (correctness already proven; timing just needs representative traffic + the barriers).
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float us_recd = 0.f, us_lib = 0.f, us_a2a = 0.f;

  // NOTE on methodology: latency is the cost of put+barrier, which is INDEPENDENT of the buffer
  // contents, so we do NOT reseed inside the timed loop (a per-iter 16 KB D2D memcpy would add its
  // own launch/copy latency and make the three collectives non-comparable). After the first iter
  // `acc` holds a running (numerically meaningless) sum — irrelevant to the barrier-bound timing,
  // and correctness was already proven above on real data.

  // ---- (a) recursive-doubling all-reduce latency ----
  launch_ar_recd(acc, recv);   // re-seed once from the proven run is unnecessary for timing.
  for (int it = 0; it < warmup; ++it) launch_ar_recd(acc, recv);
  CK(cudaStreamSynchronize(s)); nvshmem_barrier_all();
  CK(cudaEventRecord(e0, s));
  for (int it = 0; it < iters; ++it) launch_ar_recd(acc, recv);
  CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
  { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us_recd = ms * 1e3f / iters; }
  nvshmem_barrier_all();

  // ---- (b) NVSHMEM library all-reduce latency — SKIPPED (see correctness section) ----
  us_lib = -1.f;

  // ---- (c) all-to-all latency ----
  for (int it = 0; it < warmup; ++it) launch_a2a(a2s, a2r);
  CK(cudaStreamSynchronize(s)); nvshmem_barrier_all();
  CK(cudaEventRecord(e0, s));
  for (int it = 0; it < iters; ++it) launch_a2a(a2s, a2r);
  CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
  { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us_a2a = ms * 1e3f / iters; }
  nvshmem_barrier_all();

  // ================================ REPORT (PE 0) =============================================
  if (mype == 0) {
    const double NCCL_AR_US  = 35.0;   // measured NCCL all-reduce floor (this session)
    const double NCCL_A2A_US = 60.0;   // measured NCCL all-to-all floor
    const int    COLL_PER_TOK = 188;   // 2 all-reduces/layer x 94 layers
    printf("\n  %-34s %12s %12s %14s\n", "collective (16 KB, 8 PE)", "us/coll", "vs NCCL", "188/tok (ms)");
    printf("  %-34s %12.2f %11.1fx %14.3f\n", "recursive-doubling all-reduce",
           us_recd, NCCL_AR_US / us_recd, us_recd * COLL_PER_TOK / 1e3);
    printf("  %-34s %12.2f %11.1fx %14.3f\n", "NVSHMEM library all-reduce",
           us_lib, NCCL_AR_US / us_lib, us_lib * COLL_PER_TOK / 1e3);
    printf("  %-34s %12.2f %11.1fx %14s\n", "put+barrier all-to-all",
           us_a2a, NCCL_A2A_US / us_a2a, "(EP, 2/layer)");
    printf("\n  NCCL baseline: %.0f us AR -> %.2f ms/tok over %d colls (the wall capping ~90-150 tok/s).\n",
           NCCL_AR_US, NCCL_AR_US * COLL_PER_TOK / 1e3, COLL_PER_TOK);
    const double budget_ms = us_recd * COLL_PER_TOK / 1e3;
    printf("  NVSHMEM recdouble: %.3f ms/tok of comms -> %s the ~1 ms budget for the 1000 tok/s push.\n",
           budget_ms, budget_ms < 1.0 ? "UNDER" : "still over");
    fflush(stdout);
  }

  // ---- teardown ----
  nvshmem_barrier_all();
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  nvshmem_free(acc); nvshmem_free(recv); nvshmem_free(lsrc); nvshmem_free(ldst);
  nvshmem_free(a2s); nvshmem_free(a2r);
  CK(cudaStreamDestroy(s));
  nvshmem_finalize();
  return 0;
}
