// nvshmem_inkernel_bench.cu — PERSISTENT in-kernel NVSHMEM all-reduce vs host-relaunched baseline,
// for the Qwen3-235B-A22B TP=8 decode step on 8x H100 (single node, NVLink P2P, sm_90a).
//
// THE STRUCTURAL LEVER (measured this session):
//   The sharded decode issues 188 collectives/token (2 all-reduces/layer x 94 layers), each on a
//   tiny 16 KB ([HIDDEN] floats) payload.  At 16 KB the NVLink transfer itself is ~0.1 us, so the
//   measured ~17 us/collective (NVSHMEM put+barrier) is almost ENTIRELY host-launch + barrier
//   overhead, NOT bandwidth.  188 x 17 us = 3.2 ms/tok -> ~310 tok/s cap.  Bandwidth tuning
//   (LL/LL128, env knobs) is dead because the floor is launch/sync, not bytes.
//
//   => The only remaining structural lever is to ELIMINATE the per-collective host launch.  A
//   PERSISTENT kernel is launched ONCE (one collective_launch), then loops ITERS all-reduce rounds
//   ENTIRELY on-device — each round is a few device-side NVLink puts + a fence + an in-kernel
//   barrier, with NO return to the host between rounds.  This removes the host launch latency from
//   every collective after the first and leaves only the on-device put+barrier cost, which on
//   NVLink should be low single-digit us.  If in-kernel us/round << 17 us, then 188/tok drops from
//   3.2 ms to well under 1 ms and the 700-1000 tok/s push (with spec) is unblocked.
//
// WHAT THIS FILE MEASURES (apples-to-apples, same recursive-doubling all-reduce in both):
//   (A) HOST-RELAUNCHED baseline : the all-reduce kernel relaunched ITERS times from the host
//       (one collective_launch per round) — this is exactly the structure of nvshmem_comms.cu and
//       reproduces the ~17 us floor.  Timed with cudaEvents around the ITERS launches.
//   (B) PERSISTENT in-kernel      : ONE collective_launch of a kernel that loops ITERS rounds
//       internally.  Timed with cudaEvents around the SINGLE launch; divide elapsed by ITERS ->
//       per-round in-kernel latency.  This is the headline number.
//
//   Both do the identical math: an 8-PE recursive-doubling all-reduce(SUM) of a [HIDDEN]-float
//   (16 KB) symmetric buffer (log2(8)=3 put+barrier rounds).  Reported: us/round for each, the
//   speedup, and the implied 188-collective comms ms/token for each.
//
// CORRECTNESS:
//   The persistent kernel re-seeds its accumulator from a deterministic (pe, idx) function at the
//   START of every round (on-device, no host involvement), so each round all-reduces real,
//   known data.  After the timed persistent run, PE 0 copies its accumulator back and checks it
//   against the CPU reference sum over 8 PEs (tol 1e-3).  A mismatch aborts before any latency is
//   trusted.  (Re-seeding per round is the cost of a local store, negligible vs the 3 barriers, and
//   it keeps the result a true single-round all-reduce rather than a meaningless running sum.)
//
// WHY PUT + FENCE + BARRIER (not the library block collective):
//   nvshmemx_float_sum_reduce_block fails collective_launch occupancy on this build (confirmed),
//   so we use the proven device-side idiom from nvshmem_comms.cu: nvshmemx_float_put_block to the
//   peer's scratch, nvshmem_fence to order the put before the barrier, then nvshmemx_barrier_all_block.
//   The barrier is a *device* API and may be called repeatedly inside one persistent kernel — that
//   reusability is precisely what lets the kernel stay resident across ITERS rounds.
//
// IP: public NVSHMEM/CUDA only; recursive-doubling all-reduce + persistent-kernel + put/fence/barrier
//   are standard PGAS idioms.  Reuses the nvshmem_comms.cu in-repo idioms.  No proprietary names.
//   This file is self-contained and edits nothing else.
//
// ================================ BUILD (on the 8xH100 box) =====================================
//   nvcc -arch=sm_90a -O3 -rdc=true \
//        -I kernels/ -I /root/nv12/nvidia/nvshmem/include \
//        kernels/nvshmem_inkernel_bench.cu \
//        -L /root/nv12/nvidia/nvshmem/lib -lnvshmem_host -lnvshmem_device -lnvidia-ml \
//        -o /tmp/nvs_ink
//
// ================================ RUN (8 PEs, 1 node) ===========================================
//   LD_LIBRARY_PATH=/root/nv12/nvidia/nvshmem/lib:$LD_LIBRARY_PATH \
//   NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_DISABLE_IB_NATIVE=1 NVSHMEM_BOOTSTRAP=MPI \
//   mpirun -np 8 --allow-run-as-root /tmp/nvs_ink [iters]
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
// Geometry.  TP=8 decode all-reduces a [HIDDEN] fp32 vector (16 KB) twice per layer.
// ------------------------------------------------------------------------------------------------
constexpr int NPES_EXPECT = 8;      // 8x H100 on one node
constexpr int AR_N        = HIDDEN; // 4096 floats = 16 KB all-reduce payload

// Deterministic contribution of PE `pe` at index `idx`.  Bounded so the 8-term sum stays fp32-exact.
// Mirrored on the host for the CPU reference.  __host__ __device__ so both sides agree bit-for-bit.
__host__ __device__ __forceinline__ float ar_contrib(int pe, int idx) {
  return 0.001f * (float)(idx % 257) + 0.5f * (float)pe + 1.0f;
}

// ================================================================================================
// Recursive-doubling all-reduce(SUM) body — ONE round over the [n]-float symmetric buffer.
//
//   In sub-round r (r=0..log2(P)-1) PE i exchanges its current partial with partner (i XOR 2^r)
//   and both sum.  After log2(P) sub-rounds every PE holds the full sum.  P=8 -> 3 put+barrier
//   sub-rounds.  This is the exact body shared by the host-relaunched and persistent variants, so
//   the only difference measured between them is the launch structure.
//
//   acc[n]  : symmetric, in/out — starts = this PE's contribution, ends = global sum.
//   recv[n] : symmetric scratch — partner writes its partial here each sub-round.
//
//   Device-only helper (called from both kernels). __forceinline__ to keep it a single body.
// ================================================================================================
__device__ __forceinline__ void ar_recdouble_round(float* __restrict__ acc,
                                                    float* __restrict__ recv,
                                                    int n, int mype, int npes,
                                                    int tid, int nthr) {
  for (int mask = 1; mask < npes; mask <<= 1) {
    const int peer = mype ^ mask;

    // One-sided, GPU-initiated block put of the whole current partial into the peer's recv buffer.
    nvshmemx_float_put_block(recv, acc, n, peer);
    // Order the put's stores before the barrier signals completion to the peer.
    nvshmem_fence();
    // World barrier (block variant: this single block/PE performs the barrier).  Every PE has now
    // delivered this sub-round's partial into its peer's recv.
    nvshmemx_barrier_all_block();

    // Sum partner's partial into our accumulator.
    for (int i = tid; i < n; i += nthr) acc[i] += recv[i];

    // Make the summed acc visible before the next sub-round's put reads it, and ensure no PE races
    // ahead to overwrite a peer's recv before that peer has consumed it.
    __syncthreads();
    nvshmemx_barrier_all_block();
  }
}

// ================================================================================================
// (A) HOST-RELAUNCHED baseline kernel: ONE all-reduce round per launch.  Re-seeds acc from the
//     deterministic contribution first so every launch reduces real data (matches the persistent
//     kernel's per-round seeding, keeping the two strictly comparable).
// ================================================================================================
__global__ void ar_round_once(float* __restrict__ acc, float* __restrict__ recv, int n) {
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  const int tid  = threadIdx.x;
  const int nthr = blockDim.x;

  for (int i = tid; i < n; i += nthr) acc[i] = ar_contrib(mype, i);
  __syncthreads();

  ar_recdouble_round(acc, recv, n, mype, npes, tid, nthr);
}

// ================================================================================================
// (B) PERSISTENT in-kernel benchmark: launched ONCE, loops `iters` all-reduce rounds internally.
//     No return to host between rounds -> no per-collective host launch latency.  The only per-round
//     cost is the on-device put + fence + barrier (x3 sub-rounds), i.e. the true structural floor.
//
//     A leading barrier_all_block makes all PEs enter the loop together (cross-PE start fence).
//     Each round re-seeds acc on-device (cheap local store) so the reduction is a real single-round
//     all-reduce whose result we can verify, not a running sum.
// ================================================================================================
__global__ void ar_persistent(float* __restrict__ acc, float* __restrict__ recv, int n, int iters) {
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  const int tid  = threadIdx.x;
  const int nthr = blockDim.x;

  // All PEs aligned before the first timed round.
  nvshmemx_barrier_all_block();

  for (int it = 0; it < iters; ++it) {
    // Re-seed this round's contribution on-device (no host involvement).
    for (int i = tid; i < n; i += nthr) acc[i] = ar_contrib(mype, i);
    __syncthreads();

    ar_recdouble_round(acc, recv, n, mype, npes, tid, nthr);
  }
}

// ================================================================================================
// main — runs on every PE.
// ================================================================================================
int main(int argc, char** argv) {
  const int iters  = (argc > 1) ? atoi(argv[1]) : 500;   // rounds (both variants)
  const int warmup = 50;

  // ---- NVSHMEM bootstrap (multi-process; one PE per process, one GPU per PE). --------------------
  nvshmem_init();
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();

  int n_dev = 0; CK(cudaGetDeviceCount(&n_dev));
  const int dev = (n_dev > 0) ? (mype % n_dev) : 0;
  CK(cudaSetDevice(dev));

  if (mype == 0) {
    printf("NVSHMEM in-kernel comms microbench: npes=%d  visible GPUs=%d  AR payload=%d floats (%.1f KB)"
           "  iters=%d\n",
           npes, n_dev, AR_N, AR_N * 4 / 1024.0, iters);
    if (npes != NPES_EXPECT)
      printf("  NOTE: expected %d PEs (8x H100); running with %d.\n", NPES_EXPECT, npes);
  }

  cudaStream_t s; CK(cudaStreamCreate(&s));

  // ---- Symmetric-heap allocations (same VA on every PE; required for one-sided puts). ------------
  float* acc  = (float*)nvshmem_malloc(sizeof(float) * AR_N);   // recdouble in/out
  float* recv = (float*)nvshmem_malloc(sizeof(float) * AR_N);   // recdouble scratch
  if (!acc || !recv) { printf("PE %d: nvshmem_malloc failed\n", mype); nvshmem_global_exit(2); }

  // Seed acc on host once (the kernels re-seed on-device, but a clean initial value is tidy).
  std::vector<float> h_ar(AR_N);
  for (int i = 0; i < AR_N; ++i) h_ar[i] = ar_contrib(mype, i);
  CK(cudaMemcpy(acc, h_ar.data(), sizeof(float) * AR_N, cudaMemcpyHostToDevice));

  const int block = 1024;            // one CTA; 1024 threads cover 16 KB in 4 grid-stride steps.
  int n_ar = AR_N;                   // addressable copy for the collective_launch arg array.
  int it_arg = iters;                // addressable copy of the persistent round count.
  const dim3 grid1(1), blk(block);

  // collective_launch wrappers (kernels call synchronizing device APIs -> MUST use collective_launch).
  auto launch_round_once = [&](float* a, float* r){
    void* args[] = { (void*)&a, (void*)&r, (void*)&n_ar };
    int rc = nvshmemx_collective_launch((const void*)ar_round_once, grid1, blk, args, 0, s);
    if (rc != 0) { printf("PE %d: collective_launch(round_once) rc=%d\n", mype, rc); nvshmem_global_exit(3); }
  };
  auto launch_persistent = [&](float* a, float* r, int* nit){
    void* args[] = { (void*)&a, (void*)&r, (void*)&n_ar, (void*)nit };
    int rc = nvshmemx_collective_launch((const void*)ar_persistent, grid1, blk, args, 0, s);
    if (rc != 0) { printf("PE %d: collective_launch(persistent) rc=%d\n", mype, rc); nvshmem_global_exit(3); }
  };

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  // CPU reference: single-round all-reduce sum over all PEs.
  auto check_acc = [&](const char* tag){
    std::vector<float> got(AR_N);
    CK(cudaMemcpy(got.data(), acc, sizeof(float) * AR_N, cudaMemcpyDeviceToHost));
    double maxerr = 0.0; int bad = -1;
    for (int i = 0; i < AR_N; ++i) {
      double ref = 0.0; for (int p = 0; p < npes; ++p) ref += ar_contrib(p, i);
      double e = fabs((double)got[i] - ref);
      if (e > maxerr) { maxerr = e; if (e > 1e-3) bad = i; }
    }
    if (bad >= 0) {
      printf("PE %d: %s all-reduce MISMATCH at i=%d got=%g maxerr=%g\n", mype, tag, bad, got[bad], maxerr);
      nvshmem_global_exit(2);
    }
    if (mype == 0) printf("  [check] %s all-reduce OK (maxerr=%.2e)\n", tag, maxerr);
  };

  // ================================ CORRECTNESS ================================================
  // (A) host-relaunched single round.
  launch_round_once(acc, recv);
  CK(cudaStreamSynchronize(s)); nvshmem_barrier_all();
  check_acc("host-relaunched");

  // (B) persistent kernel for a few rounds, then verify the final round's result.  Each round
  //     re-seeds acc on-device and produces a true single-round all-reduce, so the last round's
  //     output equals the same CPU reference sum regardless of round count.
  { int small_it = 4;
    launch_persistent(acc, recv, &small_it);
    CK(cudaStreamSynchronize(s)); nvshmem_barrier_all();
    check_acc("persistent");
  }

  // ================================ LATENCY ===================================================
  float us_host = 0.f, us_ink = 0.f;

  // ---- (A) HOST-RELAUNCHED: one collective_launch per round, ITERS times. ----
  for (int it = 0; it < warmup; ++it) launch_round_once(acc, recv);
  CK(cudaStreamSynchronize(s)); nvshmem_barrier_all();
  CK(cudaEventRecord(e0, s));
  for (int it = 0; it < iters; ++it) launch_round_once(acc, recv);
  CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
  { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us_host = ms * 1e3f / iters; }
  nvshmem_barrier_all();

  // ---- (B) PERSISTENT: ONE collective_launch looping ITERS rounds internally. ----
  // Warmup persistent launch (own small count) so the first-touch / page-in cost is not timed.
  { int w = warmup; int w_it = w; launch_persistent(acc, recv, &w_it);
    CK(cudaStreamSynchronize(s)); nvshmem_barrier_all(); }
  it_arg = iters;
  CK(cudaEventRecord(e0, s));
  launch_persistent(acc, recv, &it_arg);
  CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
  { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us_ink = ms * 1e3f / iters; }
  nvshmem_barrier_all();

  // ================================ REPORT (PE 0) =============================================
  if (mype == 0) {
    const double NCCL_AR_US   = 35.0;  // measured NCCL all-reduce floor
    const double NVSH_HOST_US = 17.0;  // measured NVSHMEM host-launched put+barrier floor
    const int    COLL_PER_TOK = 188;   // 2 all-reduces/layer x 94 layers
    printf("\n  %-38s %12s %14s\n", "all-reduce variant (16 KB, 8 PE)", "us/round", "188/tok (ms)");
    printf("  %-38s %12.2f %14.3f\n", "host-relaunched (per-round launch)",
           us_host, us_host * COLL_PER_TOK / 1e3);
    printf("  %-38s %12.2f %14.3f\n", "PERSISTENT in-kernel (launch once)",
           us_ink,  us_ink  * COLL_PER_TOK / 1e3);
    printf("\n  in-kernel speedup vs host-relaunched : %.2fx\n", us_host / us_ink);
    printf("  in-kernel vs measured NVSHMEM host floor (%.0f us): %.2fx\n", NVSH_HOST_US, NVSH_HOST_US / us_ink);
    printf("  in-kernel vs measured NCCL floor (%.0f us)        : %.2fx\n", NCCL_AR_US,   NCCL_AR_US   / us_ink);

    const double host_ms = us_host * COLL_PER_TOK / 1e3;
    const double ink_ms  = us_ink  * COLL_PER_TOK / 1e3;
    printf("\n  comms ms/token: host-relaunched %.3f ms (~%.0f tok/s comms-cap)  ->  "
           "in-kernel %.3f ms (~%.0f tok/s comms-cap)\n",
           host_ms, 1000.0 / host_ms, ink_ms, 1000.0 / ink_ms);
    printf("  verdict: in-kernel comms is %s the ~1 ms/tok budget that unlocks the 700-1000 tok/s push.\n",
           ink_ms < 1.0 ? "UNDER" : "still over");
    fflush(stdout);
  }

  // ---- teardown ----
  nvshmem_barrier_all();
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  nvshmem_free(acc); nvshmem_free(recv);
  CK(cudaStreamDestroy(s));
  nvshmem_finalize();
  return 0;
}
