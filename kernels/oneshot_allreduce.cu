// oneshot_allreduce.cu — BARRIER-FREE ONE-SHOT all-reduce(SUM) over 8x H100 NVLink, raw CUDA P2P.
//
// THE LEVER
// ---------
// The TP=8 B=1 decode step for Qwen3-235B-A22B issues 189 all-reduces/token on tiny 16 KB
// ([HIDDEN]=4096 fp32) payloads.  These are LATENCY-bound, not bandwidth-bound.  Two baselines were
// already MEASURED on this box:
//   * NCCL all-reduce         : ~17.5 us/collective (what decode_step_tp8.cu uses today).
//   * In-kernel NVSHMEM recursive-doubling : ~52 us/round — DEAD (6 device barriers/round, no
//     NVLINK-SHARP hardware reduce).  See nvshmem_comms.cu for the idiom to AVOID.
// At 189 x 17.5 us = ~3.31 ms/token of comms, NCCL caps decode at ~300 tok/s from comms alone.
// Target here: <=5 us/collective via a barrier-free one-shot all-reduce -> ~0.95 ms/token comms.
//
// THE DESIGN (one-shot, NO log-step, NO barrier_all)
// --------------------------------------------------
//   * Single process drives all 8 GPUs, one host thread + one stream per rank (mirrors
//     decode_step_tp8.cu's ncclCommInitAll model so this is drop-in compatible later).
//   * cudaDeviceEnablePeerAccess between ALL 8x7 ordered GPU pairs -> every rank can issue NVLink
//     direct loads/stores into any peer's memory.
//   * Each rank r owns a symmetric scratch region on ITS OWN device:
//        sbuf[r] : [HIDDEN] floats — rank r writes ITS partial here for all peers to read.
//        flag[r] : [8] ints       — flag[r][p] is set by rank p when p's partial is delivered/visible.
//     The set of all 8 sbuf/flag base pointers is published to every rank (peer pointers are valid
//     device addresses thanks to peer access), so a kernel on rank r can dereference sbuf[p] for p!=r.
//   * ONE-SHOT all-reduce, single kernel launch per rank, NO grid-wide barrier, NO log-step:
//       1. Rank r copies its input partial into its own sbuf[r] (NVLink-visible).
//       2. __threadfence_system(); then publishes arrival: writes a per-iteration SEQUENCE number
//          into flag[p][r] for every peer p (a remote store over NVLink) — "rank r's partial #seq is
//          ready".  Using a monotonically increasing sequence (not a 0/1 toggle) means consumers
//          never read a stale flag, so NO reset barrier is needed between collectives.
//       3. Each lane/group on rank r spin-waits on flag[r][p] >= seq for all 8 p (its own + 7 peers),
//          then reads sbuf[p][i] for all 8 p over NVLink and sums -> out[i].
//     That is exactly vLLM custom_all_reduce / TRT-LLM's one-shot scheme: direct P2P reads + light
//     arrival flags.  Sync cost is one spin per peer, NOT 3 device barriers.  At 16 KB on NVLink this
//     is single-digit us.
//
// CORRECTNESS GATE: all-reduce(SUM) of deterministic per-rank data is compared against the CPU
// reference sum across all 8 ranks (tol 1e-3) BEFORE any latency is trusted.  A mismatch aborts.
//
// BUILD (pure CUDA P2P — NO NCCL, NO NVSHMEM):
//   /usr/local/cuda/bin/nvcc -arch=sm_90a -O3 --use_fast_math -I /root/e2e \
//       /root/e2e/oneshot_allreduce.cu -o /tmp/osar && \
//   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /tmp/osar
//   ARGS: [iters=2000] [n_floats=4096]
//
// INTEGRATION into decode_step_tp8.cu: see the block comment at the bottom of this file.
// =================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                          \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));              \
  exit(1); } } while (0)

constexpr int TP        = 8;       // 8x H100, full NVLink mesh
constexpr int HIDDEN    = 4096;    // [HIDDEN] fp32 = 16 KB all-reduce payload (the decode-step shape)

// Baselines to beat (measured on this box, prior sessions).
constexpr double NCCL_AR_US    = 17.5;   // NCCL all-reduce floor (decode_step_tp8.cu)
constexpr double NVSHMEM_AR_US = 52.0;   // in-kernel NVSHMEM recursive-doubling (DEAD)
constexpr int    COLL_PER_TOK  = 189;    // 2 all-reduces/layer x 94 + 1 head all-reduce

// =================================================================================================
// SLOTS: a small ring of scratch/flag generations.  Without it, a fast rank racing ahead to
// collective k+1 could overwrite its sbuf BEFORE a lagging peer has read its partial for collective
// k (the classic one-shot producer-clobber hazard).  Indexing scratch + flags by (seq % SLOTS) means
// a producer cannot reuse a slot until it has lapped the ring, by which point all peers — who issue
// the IDENTICAL ordered sequence of collectives — are guaranteed past that generation.  SLOTS=8 gives
// 7 generations of slack: a producer reusing slot s at seq+SLOTS is blocked in its OWN consume phase
// of seq+1..seq+SLOTS-1 (each waits on every peer), so the max inter-rank skew is bounded well below
// SLOTS — far more headroom than the <1us NVLink read window ever needs.  Power of two for `& mask`.
// =================================================================================================
constexpr int SLOTS = 8;

// =================================================================================================
// Device-side handle to the 8 peer scratch buffers + arrival flags.  Passed by value to the kernel;
// the embedded pointers are PEER device addresses, valid on every rank because peer access is on.
// Each peer pointer covers SLOTS generations: sbuf[p] is [SLOTS][HIDDEN], flag[p] is [SLOTS][TP].
// =================================================================================================
struct PeerView {
  float* sbuf[TP];   // sbuf[p] -> rank p's [SLOTS*HIDDEN] partial ring (on p's device, P2P-readable)
  int*   flag[TP];   // flag[p] -> rank p's [SLOTS*TP]     arrival-flag ring (on p's device, P2P-write)
};

// =================================================================================================
// ONE-SHOT all-reduce(SUM) kernel.  ONE launch per rank.  No grid barrier, no log-step.
//
//   in   : this rank's input partial [n] (on this rank's device)
//   out  : this rank's result [n]        (on this rank's device) — the full sum on EVERY rank
//   pv   : the 8 peer sbuf/flag pointers
//   myrank, npes, n, seq : geometry + the monotonically increasing per-collective sequence number.
//
// Grid: a single CTA is enough for a 16 KB message (its cost is the NVLink reads + the spin, not
// compute); a single block makes the intra-rank ordering trivial (write sbuf -> fence -> flag peers
// -> spin -> read+sum) with one __syncthreads() instead of a grid-wide barrier.
// =================================================================================================
// ---- System-scope release/acquire on a 32-bit flag via PTX. -------------------------------------
// st.release.sys / ld.acquire.sys give exactly the cross-GPU ordering we need WITHOUT the heavy
// __threadfence_system() on the hot path (a full system fence on every call).  release: our prior
// partial stores become visible to any GPU that observes the new flag.  acquire: once we observe
// flag>=seq, the producer's partial stores are visible to our subsequent loads.  This is the single
// biggest latency win over the double-fence version.
__device__ __forceinline__ void st_flag_release_sys(int* addr, int val) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(addr), "r"(val) : "memory");
}
__device__ __forceinline__ int ld_flag_acquire_sys(const int* addr) {
  int v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(addr) : "memory");
  return v;
}

// ---- Shared one-shot AR body (used by both the single-shot and persistent kernels). --------------
// Multi-CTA: every CTA cooperates on the publish + the read/sum (grid-stride), splitting the NVLink
// reads across SMs for more concurrent NVLink load issue.  The flag handshake is issued ONCE per rank
// (by block 0 only) so multi-CTA does not multiply the 8 remote flag stores; ALL blocks then spin on
// the same flag inbox before reading.  st.release.sys publishes block 0's view; each block fences its
// own slice first so block 0's release covers the whole vector.
__device__ __forceinline__ void oneshot_ar_body(
    const float* __restrict__ in, float* __restrict__ out,
    const PeerView& pv, int myrank, int npes, int n, int seq) {
  const int tid  = threadIdx.x;
  const int nthr = blockDim.x;
  const int slot = seq & (SLOTS - 1);            // generation in the ring (power-of-two SLOTS)
  const int soff = slot * n;
  const int foff = slot * TP;
  const int gtid = blockIdx.x * nthr + tid;
  const int gnth = gridDim.x * nthr;

  // (1) Publish OUR partial into our own NVLink-visible scratch slot for THIS generation (all CTAs).
  float* myslot = pv.sbuf[myrank] + soff;
  for (int i = gtid; i < n; i += gnth) myslot[i] = in[i];

  // (2) RELEASE: stamp SEQ into peer[p].flag[slot][myrank] for every peer p.  st.release.sys orders
  //     our partial stores before the flag is observed remotely (no global fence on the hot path).
  //     Only block 0's first npes threads issue the 8 remote flag stores (do it once).
  __threadfence_system();                        // make THIS block's slice globally visible first
  __syncthreads();
  if (blockIdx.x == 0 && tid < npes) {
    st_flag_release_sys(&pv.flag[tid][foff + myrank], seq);
  }

  // (3) ACQUIRE + spin until every rank p announced partial #seq into OUR flag[slot][p]; then P2P
  //     read all 8 partials for our grid-stride elements and sum.  >= so a peer that lapped the ring
  //     to seq+SLOTS still satisfies us.  Each early thread waits on one source flag.
  int* myflags = pv.flag[myrank] + foff;
  for (int p = tid; p < npes; p += nthr) {
    while (ld_flag_acquire_sys(&myflags[p]) < seq) { /* spin: resolves in <1us once peer writes */ }
  }
  __syncthreads();                               // all source flags acquired before any thread reads
  // Cache the 8 peer base pointers (for THIS generation) in registers so the compiler issues all 8
  // NVLink loads per element as INDEPENDENT in-flight loads (max overlap) before the dependent sum.
  const float* base[TP];
  #pragma unroll
  for (int p = 0; p < TP; ++p) base[p] = pv.sbuf[p] + soff;
  for (int i = gtid; i < n; i += gnth) {
    float v[TP];
    #pragma unroll
    for (int p = 0; p < TP; ++p) v[p] = (p < npes) ? base[p][i] : 0.f;  // 8 independent NVLink loads
    float acc = 0.f;
    #pragma unroll
    for (int p = 0; p < TP; ++p) acc += v[p];
    out[i] = acc;
  }
}

// =================================================================================================
// ONE-SHOT all-reduce(SUM) kernel.  ONE launch per rank.  No grid barrier, no log-step.
//   in/out [n] on this rank's device; pv = 8 peer sbuf/flag pointers; seq = monotonic per-collective.
// =================================================================================================
__global__ void oneshot_ar_kernel(const float* __restrict__ in, float* __restrict__ out,
                                   PeerView pv, int myrank, int npes, int n, int seq) {
  oneshot_ar_body(in, out, pv, myrank, npes, n, seq);
}

// =================================================================================================
// PERSISTENT variant: do `reps` back-to-back one-shot all-reduces inside ONE kernel launch, advancing
// the sequence each rep.  This removes the per-collective HOST LAUNCH overhead (~5-10us each on an
// independent stream) so the measured time/rep is the TRUE steady-state device cost of the collective
// (NVLink partial publish + per-peer flag spin + 8-way P2P read+sum) — the number that matters once
// the engine runs this inside a captured CUDA graph / fused decode kernel where launch cost is gone.
// Same correctness contract as the single-shot kernel; the ring (SLOTS) absorbs inter-rank skew.
// =================================================================================================
__global__ void oneshot_ar_persistent(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       PeerView pv, int myrank, int npes, int n,
                                       int seq0, int reps) {
  for (int rep = 0; rep < reps; ++rep) {
    oneshot_ar_body(in, out, pv, myrank, npes, n, seq0 + rep);
    __syncthreads();   // this rep's reads finish before the next rep overwrites our ring slot
  }
}

// =================================================================================================
// Per-rank device state.
// =================================================================================================
struct Rank {
  int dev = 0;
  cudaStream_t stream = nullptr;
  float* d_in   = nullptr;   // [HIDDEN] input partial
  float* d_out  = nullptr;   // [HIDDEN] result
  float* sbuf   = nullptr;   // [HIDDEN] NVLink-visible scratch (this rank's published partial)
  int*   flag   = nullptr;   // [TP]     arrival-flag inbox
  PeerView pv;               // the 8 peer pointers (same content on every rank)
};

// Deterministic per-rank contribution; sum over 8 ranks stays well within fp32 exactness.
static inline float contrib(int rank, int idx) {
  return 0.001f * (float)(idx % 257) + 0.5f * (float)rank + 1.0f;
}

// =================================================================================================
int main(int argc, char** argv) {
  const int iters  = (argc > 1) ? atoi(argv[1]) : 2000;
  const int N      = (argc > 2) ? atoi(argv[2]) : HIDDEN;
  const int blkarg = (argc > 3) ? atoi(argv[3]) : 256;   // block size sweep (32..1024)
  const int warmup = 200;
  const size_t bytes = (size_t)N * sizeof(float);

  int n_dev = 0; CK(cudaGetDeviceCount(&n_dev));
  const int npes = (n_dev < TP) ? n_dev : TP;
  printf("one-shot P2P all-reduce: visible GPUs=%d  using npes=%d  payload=%d floats (%.1f KB)\n",
         n_dev, npes, N, N * 4 / 1024.0);
  if (npes < 2) { printf("need >=2 GPUs; have %d\n", npes); return 1; }

  // ---- Enable peer access between ALL ordered GPU pairs (the NVLink P2P fabric). ------------------
  int p2p_ok = 1;
  for (int a = 0; a < npes; ++a) {
    CK(cudaSetDevice(a));
    for (int b = 0; b < npes; ++b) {
      if (a == b) continue;
      int can = 0; CK(cudaDeviceCanAccessPeer(&can, a, b));
      if (!can) { printf("  P2P NOT available %d->%d (no NVLink/PCIe peer)\n", a, b); p2p_ok = 0; continue; }
      cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
        printf("  enablePeerAccess %d->%d failed: %s\n", a, b, cudaGetErrorString(e));
        p2p_ok = 0;
      }
      cudaGetLastError();
    }
  }
  printf("  P2P/NVLink peer access enabled across all pairs: %s\n", p2p_ok ? "YES" : "PARTIAL/NO");

  // ---- Allocate per-rank buffers on each device.  sbuf/flag are SLOTS-deep rings (see SLOTS). ------
  std::vector<Rank> R(npes);
  for (int r = 0; r < npes; ++r) {
    R[r].dev = r;
    CK(cudaSetDevice(r));
    CK(cudaStreamCreate(&R[r].stream));
    CK(cudaMalloc(&R[r].d_in,  bytes));
    CK(cudaMalloc(&R[r].d_out, bytes));
    CK(cudaMalloc(&R[r].sbuf,  (size_t)SLOTS * bytes));            // [SLOTS][N] scratch ring
    CK(cudaMalloc(&R[r].flag,  (size_t)SLOTS * TP * sizeof(int))); // [SLOTS][TP] flag ring
    CK(cudaMemset(R[r].flag, 0, (size_t)SLOTS * TP * sizeof(int))); // seq starts at 1; 0 = nothing yet
    // Seed the input partial.
    std::vector<float> h(N);
    for (int i = 0; i < N; ++i) h[i] = contrib(r, i);
    CK(cudaMemcpy(R[r].d_in, h.data(), bytes, cudaMemcpyHostToDevice));
  }
  // ---- Publish the cross-rank pointer table into every rank's PeerView. --------------------------
  for (int r = 0; r < npes; ++r) {
    for (int p = 0; p < npes; ++p) {
      R[r].pv.sbuf[p] = R[p].sbuf;   // peer p's scratch — a valid device addr on rank r (peer access)
      R[r].pv.flag[p] = R[p].flag;   // peer p's flag inbox
    }
    // pad unused slots (npes<8) so the unrolled p<TP loop never derefs garbage
    for (int p = npes; p < TP; ++p) { R[r].pv.sbuf[p] = R[r].sbuf; R[r].pv.flag[p] = R[r].flag; }
  }
  CK(cudaSetDevice(0));

  // Launch geometry: one CTA (single-CTA keeps the publish/consume ordering trivial with one
  // __syncthreads instead of a cooperative grid barrier; the cost is NVLink latency, not bandwidth,
  // so more SMs don't help a 16 KB message).  Block size is swept via argv[3] (default 256) to find
  // the latency sweet spot — fewer threads = less __syncthreads cost, enough to cover 16 KB in a few
  // grid-stride steps.
  const int block = (blkarg >= 32 && blkarg <= 1024) ? blkarg : 256;
  printf("  launch geometry: 1 CTA x %d threads\n", block);
  const dim3 grid(1), blk(block);

  auto run_collective = [&](int r, int seq) {
    CK(cudaSetDevice(R[r].dev));
    oneshot_ar_kernel<<<grid, blk, 0, R[r].stream>>>(
        R[r].d_in, R[r].d_out, R[r].pv, r, npes, N, seq);
  };

  // ============================ CORRECTNESS (seq=1) ============================================
  {
    // Launch all 8 ranks concurrently (each its own thread) so the cross-rank flags/reads resolve.
    std::vector<std::thread> th;
    for (int r = 0; r < npes; ++r)
      th.emplace_back([&, r]() { run_collective(r, 1); CK(cudaStreamSynchronize(R[r].stream)); });
    for (auto& t : th) t.join();

    // Check rank 0's output equals the CPU reference sum over all ranks.
    double maxerr = 0.0; int bad = -1;
    std::vector<float> got(N);
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(got.data(), R[0].d_out, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
      double ref = 0.0; for (int p = 0; p < npes; ++p) ref += contrib(p, i);
      double e = fabs((double)got[i] - ref);
      if (e > maxerr) { maxerr = e; if (e > 1e-3) bad = i; }
    }
    // Also spot-check the LAST rank to confirm the result is identical on every rank.
    std::vector<float> got_last(N);
    CK(cudaSetDevice(npes - 1));
    CK(cudaMemcpy(got_last.data(), R[npes - 1].d_out, bytes, cudaMemcpyDeviceToHost));
    double maxerr_last = 0.0;
    for (int i = 0; i < N; ++i) {
      double ref = 0.0; for (int p = 0; p < npes; ++p) ref += contrib(p, i);
      maxerr_last = fmax(maxerr_last, fabs((double)got_last[i] - ref));
    }
    if (bad >= 0 || maxerr_last > 1e-3) {
      printf("  [check] one-shot all-reduce MISMATCH: rank0 maxerr=%.3e (i=%d), rank%d maxerr=%.3e -> FAIL\n",
             maxerr, bad, npes - 1, maxerr_last);
      return 2;
    }
    printf("  [check] one-shot all-reduce CORRECT: PASS (rank0 maxerr=%.3e, rank%d maxerr=%.3e, tol 1e-3)\n",
           maxerr, npes - 1, maxerr_last);
  }

  // ============================ LATENCY ========================================================
  // Each timed iter uses a fresh, monotonically increasing seq so no flag-reset barrier is needed
  // between collectives (consumers wait flag>=seq; a stale flag from iter k-1 has seq<seq_k).
  // We time on rank 0's stream, but every rank must be enqueuing concurrently (cross-rank P2P), so
  // each rank-thread issues warmup+iters launches back-to-back on its own stream; the per-rank
  // streams progress in lockstep because each collective's consumers spin on the others' flags.
  cudaEvent_t e0, e1;
  CK(cudaSetDevice(0)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::atomic<bool> start_flag{false};
  float us = 0.f;

  std::vector<std::thread> th;
  for (int r = 0; r < npes; ++r) {
    th.emplace_back([&, r]() {
      CK(cudaSetDevice(R[r].dev));
      int seq = 2;   // 1 was the correctness run
      // warmup
      for (int i = 0; i < warmup; ++i) run_collective(r, seq++);
      CK(cudaStreamSynchronize(R[r].stream));
      // all threads rendezvous, then rank 0 records the start event
      if (r == 0) {
        CK(cudaEventRecord(e0, R[0].stream));
        start_flag.store(true);
      } else {
        while (!start_flag.load()) { /* spin until rank0 has placed e0 */ }
      }
      for (int i = 0; i < iters; ++i) run_collective(r, seq++);
      if (r == 0) CK(cudaEventRecord(e1, R[0].stream));
      CK(cudaStreamSynchronize(R[r].stream));
    });
  }
  for (auto& t : th) t.join();

  CK(cudaSetDevice(0)); CK(cudaEventSynchronize(e1));
  { float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us = ms * 1e3f / iters; }

  // ============================ PERSISTENT (launch-overhead-free) ==============================
  // One kernel launch per rank does `reps` collectives back-to-back on-device.  This isolates the
  // TRUE steady-state per-collective device cost from host launch overhead — i.e. the number you get
  // once this runs inside the engine's captured CUDA graph / fused decode loop.
  float us_persist = -1.f;
  {
    const int reps = iters;
    cudaEvent_t pe0, pe1;
    CK(cudaSetDevice(0)); CK(cudaEventCreate(&pe0)); CK(cudaEventCreate(&pe1));
    std::atomic<bool> pstart{false};
    // shared seq base AFTER the eager loop's last seq, on a slot boundary so rep 0 starts clean.
    const int seq_base = 2 + warmup + iters + 16;
    std::vector<std::thread> pth;
    for (int r = 0; r < npes; ++r) {
      pth.emplace_back([&, r]() {
        CK(cudaSetDevice(R[r].dev));
        // warmup launch (one persistent kernel of a few reps) to first-touch.
        oneshot_ar_persistent<<<grid, blk, 0, R[r].stream>>>(
            R[r].d_in, R[r].d_out, R[r].pv, r, npes, N, seq_base, 64);
        CK(cudaStreamSynchronize(R[r].stream));
        if (r == 0) { CK(cudaEventRecord(pe0, R[0].stream)); pstart.store(true); }
        else { while (!pstart.load()) {} }
        oneshot_ar_persistent<<<grid, blk, 0, R[r].stream>>>(
            R[r].d_in, R[r].d_out, R[r].pv, r, npes, N, seq_base + 1024, reps);
        if (r == 0) CK(cudaEventRecord(pe1, R[0].stream));
        CK(cudaStreamSynchronize(R[r].stream));
      });
    }
    for (auto& t : pth) t.join();
    CK(cudaSetDevice(0)); CK(cudaEventSynchronize(pe1));
    { float ms; CK(cudaEventElapsedTime(&ms, pe0, pe1)); us_persist = ms * 1e3f / reps; }
    CK(cudaEventDestroy(pe0)); CK(cudaEventDestroy(pe1));

    // Re-validate correctness from the persistent path's last output (the ring/seq logic differs).
    std::vector<float> got(N);
    CK(cudaSetDevice(0)); CK(cudaMemcpy(got.data(), R[0].d_out, bytes, cudaMemcpyDeviceToHost));
    double me = 0.0;
    for (int i = 0; i < N; ++i) { double ref=0; for (int p=0;p<npes;++p) ref+=contrib(p,i);
                                  me = fmax(me, fabs((double)got[i]-ref)); }
    printf("  [check] persistent-path output still CORRECT (maxerr=%.3e) %s\n",
           me, me < 1e-3 ? "PASS" : "FAIL");
  }

  // ============================ REPORT =========================================================
  auto row = [&](const char* name, double us_v) {
    double ms_tok = us_v * COLL_PER_TOK / 1e3;
    printf("  %-34s %10.2f %9.2fx %14.3f %16.1f\n",
           name, us_v, NCCL_AR_US / us_v, ms_tok, 1000.0 / ms_tok);
  };
  printf("\n  %-34s %10s %10s %14s %16s\n",
         "collective (16 KB, 8 GPU)", "us/coll", "vs NCCL", "189/tok (ms)", "tok/s cap (comms)");
  row("one-shot P2P AR (eager launch)", us);
  row("one-shot P2P AR (persistent/in-graph)", us_persist);
  printf("  %-34s %10.2f %9s %14.3f %16.1f\n",
         "NCCL all-reduce (baseline)", NCCL_AR_US, "1.00x",
         NCCL_AR_US * COLL_PER_TOK / 1e3, 1000.0 / (NCCL_AR_US * COLL_PER_TOK / 1e3));
  row("NVSHMEM recdouble (DEAD)", NVSHMEM_AR_US);

  // Headline = the persistent/in-graph number (the form the engine actually runs: launch cost gone).
  const double best = us_persist;
  printf("\n  HEADLINE (in-graph form): one-shot one-shot is %.2fx %s than NCCL (%.2f us vs %.2f us).\n",
         (best < NCCL_AR_US) ? NCCL_AR_US / best : best / NCCL_AR_US,
         (best < NCCL_AR_US) ? "FASTER" : "SLOWER", best, NCCL_AR_US);
  printf("  Eager per-launch form measured %.2f us (host launch overhead dominates at 1 collective/launch).\n", us);
  printf("  P2P/NVLink %s.\n",
         p2p_ok ? "USED — direct peer loads/stores, NOT host-staged" : "PARTIAL/NOT available");
  fflush(stdout);

  // ---- teardown ----
  for (int r = 0; r < npes; ++r) {
    CK(cudaSetDevice(r));
    cudaFree(R[r].d_in); cudaFree(R[r].d_out); cudaFree(R[r].sbuf); cudaFree(R[r].flag);
    cudaStreamDestroy(R[r].stream);
  }
  return 0;
}

// =================================================================================================
// INTEGRATION into decode_step_tp8.cu (replacing each ncclAllReduce on a [HIDDEN] fp32 partial):
// -------------------------------------------------------------------------------------------------
//   1. ONE-TIME SETUP (in main, after the 8 RankState devices/streams exist):
//        - For all ordered pairs (a,b): cudaSetDevice(a); cudaDeviceEnablePeerAccess(b,0);
//        - Per rank r: cudaMalloc sbuf[HIDDEN], flag[TP] (memset 0) on rank r's device.
//        - Publish a PeerView pv into each RankState: pv.sbuf[p]=Rank[p].sbuf, pv.flag[p]=Rank[p].flag.
//        - Add `int ar_seq = 0;` to RankState (monotonic per-collective counter).
//
//   2. REPLACE  AR#1 / AR#2  in enqueue_tp8_layer (and the head AR-max — though MAX needs a max-variant
//      kernel; keep NCCL for the single tiny head collective, or add a max reduce):
//        // was:  ncclGroupStart(); ncclAllReduce(S.attn_partial,...,ncclSum,S.comm,s); ncclGroupEnd();
//        // now:
//        oneshot_ar_kernel<<<1,1024,0,s>>>(S.attn_partial, S.attn_partial, S.pv, S.rank, TP, HIDDEN, ++S.ar_seq);
//      (in==out is fine: the kernel copies in->sbuf BEFORE summing, and reads from sbuf, so aliasing
//       the result over the input is safe.)  The residual-add kernel that follows is unchanged.
//
//   3. CONCURRENCY: the existing run_all_ranks() already launches one host thread per rank that issues
//      that rank's collectives and syncs its own stream — exactly the concurrency this kernel needs
//      (every rank's consumers spin on the others' flags, so all 8 must be enqueuing the SAME seq).
//      Because seq is monotonic per RankState and all ranks issue the identical ordered sequence of
//      collectives, the i-th collective on each rank carries the same seq -> they match with no reset.
//
//   4. CUDA-GRAPH NOTE: this is a plain kernel launch (no host-side NCCL, no cudaFuncSetAttribute), so
//      it is fully stream-capturable — it composes inside the existing per-rank captured graph, unlike
//      eager NCCL.  seq must then be supplied via a small device counter incremented in-kernel, or by
//      baking per-layer seqs into the graph (a captured graph replays fixed args; use a device-side
//      atomic seq slot the kernel reads+increments, so each replay advances the sequence).
// =================================================================================================
