// overlap_decode.cu — COMMS/COMPUTE OVERLAP proof-of-concept for the sharded Qwen3-235B-A22B
// B=1 decode (8x H100, sm_90a).  Standard CUDA + NCCL only.
//
// =================================================================================================
// THE PROBLEM THIS FILE ATTACKS
// -------------------------------------------------------------------------------------------------
// Sharded B=1 decode is gated by collectives, not by kernels.  Per layer the residual stream must be
// summed across the 8 ranks twice (one all-reduce after the sharded O-proj, one after the sharded
// MoE-down), so 2 collectives x 94 layers = 188 collectives/token.  Each tiny ([HIDDEN]=4096 floats
// = 16 KB) all-reduce is LATENCY-floored at ~35 us on NVLink/NVSwitch (LL/LL128 do not move that
// floor for a message this small), so 188 x ~35 us ~= 6.6 ms of pure comms/token -> a hard ceiling
// of ~150 tok/s if the collective is on the critical path SERIALLY, regardless of how fast the
// kernels are.  Breaking 1000 tok/s requires the collective to HIDE behind independent compute.
//
// THE OVERLAP IDEA (two complementary, both prototyped + measured here)
// -------------------------------------------------------------------------------------------------
//   (1) CHUNKED / PIPELINED ALL-REDUCE.  Split the [HIDDEN] residual into C chunks.  Issue the
//       all-reduce of chunk c on a dedicated COMM stream the instant chunk c's partial is produced,
//       while the COMPUTE stream keeps producing chunk c+1.  The reduce of the early chunks flies
//       under the compute of the later chunks; only the LAST chunk's reduce is exposed.  This is the
//       classic "reduce-as-you-go" software pipeline (what fused-comm GEMMs and DeepEP-style dispatch
//       do); it needs no model-structure change and works for the O-proj and MoE-down combines as-is.
//
//   (2) LAYER-PIPELINE PREFETCH.  The current layer's all-reduce is independent of the NEXT layer's
//       Wqkv weight read + QKV GEMV prologue (K1) — those touch disjoint data and live on a disjoint
//       part of the residual timeline once the token is fixed.  So we launch layer L's collective on
//       the comm stream and overlap it with layer L+1's K1 QKV GEMV on the compute stream, joining
//       with a CUDA event only where the data dependency actually bites.  At B=1 K1 is a fat fp8 GEMV
//       (Wqkv 9216x4096 = ~37 MB fp8/rank unsharded, ~4.7 MB at TP=8) whose HBM read is plenty of
//       independent work to hide a 35 us collective behind.
//
// MECHANISM: two CUDA streams per rank (compute, comm) + cudaEvent_t handoffs.  cudaEventRecord on
// the producing stream, cudaStreamWaitEvent on the consuming stream — a pure GPU-side dependency, no
// host sync in the loop.  NCCL collectives are enqueued on the comm stream; NCCL is CUDA-graph- and
// multi-stream-safe as long as every rank issues the SAME collective sequence (we honor that:
// every rank issues exactly C all-reduces per "layer", bracketed in one ncclGroup).
//
// WHAT WE MEASURE: for a representative (compute kernel + collective) pair we time, over many iters,
//   - SERIAL:    compute on stream; then collective on the same stream (today's critical path).
//   - OVERLAP-1: chunked all-reduce pipelined against the compute (idea 1).
//   - OVERLAP-2: collective on comm stream || next-layer K1 GEMV on compute stream (idea 2).
// and report the overlap fraction  ov = 1 - t_overlap / t_serial  and the IMPLIED per-token comms
// after overlap (188 collectives priced at the exposed, post-overlap cost).
//
// LATENCY-PROXY DISCLAIMER (same convention as decode_step_tp8.cu / ep_moe_sharded.cu): the compute
// kernel is a real fp8 GEMV reading real (dummy) fp8 weights at the true per-rank byte volume, and
// the collective is the real NCCL all-reduce over the real [HIDDEN] message; only the weights are a
// single reused layer (latency proxy), so the measured us and overlap fraction are representative.
// Correctness of the all-reduce path is checked against a CPU cross-rank-sum reference (<1e-2).
//
// BUILD (on the 8xH100 box; NCCL via pip `nvidia-nccl-cu12`):
//   NCCL_INC=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'include'))")
//   NCCL_LIB=$(python -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'lib'))")
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -I "$NCCL_INC" \
//        kernels/overlap_decode.cu -L "$NCCL_LIB" -lnccl -o /tmp/overlap
//   LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" /tmp/overlap
//   (If NCCL is system-installed, plain `-lnccl` suffices and nccl.h is on the default include path.)
//
// =================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <nccl.h>
#include "common.cuh"
using namespace q3;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                       \
  printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));           \
  exit(1); } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) {                      \
  printf("NCCL err %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r_));           \
  exit(1); } } while (0)

// =================================================================================================
// TP=8 geometry for the representative compute kernel.  K1 (the attention prologue QKV GEMV) is the
// natural overlap partner: under TP=8 the fused QKV matrix Wqkv [QKV_OUT, HIDDEN] is column-sharded
// so each rank owns QKV_OUT/8 output rows; the GEMV reads that fp8 shard once.
// =================================================================================================
constexpr int TP          = 8;
constexpr int QKV_OUT_RANK = QKV_OUT / TP;                       // 9216/8 = 1152 rows/rank
static_assert(QKV_OUT % TP == 0, "QKV_OUT must shard evenly across TP ranks");

// Default number of chunks for the pipelined all-reduce (overridable on the command line).
constexpr int N_CHUNKS_DEFAULT = 4;

// =================================================================================================
// Device dot primitive — the in-repo fast GEMV (k5_experts.cu warp_dot_fp8 idiom), reproduced
// locally so this file is one self-contained translation unit that never edits k5/common.cuh.
//   Warp dots a contiguous K-major fp8 weight row against a staged f32 activation, split-K across the
//   32 lanes (consecutive lanes -> consecutive uint4 = 16 fp8 -> coalesced 128-bit HBM), hardware
//   fp8x2->half2 dequant, 2 accumulators for ILP.  n must be a multiple of 16 (HIDDEN=4096 is).
//   Result valid on lane 0.
// =================================================================================================
static __device__ __forceinline__ float ov_warp_dot_fp8(const fp8* __restrict__ w,
                                                         const float* __restrict__ ys,
                                                         int n, int lane) {
  float a0 = 0.f, a1 = 0.f;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(w);
  const int nv = n >> 4;                                          // 16 fp8 per uint4
  for (int v = lane; v < nv; v += 32) {                           // lanes 0..31 -> consecutive uint4
    uint4 p = wv[v];
    const unsigned* wu = reinterpret_cast<const unsigned*>(&p);
    const float* yy = ys + (v << 4);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {                                 // 4 x 32-bit words = 4 x (2 fp8 pairs)
      unsigned wq = wu[q];
      __nv_fp8x2_e4m3 lo, hi;
      lo.__x = (unsigned short)(wq & 0xffffu);
      hi.__x = (unsigned short)(wq >> 16);
      float2 fl = __half22float2((__half2)lo);
      float2 fh = __half22float2((__half2)hi);
      const float* yq = yy + (q << 2);
      a0 += yq[0]*fl.x;  a1 += yq[1]*fl.y;
      a0 += yq[2]*fh.x;  a1 += yq[3]*fh.y;
    }
  }
  float acc = a0 + a1;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                                     // valid on lane 0
}

// =================================================================================================
// Representative COMPUTE kernel: this rank's K1 QKV GEMV shard.  out[row] = scale[row] * <x, W[row]>
// over the rank's QKV_OUT_RANK rows of the fused [QKV_OUT, HIDDEN] matrix.  One warp per output row,
// grid-stride; x staged once into shared memory per CTA.  This is the independent "next-layer
// prologue" work we overlap the collective behind (idea 2), and structurally identical to k1.
// Launch with dynamic smem = HIDDEN*sizeof(float).
// =================================================================================================
extern "C" __global__ void ov_qkv_gemv(const float* __restrict__ x,
                                        const fp8* __restrict__ W,
                                        const float* __restrict__ scale,
                                        float* __restrict__ out, int rows) {
  extern __shared__ float xs[];                                   // [HIDDEN]
  for (int k = threadIdx.x; k < HIDDEN; k += blockDim.x) xs[k] = x[k];
  __syncthreads();
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int row = gwarp; row < rows; row += nwarp) {
    const float v = ov_warp_dot_fp8(W + (size_t)row * HIDDEN, xs, HIDDEN, lane);
    if (lane == 0) out[row] = v * scale[row];
  }
}

// A trivial "produce a residual chunk" kernel for the chunked-pipeline experiment: writes a
// deterministic partial into [base, base+len) so the cross-rank all-reduce(SUM) has a checkable
// result.  In the real decode this is the tail of the O-proj / MoE-down that lands chunk c; here it
// is a stand-in whose only job is to (a) be schedulable per-chunk and (b) feed a verifiable reduce.
extern "C" __global__ void ov_make_chunk(float* __restrict__ part, int base, int len, int rank) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) {
    // value = rank+1 so SUM over 8 ranks = 36 at every element -> trivial cross-rank check.
    part[base + i] = (float)(rank + 1);
  }
}

// =================================================================================================
// Per-rank state.
// =================================================================================================
struct RankState {
  int rank = 0, dev = 0;
  ncclComm_t   comm    = nullptr;
  cudaStream_t compute = nullptr;     // GEMVs / chunk producers
  cudaStream_t comms   = nullptr;     // NCCL all-reduce(s)

  // K1 QKV GEMV shard (this rank's QKV_OUT_RANK rows).
  float *x          = nullptr;        // [HIDDEN] activation (staged input)
  fp8   *Wqkv       = nullptr;        // [QKV_OUT_RANK, HIDDEN] fp8 weight shard
  float *Wqkv_scale = nullptr;        // [QKV_OUT_RANK] per-row scale
  float *qkv_out    = nullptr;        // [QKV_OUT_RANK] GEMV output

  // residual partial reduced across ranks (the collective payload).
  float *part       = nullptr;        // [HIDDEN]

  int block = 256;
};

// =================================================================================================
// Dummy-weight allocation (one reused K1 shard per rank — latency proxy; the byte volume is real).
// =================================================================================================
static inline unsigned hashu(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static inline float frnd(unsigned seed, size_t i, float scale, bool positive) {
  unsigned h = hashu((unsigned)(i * 2654435761u) ^ (seed * 40503u));
  float v = (((h % 2001) / 1000.0f) - 1.0f) * scale;
  return positive ? (fabsf(v) + 1e-3f) : v;
}

static void alloc_rank(RankState& S) {
  CK(cudaSetDevice(S.dev));
  const size_t wn = (size_t)QKV_OUT_RANK * HIDDEN;

  std::vector<fp8>   Wh(wn);
  std::vector<float> Sh(QKV_OUT_RANK), Xh(HIDDEN);
  for (size_t i = 0; i < wn; ++i)            Wh[i] = (fp8)frnd(11u + S.rank, i, 0.25f, false);
  for (int i = 0; i < QKV_OUT_RANK; ++i)     Sh[i] = frnd(23u + S.rank, i, 0.02f, true);
  for (int k = 0; k < HIDDEN; ++k)           Xh[k] = frnd(99u, k, 1.0f, false);  // replicated input

  CK(cudaMalloc(&S.Wqkv,       wn * sizeof(fp8)));
  CK(cudaMalloc(&S.Wqkv_scale, QKV_OUT_RANK * sizeof(float)));
  CK(cudaMalloc(&S.x,          HIDDEN * sizeof(float)));
  CK(cudaMalloc(&S.qkv_out,    QKV_OUT_RANK * sizeof(float)));
  CK(cudaMalloc(&S.part,       HIDDEN * sizeof(float)));
  CK(cudaMemcpy(S.Wqkv,       Wh.data(), wn * sizeof(fp8),         cudaMemcpyHostToDevice));
  CK(cudaMemcpy(S.Wqkv_scale, Sh.data(), QKV_OUT_RANK*sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(S.x,          Xh.data(), HIDDEN * sizeof(float),   cudaMemcpyHostToDevice));
}

static void free_rank(RankState& S) {
  CK(cudaSetDevice(S.dev));
  cudaFree(S.Wqkv); cudaFree(S.Wqkv_scale); cudaFree(S.x); cudaFree(S.qkv_out); cudaFree(S.part);
}

static inline int grid_for(int rows, int warps_per_cta) {
  int need = (rows + warps_per_cta - 1) / warps_per_cta;
  return std::min(std::max(need, 132), 264);                      // fill the H100, cap ~2 CTAs/SM
}

// One QKV GEMV launch on a chosen stream.
static void launch_qkv(RankState& S, cudaStream_t s) {
  const int warps = S.block >> 5;
  const int ctas  = grid_for(QKV_OUT_RANK, warps);
  const size_t smem = (size_t)HIDDEN * sizeof(float);
  ov_qkv_gemv<<<ctas, S.block, smem, s>>>(S.x, S.Wqkv, S.Wqkv_scale, S.qkv_out, QKV_OUT_RANK);
}

// =================================================================================================
// NCCL DRIVER CONTRACT (single process, ncclCommInitAll, one host thread owns all 8 comms).
// -------------------------------------------------------------------------------------------------
// NVIDIA's group-calls doc: when ONE thread owns multiple communicators, all ranks' i-th collective
// must be inside ONE ncclGroupStart/End so ncclGroupEnd does not block waiting on peers that the same
// thread has not yet reached.  So every scheme below is split into two phases:
//   (1) a per-rank COMPUTE enqueue (kernels + events only, NO NCCL) issued for all ranks, then
//   (2) a single OUTER ncclGroupStart() ... [every rank's collective(s)] ... ncclGroupEnd().
// This is the robust, deadlock-free single-thread pattern; phase (1) sets up the stream/event
// dependencies that make the phase-(2) collectives overlap the still-running compute.
// =================================================================================================

// ---- SCHEME A — SERIAL (today's critical path) ----
// Produce the full partial on the compute stream; the all-reduce runs on the SAME stream so it is
// fully exposed (no overlap).  compute-phase + collective-phase below.
static void serial_compute(RankState& S) {
  ov_make_chunk<<<132, 256, 0, S.compute>>>(S.part, 0, HIDDEN, S.rank);
}
static void serial_collective(RankState& S) {                     // inside the outer group
  NK(ncclAllReduce(S.part, S.part, HIDDEN, ncclFloat32, ncclSum, S.comm, S.compute));
}

// ---- SCHEME B — CHUNKED / PIPELINED ALL-REDUCE (idea 1) ----
// Produce chunk c on the compute stream and record event ev[c]; the comm stream waits on ev[c] only,
// then all-reduces that chunk — so reduce(c) overlaps produce(c+1..).  Only the last chunk's reduce
// is exposed.  All C sub-reduces (for all ranks) live in the single outer group.
static void chunked_compute(RankState& S, int nchunks, cudaEvent_t* ev) {
  const int base_len = (HIDDEN + nchunks - 1) / nchunks;
  for (int c = 0; c < nchunks; ++c) {
    const int base = c * base_len;
    const int len  = std::min(base_len, HIDDEN - base);
    if (len <= 0) break;
    ov_make_chunk<<<132, 256, 0, S.compute>>>(S.part, base, len, S.rank);
    CK(cudaEventRecord(ev[c], S.compute));                         // chunk c ready
    // Issue the comm stream's wait for chunk c HERE, in the (NCCL-free) compute phase, NOT
    // interleaved between ncclGroupStart/End. The wait is stream-ordered on S.comms and the
    // chunk's all-reduce is enqueued on the SAME stream at ncclGroupEnd, so the gate (wait[c]
    // then AR[c]) is preserved while keeping the NCCL group region pure NCCL calls.
    CK(cudaStreamWaitEvent(S.comms, ev[c], 0));                    // comm stream waits for chunk c only
  }
}
static void chunked_collective(RankState& S, int nchunks, cudaEvent_t* /*ev*/) {  // inside the outer group
  const int base_len = (HIDDEN + nchunks - 1) / nchunks;
  for (int c = 0; c < nchunks; ++c) {
    const int base = c * base_len;
    const int len  = std::min(base_len, HIDDEN - base);
    if (len <= 0) break;
    // No CUDA-runtime calls inside the group: the per-chunk waits were already enqueued on
    // S.comms in chunked_compute(); these all-reduces land after them in stream order.
    NK(ncclAllReduce(S.part + base, S.part + base, len, ncclFloat32, ncclSum, S.comm, S.comms));
  }
}

// ---- SCHEME C — LAYER-PIPELINE PREFETCH (idea 2) ----
// This layer's all-reduce runs on the comm stream; the NEXT layer's K1 QKV GEMV runs concurrently on
// the compute stream.  They touch disjoint data, so the collective hides behind the GEMV's HBM read.
// The collective's input partial is produced on the comm stream so the reduce has a real dependency
// chain; the GEMV is the independent work filling the comms-latency window.
static void pipeline_compute(RankState& S) {
  // produce this layer's partial on the comm stream (so it precedes the reduce on that stream)...
  ov_make_chunk<<<132, 256, 0, S.comms>>>(S.part, 0, HIDDEN, S.rank);
  // ...and launch the NEXT layer's independent prologue on the compute stream (the overlap partner).
  launch_qkv(S, S.compute);
}
static void pipeline_collective(RankState& S) {                   // inside the outer group
  NK(ncclAllReduce(S.part, S.part, HIDDEN, ncclFloat32, ncclSum, S.comm, S.comms));
}

// =================================================================================================
// Timing helpers.  We time on rank 0's relevant stream(s) with CUDA events, but every rank enqueues
// each iter (NCCL needs all ranks present) and we sync ALL streams of ALL ranks before stopping the
// timer so the measured span covers the slowest rank's exposed work.
// =================================================================================================
static void sync_all(std::vector<RankState>& R) {
  for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.compute));
                                                CK(cudaStreamSynchronize(S.comms)); }
}

int main(int argc, char** argv) {
  const int IT       = (argc > 1) ? atoi(argv[1]) : 300;
  const int NCHUNKS  = (argc > 2) ? atoi(argv[2]) : N_CHUNKS_DEFAULT;
  const int WARM     = 30;

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < TP) {
    printf("Need >= %d CUDA devices for TP=%d; found %d.\n", TP, TP, ndev); return 1;
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("== Qwen3-235B-A22B comms/compute OVERLAP PoC (B=1 decode, %d GPUs) ==\n", TP);
  printf("device0: %s  SMs=%d  TP=%d  QKV rows/rank=%d  HIDDEN=%d  chunks=%d  iters=%d\n",
         prop.name, prop.multiProcessorCount, TP, QKV_OUT_RANK, HIDDEN, NCHUNKS, IT);

  // ---- enable peer access so NCCL uses NVLink P2P (NVSwitch makes all pairs peers) ----
  for (int i = 0; i < TP; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < TP; ++j) if (i != j) {
      int can = 0; cudaDeviceCanAccessPeer(&can, i, j);
      if (can) cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  // ---- NCCL: one communicator clique across the 8 local GPUs (single-process) ----
  std::vector<RankState>  R(TP);
  std::vector<ncclComm_t> comms(TP);
  std::vector<int>        devs(TP);
  for (int r = 0; r < TP; ++r) devs[r] = r;
  NK(ncclCommInitAll(comms.data(), TP, devs.data()));

  for (int r = 0; r < TP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    CK(cudaSetDevice(r));
    CK(cudaStreamCreate(&R[r].compute));
    // high-priority comm stream so the tiny collective is not starved behind the GEMV's CTAs.
    int lo = 0, hi = 0; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi)); (void)lo;
    CK(cudaStreamCreateWithPriority(&R[r].comms, cudaStreamNonBlocking, hi));
    alloc_rank(R[r]);
  }

  // per-rank per-chunk events for the chunked pipeline.
  std::vector<std::vector<cudaEvent_t>> chunk_ev(TP, std::vector<cudaEvent_t>(NCHUNKS));
  for (int r = 0; r < TP; ++r) {
    CK(cudaSetDevice(r));
    for (int c = 0; c < NCHUNKS; ++c) CK(cudaEventCreateWithFlags(&chunk_ev[r][c], cudaEventDisableTiming));
  }

  // =============================================================================================
  // One timed/untimed step over ALL ranks, honoring the single-thread NCCL group contract:
  //   phase 1 — enqueue every rank's COMPUTE (kernels + events, no NCCL),
  //   phase 2 — ONE outer ncclGroupStart() ... every rank's COLLECTIVE ... ncclGroupEnd().
  // `compute_one(S)` and `coll_one(S)` are the per-scheme phase callables.
  // =============================================================================================
  auto run_step = [&](auto compute_one, auto coll_one) {
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); compute_one(R[r]); }
    NK(ncclGroupStart());
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); coll_one(R[r]); }
    NK(ncclGroupEnd());
  };

  // =============================================================================================
  // Correctness: after the chunked all-reduce, every element of part[] must equal SUM_r (r+1) = 36
  // (TP=8 -> 1+2+...+8 = 36).  This validates that the per-chunk event handoff reduced every chunk
  // exactly once and the comm-stream pipeline did not drop or double-count any range.
  // =============================================================================================
  for (int r = 0; r < TP; ++r) {
    CK(cudaSetDevice(r));
    CK(cudaMemsetAsync(R[r].part, 0, HIDDEN * sizeof(float), R[r].compute));
  }
  sync_all(R);
  run_step([&](RankState& S){ chunked_compute(S, NCHUNKS, chunk_ev[S.rank].data()); },
           [&](RankState& S){ chunked_collective(S, NCHUNKS, chunk_ev[S.rank].data()); });
  sync_all(R);
  {
    std::vector<float> got(HIDDEN);
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(got.data(), R[0].part, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
    const float want = (float)(TP * (TP + 1) / 2);                // 36 for TP=8
    double max_abs = 0.0;
    for (int i = 0; i < HIDDEN; ++i) max_abs = std::max(max_abs, fabs((double)got[i] - (double)want));
    printf("\ncorrectness (chunked all-reduce SUM == %g everywhere): max_abs=%.3e -> %s (<1e-2)\n",
           want, max_abs, (max_abs < 1e-2 ? "PASS" : "FAIL"));
  }

  // =============================================================================================
  // Timing.  Each scheme: WARM untimed iters, then IT timed iters; sync ALL streams of ALL ranks
  // before stopping the timer (so the span covers the slowest rank).  CUDA events on rank 0's
  // compute stream bracket the whole iter span after a full barrier.
  // =============================================================================================
  auto bench = [&](auto compute_one, auto coll_one) -> float {
    for (int it = 0; it < WARM; ++it) { run_step(compute_one, coll_one); sync_all(R); }
    cudaEvent_t e0, e1; CK(cudaSetDevice(0)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    sync_all(R);
    CK(cudaSetDevice(0));                       // sync_all left device 7 current; e0 lives on device 0
    CK(cudaEventRecord(e0, R[0].compute));
    for (int it = 0; it < IT; ++it) { run_step(compute_one, coll_one); sync_all(R); }
    CK(cudaSetDevice(0)); CK(cudaEventRecord(e1, R[0].compute)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= IT;
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return ms;
  };

  auto noop_compute = [&](RankState&){};                          // collective-only reference
  auto noop_coll    = [&](RankState&){};                          // compute-only reference

  // Standalone reference timings: bare collective and bare compute, to see the floor.
  const float ms_coll  = bench(noop_compute,
                               [&](RankState& S){ NK(ncclAllReduce(S.part, S.part, HIDDEN,
                                   ncclFloat32, ncclSum, S.comm, S.comms)); });
  const float ms_comp  = bench([&](RankState& S){ launch_qkv(S, S.compute); }, noop_coll);
  // Scheme A — serial: collective on the compute stream, after the producer, fully exposed.
  const float ms_ser   = bench([&](RankState& S){ serial_compute(S); },
                               [&](RankState& S){ serial_collective(S); });
  // Scheme B — chunked reduce-as-you-go.
  const float ms_chunk = bench([&](RankState& S){ chunked_compute(S, NCHUNKS, chunk_ev[S.rank].data()); },
                               [&](RankState& S){ chunked_collective(S, NCHUNKS, chunk_ev[S.rank].data()); });
  // Scheme C — collective on comm stream || next-layer K1 GEMV on compute stream.
  const float ms_pipe  = bench([&](RankState& S){ pipeline_compute(S); },
                               [&](RankState& S){ pipeline_collective(S); });

  // =============================================================================================
  // Report.
  // =============================================================================================
  const double coll_us = ms_coll * 1e3;
  const double comp_us = ms_comp * 1e3;
  printf("\n-- component costs (per iter) --\n");
  printf("  %-32s %10.2f us\n", "bare all-reduce [HIDDEN]", coll_us);
  printf("  %-32s %10.2f us  (QKV GEMV %d rows/rank, ~%.2f MB fp8/rank)\n",
         "bare K1 QKV GEMV (compute)", comp_us, QKV_OUT_RANK,
         (double)QKV_OUT_RANK * HIDDEN / 1e6);

  // ideal-serial = collective + compute (what you pay if neither hides).  For SchemeA the "compute"
  // is the chunk producer (tiny), so the SchemeA serial is ~ coll + producer; we report it directly.
  printf("\n-- overlap schemes (per iter, lower is better) --\n");
  printf("  %-32s %10.2f us  (collective fully exposed)\n", "A: serial (produce; reduce)", ms_ser*1e3);
  printf("  %-32s %10.2f us  (reduce-as-you-go, %d chunks)\n", "B: chunked all-reduce", ms_chunk*1e3, NCHUNKS);
  printf("  %-32s %10.2f us  (reduce || next-layer K1)\n", "C: layer-pipeline prefetch", ms_pipe*1e3);

  // Fraction of the naive serial SUM (collective + compute on one stream) saved by scheme C.
  // NOTE: this is "fraction of the serial sum saved", not "fraction of compute overlapped" — when
  // the shorter operand is fully hidden it reports short/(coll+comp); the downstream exposed_coll_us
  // is the number that actually matters and it is computed exactly below.
  const double ideal_serial_us = coll_us + comp_us;
  const double ovC = ideal_serial_us > 0 ? (1.0 - (ms_pipe*1e3) / ideal_serial_us) : 0.0;
  // How much of the collective is hidden in scheme C: serial_pair - overlapped = hidden; / coll.
  const double hidden_us = std::max(0.0, ideal_serial_us - ms_pipe*1e3);
  const double coll_hidden_frac = coll_us > 0 ? std::min(1.0, hidden_us / coll_us) : 0.0;
  // Exposed collective per layer after scheme-C overlap (what survives on the critical path).
  const double exposed_coll_us = std::max(0.0, coll_us - hidden_us);

  printf("\n-- overlap analysis (scheme C: collective hidden behind next-layer K1) --\n");
  printf("  ideal serial (coll + compute)      %10.2f us\n", ideal_serial_us);
  printf("  overlapped (scheme C)              %10.2f us\n", ms_pipe*1e3);
  printf("  serial-sum saved (1 - ov/serial)   %10.1f %%\n", 100.0 * ovC);
  printf("  collective hidden                  %10.2f us  (%.1f%% of the %.2f us collective)\n",
         hidden_us, 100.0 * coll_hidden_frac, coll_us);
  printf("  EXPOSED collective / layer         %10.2f us  (was %.2f us serial)\n", exposed_coll_us, coll_us);

  // Implied per-token comms (188 collectives = 2/layer x 94 layers).
  const int N_COLL = 2 * N_LAYERS;                                // 188
  const double serial_token_ms  = N_COLL * coll_us / 1e3;
  const double overlap_token_ms = N_COLL * exposed_coll_us / 1e3;
  printf("\n-- implied per-token comms (%d collectives = 2/layer x %d layers) --\n", N_COLL, N_LAYERS);
  printf("  SERIAL (exposed):   %6.2f ms/token  -> comms-capped at ~%.0f tok/s\n",
         serial_token_ms, serial_token_ms > 0 ? 1000.0 / serial_token_ms : 0.0);
  printf("  OVERLAPPED (C):     %6.2f ms/token  -> comms-capped at ~%.0f tok/s\n",
         overlap_token_ms, overlap_token_ms > 0 ? 1000.0 / overlap_token_ms : 1e9);
  // chunked-scheme exposed = ms_chunk - (compute it overlapped); since SchemeB's compute is the chunk
  // producer (negligible), the chunked benefit is the pipelining of the C sub-reduces themselves.
  const double chunk_exposed_us = ms_chunk * 1e3;
  printf("  CHUNKED (B) exposed all-reduce:  %.2f us/layer-combine (vs %.2f us monolithic serial all-reduce)\n",
         chunk_exposed_us, ms_ser*1e3);

  printf("\nNOTES:\n");
  printf("  * Timing uses a per-iter sync_all(), so each iter measures SINGLE-LAYER overlap latency;\n");
  printf("    the x%d per-token figure assumes zero cross-layer pipeline fill and is thus an UPPER\n", N_COLL);
  printf("    BOUND on the benefit (steady-state pipelining would do at least this well).\n");
  printf("  * Scheme C overlaps the per-layer collective with the NEXT layer's independent QKV GEMV;\n");
  printf("    the more independent compute per layer (full K1+K2+K3 prologue, ~tens of us at TP8), the\n");
  printf("    closer the exposed collective -> 0.  This PoC overlaps only K1, a LOWER bound on hiding.\n");
  printf("  * For a sub-microsecond, GPU-initiated, fully overlap-friendly path, swap the NCCL all-\n");
  printf("    reduce for an NVSHMEM put + on-device barrier (IBGDA) issued from inside the epilogue\n");
  printf("    kernel — no host round-trip, no separate launch — which removes the launch/latency floor\n");
  printf("    entirely (the DeepEP dispatch/combine model).  The stream/event scaffold here is the\n");
  printf("    same; only the collective primitive changes.\n");
  printf("== done ==\n");

  for (int r = 0; r < TP; ++r) {
    CK(cudaSetDevice(r));
    for (int c = 0; c < NCHUNKS; ++c) cudaEventDestroy(chunk_ev[r][c]);
  }
  for (int r = 0; r < TP; ++r) free_rank(R[r]);
  for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamDestroy(R[r].compute));
                                                       CK(cudaStreamDestroy(R[r].comms)); }
  for (int r = 0; r < TP; ++r) ncclCommDestroy(comms[r]);
  return 0;
}
