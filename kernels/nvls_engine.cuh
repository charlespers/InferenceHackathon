// nvls_engine.cuh — drop-in NVLS (multimem / NVLink-SHARP in-switch) all-reduce for decode_step_tp8.cu.
// ==================================================================================================
// Replaces the engine's 188 per-layer NCCL all-reduces (post-attention O-proj + post-MoE-down, each a
// [HIDDEN]=4096 fp32 = 16 KB SUM across the 8 ranks) with a single in-switch multimem reduce.  The
// validated standalone (kernels/nvls_ar.cu, commit bacad60) measured C=3.84us vs NCCL ~17us — 4.4x.
//
// This header adapts that validated multicast setup to the ENGINE's single-process / 8-thread /
// one-comm-per-rank model and to its fp32 partial dtype (nvls_ar.cu validated the fp16 payload; the
// engine's attn_partial/moe_partial are fp32, so we use the f32 multimem variants here).
//
// CONCURRENCY (the part nvls_ar.cu's single-device microbench did NOT exercise): the engine runs one
// host thread per rank, each launching its AR kernel on its own device/stream CONCURRENTLY.  A correct
// all-reduce therefore needs a cross-rank barrier so (a) every rank has written its partial into the
// MC-bound memory before ANY rank issues multimem.ld_reduce, and (b) every rank has consumed the
// reduced result before any rank overwrites it next collective.  We use an in-SWITCH barrier: a small
// MC-backed counter that every rank bumps via multimem.red (one in-switch atomic add broadcast to all
// 8 GPUs); each rank then spins on its LOCAL (unicast) view until the count reaches the generation
// target.  No host sync, no P2P flag mesh — the switch does the arrival count.
//
// DTYPE NOTE: attn_partial/moe_partial are fp32 [HIDDEN].  multimem.ld_reduce.global.add.v4.f32 sums
// 4 floats (128-bit) per instruction across all bound GPUs in one switch round-trip.  The reduction is
// EXACT fp32 add in the switch (same SUM NCCL produced), so the engine's correctness gate (<1e-2 vs the
// single-GPU reference) holds.
// ==================================================================================================
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef NVLS_HIDDEN
#define NVLS_HIDDEN 4096          // [HIDDEN] fp32 per all-reduce (engine's attn_partial/moe_partial)
#endif

#define NVLS_DCK(x) do{ CUresult crst_=(x); if(crst_!=CUDA_SUCCESS){ const char* s_; cuGetErrorString(crst_,&s_); \
  printf("NVLS DRV ERR %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1);} }while(0)
#define NVLS_RCK(x) do{ cudaError_t cest_=(x); if(cest_!=cudaSuccess){ \
  printf("NVLS RT  ERR %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(cest_)); exit(1);} }while(0)

// --------------------------------------------------------------------------------------------------
// Per-rank NVLS context.  mc_* are the MULTICAST views (multimem ops target these -> in-switch reduce);
// uc_* are this rank's local UNICAST view of the same physical pages (kernels write their partial here,
// validation reads here).  bar_* is the MC-backed arrival barrier counter (one int/generation slot).
// --------------------------------------------------------------------------------------------------
struct NvlsCtx {
  int    rank = 0, dev = 0, npes = 8;
  // AR data buffer #1 (post-attention O-proj partial) and #2 (post-MoE-down partial).  We pack BOTH
  // into one MC allocation laid out [attn HIDDEN | moe HIDDEN] so a single multicast object covers both.
  float* uc = nullptr;          // unicast base: uc[0..HIDDEN)=attn, uc[HIDDEN..2H)=moe  (this rank's view)
  float* mc = nullptr;          // multicast base (same layout) — multimem ops reduce across all ranks
  // arrival barrier (separate small MC allocation): THREE independent ring-counter arrays — phase-1
  // (before the reduce-read), phase-2 (between read and store: the in-place WAR-hazard guard), phase-3
  // (after the store) — each NVLS_BAR_SLOTS deep; multimem.red adds 1/arrival.  bar_uc/bar_mc point at
  // the base; phase-k counters live at +k*NVLS_BAR_SLOTS.
  unsigned* bar_uc = nullptr;   // this rank's local view of the barrier counters
  unsigned* bar_mc = nullptr;   // multicast view (multimem.red broadcasts the +1 into every rank's copy)
  // Per-rank DEVICE generation counter (plain per-rank memory, NOT multicast).  The kernel reads+++ it
  // each call, so the barrier generation advances on the DEVICE — independent of host enqueue order.
  // This makes the AR replay-safe inside a captured CUDA graph (host can't bump a counter on replay):
  // because all ranks issue the IDENTICAL ordered collective sequence, each rank's private counter
  // stays in lockstep, so the i-th collective carries the same gen on every rank.  Zeroed at setup.
  unsigned* gen_ctr = nullptr;
  bool   ready = false;         // true once the multicast object is wired (false -> caller falls back)
};

// --------------------------------------------------------------------------------------------------
// fp32 in-switch all-reduce(SUM) of a [n]-float region at MC offset `elt_off`, with a cross-rank
// in-switch barrier.  Grid: a few blocks (16 KB is one wave); ALL ranks launch this concurrently on
// their own stream.  `gen` is this collective's barrier generation (host passes ++ctx.bar_gen).
//
// Barrier protocol (BAR_SLOTS-deep ring so a fast rank can't clobber a slow rank's slot):
//   slot = gen % BAR_SLOTS;  target = gen/BAR_SLOTS * npes + npes  (cumulative arrivals expected).
//   Each rank: block 0 thread 0 does multimem.red.add.u32 [bar_mc+slot], 1  -> in-switch +1 broadcast
//   to EVERY rank's bar_uc[slot].  Then every block spins on the LOCAL bar_uc[slot] >= target.  Because
//   multimem.red is an in-switch atomic broadcast, after all 8 arrivals every rank sees count==target.
// --------------------------------------------------------------------------------------------------
#ifndef NVLS_BAR_SLOTS
#define NVLS_BAR_SLOTS 16
#endif

// One arrival barrier on a ring-counter array `bar` (mc view for the +1, uc view for the spin).
// `gen` is this collective's monotonic generation; slot = gen % SLOTS; cumulative target accounts for
// the (gen/SLOTS) prior full passes through this slot.  All npes ranks must call with the SAME gen.
// SINGLE-BLOCK kernel: thread 0 issues the in-switch +1 and spins; __syncthreads releases the block.
//
// MEMORY MODEL (the fix for the intermittent partial-sum race): the caller threadfence_system()s its
// data write BEFORE arriving here.  The arrival increment MUST be a RELEASE so that data write is
// ordered-before the counter bump in a way the peers' ACQUIRE spin-load can synchronize-with — a
// .relaxed red forms NO release sequence, so a peer's acquire had nothing to sync-with and could run
// its multimem.ld_reduce before this rank's partial became visible (=> a partial sum, the ~1e-2..1e-1
// uniform error).  release(+threadfenced data) -> acquire(spin) is the standard arrival-barrier
// handshake and guarantees every rank's partial is visible system-wide before ANY rank reduces.
__device__ __forceinline__ void nvls_barrier(unsigned* bar_mc, unsigned* bar_uc, unsigned gen, int npes) {
  const int slot = (int)(gen & (NVLS_BAR_SLOTS - 1));
  const unsigned target = ((gen / NVLS_BAR_SLOTS) + 1u) * (unsigned)npes;
  if (threadIdx.x == 0) {
    // in-switch atomic add of 1, broadcast to all bound GPUs' copy of bar[slot].  RELEASE: publishes
    // the threadfenced data write ahead of the +1 so the peers' acquire-load below sees it.
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;" :: "l"(bar_mc + slot) : "memory");
    unsigned v;
    do {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(bar_uc + slot) : "memory");
    } while (v < target);
  }
  __syncthreads();
}

// One-shot fp32 multimem all-reduce of the [n]-float region at MC offset `elt_off`.  SINGLE BLOCK of
// 1024 threads -> 1024*4 = 4096 = HIDDEN elements covered in one wave (no cross-block gen ambiguity).
// Device-side generation counter (gen_ctr, per-rank private) makes the barrier REPLAY-SAFE in a CUDA
// graph: it advances on the device each call, so host enqueue order is irrelevant.
//
// THREE independent barrier arrays — the in-place ld_reduce/st hazard fix.  A multimem.ld_reduce reads
// EVERY rank's bound buffer through the switch and multimem.st OVERWRITES every rank's buffer.  With all
// 8 ranks running concurrently and only a pre- and post-barrier, a fast rank's `st` (writing the SUM
// back) clobbers a buffer that a slow rank's `ld_reduce` has not yet read -> that slow read sums the
// already-summed values (e.g. 8x36) and re-broadcasts garbage; cascaded across the 188 collectives this
// produced the intermittent over-/under-counts (the 1e-2..1e-1 errors, and >>1 blowups under stress).
// The fix is a barrier BETWEEN the reduce-read and the store-write so ALL reads complete before ANY
// write.  Three ring-counter arrays (each NVLS_BAR_SLOTS deep), one per phase:
//   phase-1 bar[0*SLOTS .. 1*SLOTS): all partials written+visible  before any ld_reduce  (RAW).
//   phase-2 bar[1*SLOTS .. 2*SLOTS): all ld_reduce reads done      before any st         (WAR — the fix).
//   phase-3 bar[2*SLOTS .. 3*SLOTS): all st writes visible         before buffer reuse   (WAW/next-RAW).
__global__ void __launch_bounds__(1024) nvls_allreduce_f32_kernel(
    float* __restrict__ mc, int elt_off, int n,
    unsigned* bar_mc, unsigned* bar_uc, unsigned* gen_ctr, int npes) {
  // single block -> thread 0 reads+bumps the private gen counter; the value is this collective's gen.
  __shared__ unsigned s_gen;
  if (threadIdx.x == 0) { s_gen = *gen_ctr; *gen_ctr = s_gen + 1; }
  __syncthreads();
  const unsigned gen_in = s_gen;

  // Make THIS rank's partial (written by the prior K3 / K5b kernel into the MC-bound pages) visible
  // system-wide BEFORE we signal arrival, so a peer's multimem.ld_reduce reads the final value.
  __threadfence_system();
  // Phase 1: every rank's partial is now resident + visible in MC-bound memory (RAW guard).
  nvls_barrier(bar_mc + 0 * NVLS_BAR_SLOTS, bar_uc + 0 * NVLS_BAR_SLOTS, gen_in, npes);

  // In-switch reduce: read+sum 4 floats (128-bit) from ALL ranks via the switch into registers.  We
  // SPLIT the read from the write: every rank must finish reading before any rank overwrites (phase 2).
  float* base = mc + elt_off;
  const int i = threadIdx.x * 4;
  uint32_t a = 0, b = 0, c = 0, d = 0;
  const bool active = (i < n);
  if (active) {
    asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(base + i));
  }
  __threadfence_system();   // our reads are complete + ordered before we signal phase-2 arrival.
  // Phase 2: ALL ranks finished ld_reduce (read every input) before ANY rank's st clobbers it (WAR).
  nvls_barrier(bar_mc + 1 * NVLS_BAR_SLOTS, bar_uc + 1 * NVLS_BAR_SLOTS, gen_in, npes);

  // Broadcast the reduced result back to every rank's buffer (now safe — all reads are done).
  if (active) {
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
                 :: "l"(base + i), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
  }
  __threadfence_system();   // our st is globally visible before we signal phase-3 arrival.

  // Phase 3: all ranks issued their st (result complete everywhere) before anyone reuses the buffer.
  nvls_barrier(bar_mc + 2 * NVLS_BAR_SLOTS, bar_uc + 2 * NVLS_BAR_SLOTS, gen_in, npes);
}

// --------------------------------------------------------------------------------------------------
// One-time setup over the engine's `npes` devices (devices 0..npes-1).  Fills ctx[r] for every rank.
// Returns true if NVLS multicast is available + wired; false (with a printed reason) otherwise so the
// caller can fall back to NCCL.  Must be called from the main thread BEFORE the per-rank threads run.
// --------------------------------------------------------------------------------------------------
static bool nvls_engine_setup(std::vector<NvlsCtx>& ctx, int npes) {
  const size_t data_floats = (size_t)2 * NVLS_HIDDEN;          // attn[HIDDEN] + moe[HIDDEN]
  const size_t data_bytes  = data_floats * sizeof(float);
  const size_t bar_bytes   = (size_t)3 * NVLS_BAR_SLOTS * sizeof(unsigned);   // phase-1/2/3 ring arrays

  if (cuInit(0) != CUDA_SUCCESS) { printf("NVLS: cuInit failed\n"); return false; }

  // ---- check multicast support on device 0 ----
  CUdevice d0; if (cuDeviceGet(&d0, 0) != CUDA_SUCCESS) { printf("NVLS: cuDeviceGet failed\n"); return false; }
  int mc_supported = 0;
  cuDeviceGetAttribute(&mc_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, d0);
  if (!mc_supported) { printf("NVLS: CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED=0 on dev0 (no NVSwitch multicast)\n"); return false; }

  std::vector<CUdevice> dev(npes);
  for (int d = 0; d < npes; ++d) NVLS_DCK(cuDeviceGet(&dev[d], d));

  // helper: create one multicast object + bind one phys alloc per device + map MC va + per-device UC va.
  auto make_mc = [&](size_t want_bytes, std::vector<CUdeviceptr>& uc_va, CUdeviceptr& mc_va) -> bool {
    CUmulticastObjectProp mcp; memset(&mcp, 0, sizeof(mcp));
    mcp.numDevices  = npes;
    mcp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    mcp.size        = want_bytes;
    size_t mcgran = 0;
    NVLS_DCK(cuMulticastGetGranularity(&mcgran, &mcp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    size_t size = ((want_bytes + mcgran - 1) / mcgran) * mcgran;
    mcp.size = size;
    CUmemGenericAllocationHandle mc;
    // SOFT-fail on the create itself (the gate): if the driver refuses multicast despite the attribute
    // check, return false so the engine falls back to NCCL instead of aborting.
    CUresult mcr = cuMulticastCreate(&mc, &mcp);
    if (mcr != CUDA_SUCCESS) {
      const char* es; cuGetErrorString(mcr, &es);
      printf("NVLS: cuMulticastCreate failed -> %s (falling back to NCCL)\n", es);
      return false;
    }
    for (int d = 0; d < npes; ++d) NVLS_DCK(cuMulticastAddDevice(mc, dev[d]));

    uc_va.resize(npes);
    for (int d = 0; d < npes; ++d) {
      NVLS_RCK(cudaSetDevice(d));
      CUmemAllocationProp p; memset(&p, 0, sizeof(p));
      p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      p.location.id = d; p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
      size_t mgran = 0; NVLS_DCK(cuMemGetAllocationGranularity(&mgran, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
      size_t msize = ((size + mgran - 1) / mgran) * mgran;
      CUmemGenericAllocationHandle phys;
      NVLS_DCK(cuMemCreate(&phys, msize, &p, 0));
      NVLS_DCK(cuMulticastBindMem(mc, 0, phys, 0, size, 0));
      NVLS_DCK(cuMemAddressReserve(&uc_va[d], size, 0, 0, 0));
      NVLS_DCK(cuMemMap(uc_va[d], size, 0, phys, 0));
      CUmemAccessDesc ad; memset(&ad, 0, sizeof(ad));
      ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id = d;
      ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      NVLS_DCK(cuMemSetAccess(uc_va[d], size, &ad, 1));
    }
    // map the MULTICAST handle into a VA reachable from all devices (kernels target this).
    NVLS_RCK(cudaSetDevice(0));
    NVLS_DCK(cuMemAddressReserve(&mc_va, size, mcgran, 0, 0));
    NVLS_DCK(cuMemMap(mc_va, size, 0, mc, 0));
    std::vector<CUmemAccessDesc> ad(npes);
    for (int d = 0; d < npes; ++d) { memset(&ad[d], 0, sizeof(CUmemAccessDesc));
      ad[d].location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad[d].location.id = d;
      ad[d].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
    NVLS_DCK(cuMemSetAccess(mc_va, size, ad.data(), npes));
    return true;
  };

  std::vector<CUdeviceptr> data_uc, bar_uc;
  CUdeviceptr data_mc = 0, bar_mc = 0;
  if (!make_mc(data_bytes, data_uc, data_mc)) return false;
  if (!make_mc(bar_bytes,  bar_uc,  bar_mc )) return false;

  // zero the barrier counters on every device's unicast view, and alloc+zero a per-rank gen counter.
  std::vector<unsigned*> gen_ctr(npes, nullptr);
  for (int d = 0; d < npes; ++d) { NVLS_RCK(cudaSetDevice(d));
    NVLS_RCK(cudaMemset((void*)bar_uc[d], 0, bar_bytes));
    NVLS_RCK(cudaMalloc(&gen_ctr[d], sizeof(unsigned)));
    NVLS_RCK(cudaMemset(gen_ctr[d], 0, sizeof(unsigned))); }
  for (int d = 0; d < npes; ++d) { NVLS_RCK(cudaSetDevice(d)); NVLS_RCK(cudaDeviceSynchronize()); }

  ctx.resize(npes);
  for (int r = 0; r < npes; ++r) {
    ctx[r].rank = r; ctx[r].dev = r; ctx[r].npes = npes;
    ctx[r].uc      = reinterpret_cast<float*>(data_uc[r]);
    ctx[r].mc      = reinterpret_cast<float*>(data_mc);    // SAME mc VA on every rank (reserved on dev0,
                                                           // accessible from all via cuMemSetAccess).
    ctx[r].bar_uc  = reinterpret_cast<unsigned*>(bar_uc[r]);
    ctx[r].bar_mc  = reinterpret_cast<unsigned*>(bar_mc);
    ctx[r].gen_ctr = gen_ctr[r];
    ctx[r].ready   = true;
  }
  printf("NVLS: multicast wired over %d GPUs (data %zu B + barrier %zu B per rank).  multimem f32 AR active.\n",
         npes, data_bytes, bar_bytes);
  return true;
}

// MC-offset helpers (which half of the packed data buffer).
static constexpr int NVLS_OFF_ATTN = 0;
static constexpr int NVLS_OFF_MOE  = NVLS_HIDDEN;

// Launch the NVLS all-reduce of one [HIDDEN] region on this rank's stream (concurrent across ranks).
//   elt_off = NVLS_OFF_ATTN or NVLS_OFF_MOE.  The generation advances DEVICE-SIDE (gen_ctr) so this is
//   stream-capturable / replay-safe.  SINGLE block of HIDDEN/4 = 1024 threads covers the whole [HIDDEN].
static inline void nvls_allreduce_launch(NvlsCtx& c, int elt_off, cudaStream_t s) {
  const int threads = NVLS_HIDDEN / 4;     // 1024 threads, 4 floats each -> [HIDDEN] in one block
  nvls_allreduce_f32_kernel<<<1, threads, 0, s>>>(
      c.mc, elt_off, NVLS_HIDDEN, c.bar_mc, c.bar_uc, c.gen_ctr, c.npes);
}
