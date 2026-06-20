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
//
// ==================================================================================================
// SINGLE-BARRIER OUT-OF-PLACE (the comms win this revision ships)
// --------------------------------------------------------------------------------------------------
// The previous design used TWO in-switch barriers per AR:
//   phase-1 (RAW):  all partials written into IN  before any ld_reduce.
//   phase-2 (WAR):  all ld_reduce reads of IN done before the NEXT collective overwrites IN.
// A barrier is an in-switch atomic + a cross-rank spin == the per-AR latency floor (~7us each); the
// multimem reduce itself is sub-us.  So killing one barrier ~halves the per-AR cost.
//
// phase-2 existed ONLY to guard the cross-collective WAR on a SINGLE shared IN buffer: a fast rank that
// finished collective i could run the next layer's K3/K5b and overwrite IN before a slow rank finished
// its collective-i ld_reduce.  We remove phase-2 by reducing from a ROTATING (ping-pong) staging slot
// instead of one shared buffer:
//   * The producer (K3 / K5b / GEMM epilogue) keeps writing ONE FIXED IN pointer (graph-safe: buffers
//     never move).  attn_IN / moe_IN are that fixed landing pad.
//   * The AR kernel, as its FIRST step, copies its rank's fixed IN -> a ping-pong STAGE slot (stage =
//     gen & 1).  This copy is LOCAL (this rank only, same stream as the producer) -> no cross-rank
//     hazard on the fixed IN, and it is serialized after the producer by same-stream ordering.
//   * The single phase-1 barrier then guarantees every rank has staged before any ld_reduce.
//   * ld_reduce reads from the MC-bound STAGE slot; st writes OUT.  IN != STAGE != OUT.
// Cross-collective WAR is now on the STAGE slot, which is reused only every OTHER collective (gen i uses
// stage i&1, gen i+1 uses i+1&1, gen i+2 reuses i&1).  Between collective i and i+2 sits collective
// i+1's phase-1 barrier — a full 8-rank rendezvous.  A rank cannot write stage[i&1] for collective i+2
// until it passed i+1's barrier, which requires every peer to have arrived at i+1, which (same-stream,
// post-threadfence) requires every peer to have finished its collective-i ld_reduce.  => the WAR on the
// reused stage slot is ordered-safe with NO phase-2 barrier.  The local copy (16 KB coalesced, sub-us)
// is far cheaper than the 7us barrier it replaces.  Net: ~2 barriers -> 1 barrier per AR.
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
  // SINGLE-BARRIER OUT-OF-PLACE layout.  The MC allocation packs EIGHT [HIDDEN] regions:
  //   [attn_IN | moe_IN | attn_OUT | moe_OUT | attn_STAGE0 | moe_STAGE0 | attn_STAGE1 | moe_STAGE1]
  // = 8*HIDDEN floats.  The producers (K3 / K5b) write their partial into the IN halves (a FIXED
  // pointer, so the captured graph never has to move it).  The AR kernel copies IN -> the ping-pong
  // STAGE slot (gen&1), reduces FROM the STAGE slot, and stores the broadcast SUM TO the OUT half.
  // Because reduce-READ (STAGE) != store-WRITE (OUT) the in-collective WAR is gone, and because the
  // STAGE slot ping-pongs the cross-collective WAR is gone too -> ONE barrier (entry RAW) per AR.
  // The consumer (residual-add) reads the OUT half (S.attn_reduced / S.moe_reduced repointed to it).
  float* uc = nullptr;          // unicast base (this rank's view of all 8 regions)
  float* mc = nullptr;          // multicast base (same layout) — multimem ops reduce across all ranks
  // arrival barrier (separate small MC allocation): ONE ring-counter array (the entry RAW phase: all
  // partials staged + visible before any ld_reduce).  Single-barrier dropped the old phase-2 (WAR)
  // array — the ping-pong STAGE slot makes phase-2 unnecessary (see header note).
  // NVLS_BAR_SLOTS deep; multimem.red adds 1/arrival.
  unsigned* bar_uc = nullptr;   // this rank's local view of the barrier counters
  unsigned* bar_mc = nullptr;   // multicast view (multimem.red broadcasts the +1 into every rank's copy)
  // Per-rank DEVICE generation counter (plain per-rank memory, NOT multicast).  The kernel reads+++ it
  // each call, so the barrier generation advances on the DEVICE — independent of host enqueue order.
  // This makes the AR replay-safe inside a captured CUDA graph (host can't bump a counter on replay):
  // because all ranks issue the IDENTICAL ordered collective sequence, each rank's private counter
  // stays in lockstep, so the i-th collective carries the same gen on every rank.  Zeroed at setup.
  // The low bit of gen ALSO selects the ping-pong STAGE slot (stage = gen & 1).
  unsigned* gen_ctr = nullptr;
  bool   ready = false;         // true once the multicast object is wired (false -> caller falls back)
};

// --------------------------------------------------------------------------------------------------
// fp32 in-switch all-reduce(SUM) of a [n]-float region.  Grid: SINGLE block (16 KB is one wave); ALL
// ranks launch this concurrently on their own stream.  The generation advances DEVICE-side (gen_ctr).
//
// Barrier protocol (BAR_SLOTS-deep ring so a fast rank can't clobber a slow rank's slot):
//   slot = gen % BAR_SLOTS;  target = (gen/BAR_SLOTS + 1) * npes  (cumulative arrivals expected).
//   Each rank: thread 0 does multimem.red.add.u32 [bar_mc+slot], 1  -> in-switch +1 broadcast to EVERY
//   rank's bar_uc[slot].  Then thread 0 spins on the LOCAL bar_uc[slot] >= target.  Because
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

// One-shot SINGLE-BARRIER OUT-OF-PLACE fp32 multimem all-reduce.  SINGLE BLOCK of 1024 threads ->
// 1024*4 = 4096 = HIDDEN elements in one wave (no cross-block gen ambiguity).  Device-side generation
// counter (gen_ctr, per-rank private) makes the barrier REPLAY-SAFE in a CUDA graph and selects the
// ping-pong STAGE slot (gen & 1).
//
// ONE barrier (the comms win).  Flow per collective:
//   1. read+bump the private gen counter -> this collective's gen; stage = gen & 1.
//   2. LOCAL copy: this rank's fixed IN -> its ping-pong STAGE slot (stage_off).  No cross-rank traffic;
//      serialized after the producer (K3/K5b) by same-stream ordering -> no IN hazard.
//   3. __threadfence_system + phase-1 barrier: every rank has STAGED + made it visible (RAW guard).
//   4. multimem.ld_reduce FROM the MC-bound STAGE slot (sums all ranks' staged partials in-switch) ->
//      multimem.st TO the SEPARATE OUT buffer.  STAGE != OUT -> no in-collective WAR.
// No phase-2: the cross-collective WAR is on the STAGE slot, which ping-pongs (reused only every 2nd
// collective); the intervening collective's phase-1 barrier rendezvouses all ranks before the reuse.
// Each rank's local consumer (residual-add) reads its OUT half AFTER this kernel returns on its own
// stream — same-stream ordering, no cross-rank barrier needed for the consume.
__global__ void __launch_bounds__(1024) nvls_allreduce_f32_kernel(
    float* __restrict__ mc, float* __restrict__ uc, int in_off, int stage_off0, int out_off, int n,
    unsigned* bar_mc, unsigned* bar_uc, unsigned* gen_ctr, int npes) {
  // single block -> thread 0 reads+bumps the private gen counter; the value is this collective's gen.
  __shared__ unsigned s_gen;
  if (threadIdx.x == 0) { s_gen = *gen_ctr; *gen_ctr = s_gen + 1; }
  __syncthreads();
  const unsigned gen_in = s_gen;
  // ping-pong STAGE slot: gen even -> STAGE0, gen odd -> STAGE1 (stage_off0 is the STAGE0 base; STAGE1
  // is one HIDDEN-pair further, i.e. +2*HIDDEN past STAGE0 — set by the launch as stage stride).
  const int stage_off = stage_off0 + (int)(gen_in & 1u) * (2 * NVLS_HIDDEN);

  // Step 2: LOCAL copy of this rank's fixed IN partial -> its ping-pong STAGE slot (unicast; no switch).
  // We copy via the rank's UNICAST view so the writes are plain device stores (the MC-bound STAGE pages
  // are the same physical memory; the in-switch reduce in step 4 reads them via the MC view).
  const int i = threadIdx.x * 4;
  const bool active = (i < n);
  if (active) {
    float4 v = *reinterpret_cast<const float4*>(uc + in_off + i);
    *reinterpret_cast<float4*>(uc + stage_off + i) = v;
  }
  // Make THIS rank's staged partial visible system-wide BEFORE we signal arrival, so a peer's
  // multimem.ld_reduce reads the final value.
  __threadfence_system();
  // Phase 1: every rank's partial is now resident + visible in its MC-bound STAGE slot (RAW guard).
  nvls_barrier(bar_mc, bar_uc, gen_in, npes);

  // Step 4: in-switch reduce — read+sum 4 floats (128-bit) from ALL ranks' STAGE slot via the switch
  // into registers, then broadcast the SUM to every rank's OUT buffer.  STAGE != OUT -> no WAR.
  if (active) {
    float* stage_base = mc + stage_off;
    float* out_base   = mc + out_off;
    uint32_t a = 0, b = 0, c = 0, d = 0;
    asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(stage_base + i));
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
                 :: "l"(out_base + i), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
  }
  // No phase-2 barrier: the consumer reads OUT on the same stream after return; the next collective's
  // reuse of this STAGE slot is guarded by the intervening collective's phase-1 rendezvous (header).
}

// --------------------------------------------------------------------------------------------------
// One-time setup over the engine's `npes` devices (devices 0..npes-1).  Fills ctx[r] for every rank.
// Returns true if NVLS multicast is available + wired; false (with a printed reason) otherwise so the
// caller can fall back to NCCL.  Must be called from the main thread BEFORE the per-rank threads run.
// --------------------------------------------------------------------------------------------------
static bool nvls_engine_setup(std::vector<NvlsCtx>& ctx, int npes) {
  // SINGLE-BARRIER OUT-OF-PLACE: pack EIGHT [HIDDEN] regions in one MC object:
  //   [attn_IN | moe_IN | attn_OUT | moe_OUT | attn_STAGE0 | moe_STAGE0 | attn_STAGE1 | moe_STAGE1]
  const size_t data_floats = (size_t)8 * NVLS_HIDDEN;          // IN(2) + OUT(2) + STAGE0(2) + STAGE1(2)
  const size_t data_bytes  = data_floats * sizeof(float);
  const size_t bar_bytes   = (size_t)1 * NVLS_BAR_SLOTS * sizeof(unsigned);   // single phase (entry RAW)

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
  printf("NVLS: multicast wired over %d GPUs (data %zu B + barrier %zu B per rank).  multimem f32 AR "
         "(SINGLE-BARRIER, ping-pong stage) active.\n", npes, data_bytes, bar_bytes);
  return true;
}

// MC-offset helpers.  INPUT halves (producers write partials here; AR copies them to STAGE):
static constexpr int NVLS_OFF_ATTN = 0;
static constexpr int NVLS_OFF_MOE  = NVLS_HIDDEN;
// OUTPUT halves (AR multimem.st broadcasts the reduced result here; residual-add reads here).
static constexpr int NVLS_OUT_BASE = 2 * NVLS_HIDDEN;
static constexpr int NVLS_OFF_ATTN_OUT = NVLS_OUT_BASE + 0;
static constexpr int NVLS_OFF_MOE_OUT  = NVLS_OUT_BASE + NVLS_HIDDEN;
// STAGE halves (ping-pong scratch the AR reduces FROM).  STAGE0 at 4*HIDDEN, STAGE1 at 6*HIDDEN; for a
// given IN offset the matching STAGE0 base is in_off + 4*HIDDEN and STAGE1 is +2*HIDDEN beyond that.
static constexpr int NVLS_STAGE_BASE = 4 * NVLS_HIDDEN;
// Map an IN offset to its matching OUT offset (used by the launch + the engine's residual-add repoint).
static __host__ __device__ __forceinline__ int nvls_out_off(int in_off) { return in_off + NVLS_OUT_BASE; }
// Map an IN offset to its matching STAGE0 base (the AR adds (gen&1)*2*HIDDEN for the ping-pong).
static __host__ __device__ __forceinline__ int nvls_stage_off0(int in_off) { return in_off + NVLS_STAGE_BASE; }

// Launch the SINGLE-BARRIER OUT-OF-PLACE NVLS all-reduce of one [HIDDEN] region on this rank's stream
// (concurrent across ranks).  in_off = NVLS_OFF_ATTN or NVLS_OFF_MOE (where the producer wrote the
// partial).  The AR copies IN -> the ping-pong STAGE slot, reduces from STAGE, and broadcasts the SUM
// to nvls_out_off(in_off) (NVLS_OFF_*_OUT), which the residual-add consumes.  The generation advances
// DEVICE-side (gen_ctr) so this is stream-capturable / replay-safe.  SINGLE block of HIDDEN/4 = 1024
// threads covers the whole [HIDDEN].
static inline void nvls_allreduce_launch(NvlsCtx& c, int in_off, cudaStream_t s) {
  const int threads = NVLS_HIDDEN / 4;     // 1024 threads, 4 floats each -> [HIDDEN] in one block
  nvls_allreduce_f32_kernel<<<1, threads, 0, s>>>(
      c.mc, c.uc, in_off, nvls_stage_off0(in_off), nvls_out_off(in_off), NVLS_HIDDEN,
      c.bar_mc, c.bar_uc, c.gen_ctr, c.npes);
}
