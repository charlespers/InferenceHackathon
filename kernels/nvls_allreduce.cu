// nvls_allreduce.cu — multimem (NVLS / NVLink-SHARP in-switch) all-reduce for the B=1 8KB payload.
// THE make-or-break kernel for 1000 tok/s (docs/megakernel-build-plan.md Stage 3, docs/path-to-1000.md):
// gets the per-collective latency C from ~16us (NCCL ring) toward ~1-4us (one in-switch reduce).
//
// WHY C<=~4us MATTERS (the dual role, results-reaction-05.md): the comms can't be *faked* (stale/predicted-TP
// MEASURED DEAD — info barrier) but it CAN be HIDDEN losslessly via LOOP-C's exact DEFERRED-OVERLAP: overlap this
// EXACT reduce with the next op's HBM weight stream (NVLink vs HBM = different HW paths). The per-collective
// weight cover is ~4.3us at fp8 (AR-A overlaps the routed MoE gate/up read; AR-M overlaps the next QKV read).
// So at C<=~4us this reduce is FULLY hidden -> comms->0 -> ~roofline (~1218), LOSSLESS. At 16us it's partial
// (~half) -> ~938 with spec. => This kernel's C decides FULL vs PARTIAL lossless comms-hiding.
//
// DEFERRED-OVERLAP INTEGRATION (kernel feature, LOOP-C's schedule + my kernel; megakernel-b1.md K6):
//   In the persistent megakernel, run THIS reduce on a few SMs (multimem needs only ~2-8 SMs for 8KB) while the
//   REMAINING SMs cp.async-stream the next op's weights (per the schedule: post-attn AR-A || MoE gate/up load;
//   post-MoE AR-M || next-layer QKV load). The dependent multiply waits on the reduced activation, then runs on
//   already-resident weights. So the standalone reduce below is ALSO the overlap primitive — keep its SM
//   footprint small so it co-resides with the weight-stream warps.
//
// STATUS: starting skeleton to COMPILE + TUNE + VALIDATE on the 8xH100 box (like K5 began). The multimem PTX +
// the CU multicast setup are the essential structure; the TODOs are the box-specific bits to verify. Test this
// IN ISOLATION first (the microbench below) — its single number decides the 1000 path before any engine work.
//
// Build (per GPU process, NVSwitch required — H100 has it):
//   nvcc -O3 -arch=sm_90a nvls_allreduce.cu -lcuda -o nvls_ar   # sm_90a for Hopper multimem
//   then launch 8 processes (one/GPU) sharing the multicast object (MPI or a simple pthreads+IPC harness).
//
// References: CUDA multicast (cuMulticast*) + `multimem.ld_reduce`/`multimem.st` PTX (Hopper). The in-switch
// reduction happens in the NVSwitch (SHARP), so one multimem.ld_reduce returns the SUM across all bound GPUs.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

#define HIDDEN 4096                 // Qwen3 hidden; the all-reduce payload = HIDDEN * sizeof(elt)
#define NRANKS 8

// ---- the all-reduce kernel: each thread reduces a slice via the switch, then broadcasts ----
// mc_ptr is the MULTICAST address (maps to all ranks' buffers); reducing through it hits the in-switch SHARP.
__global__ void nvls_allreduce_half(half* __restrict__ mc_ptr, int n /* elements */) {
    // process 8 halves (16 bytes) per thread via v4.f16x2 multimem ops for coalesced 128-bit in-switch reduce.
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= n) return;
    // multimem.ld_reduce.global.add.v4.f16x2 : load+sum across all bound GPUs (one switch round-trip), 8 halves.
    uint32_t a, b, c, d;
    asm volatile(
        "multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "l"(mc_ptr + i));
    // multimem.st.global : broadcast the reduced result back to all GPUs' buffers in one op.
    asm volatile(
        "multimem.st.global.v4.f16x2 [%0], {%1,%2,%3,%4};"
        :: "l"(mc_ptr + i), "r"(a), "r"(b), "r"(c), "r"(d)
        : "memory");
    // NOTE: a true all-reduce also needs a grid-wide barrier so every rank sees the broadcast before using it.
    // At B=1 8KB this is one wave (HIDDEN/8 = 512 threads = 1-2 blocks on 2-4 SMs) -> a lightweight
    // multimem flag-based barrier (or cooperative-groups grid sync if single-block) suffices. TODO: wire it.
}

// ============================ host: multicast setup + latency microbench ============================
// The setup is the fiddly, box-specific part. Outline (per the CUDA multicast API); VERIFY return codes on box.
struct MC {
    CUmemGenericAllocationHandle mc;   // the multicast object (shared across ranks via IPC/MPI)
    CUdeviceptr mc_va;                 // mapped multicast virtual address (what the kernel uses)
    CUdeviceptr uc_va;                 // this rank's unicast (local) view of the same memory
    size_t bytes;
};

// TODO(box): full multi-process setup — cuMulticastCreate on rank0, export/import the handle (cuMemExportToShareableHandle
// + IPC), each rank cuMulticastAddDevice + cuMulticastBindMem(local alloc), then cuMemMap the MC va. This skeleton
// shows the single-process *shape*; the real win needs the 8-process bind so the switch reduces across all 8.
static int mc_setup(MC* m, size_t bytes) {
    (void)m; (void)bytes;
    // CUmulticastObjectProp prop{ .numDevices=NRANKS, .size=bytes, .handleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR };
    // cuMulticastCreate(&m->mc, &prop);  ... addDevice/bindMem/map ...  return 0 on success.
    return -1; // not wired in this skeleton — fill in on box.
}

int main() {
    const int n = HIDDEN;                  // 4096 halves = 8KB (bf16); for fp8 use 4KB / fp8 multimem variant
    MC m;
    if (mc_setup(&m, (size_t)n * sizeof(half)) != 0) {
        printf("nvls_allreduce: multicast setup not wired (skeleton). Fill mc_setup() on box, then:\n");
        printf("  - microbench: 200 warmup + 1000 timed all-reduces of %dB, report MEAN + p50/p99 us = C.\n", n*2);
        printf("  - validate: each rank writes rank_id to its buffer; after AR every elt == sum(0..7)=28. Bit-check.\n");
        printf("  - compare C vs `nccl-tests all_reduce_perf -b 8192 -e 8192` (the ~16us baseline).\n");
        printf("  DECISION (docs/path-to-1000.md): C<=4us -> 1000 ON; ~8us -> needs stale-TP/int4; ~16us -> custom AR still the only way.\n");
        return 0;
    }
    // ---- timed loop (on box) ----
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    int threads = 256, blocks = (n/8 + threads - 1) / threads;   // n/8 elements-of-work
    for (int w = 0; w < 200; ++w) nvls_allreduce_half<<<blocks, threads>>>((half*)m.mc_va, n);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    const int ITERS = 1000;
    for (int it = 0; it < ITERS; ++it) nvls_allreduce_half<<<blocks, threads>>>((half*)m.mc_va, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    printf("nvls_allreduce: C = %.2f us/collective (8KB, 8 GPUs)  [baseline NCCL ring ~16us]\n", ms * 1e3 / ITERS);
    printf("  -> 188 collectives x C = %.2f ms comms. 1000-tok/s gate: C<=4us.\n", 188 * ms * 1e3 / ITERS / 1e3);
    return 0;
}
