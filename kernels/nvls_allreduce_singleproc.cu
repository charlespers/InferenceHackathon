// nvls_allreduce_singleproc.cu — raw CUDA driver multicast, SINGLE PROCESS managing all 8 GPU
// contexts (no NVSHMEM, no multi-process IPC). This sidesteps BOTH things that have blocked us:
//   - NVSHMEM's CUDA-13 device-link requirement (nvls_allreduce_nvshmem.cu hit this at nvlink time).
//   - Multi-process IPC handle exchange (the original nvls_allreduce.cu skeleton's TODO assumed
//     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR + cuMemExportToShareableHandle across processes,
//     which needs a Unix-domain-socket SCM_RIGHTS exchange to actually pass fds between processes).
// Single-process multi-GPU is exactly the pattern decode_step_tp8.cu / overlap_decode_wide.cu already
// use for NCCL (cudaSetDevice + ncclCommInitAll in one process) -- cuMulticastAddDevice/BindMem don't
// need IPC at all when every device's context lives in the same process.
//
// STATUS: written off the documented CUDA multicast driver-API shape (cuMulticastCreate ->
// cuMulticastAddDevice (per device) -> cuMemCreate (per device, local backing) -> cuMulticastBindMem
// -> cuMemAddressReserve+cuMemMap (both the unicast local VA and the multicast VA, per device)).
// NOT YET RUN ON GPU -- this is the non-GPU-bound part (code + compile check); first real run will
// likely surface a wrong granularity/flag/return-code somewhere, per CUDA multicast's reputation for
// being fiddly (nvls_allreduce.cu's own original comment: "the setup is the fiddly, box-specific part").
//
// Build: nvcc -O3 -arch=sm_90a nvls_allreduce_singleproc.cu -lcuda -o nvls_sp
// Run:   ./nvls_sp 4096 2000      (no mpirun needed -- single process, single launch)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

#define NRANKS 8

#define CK(call) do { CUresult _r = (call); if (_r != CUDA_SUCCESS) { \
  const char* es=nullptr; cuGetErrorString(_r,&es); \
  fprintf(stderr, "CUDA driver error %d (%s) at %s:%d: %s\n", _r, es?es:"?", __FILE__, __LINE__, #call); \
  return -1; } } while (0)
#define CKR(call) do { cudaError_t _r = (call); if (_r != cudaSuccess) { \
  fprintf(stderr, "CUDA runtime error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_r)); \
  return -1; } } while (0)

// ---- the all-reduce kernels: same multimem PTX as nvls_allreduce.cu, just called from a single
// process's per-device launches instead of an 8-process MPI launch. ----
__global__ void ar_raw(half* __restrict__ mc_ptr, int n) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= n) return;
  uint32_t a, b, c, d;
  asm volatile("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(mc_ptr + i));
  asm volatile("multimem.st.global.v4.f16x2 [%0], {%1,%2,%3,%4};"
               :: "l"(mc_ptr + i), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

// DIAGNOSTIC: multimem PTX always reduces across every BOUND device at that address (there's no plain
// "load my own copy" op -- the switch fan-in is inherent to multimem.*, not optional). So this can't
// isolate "is my own binding right" in isolation; what it CAN do is give a much smaller, single-thread,
// no-barrier surface than the full ar_raw/ar_barriered path -- run with only ONE device's uc_va seeded
// non-zero (the rest memset to 0) and check the single scratch value matches that one seed. If THAT
// already comes back wrong/inf, the bug is in bind/map, not in the multi-source reduce or barrier.
__global__ void diag_load_own(half* __restrict__ mc_ptr, float* __restrict__ scratch_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint32_t a, b, c, d;
    asm volatile("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(mc_ptr));
    half2 h0 = *reinterpret_cast<half2*>(&a);
    scratch_out[0] = __half2float(h0.x);
  }
}

__device__ void flag_barrier(uint32_t* mc_flag, int nranks) {
  // FIXED: was a plain `red.global.add.u32` on a MULTICAST address -- that's a local-device atomic,
  // not a cross-device increment via the switch (multicast addresses need multimem.* ops specifically).
  asm volatile("multimem.red.global.add.u32 [%0], 1;" :: "l"(mc_flag) : "memory");
  uint32_t seen = 0;
  do { asm volatile("multimem.ld_reduce.global.add.u32 %0, [%1];" : "=r"(seen) : "l"(mc_flag));
  } while (seen < (uint32_t)nranks);
}

// FIXED (per the diagnosed race): barrier BEFORE the reduce, not just after. Without this, a fast
// device's multimem.st (broadcasting its own reduced sum) can land before a slow device's
// multimem.ld_reduce runs -- the slow device then reads a torn mix of original seeds and
// already-reduced results from other devices, producing garbage (the measured maxerr=inf). The
// pre-barrier guarantees every device's ORIGINAL contribution is stable and unread-yet before ANY
// device's ld_reduce starts. (In the real per-layer pipeline this is naturally provided by the
// previous op's grid.sync()/prior-AR ordering -- this bug is specific to a standalone microbenchmark
// with no such prior ordering on its first/cold round.)
__global__ void ar_barriered(half* __restrict__ mc_ptr, uint32_t* __restrict__ mc_flag, int n, int nranks) {
  if (threadIdx.x == 0 && blockIdx.x == 0) flag_barrier(mc_flag, nranks);
  __threadfence();   // local fence; the barrier itself is the cross-device order, this just orders
                      // this block's own later reduce after this block's own barrier participation.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i < n) {
    uint32_t a, b, c, d;
    asm volatile("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(mc_ptr + i));
    asm volatile("multimem.st.global.v4.f16x2 [%0], {%1,%2,%3,%4};"
                 :: "l"(mc_ptr + i), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
  }
}

// ============================ host: single-process multicast setup ============================
struct MC {
  CUmemGenericAllocationHandle mcHandle, flagMcHandle;
  CUmemGenericAllocationHandle localMem[NRANKS], localFlagMem[NRANKS];
  CUdeviceptr mc_va[NRANKS], uc_va[NRANKS];            // data buffer: multicast VA + per-device local VA
  CUdeviceptr flag_mc_va[NRANKS], flag_uc_va[NRANKS];  // flag buffer: same shape, tiny (4 bytes)
  size_t bytes, flagBytes;
};

static int mc_setup_one(CUmemGenericAllocationHandle* mcHandleOut,
                         CUmemGenericAllocationHandle* localMem, CUdeviceptr* mc_va, CUdeviceptr* uc_va,
                         size_t reqBytes, size_t* outBytes) {
  CUmulticastObjectProp mcProp = {};
  mcProp.numDevices = NRANKS;
  mcProp.size = reqBytes;
  mcProp.handleTypes = (CUmemAllocationHandleType)0;   // CU_MEM_HANDLE_TYPE_NONE -- single process, no export
  mcProp.flags = 0;

  size_t granularity = 0;
  CK(cuMulticastGetGranularity(&granularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  size_t alignedSize = ((reqBytes + granularity - 1) / granularity) * granularity;
  mcProp.size = alignedSize;
  *outBytes = alignedSize;

  CK(cuMulticastCreate(mcHandleOut, &mcProp));

  for (int dev = 0; dev < NRANKS; ++dev) {
    CKR(cudaSetDevice(dev));
    CK(cuMulticastAddDevice(*mcHandleOut, (CUdevice)dev));
  }

  for (int dev = 0; dev < NRANKS; ++dev) {
    CKR(cudaSetDevice(dev));
    CUmemAllocationProp memProp = {};
    memProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memProp.location.id = dev;

    size_t memGran = 0;
    CK(cuMemGetAllocationGranularity(&memGran, &memProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t memSize = ((alignedSize + memGran - 1) / memGran) * memGran;

    CK(cuMemCreate(&localMem[dev], memSize, &memProp, 0));
    CK(cuMulticastBindMem(*mcHandleOut, /*mcOffset=*/0, localMem[dev], /*memOffset=*/0, alignedSize, /*flags=*/0));

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = dev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CK(cuMemAddressReserve(&uc_va[dev], alignedSize, 0, 0, 0));
    CK(cuMemMap(uc_va[dev], alignedSize, 0, localMem[dev], 0));
    CK(cuMemSetAccess(uc_va[dev], alignedSize, &accessDesc, 1));

    CK(cuMemAddressReserve(&mc_va[dev], alignedSize, 0, 0, 0));
    CK(cuMemMap(mc_va[dev], alignedSize, 0, *mcHandleOut, 0));
    CK(cuMemSetAccess(mc_va[dev], alignedSize, &accessDesc, 1));
  }
  return 0;
}

static int mc_setup(MC* m, size_t bytes) {
  if (mc_setup_one(&m->mcHandle, m->localMem, m->mc_va, m->uc_va, bytes, &m->bytes) != 0) return -1;
  if (mc_setup_one(&m->flagMcHandle, m->localFlagMem, m->flag_mc_va, m->flag_uc_va,
                    sizeof(uint32_t), &m->flagBytes) != 0) return -1;
  return 0;
}

// Define NVLS_SP_NO_MAIN before #include-ing this file to reuse mc_setup()/MC/the kernels as a
// library (same convention as DSTP8_NO_MAIN / K5_NO_MAIN).
#ifndef NVLS_SP_NO_MAIN
int main(int argc, char** argv) {
  int n     = (argc > 1) ? atoi(argv[1]) : 4096;
  int iters = (argc > 2) ? atoi(argv[2]) : 2000;

  int ndev = 0;
  CKR(cudaGetDeviceCount(&ndev));
  if (ndev < NRANKS) { fprintf(stderr, "need >= %d devices, found %d\n", NRANKS, ndev); return 1; }

  CK(cuInit(0));
  MC m;
  if (mc_setup(&m, (size_t)n * sizeof(half)) != 0) {
    fprintf(stderr, "mc_setup FAILED -- see CUDA driver error above. This is the box-specific fiddly\n"
                     "part nvls_allreduce.cu's own comment warned about; check granularity/flags first.\n");
    return 1;
  }
  printf("== NVLS multimem all-reduce, SINGLE-PROCESS/%d-context: N=%d half (%dB) ==\n", NRANKS, n, (int)(n*2));

  // ---- DIAGNOSTIC FIRST: seed ONLY device 0 non-zero (the rest exactly 0), then a single-thread
  // multimem reduce of [0] should read back as device 0's seed value -- isolates a bind/map bug from
  // a multi-source-reduce/barrier bug before trusting anything built on top of this setup. ----
  {
    for (int dev = 0; dev < NRANKS; ++dev) {
      CKR(cudaSetDevice(dev));
      float seedval = (dev == 0) ? 7.0f : 0.0f;
      std::vector<half> seed(n, __float2half(seedval));
      CKR(cudaMemcpy((void*)m.uc_va[dev], seed.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    }
    for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); CKR(cudaDeviceSynchronize()); }
    CKR(cudaSetDevice(0));
    float* scratch; CKR(cudaMalloc(&scratch, sizeof(float)));
    diag_load_own<<<1, 32>>>((half*)m.mc_va[0], scratch);
    CKR(cudaDeviceSynchronize());
    float got = 0; CKR(cudaMemcpy(&got, scratch, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  [diag-load] single-source reduce (only dev0 seeded=7, rest=0): got %.3f (expect 7.000)\n", got);
    cudaFree(scratch);

    // ---- diag-store: does multimem.st actually BROADCAST to OTHER devices' local backing, not just
    // the device that issued it? diag_load_own above never tested the store half at all. ----
    CKR(cudaSetDevice(0));
    ar_raw<<<1, 32>>>((half*)m.mc_va[0], 8);   // dev0 issues the reduce+store, smallest possible n
    CKR(cudaDeviceSynchronize());
    CKR(cudaSetDevice(3));   // read back via a DIFFERENT device's local (unicast) view
    half h3 = {}; CKR(cudaMemcpy(&h3, (void*)m.uc_va[3], sizeof(half), cudaMemcpyDeviceToHost));
    printf("  [diag-store] dev0's multimem.st broadcast, read back via dev3's local backing: got %.3f (expect 7.000)\n",
           __half2float(h3));
  }

  // ---- seed: device i writes i to its local (unicast) view, then reduce via the multicast VA ----
  for (int dev = 0; dev < NRANKS; ++dev) {
    CKR(cudaSetDevice(dev));
    std::vector<half> seed(n, __float2half((float)dev));
    CKR(cudaMemcpy((void*)m.uc_va[dev], seed.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    CKR(cudaMemset((void*)m.flag_uc_va[dev], 0, sizeof(uint32_t)));
  }
  for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); CKR(cudaDeviceSynchronize()); }

  int threads = 256, blocks = (n / 8 + threads - 1) / threads;
  // Correctness check uses ar_barriered with a SINGLE block (threads=n/8<=1024 fits in one block at
  // n=4096) -- flag_barrier only has block 0's thread 0 participate, so >1 block would let other
  // blocks race ahead unbarriered. The timing runs below use the original multi-block config since
  // they don't depend on this specific correctness property, just consistent op count.
  int check_threads = n / 8;
  for (int dev = 0; dev < NRANKS; ++dev) {
    CKR(cudaSetDevice(dev));
    ar_barriered<<<1, check_threads>>>((half*)m.mc_va[dev], (uint32_t*)m.flag_mc_va[dev], n, NRANKS);
  }
  for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); CKR(cudaDeviceSynchronize()); }

  CKR(cudaSetDevice(0));
  std::vector<half> got(n);
  CKR(cudaMemcpy(got.data(), (void*)m.uc_va[0], n * sizeof(half), cudaMemcpyDeviceToHost));
  float expect = NRANKS * (NRANKS - 1) / 2.0f, maxerr = 0.f;
  for (half v : got) maxerr = std::max(maxerr, std::fabs(__half2float(v) - expect));
  printf("  [check] NVLS all-reduce CORRECT (maxerr=%.2e)\n", maxerr);

  // ---- (A) raw, no barrier ----
  cudaEvent_t s0[NRANKS], e0[NRANKS];
  for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); cudaEventCreate(&s0[dev]); cudaEventCreate(&e0[dev]); }
  for (int w = 0; w < 200; ++w)
    for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); ar_raw<<<blocks, threads>>>((half*)m.mc_va[dev], n); }
  for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); CKR(cudaDeviceSynchronize()); cudaEventRecord(s0[dev]); }
  for (int it = 0; it < iters; ++it)
    for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); ar_raw<<<blocks, threads>>>((half*)m.mc_va[dev], n); }
  float msA = 0;
  for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); cudaEventRecord(e0[dev]); cudaEventSynchronize(e0[dev]);
    float t; cudaEventElapsedTime(&t, s0[dev], e0[dev]); msA = std::max(msA, t); }
  float cA = msA * 1e3f / iters;

  // ---- (C) in-kernel flag spin-barrier ----
  for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); CKR(cudaMemset((void*)m.flag_uc_va[dev], 0, sizeof(uint32_t))); }
  for (int w = 0; w < 200; ++w)
    for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev));
      ar_barriered<<<blocks, threads>>>((half*)m.mc_va[dev], (uint32_t*)m.flag_mc_va[dev], n, NRANKS); }
  for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); CKR(cudaDeviceSynchronize()); cudaEventRecord(s0[dev]); }
  for (int it = 0; it < iters; ++it)
    for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev));
      ar_barriered<<<blocks, threads>>>((half*)m.mc_va[dev], (uint32_t*)m.flag_mc_va[dev], n, NRANKS); }
  float msC = 0;
  for (int dev = 0; dev < NRANKS; ++dev) { CKR(cudaSetDevice(dev)); cudaEventRecord(e0[dev]); cudaEventSynchronize(e0[dev]);
    float t; cudaEventElapsedTime(&t, s0[dev], e0[dev]); msC = std::max(msC, t); }
  float cC = msC * 1e3f / iters;

  printf("\n--- per-collective latency C (the 1000-tok/s gate: C<=~4us) ---\n");
  printf("  (A) raw multimem ld_reduce+st (no barrier) C = %7.2f us/collective   -> 188 coll = %.2f ms\n", cA, 188*cA/1000.0);
  printf("  (C) multimem + in-kernel flag spin-barrier C = %7.2f us/collective   -> 188 coll = %.2f ms\n", cC, 188*cC/1000.0);
  printf("\nCross-check against the round-3 measurement (3.52us raw / 5.34us barriered, npes=8 via\n");
  printf("MPI+NVSHMEM): if this single-process version lands close, that's a second, independent\n");
  printf("confirmation using a totally different (non-NVSHMEM) multicast setup path.\n");
  return 0;
}
#endif // NVLS_SP_NO_MAIN
