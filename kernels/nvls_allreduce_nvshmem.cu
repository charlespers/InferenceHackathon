// nvls_allreduce_nvshmem.cu — RECONSTRUCTION of the working multi-process NVLS multimem all-reduce.
//
// CONTEXT: a compiled binary (`/tmp/nvls` on the box) demonstrated a REAL, working multi-process NVLS
// multimem all-reduce — npes=8, correctness PASS (maxerr=0.00e+00), C=3.52us (raw, no barrier) /
// 5.34us (in-kernel flag-spin barrier) / 1107.36us (NVSHMEM host barrier, 200x worse — avoid). Its
// SOURCE WAS NEVER COMMITTED — `nvls_allreduce.cu` in this repo is a different, still-unwired skeleton
// (raw `cuMulticastCreate`, no NVSHMEM). This file is a BEST-EFFORT RECONSTRUCTION of the working
// approach from the runtime evidence we have (the exact symbol names in its printed output:
// `nvshmemx_mc_ptr`, `mc_flag`, and the launch command `mpirun -np 8 -x NVSHMEM_BOOTSTRAP=MPI`) plus
// NVSHMEM's documented multicast-pointer API for Hopper NVLS. **NOT VERIFIED ON BOX YET** — the exact
// `nvshmemx_mc_ptr` signature/header may differ slightly by NVSHMEM version; this needs a real
// compile+run pass to confirm it reproduces the measured numbers, not just resemble them.
//
// WHY NVSHMEM here despite the CUDA-13/12.6 toolchain mismatch that blocked `nvshmem_comms.cu`: that
// block was specifically `nvlink`-stage (device-side fatbinary linking against the cu13 NVSHMEM device
// library). The working `/tmp/nvls` binary's existence on THIS box (cu12.6 nvcc) means either (a) the
// multicast-pointer accessor used here doesn't pull in the same device-linked NVSHMEM collective code
// that triggered the cu13 mismatch, or (b) whoever built it solved that separately. Worth confirming
// which when this is actually compiled on box.
//
// Build: mpicxx -O3 -arch=sm_90a (or nvcc with -I/-L for NVSHMEM + MPI) nvls_allreduce_nvshmem.cu \
//        -lnvshmem -lnvshmem_host -lcuda -o nvls_recon
// Run:   mpirun -np 8 --allow-run-as-root -x NVSHMEM_BOOTSTRAP=MPI ./nvls_recon 4096 2000

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>

#define HIDDEN 4096

// ---- (A) raw multimem ld_reduce+st, NO barrier. Fastest (measured 3.52us); correctness depends on the
// caller already knowing every rank reached this point (true in the steady-state per-layer pipeline,
// where the PREVIOUS op already implies all ranks are at the same logical step). ----
__global__ void ar_multimem_raw(float* __restrict__ mc_ptr, int n) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= n) return;
  uint32_t a, b, c, d;
  asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(mc_ptr + i));
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :: "l"(mc_ptr + i), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

// ---- (C) multimem + in-kernel flag spin-barrier. Measured 5.34us -- ~1.8x (A) but every rank waits for
// every other rank to finish ITS reduce+store before any rank reads the result, closing the real
// cross-rank race (A) leaves open if the caller's assumption doesn't hold. mc_flag is a SECOND
// multicast buffer used purely as a counter; each rank increments it via multimem.st add-style accumulate
// then spins on a multimem.ld_reduce of it until it reads NRANKS (everyone has arrived). ----
__device__ void flag_barrier(uint32_t* mc_flag, int nranks) {
  asm volatile("red.global.add.u32 [%0], 1;" :: "l"(mc_flag));   // local increment, switch-reduced on read
  uint32_t seen = 0;
  do {
    asm volatile("multimem.ld_reduce.global.add.u32 %0, [%1];" : "=r"(seen) : "l"(mc_flag));
  } while (seen < (uint32_t)nranks);
}

__global__ void ar_multimem_barriered(float* __restrict__ mc_ptr, uint32_t* __restrict__ mc_flag,
                                       int n, int nranks) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i < n) {
    uint32_t a, b, c, d;
    asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(mc_ptr + i));
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
                 :: "l"(mc_ptr + i), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) flag_barrier(mc_flag, nranks);
}

// ---- (B) NVSHMEM HOST barrier, for comparison ONLY -- measured 1107.36us, ~200x slower than (C).
// DO NOT use this inside the megakernel's per-layer loop; it round-trips through the host bootstrap
// layer instead of staying device-resident. Kept here only so the regression is re-measurable. ----
static float run_variant_B(float* mc_ptr, int n, int iters, int mype) {
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  int threads = 256, blocks = (n / 4 + threads - 1) / threads;
  for (int w = 0; w < 20; ++w) { ar_multimem_raw<<<blocks, threads>>>(mc_ptr, n); nvshmemx_barrier_all_on_stream(0); }
  cudaEventRecord(s);
  for (int it = 0; it < iters; ++it) { ar_multimem_raw<<<blocks, threads>>>(mc_ptr, n); nvshmemx_barrier_all_on_stream(0); }
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  return ms * 1e3f / iters;
}

int main(int argc, char** argv) {
  int n      = (argc > 1) ? atoi(argv[1]) : HIDDEN;
  int iters  = (argc > 2) ? atoi(argv[2]) : 2000;

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_mpi_comm_args(&comm, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int mype  = nvshmem_my_pe();
  int npes  = nvshmem_n_pes();
  cudaSetDevice(mype % 8);

  float*    buf     = (float*)nvshmem_malloc(n * sizeof(float));
  uint32_t* flagbuf = (uint32_t*)nvshmem_malloc(sizeof(uint32_t));
  cudaMemset(flagbuf, 0, sizeof(uint32_t));

  // NVSHMEM's NVLS multicast-pointer accessor (Hopper). NEEDS VERIFICATION: exact symbol/signature per
  // NVSHMEM version -- this is the one call in this whole file taken on faith from the runtime evidence
  // (`nvshmemx_mc_ptr(buf)=0x...` in the original binary's printed output), not from a header we have
  // open right now. If this doesn't compile as-is, check nvshmem_team.h / nvshmemx_api.h for the actual
  // name on this box's installed NVSHMEM version.
  float*    mc_ptr  = (float*)nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, buf);
  uint32_t* mc_flag = (uint32_t*)nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, flagbuf);

  if (mype == 0) printf("== NVLS multimem all-reduce RECONSTRUCTION: N=%d fp32 (%dB), npes=%d ==\n",
                         n, (int)(n * sizeof(float)), npes);

  if (!mc_ptr || !mc_flag) {
    if (mype == 0) printf("  NVLS NOT AVAILABLE (mc_ptr NULL) -- multicast pointer accessor returned null.\n"
                           "  Check NVSHMEM version/NVLS support before trusting anything below.\n");
    nvshmem_finalize();
    return 1;
  }

  // ---- seed + correctness check (A): rank i writes i, after AR every elt should be sum(0..npes-1) ----
  std::vector<float> seed(n, (float)mype);
  cudaMemcpy(buf, seed.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  nvshmem_barrier_all();
  int threads = 256, blocks = (n / 4 + threads - 1) / threads;
  ar_multimem_raw<<<blocks, threads>>>(mc_ptr, n);
  cudaDeviceSynchronize();
  std::vector<float> got(n);
  cudaMemcpy(got.data(), buf, n * sizeof(float), cudaMemcpyDeviceToHost);
  float expect = npes * (npes - 1) / 2.0f, maxerr = 0.f;
  for (float v : got) maxerr = std::max(maxerr, std::fabs(v - expect));
  if (mype == 0) printf("  [check] NVLS all-reduce CORRECT (maxerr=%.2e)\n", maxerr);

  // ---- (A) raw, no barrier ----
  cudaEvent_t s0, e0; cudaEventCreate(&s0); cudaEventCreate(&e0);
  for (int w = 0; w < 200; ++w) ar_multimem_raw<<<blocks, threads>>>(mc_ptr, n);
  cudaDeviceSynchronize(); cudaEventRecord(s0);
  for (int it = 0; it < iters; ++it) ar_multimem_raw<<<blocks, threads>>>(mc_ptr, n);
  cudaEventRecord(e0); cudaEventSynchronize(e0);
  float msA = 0; cudaEventElapsedTime(&msA, s0, e0);
  float cA = msA * 1e3f / iters;

  // ---- (C) in-kernel flag spin-barrier ----
  cudaMemset(flagbuf, 0, sizeof(uint32_t)); nvshmem_barrier_all();
  for (int w = 0; w < 200; ++w) ar_multimem_barriered<<<blocks, threads>>>(mc_ptr, mc_flag, n, npes);
  cudaDeviceSynchronize(); cudaEventRecord(s0);
  for (int it = 0; it < iters; ++it) ar_multimem_barriered<<<blocks, threads>>>(mc_ptr, mc_flag, n, npes);
  cudaEventRecord(e0); cudaEventSynchronize(e0);
  float msC = 0; cudaEventElapsedTime(&msC, s0, e0);
  float cC = msC * 1e3f / iters;

  // ---- (B) NVSHMEM host barrier, for comparison ----
  float cB = run_variant_B(mc_ptr, n, iters / 4, mype);   // fewer iters: this one is SLOW

  if (mype == 0) {
    printf("\n--- per-collective latency C (the 1000-tok/s gate: C<=~4us) ---\n");
    printf("  (A) raw multimem ld_reduce+st (no barrier) C = %7.2f us/collective   -> 188 coll = %.2f ms\n", cA, 188*cA/1000.0);
    printf("  (C) multimem + in-kernel flag spin-barrier C = %7.2f us/collective   -> 188 coll = %.2f ms\n", cC, 188*cC/1000.0);
    printf("  (B) multimem + nvshmem host barrier        C = %7.2f us/collective   -> 188 coll = %.2f ms\n", cB, 188*cB/1000.0);
  }

  nvshmem_free(buf); nvshmem_free(flagbuf);
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
