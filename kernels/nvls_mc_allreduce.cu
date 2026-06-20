// nvls_mc_allreduce.cu — NVLS (NVLink-SHARP in-switch) all-reduce for the B=1 residual [HIDDEN] fp32,
// using NVSHMEM's already-set-up multicast team (nvshmemx_mc_ptr) instead of raw cuMulticast wiring.
//
// THE make-or-break primitive for the barrier-free megakernel (docs/path-to-1000.md Stage 3):
//   multimem.ld_reduce.add  — ONE instruction loads+sums a 128-bit slice across ALL 8 GPUs IN THE SWITCH.
//   multimem.st             — ONE instruction broadcasts the result back to all 8 GPUs' buffers.
// The reduction is done by the NVSwitch (SHARP), not by per-PE local sums + puts.  At B=1 8-16KB the
// data movement is trivial; what matters is the per-collective LATENCY C and how cheap the surrounding
// barrier can be made.  This file measures C three ways:
//   (1) multimem ops ONLY (no barrier)        — the raw in-switch reduce cost
//   (2) multimem + nvshmem device barrier      — a correct standalone all-reduce
//   (3) multimem + a multimem flag spin-barrier — the cheap barrier the megakernel will use in-kernel
// and VALIDATES correctness (each PE seeds its rank; after AR every elt == sum_{p<npes} contribution).
//
// nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, buf) returns the multicast VA for a symmetric `buf` IF the system
// is NVLS-capable (H100 + NVSwitch).  Returns NULL otherwise -> we report and bail (no silent fallback).
//
// BUILD (8xH100 box):
//   nvcc -arch=sm_90a -O3 -rdc=true -I kernels/ -I $NVS_INC -I $MPI_INC kernels/nvls_mc_allreduce.cu \
//     $NVS_LIB/libnvshmem_device.a -L $NVS_LIB -lnvshmem_host -L $MPI_LIB -lmpi -Xlinker -rpath,$NVS_LIB -o /tmp/nvls
// RUN:
//   LD_LIBRARY_PATH=$NVS_LIB:$MPI_LIB NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_BOOTSTRAP=MPI \
//     mpirun -np 8 --allow-run-as-root /tmp/nvls
// =================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); nvshmem_global_exit(1);} } while(0)

#ifndef HIDDEN
#define HIDDEN 4096
#endif

// ---- multimem device helpers (fp32, 4-wide = 128-bit coalesced in-switch reduce) -----------------
// multimem.ld_reduce.global.add.v4.f32 : sum 4 floats at [mc+off] across all bound GPUs (one switch op).
static __device__ __forceinline__ void mm_ld_reduce_f32x4(const float* mc, float& a, float& b, float& c, float& d) {
  asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "l"(mc) : "memory");
}
static __device__ __forceinline__ void mm_st_f32x4(float* mc, float a, float b, float c, float d) {
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :: "l"(mc), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

// ---- (A) raw in-switch reduce: each thread reduces+broadcasts its 4-float slice (NO barrier) -------
// CRITICAL: each element must be reduced by EXACTLY ONE PE — else PE q re-reads the sum PE p already
// broadcast and re-sums it (exponential blowup).  Partition: PE `pe` owns elements [pe*chunk,(pe+1)*chunk).
// multimem.ld_reduce reads ALL PEs' copies of that element; multimem.st broadcasts the sum to ALL copies.
__global__ void nvls_ar_raw(float* mc, int n, int pe, int npes) {
  const int chunk = ((n / 4) + npes - 1) / npes * 4;     // elements per PE, 4-aligned
  const int lo = pe * chunk, hi = min(n, lo + chunk);
  for (int i = lo + (blockIdx.x*blockDim.x + threadIdx.x)*4; i < hi; i += gridDim.x*blockDim.x*4) {
    float a,b,c,d; mm_ld_reduce_f32x4(mc + i, a,b,c,d); mm_st_f32x4(mc + i, a,b,c,d);
  }
}

// ---- (C) all-reduce with an in-kernel MULTIMEM flag spin-barrier (the megakernel's cheap barrier) --
// A device-side all-PE barrier built on multimem: PE arrives by add-ing 1 to a multicast counter (lands
// on all PEs in-switch), then spins until the counter (read via multimem reduce = global arrival sum)
// reaches npes * generation.  No host round-trip, no nvshmem collective.  One block drives it.
__device__ unsigned g_bar_gen = 0;
__global__ void nvls_ar_barrier(float* mc, int n, unsigned* mc_flag, int pe, int npes) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // phase 1: reduce+broadcast ONLY this PE's disjoint slice (each element reduced exactly once)
  const int chunk = ((n / 4) + npes - 1) / npes * 4;
  const int lo = pe * chunk, hi = min(n, lo + chunk);
  for (int i = lo + tid*4; i < hi; i += gridDim.x*blockDim.x*4) {
    float a,b,c,d; mm_ld_reduce_f32x4(mc + i, a,b,c,d); mm_st_f32x4(mc + i, a,b,c,d);
  }
  // phase 2: one thread/PE arrives at the multimem flag-barrier and waits for all npes arrivals, so
  // every PE's st on its slice is visible before any PE reads the full buffer.  No host round-trip.
  if (tid == 0) {
    g_bar_gen++;
    unsigned target = g_bar_gen * (unsigned)npes;
    __threadfence_system();                              // our st's are globally visible before arrive
    asm volatile("multimem.red.global.add.u32 [%0], 1;" :: "l"(mc_flag) : "memory");
    unsigned got = 0;
    do {
      asm volatile("multimem.ld_reduce.global.add.u32 %0, [%1];" : "=r"(got) : "l"(mc_flag) : "memory");
    } while (got < target);
  }
}

int main(int argc, char** argv) {
  const int N = (argc > 1) ? atoi(argv[1]) : HIDDEN;  // elements (fp32)
  const int ITERS = (argc > 2) ? atoi(argv[2]) : 1000;
  const int WARM = 200;

  nvshmem_init();
  const int pe = nvshmem_my_pe(), npes = nvshmem_n_pes();
  int ndev=0; CK(cudaGetDeviceCount(&ndev)); CK(cudaSetDevice(pe % ndev));

  // symmetric payload + a symmetric u32 flag for the multimem barrier
  float* buf = (float*)nvshmem_malloc(sizeof(float) * N);
  unsigned* flag = (unsigned*)nvshmem_malloc(sizeof(unsigned));
  if (!buf || !flag) { printf("PE %d: nvshmem_malloc failed\n", pe); nvshmem_global_exit(2); }
  CK(cudaMemset(flag, 0, sizeof(unsigned)));

  // the multicast VAs (NULL if this system has no NVLS/NVSwitch multicast)
  float*    mc      = (float*)   nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, buf);
  unsigned* mc_flag = (unsigned*)nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, flag);
  if (pe == 0) {
    printf("== NVLS multimem all-reduce: N=%d fp32 (%d KB), npes=%d ==\n", N, N*4/1024, npes);
    printf("nvshmemx_mc_ptr(buf)=%p  mc_flag=%p\n", (void*)mc, (void*)mc_flag);
  }
  if (!mc || !mc_flag) {
    if (pe == 0) printf("  NVLS NOT AVAILABLE on this system (mc_ptr NULL). multimem all-reduce impossible here.\n");
    nvshmem_finalize(); return 0;
  }

  const int threads = 256;
  const int blocks_raw = (N/4 + threads - 1) / threads;

  // ---- correctness: PE p seeds buf[i] = p*1000 + (i%100); after AR every elt == sum_p(p*1000+(i%100)) ----
  {
    std::vector<float> h(N);
    for (int i=0;i<N;i++) h[i] = (float)pe*1000.0f + (float)(i%100);
    CK(cudaMemcpy(buf, h.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
    CK(cudaMemset(flag, 0, sizeof(unsigned)));
    nvshmem_barrier_all();
    nvls_ar_barrier<<<blocks_raw, threads>>>(mc, N, mc_flag, pe, npes);
    CK(cudaDeviceSynchronize()); nvshmem_barrier_all();
    std::vector<float> got(N); CK(cudaMemcpy(got.data(), buf, sizeof(float)*N, cudaMemcpyDeviceToHost));
    double maxerr=0; int bad=-1;
    for (int i=0;i<N;i++){ double ref=0; for(int p=0;p<npes;p++) ref += (double)p*1000.0+(double)(i%100);
      double e=fabs((double)got[i]-ref); if(e>maxerr){maxerr=e; if(e>1e-1)bad=i;} }
    if (pe==0) {
      if (bad>=0) printf("  [check] NVLS all-reduce MISMATCH at i=%d got=%g maxerr=%g\n", bad, got[bad], maxerr);
      else        printf("  [check] NVLS all-reduce CORRECT (maxerr=%.2e)\n", maxerr);
    }
  }

  cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  auto bench = [&](const char* name, int mode){
    // mode 0: raw multimem only ; 1: multimem + nvshmem device barrier ; 2: multimem flag spin-barrier
    for (int w=0; w<WARM; ++w) {
      if (mode==2) nvls_ar_barrier<<<blocks_raw,threads>>>(mc,N,mc_flag,pe,npes);
      else { nvls_ar_raw<<<blocks_raw,threads>>>(mc,N,pe,npes); if (mode==1){ CK(cudaDeviceSynchronize()); nvshmem_barrier_all(); } }
    }
    CK(cudaDeviceSynchronize()); nvshmem_barrier_all();
    CK(cudaEventRecord(t0));
    for (int it=0; it<ITERS; ++it) {
      if (mode==2) nvls_ar_barrier<<<blocks_raw,threads>>>(mc,N,mc_flag,pe,npes);
      else { nvls_ar_raw<<<blocks_raw,threads>>>(mc,N,pe,npes); if (mode==1){ CK(cudaDeviceSynchronize()); nvshmem_barrier_all(); } }
    }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms=0; CK(cudaEventElapsedTime(&ms,t0,t1));
    if (pe==0) printf("  %-42s C = %6.2f us/collective   -> 188 coll = %5.2f ms\n",
                      name, ms*1e3/ITERS, 188.0*ms*1e3/ITERS/1e3);
  };
  if (pe==0) printf("\n--- per-collective latency C (the 1000-tok/s gate: C<=~4us) ---\n");
  bench("(A) raw multimem ld_reduce+st (no barrier)", 0);
  bench("(C) multimem + in-kernel flag spin-barrier", 2);
  bench("(B) multimem + nvshmem host barrier",        1);

  if (pe==0) {
    printf("\nbaselines this session: NCCL ring AR ~35us ; NVSHMEM put+barrier ~17us (isolated), 52us in-loop.\n");
    printf("if (C) <= ~4us, the barrier-free megakernel's 188 ARs cost <0.8ms -> 1000 tok/s is reachable.\n");
  }
  nvshmem_finalize();
  return 0;
}
