// nvls_ar.cu — single-process 8-GPU NVLS (multimem / NVLink-SHARP in-switch) all-reduce microbench.
// Implements mc_setup for kernels/nvls_allreduce.cu's skeleton: CUDA multicast objects + multimem PTX.
// Measures C (per-collective latency) for the 8KB B=1 payload vs the ~33us NCCL floor (E0).
// Build: nvcc -O3 -arch=sm_90a nvls_ar.cu -lcuda -o nvls_ar  ; run on 8xH100 NVSwitch.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NRANKS 8
#define HIDDEN 4096                 // halves -> 8 KB payload (the per-collective B=1 message)

#define DCK(x) do{ CUresult crst_=(x); if(crst_!=CUDA_SUCCESS){ const char* s_; cuGetErrorString(crst_,&s_); \
  printf("DRV ERR %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1);} }while(0)
#define RCK(x) do{ cudaError_t cest_=(x); if(cest_!=cudaSuccess){ \
  printf("RT  ERR %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(cest_)); exit(1);} }while(0)

__global__ void fill_half(half* p, int n, half v){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=v; }

// in-switch all-reduce: one multimem.ld_reduce (sum across all 8 GPUs) + multimem.st (broadcast).
__global__ void nvls_ar_half(half* __restrict__ mc, int n){
    int i = (blockIdx.x*blockDim.x + threadIdx.x) * 8;          // 8 halves (128-bit) per thread
    if (i >= n) return;
    uint32_t a,b,c,d;
    asm volatile("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "l"(mc+i));
    asm volatile("multimem.st.global.v4.f16x2 [%0], {%1,%2,%3,%4};"
                 :: "l"(mc+i),"r"(a),"r"(b),"r"(c),"r"(d) : "memory");
}

int main(){
    DCK(cuInit(0));
    CUdevice dev[NRANKS]; CUcontext ctx[NRANKS];
    for(int d=0; d<NRANKS; ++d){ DCK(cuDeviceGet(&dev[d], d)); DCK(cuDevicePrimaryCtxRetain(&ctx[d], dev[d])); }

    // ---- multicast object spanning all 8 GPUs ----
    CUmulticastObjectProp mcp; memset(&mcp,0,sizeof(mcp));
    mcp.numDevices  = NRANKS;
    mcp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    mcp.size        = (size_t)HIDDEN*sizeof(half);
    size_t mcgran=0; DCK(cuMulticastGetGranularity(&mcgran,&mcp,CU_MULTICAST_GRANULARITY_RECOMMENDED));
    size_t size = ((mcp.size + mcgran - 1)/mcgran)*mcgran;        // align up
    mcp.size = size;
    CUmemGenericAllocationHandle mc; DCK(cuMulticastCreate(&mc,&mcp));
    for(int d=0; d<NRANKS; ++d) DCK(cuMulticastAddDevice(mc, dev[d]));

    // ---- per-device physical alloc, bind to MC, and a local (unicast) mapping for init/validate ----
    CUmemGenericAllocationHandle phys[NRANKS]; CUdeviceptr uc[NRANKS];
    for(int d=0; d<NRANKS; ++d){
        RCK(cudaSetDevice(d));
        CUmemAllocationProp p; memset(&p,0,sizeof(p));
        p.type=CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type=CU_MEM_LOCATION_TYPE_DEVICE; p.location.id=d;
        p.requestedHandleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        size_t mgran=0; DCK(cuMemGetAllocationGranularity(&mgran,&p,CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        size_t msize=((size+mgran-1)/mgran)*mgran;
        DCK(cuMemCreate(&phys[d], msize, &p, 0));
        DCK(cuMulticastBindMem(mc, 0, phys[d], 0, size, 0));
        DCK(cuMemAddressReserve(&uc[d], size, 0, 0, 0));
        DCK(cuMemMap(uc[d], size, 0, phys[d], 0));
        CUmemAccessDesc ad; memset(&ad,0,sizeof(ad));
        ad.location.type=CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id=d; ad.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        DCK(cuMemSetAccess(uc[d], size, &ad, 1));
    }

    // ---- map the multicast handle into a VA accessible from all devices (the kernel uses this) ----
    RCK(cudaSetDevice(0));
    CUdeviceptr mc_va; DCK(cuMemAddressReserve(&mc_va, size, mcgran, 0, 0)); DCK(cuMemMap(mc_va, size, 0, mc, 0));
    std::vector<CUmemAccessDesc> ad(NRANKS);
    for(int d=0; d<NRANKS; ++d){ memset(&ad[d],0,sizeof(CUmemAccessDesc));
        ad[d].location.type=CU_MEM_LOCATION_TYPE_DEVICE; ad[d].location.id=d; ad[d].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
    DCK(cuMemSetAccess(mc_va, size, ad.data(), NRANKS));

    // ---- init: each GPU d's buffer = d (so the all-reduce SUM = 0+1+..+7 = 28) ----
    for(int d=0; d<NRANKS; ++d){ RCK(cudaSetDevice(d)); fill_half<<<(HIDDEN+255)/256,256>>>((half*)uc[d], HIDDEN, __float2half((float)d)); }
    for(int d=0; d<NRANKS; ++d){ RCK(cudaSetDevice(d)); RCK(cudaDeviceSynchronize()); }

    // ---- run one all-reduce on device 0 (the switch reduces across all 8) + validate ----
    RCK(cudaSetDevice(0));
    int threads=256, blocks=((HIDDEN/8)+threads-1)/threads;
    nvls_ar_half<<<blocks,threads>>>((half*)mc_va, HIDDEN); RCK(cudaDeviceSynchronize());
    bool ok=true; for(int d=0; d<NRANKS && ok; ++d){ RCK(cudaSetDevice(d));
        std::vector<half> h(HIDDEN); RCK(cudaMemcpy(h.data(),(void*)uc[d],HIDDEN*sizeof(half),cudaMemcpyDeviceToHost));
        for(int i=0;i<HIDDEN;++i){ float v=__half2float(h[i]); if(v<27.5f||v>28.5f){ printf("VALIDATE FAIL gpu%d [%d]=%.1f (want 28)\n",d,i,v); ok=false; break; } } }
    printf("validate: %s (every elt == sum(0..7)=28 on all 8 GPUs)\n", ok?"PASS":"FAIL");

    // ---- timed microbench: C = mean us per in-switch all-reduce ----
    RCK(cudaSetDevice(0));
    cudaEvent_t s,e; RCK(cudaEventCreate(&s)); RCK(cudaEventCreate(&e));
    for(int w=0; w<200; ++w) nvls_ar_half<<<blocks,threads>>>((half*)mc_va, HIDDEN);
    RCK(cudaDeviceSynchronize());
    const int IT=1000; RCK(cudaEventRecord(s));
    for(int it=0; it<IT; ++it) nvls_ar_half<<<blocks,threads>>>((half*)mc_va, HIDDEN);
    RCK(cudaEventRecord(e)); RCK(cudaEventSynchronize(e));
    float ms=0; RCK(cudaEventElapsedTime(&ms,s,e)); double C=ms*1e3/IT;
    printf("\nNVLS in-switch all-reduce: C = %.2f us/collective (8KB, 8 GPUs)   [NCCL E0 baseline ~33us]\n", C);
    printf("  -> 188 collectives x C = %.2f ms/token comms.   1000-tok/s gate: C <= 4us.\n", 188*C/1e3);
    printf("  -> verdict: %s\n", C<=4.0?"C<=4us -> FULL lossless comms-hide possible (1000 path ON)":
                                 C<=8.0?"~8us -> partial hide; pair with spec/int4":"still >8us -> needs deferred-overlap/tuning");
    return 0;
}
