// nvls_precision_test.cu — is the in-switch multimem.ld_reduce.add.f32 NUMERICALLY exact, or lossy?
// The engine stress test uses integer values (1..8, sum=36) which are exact in ANY precision, so it
// only validates the barrier/no-race, NOT the reduction precision.  The decode's partials are
// FRACTIONAL O(1) fp32 — if the switch accumulates in reduced precision, integers pass but fractions
// err ~1e-2 (exactly the decode gate's 1.75e-2 on BOTH ARs).  This probe writes fractional O(1) data
// per rank, all-reduces via the EXACT engine path, and compares to a double-precision reference.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <thread>
#include "nvls_engine.cuh"
#define RCK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ printf("RT ERR %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); exit(1);} }while(0)

// rank r's element i: fractional, O(1), NOT exactly representable when summed across ranks.
__device__ __host__ inline float val(int r, int i){ return 0.1f * (float)(r+1) * cosf(0.013f*(float)i + 0.7f*(float)r); }
__global__ void fill_frac(float* p, int n, int r){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) p[i]=val(r,i); }

int main(){
  const int npes=8;
  std::vector<NvlsCtx> ctx;
  if(!nvls_engine_setup(ctx, npes)){ printf("setup FAILED\n"); return 1; }
  std::vector<cudaStream_t> str(npes);
  for(int d=0; d<npes; ++d){ RCK(cudaSetDevice(d)); RCK(cudaStreamCreate(&str[d])); }

  // each rank writes fractional O(1) data into the ATTN half, then all-reduces (concurrent).
  std::vector<std::thread> th;
  for(int r=0;r<npes;++r) th.emplace_back([&,r]{
    RCK(cudaSetDevice(r)); cudaStream_t s=str[r];
    fill_frac<<<(NVLS_HIDDEN+255)/256,256,0,s>>>(ctx[r].uc + NVLS_OFF_ATTN, NVLS_HIDDEN, r);
    nvls_allreduce_launch(ctx[r], NVLS_OFF_ATTN, s);
    RCK(cudaStreamSynchronize(s));
  });
  for(auto&t:th) t.join();

  // double-precision reference sum per element; compare to the reduced result on rank 0's uc.
  std::vector<float> got(NVLS_HIDDEN);
  RCK(cudaSetDevice(0));
  RCK(cudaMemcpy(got.data(), ctx[0].uc + NVLS_OFF_ATTN, NVLS_HIDDEN*sizeof(float), cudaMemcpyDeviceToHost));
  double maxerr=0, refmax=0; int worst=-1;
  for(int i=0;i<NVLS_HIDDEN;++i){
    double ref=0; for(int r=0;r<npes;++r) ref += (double)val(r,i);
    double e=fabs((double)got[i]-ref); if(e>maxerr){maxerr=e; worst=i;}
    if(fabs(ref)>refmax) refmax=fabs(ref);
  }
  printf("\n== multimem.ld_reduce.add.f32 PRECISION on fractional O(1) data ==\n");
  printf("  ref max|.| = %.4f   max|got-ref| = %.3e  (worst elt %d: got=%.6f ref=%.6f)\n",
         refmax, maxerr, worst, got[worst], [&]{double r=0;for(int k=0;k<npes;++k)r+=(double)val(k,worst);return r;}());
  printf("  VERDICT: %s\n", maxerr < 1e-4 ? "EXACT f32 reduce (bug is NOT precision -> integration/visibility)"
                          : maxerr < 5e-2 ? "LOSSY in-switch reduce (REDUCED-PRECISION accumulate) -> this IS the 1.75e-2 root cause"
                                          : "WILDLY off (missing contributions -> race/addressing)");
  return 0;
}
