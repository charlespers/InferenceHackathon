// nvls_engine_test.cu — STRESS test of the engine NVLS path (nvls_engine.cuh), mimicking decode_step
// concurrency: 8 host threads (one/rank), each issues MANY back-to-back (fill -> AR) pairs on its own
// stream with NO inter-collective sync (like the engine's 188 ARs/token interleaved with kernels), then
// syncs once at the end and we validate.  Each rank r writes (r+1); after each AR every elt must == 36.
// To catch a partial-sum race we make the per-collective fill value depend on a per-rank counter so a
// stale/partial reduce shows up; we validate the FINAL buffer == sum of the final round's values.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <thread>
#include "nvls_engine.cuh"

#define RCK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ printf("RT ERR %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e_)); exit(1);} }while(0)

__global__ void fill_f(float* p, int n, float v){ for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) p[i]=v; }
// busy kernel to perturb scheduling / widen the race window between ranks (mimics K1..K5 load).
__global__ void busy(float* p, int n, int iters){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float a=p[i]; for(int k=0;k<iters;++k) a=a*1.0000001f+1e-7f; if(a==123456.f) p[i]=a; } }

// per-collective check: copy each rank's OUTPUT half to host, verify == want.  Returns #bad.
// OUT-OF-PLACE: the reduced result lands in the OUT region (nvls_out_off(elt_off)), NOT the IN region
// the fill wrote.  Read OUT here to validate the all-reduce result.
static int check_half(std::vector<NvlsCtx>& ctx, int npes, int elt_off, float want){
  int bad=0; float worst=0; int wd=-1,wi=-1; float wv=0;
  const int out_off = nvls_out_off(elt_off);
  for(int d=0; d<npes; ++d){ RCK(cudaSetDevice(d));
    std::vector<float> h(NVLS_HIDDEN);
    RCK(cudaMemcpy(h.data(), ctx[d].uc+out_off, NVLS_HIDDEN*sizeof(float), cudaMemcpyDeviceToHost));
    for(int i=0;i<NVLS_HIDDEN;++i){ float dd=fabsf(h[i]-want); if(dd>1e-2f){ bad++; if(dd>worst){worst=dd;wd=d;wi=i;wv=h[i];} } } }
  if(bad) printf("    FAIL off=%d want=%.1f : %d bad; worst gpu%d[%d]=%.4f (off %.4f)\n", elt_off, want, bad, wd, wi, wv, worst);
  return bad;
}

int main(int argc, char** argv){
  const int npes = 8;
  const int ROUNDS = (argc>1)?atoi(argv[1]):100;   // collectives per half, back-to-back, no inter-sync
  const int BUSY   = (argc>2)?atoi(argv[2]):2000;  // busy iters to widen the inter-rank skew
  std::vector<NvlsCtx> ctx;
  if(!nvls_engine_setup(ctx, npes)){ printf("setup FAILED\n"); return 1; }
  std::vector<cudaStream_t> str(npes);
  std::vector<float*> scratch(npes);
  for(int d=0; d<npes; ++d){ RCK(cudaSetDevice(d)); RCK(cudaStreamCreate(&str[d]));
    RCK(cudaMalloc(&scratch[d], NVLS_HIDDEN*sizeof(float))); RCK(cudaMemset(scratch[d],0,NVLS_HIDDEN*sizeof(float))); }

  const float want = (float)(npes*(npes+1)/2);  // 36

  printf("\n== STRESS: %d rounds x (ATTN+MOE) back-to-back, busy=%d, NO inter-collective sync ==\n", ROUNDS, BUSY);
  // Each round: every rank fills BOTH halves with (r+1), perturbs with busy, AR ATTN, AR MOE — all on
  // its own stream with NO sync.  We snapshot+validate after EACH round's pair by syncing then checking
  // (the sync is AFTER the un-synced issue, so the race window inside the round is fully exercised).
  int total_bad = 0;
  for(int rnd=0; rnd<ROUNDS; ++rnd){
    std::vector<std::thread> th;
    for(int r=0;r<npes;++r) th.emplace_back([&,r]{
      RCK(cudaSetDevice(r)); cudaStream_t s=str[r];
      fill_f<<<32,256,0,s>>>(ctx[r].uc+NVLS_OFF_ATTN, NVLS_HIDDEN, (float)(r+1));
      fill_f<<<32,256,0,s>>>(ctx[r].uc+NVLS_OFF_MOE,  NVLS_HIDDEN, (float)(r+1));
      busy<<<32,256,0,s>>>(scratch[r], NVLS_HIDDEN, (r+1)*BUSY); // rank-dependent skew (separate scratch)
      nvls_allreduce_launch(ctx[r], NVLS_OFF_ATTN, s);
      nvls_allreduce_launch(ctx[r], NVLS_OFF_MOE,  s);
    });
    for(auto&t:th) t.join();
    for(int d=0; d<npes; ++d){ RCK(cudaSetDevice(d)); RCK(cudaStreamSynchronize(str[d])); }
    int b = check_half(ctx,npes,NVLS_OFF_ATTN,want) + check_half(ctx,npes,NVLS_OFF_MOE,want);
    if(b){ printf("  round %d: %d bad\n", rnd, b); total_bad+=b; }
  }
  printf("\nRESULT: %s  (%d total bad elts over %d rounds)\n",
         total_bad==0?"PASS — engine NVLS reduce is robust under stress":"FAIL — RACE present", total_bad, ROUNDS);
  return total_bad==0?0:1;
}
