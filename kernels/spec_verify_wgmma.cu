// spec_verify_wgmma.cu — does fp8 TENSOR-CORE batched verify amortize the weight read?
//
// THE make-or-break for B=1 spec → 500 tok/s.  The CUDA-core verify (spec_verify_bench.cu) was
// ALU-bound: T=3 rows cost 2.66× T=1 (NO amortization → spec is a net loss).  A tensor-core expert
// GEMM (prefill_wgmma::run_expert, fp8→fp16 HMMA) does the M-row matmul on the tensor cores, so the
// per-row arithmetic is ~free and the cost should stay WEIGHT-BOUND: reading the expert's 18.9 MB of
// gate/up/down ONCE serves all T draft rows.  If timing is ~FLAT in T, spec verify amortizes and
// B=1 spec multiplies throughput; if it grows ~linearly, tensor cores didn't fix it either.
//
// We time ONE expert's gate+up+down (the EP per-GPU unit — each of 8 GPUs owns 16 experts; at the
// route-aware small union ~1 fires/GPU and batches all T draft rows) at T ∈ {1,2,4,8}.
//
// BUILD: nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -DPREFILL_WGMMA_NO_MAIN \
//          kernels/spec_verify_wgmma.cu -o /tmp/svw
// RUN:   CUDA_VISIBLE_DEVICES=0 /tmp/svw
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

#define PREFILL_WGMMA_NO_MAIN 1
#include "prefill_wgmma.cu"   // brings in prefill_wgmma::run_expert + helpers

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

template<class T> static T* devv(const std::vector<T>& h){ T* d; CK(cudaMalloc(&d,h.size()*sizeof(T))); CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice)); return d; }

int main(){
  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  const int H=HIDDEN, I=MOE_INTER;
  // one expert's weights (the per-GPU EP unit): gate[I,H] + up[I,H] + down[H,I] fp8
  const double expert_bytes = (double)2*I*H + (double)H*I;   // fp8 bytes read per verify
  printf("=== spec_verify_wgmma: tensor-core batched verify amortization (1 expert) ===\n");
  printf("device %s  HBM peak=%.0f GB/s  expert weight read=%.1f MB\n",
         prop.name, 3350.0, expert_bytes/1e6);

  std::vector<fp8> Wg((size_t)I*H,fp8(0.02f)), Wu((size_t)I*H,fp8(0.02f)), Wd((size_t)H*I,fp8(0.02f));
  std::vector<float> Sg(I,0.02f),Su(I,0.02f),Sd(H,0.02f);
  fp8 *dWg=devv(Wg),*dWu=devv(Wu),*dWd=devv(Wd);
  float *dSg=devv(Sg),*dSu=devv(Su),*dSd=devv(Sd);

  printf("\n  %4s %12s %12s %10s %12s\n","T","ms/expert","GB/s","%HBM","ms/T=1");
  double t1=0;
  for (int T : {1,2,4,8}) {
    std::vector<float> X((size_t)T*H,0.01f), rw(T,0.125f); std::vector<int> tid(T);
    for(int i=0;i<T;++i) tid[i]=i;
    float* dX=devv(X); float* drw=devv(rw); int* dtid=devv(tid);
    float* dres; CK(cudaMalloc(&dres,(size_t)T*H*sizeof(float))); CK(cudaMemset(dres,0,(size_t)T*H*sizeof(float)));
    float *dXe,*dH,*dXeAct,*dHAct; fp8 *dXeQ,*dHQ;
    CK(cudaMalloc(&dXe,(size_t)T*H*sizeof(float)));   CK(cudaMalloc(&dH,(size_t)T*I*sizeof(float)));
    CK(cudaMalloc(&dXeQ,(size_t)T*H*sizeof(fp8)));    CK(cudaMalloc(&dHQ,(size_t)T*I*sizeof(fp8)));
    CK(cudaMalloc(&dXeAct,(size_t)T*sizeof(float)));  CK(cudaMalloc(&dHAct,(size_t)T*sizeof(float)));
    auto run=[&]{ prefill_wgmma::run_expert(dX,dtid,drw,dWg,dSg,dWu,dSu,dWd,dSd,dXe,dXeQ,dXeAct,dH,dHQ,dHAct,dres,T); };
    for(int i=0;i<10;++i) run(); CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    const int it=100; CK(cudaEventRecord(a));
    for(int i=0;i<it;++i) run();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); ms/=it;
    if(T==1) t1=ms;
    double gbps=expert_bytes/(ms*1e-3)/1e9;
    printf("  %4d %12.4f %12.1f %9.1f%% %12.2f\n", T, ms, gbps, 100.0*gbps/3350.0, ms/t1);
    cudaFree(dX);cudaFree(drw);cudaFree(dtid);cudaFree(dres);
    cudaFree(dXe);cudaFree(dH);cudaFree(dXeQ);cudaFree(dHQ);cudaFree(dXeAct);cudaFree(dHAct);
    cudaEventDestroy(a);cudaEventDestroy(b);
  }
  printf("\nINTERPRETATION: ms/T=1 ~1.0 across T => WEIGHT-BOUND (amortized) => spec verify multiplies B=1.\n");
  printf("                ms/T=1 ~T => still compute-bound (tensor cores didn't fix it).\n");

  // ---- GROUPED GEMM: occupancy fix — one big-N GEMM over the expert UNION (U experts' gate cols) ----
  // Per-expert GEMM at small T launches grid(N/64, T/64) ~= 24 blocks (18% of SMs).  Spanning N over
  // U union experts (N = U*MOE_INTER) gives U*24 blocks -> full occupancy -> should saturate HBM while
  // staying weight-bound + flat in T.  This is the verify's actual shape (route-aware union ~16).
  printf("\n=== grouped gate GEMM over the expert union (N = U*MOE_INTER, full occupancy) ===\n");
  printf("  %4s %4s %12s %12s %10s %12s\n","U","T","ms","GB/s","%HBM","ms/T=1");
  for (int U : {16}) {
    const int N = U * I;
    std::vector<fp8> W((size_t)N*H, fp8(0.02f)); std::vector<float> S(N, 0.02f);
    fp8* dW=devv(W); float* dS=devv(S);
    const double bytes = (double)N*H;          // fp8 weight read
    double tt1=0;
    for (int T : {1,8,16,32,64,128}) {
      std::vector<float> X((size_t)T*H,0.01f);
      float* dX=devv(X); fp8* dXq; float* dAct; float* dY;
      CK(cudaMalloc(&dXq,(size_t)T*H*sizeof(fp8))); CK(cudaMalloc(&dAct,(size_t)T*sizeof(float)));
      CK(cudaMalloc(&dY,(size_t)T*N*sizeof(float)));
      auto run=[&]{ prefill_wgmma::project(dX,dXq,dAct,dW,dS,dY,T,N,H); };
      for(int i=0;i<10;++i) run(); CK(cudaDeviceSynchronize());
      cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
      const int it=50; CK(cudaEventRecord(a)); for(int i=0;i<it;++i) run();
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); ms/=it; if(T==1) tt1=ms;
      double gbps=bytes/(ms*1e-3)/1e9;
      printf("  %4d %4d %12.4f %12.1f %9.1f%% %12.2f\n", U,T,ms,gbps,100.0*gbps/3350.0,ms/tt1);
      cudaFree(dX);cudaFree(dXq);cudaFree(dAct);cudaFree(dY);
      cudaEventDestroy(a);cudaEventDestroy(b);
    }
    cudaFree(dW);cudaFree(dS);
  }
  printf("\n--- B=1 projection IF weight-bound ---\n");
  printf("  per-GPU EP weight/token ~3.35 GB; at the measured %%HBM above, one verify produces\n");
  printf("  E[accepted]+1 tokens.  effective tok/s = emitted / verify_time(full-model-at-that-eff).\n");
  return 0;
}
