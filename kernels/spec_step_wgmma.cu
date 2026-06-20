// spec_step_wgmma.cu — MEASURED end-to-end B=1 spec-verify decode step (per-GPU EP shard, 1 GPU proxy).
//
// Runs the REAL per-PE per-layer weight volume through the fp8 tensor-core GEMM (prefill_wgmma::project,
// the kernel measured at ~11% HBM, weight-bound/flat-in-M) at M = gamma+1 draft rows, x N_LAYERS,
// then reports effective tok/s = (E[accepted]+1) / (compute + NVLS comms).  Latency proxy: one layer's
// dummy weights reused x94 (shapes/volumes/M = real per-PE EP shard), same contract as decode_step.cu.
//
// Per-PE per-layer (EP=8, TP=8 attention):
//   qkv   : [QKV_OUT_RANK=2048, HIDDEN]      O-proj: [HIDDEN, Q_DIM_RANK=1024]
//   MoE   : route-aware union ~16 experts total -> U_PE = union/8 on this PE; each gate+up [2*1536,4096]
//           + down [4096,1536].  (U_PE swept; 1=top-8 only, 2=small tree, 3=bigger tree.)
//
// BUILD: nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -DPREFILL_WGMMA_NO_MAIN -DBM=16 \
//          kernels/spec_step_wgmma.cu -o /tmp/sstep
// RUN:   CUDA_VISIBLE_DEVICES=0 /tmp/sstep [M=8] [U_PE=2] [alpha=0.7]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;
#define PREFILL_WGMMA_NO_MAIN 1
#include "prefill_wgmma.cu"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// TP=8 / EP=8 per-PE shard shapes
static const int QKV_OUT_RANK = 2048;     // (Q_DIM_RANK 1024 + 2*KV_DIM 512) ~ per-PE QKV rows
static const int Q_DIM_RANK   = 1024;     // per-PE attention output width

struct Buf { fp8* W; float* S; float* Y; fp8* Xq; float* act; float* X; };
static Buf mkbuf(int M, int N, int K) {
  Buf b;
  std::vector<fp8> w((size_t)N*K, fp8(0.02f)); std::vector<float> s(N, 0.02f);
  std::vector<float> x((size_t)M*K, 0.01f);
  CK(cudaMalloc(&b.W,(size_t)N*K*sizeof(fp8)));   CK(cudaMemcpy(b.W,w.data(),w.size()*sizeof(fp8),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&b.S,(size_t)N*sizeof(float)));   CK(cudaMemcpy(b.S,s.data(),s.size()*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&b.X,(size_t)M*K*sizeof(float))); CK(cudaMemcpy(b.X,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&b.Xq,(size_t)M*K*sizeof(fp8)));  CK(cudaMalloc(&b.act,(size_t)M*sizeof(float)));
  CK(cudaMalloc(&b.Y,(size_t)M*N*sizeof(float)));
  return b;
}

int main(int argc, char** argv){
  const int   M     = (argc>1)?atoi(argv[1]):8;     // draft rows verified (gamma+1)
  const int   U_PE  = (argc>2)?atoi(argv[2]):2;     // union experts on THIS PE (union/8)
  const float alpha = (argc>3)?atof(argv[3]):0.7f;  // per-position accept prob
  const int   GAMMA = M-1;
  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));

  // per-PE per-layer GEMMs (one layer's weights, reused x N_LAYERS)
  Buf qkv  = mkbuf(M, QKV_OUT_RANK,            HIDDEN);        // attention QKV (TP)
  Buf op   = mkbuf(M, HIDDEN,                  Q_DIM_RANK);    // attention O-proj (TP)
  Buf gu   = mkbuf(M, U_PE * 2 * MOE_INTER,    HIDDEN);        // grouped gate+up over the PE's union
  Buf dn   = mkbuf(M, U_PE * HIDDEN,           MOE_INTER);     // grouped down over the PE's union

  const double per_layer_bytes =
      (double)QKV_OUT_RANK*HIDDEN + (double)HIDDEN*Q_DIM_RANK +
      (double)U_PE*2*MOE_INTER*HIDDEN + (double)U_PE*HIDDEN*MOE_INTER;
  const double total_gb = per_layer_bytes * N_LAYERS / 1e9;

  auto layer = [&](cudaStream_t s){
    prefill_wgmma::project(qkv.X, qkv.Xq, qkv.act, qkv.W, qkv.S, qkv.Y, M, QKV_OUT_RANK,         HIDDEN, s);
    prefill_wgmma::project(op.X,  op.Xq,  op.act,  op.W,  op.S,  op.Y,  M, HIDDEN,               Q_DIM_RANK, s);
    prefill_wgmma::project(gu.X,  gu.Xq,  gu.act,  gu.W,  gu.S,  gu.Y,  M, U_PE*2*MOE_INTER,     HIDDEN, s);
    prefill_wgmma::project(dn.X,  dn.Xq,  dn.act,  dn.W,  dn.S,  dn.Y,  M, U_PE*HIDDEN,          MOE_INTER, s);
  };
  cudaStream_t s; CK(cudaStreamCreate(&s));
  for(int w=0;w<3;++w){ for(int l=0;l<N_LAYERS;++l) layer(s); } CK(cudaStreamSynchronize(s));

  // ---- per-component breakdown: time each GEMM type over N_LAYERS to find the drag ----
  {
    auto timeit = [&](const char* nm, Buf& bb, int N, int K)->double{
      cudaEvent_t x,y; CK(cudaEventCreate(&x)); CK(cudaEventCreate(&y));
      const int it=20; CK(cudaEventRecord(x,s));
      for(int i=0;i<it;++i) for(int l=0;l<N_LAYERS;++l)
        prefill_wgmma::project(bb.X,bb.Xq,bb.act,bb.W,bb.S,bb.Y,M,N,K,s);
      CK(cudaEventRecord(y,s)); CK(cudaEventSynchronize(y));
      float mm=0; CK(cudaEventElapsedTime(&mm,x,y)); mm/=it;
      double gb=(double)N*K*N_LAYERS/1e9, gbps=gb/(mm/1000.0);
      printf("  %-10s N=%-6d K=%-5d : %7.2f ms  (%.1f GB/s, %.1f%% HBM)\n", nm,N,K,mm,gbps,100.0*gbps/3350.0);
      cudaEventDestroy(x);cudaEventDestroy(y); return mm;
    };
    printf("\n-- per-component breakdown (x%d layers) --\n", N_LAYERS);
    timeit("qkv",   qkv, QKV_OUT_RANK,         HIDDEN);
    timeit("oproj", op,  HIDDEN,               Q_DIM_RANK);
    timeit("gate+up",gu, U_PE*2*MOE_INTER,     HIDDEN);
    timeit("down",  dn,  U_PE*HIDDEN,          MOE_INTER);
  }

  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  const int IT=20; CK(cudaEventRecord(a,s));
  for(int i=0;i<IT;++i) for(int l=0;l<N_LAYERS;++l) layer(s);
  CK(cudaEventRecord(b,s)); CK(cudaEventSynchronize(b));
  float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); ms/=IT;   // compute ms / verify-step

  // measured NVLS comms: 2 all-reduces/layer x N_LAYERS x 5.2us (kernels/nvls_mc_allreduce.cu)
  const double comms_ms = 2.0 * N_LAYERS * 5.2 / 1e3;
  const double step_ms  = ms + comms_ms;

  // emitted tokens / verify = E[accepted] + 1 bonus ; E[accepted] = sum_{i=1..gamma} alpha^i
  double eacc=0, pre=1; for(int i=0;i<GAMMA;++i){ pre*=alpha; eacc+=pre; }
  const double emitted = eacc + 1.0;
  const double gbps = total_gb/(ms/1000.0), mbu = 100.0*gbps/3350.0;

  printf("== spec-verify decode step (1-GPU per-PE EP proxy, tensor-core verify) ==\n");
  printf("device %s  M(draft rows)=%d  gamma=%d  U_PE(union/8)=%d  alpha=%.2f\n", prop.name, M, GAMMA, U_PE, alpha);
  printf("per-PE weight/token: %.1f MB/layer x %d = %.2f GB\n", per_layer_bytes/1e6, N_LAYERS, total_gb);
  printf("\n  compute/verify-step : %7.2f ms   (%.1f GB/s, %.1f%% HBM — tensor-core, M-amortized)\n", ms, gbps, mbu);
  printf("  + NVLS comms        : %7.2f ms   (188 x 5.2us, measured)\n", comms_ms);
  printf("  = total/verify-step : %7.2f ms\n", step_ms);
  printf("  emitted/verify      : %.2f tokens (E[acc]=%.2f + bonus)\n", emitted, eacc);
  printf("\n  >>> effective decode throughput : %.0f tok/s <<<\n", 1000.0*emitted/step_ms);
  printf("  (no-spec single-token at this eff: %.0f tok/s ; vLLM bf16 baseline 85.7)\n", 1000.0/step_ms);
  return 0;
}
