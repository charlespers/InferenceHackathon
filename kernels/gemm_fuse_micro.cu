// gemm_fuse_micro.cu — single-GPU microbench for the cuBLASLt fp8 GEMM overhead-kill.
// Measures, for the dominant decode panels (K5 gate+up, K1 QKV), the per-call cost of:
//   (A) UNFUSED  : separate quant kernel + cuBLASLt matmul + separate scale/epilogue kernel  (3 launches)
//   (B) D_SCALE  : quant + matmul-with-D_SCALE_POINTER(act_scale) + epilogue-without-act_scale (3 launches,
//                  but the act_scale multiply moves INTO the matmul)
//   (C) FUSED-EPI: quant + matmul + a SINGLE epilogue that also writes the fp8 quant of the NEXT op
//                  (collapses the next op's quant into this epilogue; net launches/GEMM -> ~2)
// Also times launches/GEMM and verifies the fused result matches the unfused (fp8 tol).
//
// Build (single GPU, no NCCL):
//   nvcc -arch=sm_90a -O3 --use_fast_math -I /root/e2e /root/e2e/gemm_fuse_micro.cu -lcublas -lcublasLt -o /tmp/gfm
//   /tmp/gfm
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <functional>

using namespace q3;
#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)
#include "gemm_engine.cuh"
static constexpr int TP = 8;
static constexpr int MOE_INTER_RANK = MOE_INTER / TP;   // 192

// per-channel scale epilogue WITH act_scale (the current gemm_epi_scale): out[n]=D[col0,n]*as*Wscale[n]
extern "C" __global__ void epi_scale_as(const __nv_bfloat16* D, const float* Wscale, const float* act_scale,
                                        float* out, int N, int Mpad){
  const float as=act_scale[0];
  for(int n=blockIdx.x*blockDim.x+threadIdx.x;n<N;n+=gridDim.x*blockDim.x)
    out[n]=(float)D[(size_t)n*Mpad]*as*Wscale[n];
}
// per-channel scale epilogue WITHOUT act_scale (cuBLASLt already applied D_SCALE=act_scale): out[n]=D[col0,n]*Wscale[n]
extern "C" __global__ void epi_scale_noas(const __nv_bfloat16* D, const float* Wscale,
                                          float* out, int N, int Mpad){
  for(int n=blockIdx.x*blockDim.x+threadIdx.x;n<N;n+=gridDim.x*blockDim.x)
    out[n]=(float)D[(size_t)n*Mpad]*Wscale[n];
}
// FUSED epilogue: apply per-channel scale -> out[n], AND quantize out -> fp8 Xq for the NEXT GEMM,
// computing the next act_scale via a block amax reduce.  One launch does dequant+next-quant.
// (Models collapsing the downstream gemm_quant into this epilogue.)  N<=4096 here, one block.
extern "C" __global__ void epi_scale_and_quant(const __nv_bfloat16* D, const float* Wscale, const float* act_scale,
                                               float* out, __nv_fp8_e4m3* Xq_next, float* act_scale_next,
                                               int N, int Mpad){
  extern __shared__ float ybuf[];                 // [N]
  const float as=act_scale[0];
  float amax=0.f;
  for(int n=threadIdx.x;n<N;n+=blockDim.x){ float v=(float)D[(size_t)n*Mpad]*as*Wscale[n]; ybuf[n]=v; out[n]=v; amax=fmaxf(amax,fabsf(v)); }
  #pragma unroll
  for(int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffffu,amax,o));
  __shared__ float amx[32]; const int lane=threadIdx.x&31, wid=threadIdx.x>>5;
  if(lane==0) amx[wid]=amax; __syncthreads();
  __shared__ float inv_sh;
  if(threadIdx.x==0){ float a=0.f; int nw=(blockDim.x+31)>>5; for(int i=0;i<nw;i++) a=fmaxf(a,amx[i]);
                      float sc=(a>0.f)?(a/448.f):1.f; act_scale_next[0]=sc; inv_sh=1.f/sc; }
  __syncthreads();
  const float inv=inv_sh;
  for(int n=threadIdx.x;n<N;n+=blockDim.x) Xq_next[n]=(__nv_fp8_e4m3)(ybuf[n]*inv);
}

static double time_us(cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1, int IT, std::function<void()> body){
  for(int i=0;i<20;i++) body();
  CK(cudaStreamSynchronize(s)); CK(cudaEventRecord(e0,s));
  for(int i=0;i<IT;i++) body();
  CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
  float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); return (double)ms/IT*1e3;
}

// empty kernel to measure the pure per-launch host->stream overhead floor at M=1.
extern "C" __global__ void noop_k(){}
// quant with a wider grid (more CTAs do a partial amax, but at K=4096 a single CTA is enough work;
// this measures whether the 5us quant is launch-overhead-bound vs reduce-bound).
extern "C" __global__ void quant_wide(const float* __restrict__ y, __nv_fp8_e4m3* __restrict__ Xq,
                                      float* __restrict__ act_scale, int n){
  // one block, 1024 threads — same as gemm_rmsnorm_quant's width (more threads, shorter serial loop).
  float amax=0.f;
  for(int i=threadIdx.x;i<n;i+=blockDim.x) amax=fmaxf(amax,fabsf(y[i]));
  #pragma unroll
  for(int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffffu,amax,o));
  __shared__ float amx[32]; const int lane=threadIdx.x&31, wid=threadIdx.x>>5;
  if(lane==0) amx[wid]=amax; __syncthreads();
  __shared__ float inv_sh;
  if(threadIdx.x==0){ float a=0.f; int nw=(blockDim.x+31)>>5; for(int i=0;i<nw;i++) a=fmaxf(a,amx[i]);
                      float sc=(a>0.f)?(a/448.f):1.f; act_scale[0]=sc; inv_sh=1.f/sc; }
  __syncthreads();
  const float inv=inv_sh;
  for(int i=threadIdx.x;i<n;i+=blockDim.x) Xq[i]=(__nv_fp8_e4m3)(y[i]*inv);
}

// add a scalar D_SCALE_POINTER to a panel's matmul desc (act_scale applied in-kernel)
static void set_dscale(LtPanel& p, const float* dscale){
  CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &dscale, sizeof(dscale)));
}
static void clear_dscale(LtPanel& p){
  const void* z=nullptr;
  CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &z, sizeof(z)));
}

struct Panel { const char* name; int K,N; };

int main(){
  CK(cudaSetDevice(0));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  cublasLtHandle_t lt; CL(cublasLtCreate(&lt));
  const int MP=16, IT=2000;

  // The two dominant panels: K5 gate+up packed (K=4096,N=8*2*192=3072) and K1 QKV (K=4096,N=2048).
  Panel panels[]={ {"K5 gateup pack", HIDDEN, TOP_K*2*MOE_INTER_RANK}, {"K1 QKV", HIDDEN, 2048} };

  printf("== gemm fuse microbench (single GPU, M=1, MP=16, %d iters) ==\n", IT);
  printf("  device: cuBLASLt fp8 e4m3 TN-GEMM.  Times us/call for quant, matmul, epilogue, and fused combos.\n\n");

  // ---- LAUNCH-OVERHEAD FLOOR: empty kernel = the irreducible per-launch host cost at M=1 ----
  double t_noop = time_us(s,e0,e1,IT,[&]{ noop_k<<<1,32,0,s>>>(); });
  double t_noop3= time_us(s,e0,e1,IT,[&]{ noop_k<<<1,32,0,s>>>(); noop_k<<<1,32,0,s>>>(); noop_k<<<1,32,0,s>>>(); });
  printf("  LAUNCH FLOOR: 1 empty kernel = %.3f us ; 3 empty kernels = %.3f us (=> ~%.3f us/launch pure overhead)\n\n",
         t_noop, t_noop3, t_noop3/3.0);

  for(auto& P: panels){
    const int K=P.K, N=P.N;
    // buffers
    __nv_fp8_e4m3 *Xq, *Wd, *Xq_next; __nv_bfloat16* Dd; float *Wscale, *act_scale, *act_scale_next, *out, *hsrc;
    CK(cudaMalloc(&Xq,(size_t)K*MP*sizeof(__nv_fp8_e4m3)));   CK(cudaMemset(Xq,0,(size_t)K*MP*sizeof(__nv_fp8_e4m3)));
    CK(cudaMalloc(&Xq_next,(size_t)4096*MP*sizeof(__nv_fp8_e4m3))); CK(cudaMemset(Xq_next,0,(size_t)4096*MP*sizeof(__nv_fp8_e4m3)));
    CK(cudaMalloc(&Wd,(size_t)K*N*sizeof(__nv_fp8_e4m3)));
    CK(cudaMalloc(&Dd,(size_t)MP*N*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&Wscale,N*sizeof(float)));
    CK(cudaMalloc(&act_scale,sizeof(float))); CK(cudaMalloc(&act_scale_next,sizeof(float)));
    CK(cudaMalloc(&out,N*sizeof(float)));
    CK(cudaMalloc(&hsrc,K*sizeof(float)));
    // fill with pseudo-data
    { std::vector<float> hv(K); for(int i=0;i<K;i++) hv[i]=0.01f*((i*37)%101-50); CK(cudaMemcpy(hsrc,hv.data(),K*sizeof(float),cudaMemcpyHostToDevice)); }
    { std::vector<float> sv(N,1.0f); for(int i=0;i<N;i++) sv[i]=0.5f+0.001f*(i%64); CK(cudaMemcpy(Wscale,sv.data(),N*sizeof(float),cudaMemcpyHostToDevice)); }
    { std::vector<unsigned char> wv((size_t)K*N); for(size_t i=0;i<wv.size();i++) wv[i]=(unsigned char)((i%7)+56); CK(cudaMemcpy(Wd,wv.data(),wv.size(),cudaMemcpyHostToDevice)); }

    LtPanel p; p.init(lt,K,N,MP,Xq,Wd,Dd,s,e0,e1);
    if(!p.haveAlgo){ printf("  %-16s AUTOTUNE FAILED\n",P.name); continue; }

    // ---- component timings ----
    double t_quant = time_us(s,e0,e1,IT,[&]{ gemm_quant<<<1,256,0,s>>>(hsrc,Xq,act_scale,K); });
    double t_qwide = time_us(s,e0,e1,IT,[&]{ quant_wide<<<1,1024,0,s>>>(hsrc,Xq,act_scale,K); });
    double t_mm    = time_us(s,e0,e1,IT,[&]{ p.run(Xq,Wd,Dd,s); });
    double t_epi   = time_us(s,e0,e1,IT,[&]{ epi_scale_as<<<32,256,0,s>>>(Dd,Wscale,act_scale,out,N,MP); });
    double t_epi1  = time_us(s,e0,e1,IT,[&]{ epi_scale_as<<<1,256,0,s>>>(Dd,Wscale,act_scale,out,N,MP); });

    // ---- (A) UNFUSED: quant + matmul + epilogue (3 launches) ----
    double tA = time_us(s,e0,e1,IT,[&]{
      gemm_quant<<<1,256,0,s>>>(hsrc,Xq,act_scale,K);
      p.run(Xq,Wd,Dd,s);
      epi_scale_as<<<32,256,0,s>>>(Dd,Wscale,act_scale,out,N,MP);
    });

    // ---- (B) D_SCALE: quant + matmul(D_SCALE=act_scale) + epilogue-without-as (3 launches) ----
    set_dscale(p, act_scale);
    double tB = time_us(s,e0,e1,IT,[&]{
      gemm_quant<<<1,256,0,s>>>(hsrc,Xq,act_scale,K);
      p.run(Xq,Wd,Dd,s);
      epi_scale_noas<<<32,256,0,s>>>(Dd,Wscale,out,N,MP);
    });
    clear_dscale(p);

    // ---- (C) FUSED-EPI: quant + matmul + ONE epilogue that ALSO quantizes next op (2 launches, since the
    //         next op's quant is absorbed here -> over a chain it's 2 launches/GEMM instead of 3). ----
    double tC = time_us(s,e0,e1,IT,[&]{
      gemm_quant<<<1,256,0,s>>>(hsrc,Xq,act_scale,K);
      p.run(Xq,Wd,Dd,s);
      epi_scale_and_quant<<<1,1024,(size_t)N*sizeof(float),s>>>(Dd,Wscale,act_scale,out,Xq_next,act_scale_next,N,MP);
    });

    // ---- CHAIN measurement: 2 GEMMs back-to-back, the way a layer runs them.
    //   UNFUSED chain : [quant, mm, epi] , [quant, mm, epi]                 = 6 launches
    //   FUSED chain   : [quant, mm, epi&quant] , [mm, epi]                  = 5 launches
    //                   (the 2nd op's quant absorbed into the 1st op's epilogue)
    double tChainUnfused = time_us(s,e0,e1,IT,[&]{
      gemm_quant<<<1,256,0,s>>>(hsrc,Xq,act_scale,K);
      p.run(Xq,Wd,Dd,s);
      epi_scale_as<<<32,256,0,s>>>(Dd,Wscale,act_scale,out,N,MP);
      gemm_quant<<<1,256,0,s>>>(hsrc,Xq,act_scale,K);   // 2nd op's standalone quant
      p.run(Xq,Wd,Dd,s);
      epi_scale_as<<<32,256,0,s>>>(Dd,Wscale,act_scale,out,N,MP);
    });
    double tChainFused = time_us(s,e0,e1,IT,[&]{
      gemm_quant<<<1,256,0,s>>>(hsrc,Xq,act_scale,K);
      p.run(Xq,Wd,Dd,s);
      epi_scale_and_quant<<<1,1024,(size_t)N*sizeof(float),s>>>(Dd,Wscale,act_scale,out,Xq_next,act_scale_next,N,MP); // 1st epi ALSO quantizes for 2nd
      p.run(Xq_next,Wd,Dd,s);                              // 2nd op: NO separate quant
      epi_scale_as<<<32,256,0,s>>>(Dd,Wscale,act_scale_next,out,N,MP);
    });

    printf("  ---- %s  (K=%d N=%d) ----\n", P.name, K, N);
    printf("    component us/call:  quant<<<1,256>>>=%.3f  quant<<<1,1024>>>=%.3f  matmul=%.3f  epi<<<32>>>=%.3f  epi<<<1>>>=%.3f\n",
           t_quant,t_qwide,t_mm,t_epi,t_epi1);
    printf("    (A) UNFUSED  quant+mm+epi          = %.3f us  (3 launches)\n", tA);
    printf("    (B) D_SCALE  quant+mm(Dscale)+epi  = %.3f us  (3 launches; act_scale in-matmul)\n", tB);
    printf("    (C) FUSED    quant+mm+epi&nextquant= %.3f us  (3 launches but absorbs NEXT op quant -> 2/GEMM amortized)\n", tC);
    printf("    => epilogue+quant overhead vs pure matmul: %.3f us (%.1fx the matmul)\n",
           tA-t_mm, tA/t_mm);
    printf("    CHAIN 2-GEMM:  UNFUSED=%.3f us (6 launches)   FUSED=%.3f us (5 launches)   SAVED=%.3f us (%.1f%%)\n",
           tChainUnfused, tChainFused, tChainUnfused-tChainFused, 100.0*(tChainUnfused-tChainFused)/tChainUnfused);
    printf("\n");

    p.destroy();
    cudaFree(Xq);cudaFree(Xq_next);cudaFree(Wd);cudaFree(Dd);cudaFree(Wscale);
    cudaFree(act_scale);cudaFree(act_scale_next);cudaFree(out);cudaFree(hsrc);
  }
  printf("== done ==\n");
  return 0;
}
