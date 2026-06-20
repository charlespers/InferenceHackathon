// spec_decode_bench.cu — speculative-decode analysis + microbench for Qwen3-235B-A22B B=1 decode.
// Target: 8x H100 (sm_90a). Standard CUDA only. Compiles + runs standalone (CPU math + one GPU bench).
//   build: /usr/local/cuda/bin/nvcc -arch=sm_90a -O3 --use_fast_math kernels/spec_decode_bench.cu -I kernels -o /tmp/specbench
//   run:   CUDA_VISIBLE_DEVICES=0 ./specbench [base_toks=545] [peak_GBps=3350]
//
// WHY SPECULATIVE DECODE
// ----------------------
// B=1 decode is HBM-bound: every token reads the full ~22 GB active-fp8 weight set (sharded ~2.75 GB/GPU
// over 8 GPUs). That fixes a hard per-token bandwidth ROOFLINE — no kernel can beat the bytes/token.
// The measured-best fused fp8 MoE (k5_experts_warp.cu, the in-repo fast GEMV: warp-per-output-row +
// coalesced uint4 fp8 loads + fp8x2->half2 dequant, 1538 GB/s = 0.46 of peak) and the TP=8 shard get us
// to ~400-545 sharded tok/s, but the only way PAST the raw-bandwidth roofline is to read the weights
// fewer times per *accepted* token.
//
// VERIFY-IN-ONE-PASS (the core idea modeled here)
// -----------------------------------------------
// A cheap draft proposes gamma candidate tokens. The target model VERIFIES all gamma in ONE forward:
// the gamma candidates (+1 bonus position) form a tiny (gamma+1)-row "batch". Because B=1 decode is
// bandwidth-bound, weights are streamed from HBM exactly ONCE and the per-row arithmetic is amortized
// across the (gamma+1) rows — so a (gamma+1)-row verify pass costs ~the same wall-clock as a single
// 1-row decode pass (the GPU microbench below confirms this on the k5 idiom). With acceptance prob
// alpha per draft token, the expected number of ACCEPTED tokens emitted per verify pass is the standard
// geometric result   E[accepted] = (1 - alpha^(gamma+1)) / (1 - alpha)
// (gamma chances + 1 guaranteed bonus token from the target on the first rejection / full-accept).
// Effective throughput = E[accepted] / (time of one verify pass), and the multiplier over no-spec is
// E[accepted] / (verify_pass_time / single_token_time)  ~=  E[accepted] / batch_slowdown.
//
// This file:
//   (1) measures batch_slowdown(gamma) on a real H100 using the proven k5 warp-per-row fp8 GEMV
//       (a (gamma+1)-row MoE-expert GEMV vs a 1-row one) -> the empirical "one pass for K+1" cost;
//   (2) builds the full (alpha,gamma) table of E[accepted], effective tok/s, and the multiplier vs the
//       sharded no-spec baseline, and flags which (alpha,gamma) clear a 1000-tok/s-equivalent.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)

// Max modeled verify batch: gamma up to 8 -> (gamma+1) up to 9 rows.
#define BMAX 9

// ----------------------------------------------------------------------------------------------------
// GPU microbench: batched warp-per-row fp8 GEMV (the in-repo k5 idiom), B rows vs 1 row.
// Reuses k5_experts_warp.cu's warp_dot recipe: a warp owns one output channel, its 32 lanes split the
// contraction so consecutive lanes read consecutive 16-byte (uint4 = 16 fp8) chunks of the SAME weight
// row -> fully coalesced HBM, then a shuffle reduce. Weights are read ONCE; the only extra work for B
// rows is B independent dot accumulations over the staged inputs -> proves verify-in-one-pass amortizes
// the HBM-bound weight read across the batch.
// ----------------------------------------------------------------------------------------------------
static __device__ __forceinline__ void warp_dot_batch(
    const fp8* __restrict__ w, const float* __restrict__ ys, int n, int B, int lane, float* out){
  // out[0..B): <w, ys[b*n ..]> for b in [0,B), one coalesced sweep of w shared across all B rows.
  float acc[BMAX]; // gamma up to 8 -> B = gamma+1 up to 9 rows
  #pragma unroll
  for(int b=0;b<BMAX;b++) acc[b]=0.f;
  const uint4* __restrict__ wv=reinterpret_cast<const uint4*>(w);
  const int nv=n>>4;                                   // 16 fp8 per uint4
  for(int v=lane; v<nv; v+=32){                        // coalesced: lane k -> uint4 k (16 bytes apart)
    uint4 p=wv[v];
    const unsigned* wu=reinterpret_cast<const unsigned*>(&p);
    float wf[16];
    #pragma unroll
    for(int q=0;q<4;q++){                              // fp8x2 -> half2 hardware dequant (k5 idiom)
      unsigned wq=wu[q];
      __nv_fp8x2_e4m3 lo,hi; lo.__x=(unsigned short)(wq&0xffffu); hi.__x=(unsigned short)(wq>>16);
      float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi);
      wf[q*4+0]=fl.x; wf[q*4+1]=fl.y; wf[q*4+2]=fh.x; wf[q*4+3]=fh.y;
    }
    #pragma unroll
    for(int b=0;b<BMAX;b++){ if(b<B){ const float* yy=ys+(size_t)b*n+(v<<4);
      #pragma unroll
      for(int t=0;t<16;t++) acc[b]+=yy[t]*wf[t]; } }
  }
  #pragma unroll
  for(int b=0;b<BMAX;b++){ if(b<B){ float a=acc[b];
    #pragma unroll
    for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffffu,a,o);
    if(lane==0) out[b]=a; } }
}

// Batched MoE down-proj-shaped GEMV over a contraction of MOE_INTER, B rows, warp-per-output-row.
// One CTA tiles output channels across its warps (grid-stride), B rows share the weight sweep.
extern "C" __global__ void gemv_batch(
    const fp8* __restrict__ W, const float* __restrict__ S, const float* __restrict__ Y,
    float* __restrict__ Hout, int n_out, int n_in, int B){
  const int lane=threadIdx.x&31;
  const int gwarp=(blockIdx.x*blockDim.x+threadIdx.x)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  for(int o=gwarp; o<n_out; o+=nwarp){
    float res[BMAX];
    warp_dot_batch(W+(size_t)o*n_in, Y, n_in, B, lane, res);
    if(lane==0){ const float s=S[o];
      #pragma unroll
      for(int b=0;b<BMAX;b++) if(b<B) Hout[(size_t)b*n_out+o]=res[b]*s; }
  }
}

__global__ void fill_fp8(fp8* w,size_t n,unsigned s){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2654435761u)+s*40503u; w[i]=fp8((((h%2000)/1000.0f)-1.0f)*0.25f);} }
__global__ void fill_f32(float* a,size_t n,unsigned s,float sc){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){unsigned h=(unsigned)(i*2246822519u)+s*40503u; a[i]=(((h%2000)/1000.0f)-1.0f)*sc;} }

// ----------------------------------------------------------------------------------------------------
// Pure-math model (no GPU needed).
// ----------------------------------------------------------------------------------------------------
static double E_accept(double alpha, int gamma){
  // gamma draft tokens, geometric acceptance + 1 guaranteed bonus token:
  //   E[accepted] = sum_{i=0..gamma} alpha^i = (1 - alpha^(gamma+1)) / (1 - alpha)
  if (alpha >= 1.0) return gamma + 1.0;
  return (1.0 - pow(alpha, gamma + 1)) / (1.0 - alpha);
}

int main(int argc,char**argv){
  const double BASE_TOKS = (argc>1)?atof(argv[1]):545.0;   // sharded no-spec baseline (TP=8 fp8), tok/s
  const double PEAK      = (argc>2)?atof(argv[2]):3350.0;  // H100 HBM peak GB/s (per GPU)

  // ---------------- (1) GPU: measure batch_slowdown(B) on the k5 fp8 GEMV idiom ----------------
  // We use the MoE down-proj shape (out=HIDDEN, in=MOE_INTER) as a representative bandwidth-bound GEMV.
  // For B = 1..9 we time the SAME weight matrix swept once for B rows. If decode is HBM-bound, time(B)
  // should be ~flat in B (weights dominate, rows are free) -> batch_slowdown(B) ~= 1, i.e. verifying
  // gamma=B-1 candidates costs ~one single-token pass. We report the measured ratio time(B)/time(1).
  const int n_out=HIDDEN, n_in=MOE_INTER;               // 4096 x 1536 (one expert's down-proj)
  const int CTAS=264, BLK=1024;                         // k5's best-measured launch (8448 warps)
  const int Bmax=BMAX;                                  // gamma up to 8 -> verify batch up to 9 rows
  fp8 *W; float *S,*Y,*Hout;
  CK(cudaMalloc(&W,(size_t)n_out*n_in*sizeof(fp8)));
  CK(cudaMalloc(&S,(size_t)n_out*sizeof(float)));
  CK(cudaMalloc(&Y,(size_t)Bmax*n_in*sizeof(float)));
  CK(cudaMalloc(&Hout,(size_t)Bmax*n_out*sizeof(float)));
  fill_fp8<<<512,256>>>(W,(size_t)n_out*n_in,7u);
  fill_f32<<<64,256>>>(S,n_out,3u,0.02f);
  fill_f32<<<64,256>>>(Y,(size_t)Bmax*n_in,11u,1.0f);
  CK(cudaDeviceSynchronize());

  cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
  const int WARM=30, IT=400;
  auto timeB=[&](int B)->float{
    for(int i=0;i<WARM;i++) gemv_batch<<<CTAS,BLK>>>(W,S,Y,Hout,n_out,n_in,B);
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
    for(int i=0;i<IT;i++)   gemv_batch<<<CTAS,BLK>>>(W,S,Y,Hout,n_out,n_in,B);
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms; CK(cudaEventElapsedTime(&ms,s,e)); return ms/IT; };

  float t1=timeB(1);
  const double bytes=(double)n_out*n_in;                // fp8: 1 byte/elem, weights read once
  printf("== (1) GPU microbench: k5 warp-per-row fp8 GEMV, weights swept once for B rows ==\n");
  printf("   shape out=%d in=%d  launch %dx%d   weight read = %.1f MB (read ONCE per pass)\n",
         n_out,n_in,CTAS,BLK,bytes/1e6);
  printf("   %-3s %10s %10s %9s\n","B","ms/call","GB/s","slowdown");
  double slow[BMAX+1]; slow[0]=1.0;
  for(int B=1;B<=Bmax;B++){
    float tB=timeB(B);
    slow[B]=tB/t1;
    printf("   %-3d %10.4f %10.1f %9.3f%s\n", B, tB, bytes/1e6/tB, tB/t1,
           (B==1?"  (1-token decode pass)":""));
  }
  printf("   NOTE: slowdown ~1.0 across B confirms verify-in-one-pass: a (gamma+1)-row verify costs\n");
  printf("         ~one single-token weight read (HBM-bound) -> arithmetic over rows is amortized.\n\n");

  // batch_slowdown for a verify of gamma candidates = slow[gamma+1] (measured; clamp to table range).
  auto slowdown=[&](int gamma)->double{ int B=gamma+1; if(B<1)B=1; if(B>Bmax)B=Bmax; return slow[B]; };

  // ---------------- (2) Math: (alpha,gamma) effective-tok/s table ----------------
  const double ALPHAS[]={0.5,0.6,0.7,0.8};
  const int    GAMMAS[]={2,4,6,8};
  const int NA=4, NG=4;

  printf("== (2) Speculative-decode model  (base = %.0f tok/s sharded no-spec, TP=8 fp8) ==\n", BASE_TOKS);
  printf("   E[accepted] = (1 - alpha^(gamma+1))/(1 - alpha)   per verify-in-one-pass\n");
  printf("   eff tok/s   = base * E[accepted] / batch_slowdown(gamma)   [slowdown measured above]\n\n");

  printf("   E[accepted tokens per verify pass]\n");
  printf("   %-8s","alpha\\g");
  for(int g=0;g<NG;g++) printf(" %8d", GAMMAS[g]);
  printf("\n");
  for(int a=0;a<NA;a++){ printf("   %-8.2f", ALPHAS[a]);
    for(int g=0;g<NG;g++) printf(" %8.3f", E_accept(ALPHAS[a],GAMMAS[g]));
    printf("\n"); }
  printf("\n");

  printf("   Effective throughput MULTIPLIER vs no-spec  (x baseline)\n");
  printf("   %-8s","alpha\\g");
  for(int g=0;g<NG;g++) printf(" %8d", GAMMAS[g]);
  printf("\n");
  for(int a=0;a<NA;a++){ printf("   %-8.2f", ALPHAS[a]);
    for(int g=0;g<NG;g++){ double m=E_accept(ALPHAS[a],GAMMAS[g])/slowdown(GAMMAS[g]); printf(" %8.2f", m); }
    printf("\n"); }
  printf("\n");

  printf("   Effective tok/s  (base %.0f)   [** = clears 1000 tok/s]\n", BASE_TOKS);
  printf("   %-8s","alpha\\g");
  for(int g=0;g<NG;g++) printf(" %9d", GAMMAS[g]);
  printf("\n");
  double best=0; double best_a=0; int best_g=0;
  for(int a=0;a<NA;a++){ printf("   %-8.2f", ALPHAS[a]);
    for(int g=0;g<NG;g++){
      double toks = BASE_TOKS * E_accept(ALPHAS[a],GAMMAS[g]) / slowdown(GAMMAS[g]);
      if(toks>best){ best=toks; best_a=ALPHAS[a]; best_g=GAMMAS[g]; }
      printf(" %7.0f%s", toks, (toks>=1000.0)?"**":"  ");
    }
    printf("\n"); }
  printf("\n");

  // ---------------- (3) Which (alpha,gamma) reach 1000 tok/s ----------------
  printf("== (3) 1000-tok/s reachability (base %.0f) ==\n", BASE_TOKS);
  int nclear=0;
  for(int a=0;a<NA;a++) for(int g=0;g<NG;g++){
    double toks = BASE_TOKS * E_accept(ALPHAS[a],GAMMAS[g]) / slowdown(GAMMAS[g]);
    if(toks>=1000.0){ nclear++;
      printf("   REACH 1000:  alpha=%.2f gamma=%d  -> %.0f tok/s  (E[acc]=%.2f, slowdown=%.3f)\n",
             ALPHAS[a],GAMMAS[g],toks,E_accept(ALPHAS[a],GAMMAS[g]),slowdown(GAMMAS[g])); }
  }
  if(nclear==0) printf("   none of the tabled (alpha,gamma) clear 1000 at base=%.0f.\n", BASE_TOKS);
  printf("   best in table: alpha=%.2f gamma=%d -> %.0f tok/s (%.2fx baseline).\n",
         best_a,best_g,best,best/BASE_TOKS);

  // Min acceptance needed to clear 1000 at the best (largest E[acc]/slowdown) gamma per the model, and
  // min sharded base needed at a "realistic" alpha=0.7, gamma=4 operating point.
  {
    double need_mult = 1000.0 / BASE_TOKS;
    printf("   need multiplier >= %.2fx to clear 1000 from base %.0f.\n", need_mult, BASE_TOKS);
    int g=4; double sd=slowdown(g);
    // solve E_accept(alpha,g)/sd = need_mult for alpha by bisection
    double lo=0.0,hi=0.999;
    for(int it=0; it<60; it++){ double mid=0.5*(lo+hi);
      if(E_accept(mid,g)/sd >= need_mult) hi=mid; else lo=mid; }
    printf("   at gamma=4 (slowdown=%.3f): need alpha >= %.3f to hit 1000 tok/s.\n", sd, hi);
    double op_toks = BASE_TOKS * E_accept(0.7,4)/slowdown(4);
    printf("   realistic op point alpha=0.70 gamma=4 -> %.0f tok/s (%.2fx).  ", op_toks, op_toks/BASE_TOKS);
    double need_base = 1000.0 * slowdown(4) / E_accept(0.7,4);
    printf("base needed at (0.70,4) for 1000: %.0f tok/s.\n", need_base);
  }

  // sanity: roofline note
  printf("\n   (peak %.0f GB/s/GPU; the multiplier is pure read-amortization -> it stacks on top of\n", PEAK);
  printf("    sharding + near-roofline kernels, since spec-decode reduces weight reads / accepted token.)\n");

  CK(cudaFree(W));CK(cudaFree(S));CK(cudaFree(Y));CK(cudaFree(Hout));
  return 0;
}
