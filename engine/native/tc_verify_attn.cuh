// tc_verify_attn.cuh — DROP-IN tensor-core spec-VERIFY attention for the native decode engine.
// =============================================================================================
// WHAT: the M=k (draft-width) verify-step attention, FLAT in M (tensor cores) — the kernel that
// removes the K2 "spec free-ride blocker" (warp-shuffle k2 scales ~4x@M=8; this is ~1.0x@M=8).
// Validated vs CPU fp32 (results/mk_tree_attn/: tc_verify_tree.cu chain/tree, tc_verify_fp8.cu):
//   correctness rel <0.5%; flatness 1.09x@16 nodes ctx4096; net spec ~2.6x->~3.3-3.4x@k=8 (K2_FLATNESS_AB.md).
//
// DECOMPOSITION (per draft node m attends [context KV [0,ctx)] U [its ancestor draft nodes]):
//   (A) CONTEXT  = cuBLAS TC GEMM  S[ctx x M] = scale*K^T*Q ; coalesced softmax -> P + (mx,sm) ; O_ctx = V*P
//   (B) DRAFT-SELF = warp kernel, node m walks parent[] ancestors (tree mask) -> O_d + (mx,sm)
//   (C) MERGE    = online-softmax combine of the two normalized partials.
// M=1 (plain decode) reduces to context-only; for M=1 the engine should prefer the warp k2_flash_decode
// (tensor cores UNDERFILL the MMA M-tile at M=1: ~67us TC vs ~41us warp). Use THIS for M>=4 (verify).
//
// INTEGRATION — EXACT seam in Charles's decode_step_tp8.cu (the `tp8_k2_launch_mq(RankState& S, int M)`
// path; currently warp-shuffle k2 multi-query which SCALES ~4x@M=8 — replace with this flat TC verify):
//   Per-rank (TP=8): H = Q_HEADS_RANK = 8 (NOT 64), Q_DIM_RANK = 1024. KV cache is REPLICATED full
//   (4 KV heads); a rank's 8 Q heads all map to ONE kv head (kvh = (rank*8 + local)/GQA_GROUP, 8<16).
//   RankState buffers to bind:
//     S.q_mq       : [SPEC_MMAX * Q_DIM_RANK]  FLOAT  (M normed+roped draft queries) -> convert to fp16 for TC
//     S.kv_k/kv_v  : fp8 e4m3 [ctx_len, KV_DIM] + S.kv_k_scale/kv_v_scale [KV_DIM] (per-channel) -> dequant
//                    to fp16 (M-INDEPENDENT, flatness-safe) OR cuBLASLt native-fp8 GEMM (skips materialize)
//     S.attn_out_mq: [SPEC_MMAX * Q_DIM_RANK]  FLOAT  output (write here)
//   Per-query causal masking: the engine appends the M draft tokens' K/V to the cache and gives each
//   query its own ctx_len (draft i attends [0, ctx + i)); so the draftK/draftV/parent path here can be
//   FOLDED INTO the context GEMM by per-query ctx — OR kept separate (this header's (B) draft-self) if the
//   draft K/V are NOT yet in the cache. Confirm with Charles which the mq path does.
//   This header's CURRENT signature (standalone, fp16, H=N_Q_HEADS): Q/K/V/draftK/draftV/parent/out fp16,
//   workspace S[H*MMAX*ctx], Oc/Od[H*MMAX*HEAD_DIM] fp16, mxc/smc/mxd/smd[H*MMAX] fp32. To use in-engine:
//   pass H=Q_HEADS_RANK, convert q_mq float->fp16, dequant fp8 KV->fp16.
// *** OPEN RISK (needs on-box validation, blocked by live-demo lock): flatness was measured at H=64 heads
//   (lots of batched-GEMM SM fill). At H=8 heads/rank the batched GEMM is 8x smaller -> may UNDERFILL the
//   SMs at small M and lose some efficiency (flatness in M should hold — M is free cols — but absolute
//   per-rank efficiency must be re-measured at H=8). Queued: tcv H=8 sweep when the box frees post-demo. ***
// TODO(engine): per-node RoPE pos_id on Q (tree_attn.h has the map); cuBLASLt native-fp8 path; numerics
//   gate sdpa_tree_ref.h. M=1 plain decode KEEPS the warp k2 (TC underfills at M=1).
// =============================================================================================
#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cfloat>
#include "common.cuh"

namespace tcv {
using namespace q3;   // dims (HEAD_DIM, N_Q_HEADS, GQA_GROUP, ...) live in q3 (common.cuh)
constexpr int VPL = HEAD_DIM/32;

static __global__ void softmax_ctx(__half* S,float* mxo,float* smo,int H,int M,int MMAX,int ctx){
  int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(gw>=H*M)return;
  int hh=gw/M,m=gw%M; size_t base=(size_t)hh*MMAX*ctx+(size_t)m*ctx; float mx=-FLT_MAX;
  for(int t=lane;t<ctx;t+=32) mx=fmaxf(mx,(float)S[base+t]);
  #pragma unroll
  for(int o=16;o>0;o>>=1) mx=fmaxf(mx,__shfl_xor_sync(~0u,mx,o));
  float sm=0; for(int t=lane;t<ctx;t+=32){float e=__expf((float)S[base+t]-mx);S[base+t]=__float2half(e);sm+=e;}
  #pragma unroll
  for(int o=16;o>0;o>>=1) sm+=__shfl_xor_sync(~0u,sm,o);
  float inv=sm>0?1.f/sm:0; for(int t=lane;t<ctx;t+=32) S[base+t]=__float2half((float)S[base+t]*inv);
  if(lane==0){mxo[(size_t)hh*MMAX+m]=mx;smo[(size_t)hh*MMAX+m]=sm;}
}
static __global__ void draft_self_tree(const __half* Q,const __half* dK,const __half* dV,const int* parent,
                                       __half* Od,float* mxo,float* smo,int H,int M,int MMAX,int d){
  int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31; if(gw>=H*M)return;
  // ENGINE layout: Q/draftK/draftV are [M][H][d] (query-major, = [M][Q_DIM_RANK]); per-query stride = H*d.
  int hh=gw/M,m=gw%M; float scale=rsqrtf((float)d); const __half* q=Q+((size_t)m*H+hh)*d; float qr[VPL];
  #pragma unroll
  for(int c=0;c<VPL;c++) qr[c]=(float)q[lane*VPL+c];
  float mx=-FLT_MAX,sm=0,acc[VPL];
  #pragma unroll
  for(int c=0;c<VPL;c++) acc[c]=0;
  for(int j=m;j>=0;j=parent[j]){
    const __half* k=dK+((size_t)j*H+hh)*d; float p=0;
    #pragma unroll
    for(int c=0;c<VPL;c++) p+=qr[c]*(float)k[lane*VPL+c];
    #pragma unroll
    for(int o=16;o>0;o>>=1) p+=__shfl_xor_sync(~0u,p,o);
    float s=p*scale; const __half* v=dV+((size_t)j*H+hh)*d;
    float mn=fmaxf(mx,s),corr=__expf(mx-mn),pe=__expf(s-mn); sm=sm*corr+pe;
    #pragma unroll
    for(int c=0;c<VPL;c++) acc[c]=acc[c]*corr+pe*(float)v[lane*VPL+c];
    mx=mn; if(parent[j]<0) break;
  }
  float inv=sm>0?1.f/sm:0; __half* o=Od+((size_t)hh*MMAX+m)*d;
  #pragma unroll
  for(int c=0;c<VPL;c++) o[lane*VPL+c]=__float2half(acc[c]*inv);
  if(lane==0){mxo[(size_t)hh*MMAX+m]=mx;smo[(size_t)hh*MMAX+m]=sm;}
}
static __global__ void merge(const __half* Oc,const float* mxc,const float* smc,const __half* Od,
                             const float* mxd,const float* smd,__half* O,int H,int M,int MMAX,int d){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=H*M*d)return;
  int c=idx%d,hm=idx/d,m=hm%M,hh=hm/M; size_t st=(size_t)hh*MMAX+m;
  float mc=mxc[st],sc=smc[st],md=mxd[st],sd=smd[st],mg=fmaxf(mc,md);
  float wc=sc*__expf(mc-mg),wd=sd*__expf(md-mg),den=wc+wd;
  float oc=(float)Oc[((size_t)hh*MMAX+m)*d+c],od=(float)Od[((size_t)hh*MMAX+m)*d+c];
  O[((size_t)m*H+hh)*d+c]=__float2half(den>0?(oc*wc+od*wd)/den:0.f);  // out = ENGINE [M][H][d] (attn_out_mq)
}

// Launch the full verify attention. Returns 0 on success. Requires cublas handle in TENSOR_OP math mode.
// Layouts column-major per cuBLAS: K/V stored [HEAD_DIM x ctx] per head (lda=HEAD_DIM); Q [HEAD_DIM x MMAX].
static inline int verify_attn(cublasHandle_t cb,
    const __half* Q, const __half* K, const __half* V,
    const __half* draftK, const __half* draftV, const int* parent,
    int ctx, int M, int MMAX, int n_heads,
    __half* S, __half* Oc, __half* Od, float* mxc, float* smc, float* mxd, float* smd,
    __half* out, cudaStream_t stream=0){
  // n_heads = Q heads this call covers. In-engine: Q_HEADS_RANK (=8 at TP8), NOT N_Q_HEADS. K/V are the
  // (replicated) cache for the kv head(s) those Q heads map to; here each of the H heads has its own K/V
  // slab [ctx x HEAD_DIM] (GQA broadcast = caller duplicates the kv-head slab across its Q heads).
  const int H=n_heads, d=HEAD_DIM; const float scale=1.f/sqrtf((float)d), zero=0.f, one=1.f;
  cublasSetStream(cb, stream);
  // (A) context: S[ctx x M] = scale * K^T(ctx x d) * Q(d x M)
  if(cublasGemmStridedBatchedEx(cb,CUBLAS_OP_T,CUBLAS_OP_N, ctx,M,d, &scale,
      K,CUDA_R_16F,d,(long long)ctx*d, Q,CUDA_R_16F,H*d,(long long)d, &zero,  // Q engine [M][H][d]: ldb=H*d, batch-stride=d
      S,CUDA_R_16F,ctx,(long long)MMAX*ctx, H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)!=CUBLAS_STATUS_SUCCESS) return 1;
  int w=H*M, blk=128;
  softmax_ctx<<<(w*32+blk-1)/blk,blk,0,stream>>>(S,mxc,smc,H,M,MMAX,ctx);
  if(cublasGemmStridedBatchedEx(cb,CUBLAS_OP_N,CUBLAS_OP_N, d,M,ctx, &one,
      V,CUDA_R_16F,d,(long long)ctx*d, S,CUDA_R_16F,ctx,(long long)MMAX*ctx, &zero,
      Oc,CUDA_R_16F,d,(long long)MMAX*d, H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)!=CUBLAS_STATUS_SUCCESS) return 2;
  // (B) draft-self (tree mask) + (C) merge
  draft_self_tree<<<(w*32+blk-1)/blk,blk,0,stream>>>(Q,draftK,draftV,parent,Od,mxd,smd,H,M,MMAX,d);
  int tot=H*M*d; merge<<<(tot+255)/256,256,0,stream>>>(Oc,mxc,smc,Od,mxd,smd,out,H,M,MMAX,d);
  return 0;
}
} // namespace tcv
