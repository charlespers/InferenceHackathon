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
// INTEGRATION (Charles's engine):
//   - Q   : [M][N_Q_HEADS][HEAD_DIM] fp16  (normed+roped draft queries; per-head RMSNorm + RoPE upstream)
//   - K/V : context cache. This header takes fp16 [N_Q_HEADS][ctx][HEAD_DIM]. For the fp8 e4m3 cache
//           (per-channel scale), dequant-on-load to fp16 is M-INDEPENDENT (flatness preserved) — OR wire
//           cuBLASLt NATIVE fp8 GEMM to skip the dequant (recommended; the dequant proxy added ~210us of
//           materialization, see tc_verify_fp8.cu honest caveat). GQA: pass K/V broadcast per Q head, or
//           adapt strides (KV head = qh/GQA_GROUP).
//   - draftK/draftV : [M][N_Q_HEADS][HEAD_DIM] fp16 (the draft tokens' own K/V from the verify forward)
//   - parent : [M] int, parent[m] = ancestor of node m (-1 for root). For a CHAIN: parent[m]=m-1.
//   - out : [M][N_Q_HEADS][HEAD_DIM] fp16 attention output.
//   Workspace (caller allocates once, sized for MMAX): S[H*MMAX*ctx], Oc/Od[H*MMAX*HEAD_DIM] fp16;
//     mxc/smc/mxd/smd[H*MMAX] fp32.
// TODO(engine): per-node RoPE pos_id is the caller's job on Q (tree_attn.h has the pos_id map); add fp8
//   native-GEMM path; tune for GQA broadcast. Numerics gate: sdpa_tree_ref.h / the *_ref CPU checks.
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
  int hh=gw/M,m=gw%M; float scale=rsqrtf((float)d); const __half* q=Q+((size_t)hh*MMAX+m)*d; float qr[VPL];
  #pragma unroll
  for(int c=0;c<VPL;c++) qr[c]=(float)q[lane*VPL+c];
  float mx=-FLT_MAX,sm=0,acc[VPL];
  #pragma unroll
  for(int c=0;c<VPL;c++) acc[c]=0;
  for(int j=m;j>=0;j=parent[j]){
    const __half* k=dK+((size_t)hh*MMAX+j)*d; float p=0;
    #pragma unroll
    for(int c=0;c<VPL;c++) p+=qr[c]*(float)k[lane*VPL+c];
    #pragma unroll
    for(int o=16;o>0;o>>=1) p+=__shfl_xor_sync(~0u,p,o);
    float s=p*scale; const __half* v=dV+((size_t)hh*MMAX+j)*d;
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
  O[((size_t)hh*MMAX+m)*d+c]=__float2half(den>0?(oc*wc+od*wd)/den:0.f);
}

// Launch the full verify attention. Returns 0 on success. Requires cublas handle in TENSOR_OP math mode.
// Layouts column-major per cuBLAS: K/V stored [HEAD_DIM x ctx] per head (lda=HEAD_DIM); Q [HEAD_DIM x MMAX].
static inline int verify_attn(cublasHandle_t cb,
    const __half* Q, const __half* K, const __half* V,
    const __half* draftK, const __half* draftV, const int* parent,
    int ctx, int M, int MMAX,
    __half* S, __half* Oc, __half* Od, float* mxc, float* smc, float* mxd, float* smd,
    __half* out, cudaStream_t stream=0){
  const int H=N_Q_HEADS, d=HEAD_DIM; const float scale=1.f/sqrtf((float)d), zero=0.f, one=1.f;
  cublasSetStream(cb, stream);
  // (A) context: S[ctx x M] = scale * K^T(ctx x d) * Q(d x M)
  if(cublasGemmStridedBatchedEx(cb,CUBLAS_OP_T,CUBLAS_OP_N, ctx,M,d, &scale,
      K,CUDA_R_16F,d,(long long)ctx*d, Q,CUDA_R_16F,d,(long long)MMAX*d, &zero,
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
