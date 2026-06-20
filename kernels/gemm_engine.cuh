// gemm_engine.cuh — cuBLASLt fp8 e4m3 TN-GEMM engine for the TP=8 per-rank decode forward.
// ============================================================================================
// PURPOSE: replace decode_step_tp8.cu's hand-rolled M=1 GEMV kernels (occupancy-starved, ~21% MBU,
// ~8.5 ms/forward) with cuBLASLt fp8 tensor-core GEMM (the validated fast path: ~2.7 ms/forward,
// FLAT in M, T(16)~=T(1)).  The GEMM recipe is copied verbatim from the proven
// spec_verify_forward_gemm.cu / spec_decode_loop.cu (TN layout A^T*B, both operands K-major,
// FAST_ACCUM, autotuned-once-at-Mmax-then-pinned to defeat cuBLASLt's small-M heuristic zig-zag).
//
// MATH (matches spec_verify_forward_gemm.cu):
//   D[M,N] (col-major, ldd=Mpad) = X^T[M,K] @ W[K,N]
//     A = X : col-major [K, Mpad], lda=K, opA=T   (the M activation columns, K-major = one fp8/elt)
//     B = W : col-major [K, N],    ldb=K, opB=N   (weights, stored row-major [N,K] == col-major [K,N])
//   so D[m,n] = sum_k X[k,m] * W[n,k]  — exactly the per-row GEMV the decode kernels computed, but
//   batched over M activation columns and run on wgmma tensor cores.
// The decode weights (Wqkv/Wo/Wgu/Wd/Wgate/Wlm) are ALREADY stored K-major (row o is HIDDEN/K_in
// contiguous), i.e. col-major [K,N] with ldb=K — drops straight into B with NO repack.
//
// PER-OUTPUT-CHANNEL SCALE: the decode weights carry a per-output-channel fp32 dequant scale[N].
// cuBLASLt fp8 supports a D scale only as a single scalar, so we DON'T fold the per-channel scale
// into the GEMM; we apply it in the existing per-step epilogues (k1_epilogue's QK-norm reads the
// scaled proj; k5a applies gate/up scale + SiLU; k3/lmhead apply scale in a tiny epilogue).  The
// GEMM therefore computes the RAW fp8 dot D[m,n]=sum_k X[k,m]*W[n,k]; scale is applied downstream,
// identically to how the GEMV kernels applied `r * Wscale[o]`.
//
// fp8 e4m3 is the SHIP precision (Qwen3-235B weights are e4m3 block-scaled).  Activations are
// quantized to e4m3 with a per-tensor amax scale right before each GEMM (the small extra quantize
// pass is the cost of using tensor cores; it is dwarfed by the GEMV->GEMM win).  spec_verify_*
// proved the e4m3 GEMM is bit-exact wgmma; the only delta vs fp32 is the inherent e4m3 rounding,
// which the model already trains with — the cross-rank correctness gate (<1e-2) is preserved
// because BOTH the sharded GEMM path and the reference run the SAME fp8 weights+activations.
// ============================================================================================
#pragma once
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>

#define CL(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  printf("cuBLASLt err %s:%d: %d\n",__FILE__,__LINE__,(int)s_); exit(1);} } while(0)

namespace q3 {

// --------------------------------------------------------------------------------------------
// One cuBLASLt fp8 TN-GEMM for a fixed (K,N) panel.  Autotuned ONCE at Mmax (16) over up to 16
// heuristic candidates; the fastest algo is pinned and reused for every M<=Mmax (fp8 always runs
// the 16-wide tensor tile so the pinned kernel is optimal across M — the flatness the bench proved).
// --------------------------------------------------------------------------------------------
struct LtPanel {
  cublasLtHandle_t lt = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t aL=nullptr, bL=nullptr, dL=nullptr;
  cublasLtMatmulPreference_t pref=nullptr;
  cublasLtMatmulHeuristicResult_t heur{};
  void* ws=nullptr; size_t wsBytes = 32ull<<20;     // 32 MB workspace/panel
  int K=0, N=0, Mpad=0, align=16;                   // fp8 e4m3 -> M rounded up to 16
  bool haveAlgo=false;

  void init(cublasLtHandle_t lt_, int K_, int N_, int Mmax,
            const void* Xd, const void* Wd, void* Dd, cudaStream_t s,
            cudaEvent_t ev0, cudaEvent_t ev1) {
    lt=lt_; K=K_; N=N_;
    CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tA=CUBLAS_OP_T, tB=CUBLAS_OP_N;
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,&tA,sizeof(tA)));
    CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,&tB,sizeof(tB)));
    int8_t fa=1; CL(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fa,sizeof(fa)));
    CK(cudaMalloc(&ws,wsBytes));
    CL(cublasLtMatmulPreferenceCreate(&pref));
    CL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsBytes,sizeof(wsBytes)));
    Mpad = ((Mmax + align - 1) / align) * align;    // 16
    CL(cublasLtMatrixLayoutCreate(&aL, CUDA_R_8F_E4M3, K, Mpad, K));   // A col-major [K,Mpad]
    CL(cublasLtMatrixLayoutCreate(&bL, CUDA_R_8F_E4M3, K, N,    K));   // B col-major [K,N]
    CL(cublasLtMatrixLayoutCreate(&dL, CUDA_R_16BF,    Mpad, N, Mpad));// D col-major [Mpad,N] bf16
    // autotune at Mpad over heuristic candidates, keep the fastest (defeats small-M dip).
    const int NC=16; cublasLtMatmulHeuristicResult_t cand[NC]; int got=0;
    cublasStatus_t st=cublasLtMatmulAlgoGetHeuristic(lt,op,aL,bL,dL,dL,pref,NC,cand,&got);
    if (st!=CUBLAS_STATUS_SUCCESS||got==0){ haveAlgo=false; return; }
    const float alpha=1.f,beta=0.f; double best=1e30; int bi=-1;
    for (int c=0;c<got;c++){
      auto one=[&](){ return cublasLtMatmul(lt,op,&alpha,Xd,aL,Wd,bL,&beta,Dd,dL,Dd,dL,&cand[c].algo,ws,wsBytes,s); };
      if (one()!=CUBLAS_STATUS_SUCCESS) continue;
      for (int w=0;w<5;w++) one();
      cudaStreamSynchronize(s); cudaEventRecord(ev0,s);
      for (int r=0;r<20;r++) one();
      cudaEventRecord(ev1,s); cudaEventSynchronize(ev1);
      float ms; cudaEventElapsedTime(&ms,ev0,ev1); ms/=20;
      if (ms<best){ best=ms; bi=c; }
    }
    if (bi<0){ haveAlgo=false; return; }
    heur=cand[bi]; haveAlgo=true;
  }
  // run D[Mpad,N] = X^T @ W on stream s.  X must be col-major [K,Mpad] fp8, zero-padded for m>=M.
  void run(const void* Xd, const void* Wd, void* Dd, cudaStream_t s) const {
    const float alpha=1.f,beta=0.f;
    CL(cublasLtMatmul(lt,op,&alpha,Xd,aL,Wd,bL,&beta,Dd,dL,Dd,dL,&heur.algo,ws,wsBytes,s));
  }
  // Time IT runs at verify width M (rounded up to the fp8 16-wide tile = identical kernel for M<=16,
  // which is exactly the flatness the spec-verify bench proved).  Returns us/call.  Rebuilds the A/D
  // layouts at Mpad(M) using the SAME pinned algo, then restores the M=GEMM_MMAX layouts.
  double time_at_M(int M, const void* Xd, const void* Wd, void* Dd, cudaStream_t s,
                   cudaEvent_t ev0, cudaEvent_t ev1, int WARM, int IT) {
    int mp = ((M + align - 1) / align) * align;
    cublasLtMatrixLayout_t aM=nullptr, dM=nullptr;
    CL(cublasLtMatrixLayoutCreate(&aM, CUDA_R_8F_E4M3, K, mp, K));
    CL(cublasLtMatrixLayoutCreate(&dM, CUDA_R_16BF,    mp, N, mp));
    const float alpha=1.f,beta=0.f;
    auto one=[&](){ return cublasLtMatmul(lt,op,&alpha,Xd,aM,Wd,bL,&beta,Dd,dM,Dd,dM,&heur.algo,ws,wsBytes,s); };
    for (int i=0;i<WARM;i++) one();
    cudaStreamSynchronize(s); cudaEventRecord(ev0,s);
    for (int i=0;i<IT;i++) one();
    cudaEventRecord(ev1,s); cudaEventSynchronize(ev1);
    float ms; cudaEventElapsedTime(&ms,ev0,ev1);
    cublasLtMatrixLayoutDestroy(aM); cublasLtMatrixLayoutDestroy(dM);
    return (double)ms/IT*1e3;   // us/call
  }
  void destroy() {
    if (aL) cublasLtMatrixLayoutDestroy(aL);
    if (bL) cublasLtMatrixLayoutDestroy(bL);
    if (dL) cublasLtMatrixLayoutDestroy(dL);
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (op) cublasLtMatmulDescDestroy(op);
    if (ws) cudaFree(ws);
    aL=bL=dL=nullptr; pref=nullptr; op=nullptr; ws=nullptr; haveAlgo=false;
  }
};

// --------------------------------------------------------------------------------------------
// fp8 ACTIVATION QUANTIZE kernels.  cuBLASLt fp8 GEMM needs the activation operand as e4m3,
// K-major (col-major [K,Mpad]).  At B=1 (M=1) the activation is a single [K] vector; we quantize
// with a per-tensor scale = AMAX/448 (e4m3 max) so values use the full e4m3 range.  We bake the
// quantize scale into the DOWNSTREAM per-channel weight-scale application (the GEMM output must be
// multiplied by act_scale to undo the activation quantization) — see the epilogues in
// decode_step_tp8.cu, which fold act_scale into the existing `* Wscale[o]` step.
//
// We compute act_scale on-device (one reduce) and write 1/act_scale-quantized fp8 to Xq.  For the
// proxy bench correctness, a FIXED conservative scale also works; we use a real amax reduce so the
// path matches a production engine and keeps quant error minimal.
// --------------------------------------------------------------------------------------------

// Fused RMSNorm(h) -> y, then quantize y to fp8 e4m3 with per-tensor amax scale.
//   Xq[k] = quantize(y[k]),  y[k] = h[k]*rinv*w_norm[k].   act_scale[0] = amax(y)/448.
//   The GEMM then computes raw_dot[n] = sum_k Xq[k]*W[n,k]; the true value is
//   raw_dot[n]*act_scale[0]*Wscale[n].  (act_scale folded downstream.)  M=1 path (one CTA).
extern "C" __global__ void gemm_rmsnorm_quant(
    const float* __restrict__ h, const float* __restrict__ w_norm,
    __nv_fp8_e4m3* __restrict__ Xq, float* __restrict__ act_scale, int n) {
  extern __shared__ float ybuf[];                       // [n] normed activation
  // ---- block-wide RMSNorm ----
  float part=0.f;
  for (int i=threadIdx.x;i<n;i+=blockDim.x){ float v=h[i]; part+=v*v; }
  #pragma unroll
  for (int o=16;o>0;o>>=1) part+=__shfl_down_sync(0xffffffffu,part,o);
  __shared__ float wss[32]; const int lane=threadIdx.x&31, wid=threadIdx.x>>5;
  if (lane==0) wss[wid]=part; __syncthreads();
  __shared__ float rinv_sh;
  if (threadIdx.x==0){ float ss=0.f; int nw=(blockDim.x+31)>>5; for(int i=0;i<nw;i++) ss+=wss[i];
                       rinv_sh=rsqrtf(ss/n+RMS_EPS); }
  __syncthreads();
  const float rinv=rinv_sh;
  // ---- write normed y to smem + per-thread amax ----
  float amax=0.f;
  for (int i=threadIdx.x;i<n;i+=blockDim.x){ float v=h[i]*rinv*w_norm[i]; ybuf[i]=v; amax=fmaxf(amax,fabsf(v)); }
  #pragma unroll
  for (int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffffu,amax,o));
  __shared__ float amx[32];
  if (lane==0) amx[wid]=amax; __syncthreads();
  __shared__ float sc_sh, inv_sh;
  if (threadIdx.x==0){ float a=0.f; int nw=(blockDim.x+31)>>5; for(int i=0;i<nw;i++) a=fmaxf(a,amx[i]);
                       float sc=(a>0.f)?(a/448.0f):1.0f; sc_sh=sc; inv_sh=1.0f/sc; act_scale[0]=sc; }
  __syncthreads();
  const float inv=inv_sh;
  for (int i=threadIdx.x;i<n;i+=blockDim.x) Xq[i]=(__nv_fp8_e4m3)(ybuf[i]*inv);
}

// Quantize an already-prepared fp32 activation vector y[n] -> fp8 e4m3 (no RMSNorm).  Used for K3
// (attn_out), K5a (post-attn residual y), K5b (a_glb), lm_head (hn).  act_scale[0]=amax(y)/448.
extern "C" __global__ void gemm_quant(
    const float* __restrict__ y, __nv_fp8_e4m3* __restrict__ Xq,
    float* __restrict__ act_scale, int n) {
  float amax=0.f;
  for (int i=threadIdx.x;i<n;i+=blockDim.x) amax=fmaxf(amax,fabsf(y[i]));
  #pragma unroll
  for (int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffffu,amax,o));
  __shared__ float amx[32]; const int lane=threadIdx.x&31, wid=threadIdx.x>>5;
  if (lane==0) amx[wid]=amax; __syncthreads();
  __shared__ float inv_sh;
  if (threadIdx.x==0){ float a=0.f; int nw=(blockDim.x+31)>>5; for(int i=0;i<nw;i++) a=fmaxf(a,amx[i]);
                       float sc=(a>0.f)?(a/448.0f):1.0f; act_scale[0]=sc; inv_sh=1.0f/sc; }
  __syncthreads();
  const float inv=inv_sh;
  for (int i=threadIdx.x;i<n;i+=blockDim.x) Xq[i]=(__nv_fp8_e4m3)(y[i]*inv);
}

} // namespace q3
