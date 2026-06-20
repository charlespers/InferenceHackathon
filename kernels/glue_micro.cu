// glue_micro.cu — microbench the per-layer "glue" kernels (quant / rmsnorm-quant / epilogues / k5b
// down GEMV / K2 reduce) that the segment profile showed dominate the kernels floor beyond the GEMM
// panels.  Goal: find the cheapest config for each, on 1 GPU, realistic Qwen3-235B TP=8 shard shapes.
//
// Build: nvcc -arch=sm_90a -O3 --use_fast_math -I /root/e2e glue_micro.cu -lcuda -o /tmp/glue_micro
// Run:   /tmp/glue_micro
#define DSTP8_NO_MAIN
#include "decode_step_tp8.cu"   // pulls in all kernels + shapes (TP, HIDDEN, etc.)

template <typename F>
static double timeit(int reps, F launch) {
  cudaStream_t s; CK(cudaStreamCreate(&s));
  for (int i=0;i<50;i++) launch(s);
  CK(cudaStreamSynchronize(s));
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a,s));
  for (int i=0;i<reps;i++) launch(s);
  CK(cudaEventRecord(b,s)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms,a,b));
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b)); CK(cudaStreamDestroy(s));
  return (double)ms/reps*1e3; // us/launch
}

int main() {
  CK(cudaSetDevice(0));
  const int REPS = 4000;
  // buffers
  float *h, *w_norm, *act_scale, *out, *Wscale, *attn_out, *a_glb, *sel_w_f;
  __nv_fp8_e4m3 *xq;
  __nv_bfloat16 *D;
  int *sel_idx;
  CK(cudaMalloc(&h, HIDDEN*sizeof(float)));
  CK(cudaMalloc(&w_norm, HIDDEN*sizeof(float)));
  CK(cudaMalloc(&act_scale, 16*sizeof(float)));
  CK(cudaMalloc(&out, HIDDEN*sizeof(float)));
  CK(cudaMalloc(&Wscale, HIDDEN*sizeof(float)));
  CK(cudaMalloc(&attn_out, Q_DIM_RANK*sizeof(float)));
  CK(cudaMalloc(&xq, (size_t)HIDDEN*16*sizeof(__nv_fp8_e4m3)));
  CK(cudaMalloc(&D, (size_t)16*HIDDEN*sizeof(__nv_bfloat16)));
  CK(cudaMemset(D,0,(size_t)16*HIDDEN*sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&a_glb, (size_t)TOP_K*MOE_INTER_RANK*sizeof(float)));
  CK(cudaMalloc(&sel_idx, TOP_K*sizeof(int)));
  CK(cudaMalloc(&sel_w_f, TOP_K*sizeof(float)));
  // expert weight ptr arrays for k5b down GEMV
  const size_t d_n = (size_t)HIDDEN*MOE_INTER_RANK;
  std::vector<fp8*> Wd_dp(TOP_K); std::vector<float*> Sd_dp(TOP_K);
  for (int e=0;e<TOP_K;++e){ CK(cudaMalloc(&Wd_dp[e], d_n*sizeof(fp8))); CK(cudaMalloc(&Sd_dp[e], HIDDEN*sizeof(float))); }
  std::vector<fp8*> Wd_full(N_EXPERTS); std::vector<float*> Sd_full(N_EXPERTS);
  for (int e=0;e<N_EXPERTS;++e){ Wd_full[e]=Wd_dp[e%TOP_K]; Sd_full[e]=Sd_dp[e%TOP_K]; }
  const fp8** Wd_d; const float** Wd_scale_d;
  CK(cudaMalloc(&Wd_d, N_EXPERTS*sizeof(fp8*))); CK(cudaMemcpy(Wd_d, Wd_full.data(), N_EXPERTS*sizeof(fp8*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&Wd_scale_d, N_EXPERTS*sizeof(float*))); CK(cudaMemcpy(Wd_scale_d, Sd_full.data(), N_EXPERTS*sizeof(float*), cudaMemcpyHostToDevice));
  float* moe_partial; CK(cudaMalloc(&moe_partial, HIDDEN*sizeof(float)));
  { std::vector<int> si(TOP_K); for(int i=0;i<TOP_K;++i) si[i]=i;
    CK(cudaMemcpy(sel_idx, si.data(), TOP_K*sizeof(int), cudaMemcpyHostToDevice)); }

  CK(cudaFuncSetAttribute(gemm_rmsnorm_quant, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(HIDDEN*sizeof(float))));
  const size_t hsmem = (size_t)HIDDEN*sizeof(float);

  printf("== GLUE MICROBENCH (1 GPU, HIDDEN=%d, Q_DIM_RANK=%d, %d reps) ==\n", HIDDEN, Q_DIM_RANK, REPS);
  printf("%-46s %10s\n","kernel/config","us/launch");

  // --- gemm_rmsnorm_quant: 1-CTA 1024 (current K1/K4) vs multi-CTA variants ---
  printf("%-46s %10.3f\n","rmsnorm_quant <<<1,1024>>> (CURRENT K1/K4)",
    timeit(REPS,[&](cudaStream_t s){ gemm_rmsnorm_quant<<<1,1024,hsmem,s>>>(h,w_norm,xq,act_scale,HIDDEN); }));
  printf("%-46s %10.3f\n","rmsnorm_quant <<<1,512>>>",
    timeit(REPS,[&](cudaStream_t s){ gemm_rmsnorm_quant<<<1,512,hsmem,s>>>(h,w_norm,xq,act_scale,HIDDEN); }));

  // --- gemm_quant: 1-CTA 1024 (K3,lmhead) vs 32x256 (K5) ---
  printf("%-46s %10.3f\n","quant <<<1,1024>>> over Q_DIM_RANK (CURRENT K3)",
    timeit(REPS,[&](cudaStream_t s){ gemm_quant<<<1,1024,0,s>>>(attn_out,xq,act_scale,Q_DIM_RANK); }));
  printf("%-46s %10.3f\n","quant <<<1,1024>>> over HIDDEN (CURRENT lmhead)",
    timeit(REPS,[&](cudaStream_t s){ gemm_quant<<<1,1024,0,s>>>(h,xq,act_scale,HIDDEN); }));
  printf("%-46s %10.3f\n","quant <<<32,256>>> over HIDDEN (CURRENT K5)",
    timeit(REPS,[&](cudaStream_t s){ gemm_quant<<<32,256,0,s>>>(h,xq,act_scale,HIDDEN); }));

  // --- epilogues ---
  printf("%-46s %10.3f\n","gemm_epi_scale <<<32,256>>> (K3)",
    timeit(REPS,[&](cudaStream_t s){ gemm_epi_scale<<<32,256,0,s>>>(D,Wscale,act_scale,out,HIDDEN,16); }));
  printf("%-46s %10.3f\n","gemm_epi_k5a <<<64,256>>> (K5a SiLU)",
    timeit(REPS,[&](cudaStream_t s){ gemm_epi_k5a<<<64,256,0,s>>>(D,sel_idx,Wd_scale_d,act_scale,a_glb,TOP_K,16); }));

  // --- k5b down GEMV (current: RB=16, block 512) ---
  K5Launch k5; k5.block=512;
  { int wpc=k5.block>>5; auto ctas_for=[&](int rows,int R){int g=(rows+R-1)/R;int n=(g+wpc-1)/wpc;return std::min(std::max(n,132),264);};
    k5.ctasB=ctas_for(TOP_K*HIDDEN,TP8_K5_RB); k5.smemB=(size_t)TOP_K*MOE_INTER_RANK*sizeof(float); }
  CK(cudaFuncSetAttribute(tp8_k5b_down, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)k5.smemB));
  printf("%-46s %10.3f\n","tp8_k5b_down (CURRENT RB,block512)",
    timeit(REPS,[&](cudaStream_t s){ tp8_k5b_down<<<k5.ctasB,k5.block,k5.smemB,s>>>(sel_idx,sel_w_f,Wd_d,Wd_scale_d,a_glb,moe_partial,TOP_K); }));

  // --- residual_add ---
  printf("%-46s %10.3f\n","tp8_residual_add <<<32,256>>>",
    timeit(REPS,[&](cudaStream_t s){ tp8_residual_add<<<32,256,0,s>>>(h,out,out); }));

  // --- final norm ---
  printf("%-46s %10.3f\n","tp8_final_norm <<<1,256>>>",
    timeit(REPS,[&](cudaStream_t s){ tp8_final_norm<<<1,256,0,s>>>(h,w_norm,out); }));

  printf("== done ==\n");
  return 0;
}
