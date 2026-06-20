// K4 — MoE router, fully on-device (no host sync).
// post-RMSNorm -> gate GEMV (4096->128) -> fp32 softmax over 128 -> top-8 -> renormalize
// selected weights to sum 1 (norm_topk_prob=true). No shared expert.
#include "common.cuh"
using namespace q3;

extern "C" __global__ void k4_router(
    const float* __restrict__ h,            // [HIDDEN] residual (post-attn)
    const float* __restrict__ w_post_norm,  // [HIDDEN]
    const fp8*  __restrict__ Wgate, const float* __restrict__ Wgate_scale, // [N_EXPERTS, HIDDEN]
    int* __restrict__ sel_idx,              // [TOP_K] selected expert ids (out)
    float* __restrict__ sel_w) {            // [TOP_K] renormalized gate weights (out)
  __shared__ float y[HIDDEN];
  __shared__ float logits[N_EXPERTS];
  float ri = rms_inv(h, HIDDEN);
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) y[i] = h[i] * ri * w_post_norm[i];
  __syncthreads();
  // gate GEMV: 128 experts (one thread/expert is fine; 128 is small)
  for (int e = threadIdx.x; e < N_EXPERTS; e += blockDim.x) {
    float acc = 0.f; const fp8* g = Wgate + (size_t)e * HIDDEN;
    for (int k = 0; k < HIDDEN; ++k) acc += y[k] * deq(g[k], Wgate_scale[e]);
    logits[e] = acc;
  }
  __syncthreads();
  // fp32 softmax over all 128, then top-8, then renormalize the 8 to sum 1.
  // TODO(on-box): block-wide max/sum reduce; top-8 via 8-pass argmax or bitonic; lane-parallel.
  if (threadIdx.x == 0) {
    float mx = -1e30f; for (int e=0;e<N_EXPERTS;++e) mx = fmaxf(mx, logits[e]);
    float sum = 0.f;   for (int e=0;e<N_EXPERTS;++e) sum += __expf(logits[e]-mx);
    // probs[e] = expf(logits[e]-mx)/sum;  pick top-8, renormalize:
    float chosen = 0.f;
    for (int s=0;s<TOP_K;++s){ int bi=-1; float bv=-1e30f;
      for(int e=0;e<N_EXPERTS;++e){ float p=__expf(logits[e]-mx)/sum; bool taken=false;
        for(int j=0;j<s;++j) if(sel_idx[j]==e) taken=true;
        if(!taken && p>bv){bv=p;bi=e;} }
      sel_idx[s]=bi; sel_w[s]=bv; chosen+=bv; }
    for (int s=0;s<TOP_K;++s) sel_w[s]/=chosen;            // renormalize to sum 1
  }
  // TODO(on-box): parallelize top-8 (the O(128*8) loop is fine but lane-parallel is cleaner);
  //               keep sel_idx/sel_w in device mem so K5 + EP dispatch read them with no sync.
}
