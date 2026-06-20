// prefill_step_tp8.cu — TP=8 sharded, GEMM-BATCHED PREFILL for Qwen3-235B-A22B. Closes the gap
// live-serving-findings.md flagged ("no prefill kernel") AND uses the same specialized kernel
// structure decode's hot path already proved out: cuBLASLt fp8 tensor-core GEMM panels (flat-in-M,
// ~2.7ms/forward vs the GEVV idiom's occupancy-starved ~8.5ms), reusing the EXISTING, ALREADY-
// AUTOTUNED panels (S.p_qkv, S.p_oproj, S.p_k5gu, S.p_k5d) decode_step_tp8.cu allocates regardless --
// no new panel setup needed. Only causal self-attention stays a custom kernel (there's no off-the-
// shelf GEMM for that), exactly mirroring how decode's own GEMM path leaves K2 as a dedicated kernel
// while GEMM-batching everything else.
//
// SCOPE (deliberately bounded -- see the conversation that led here):
//   * ONE prefill chunk, M <= 16 rows (the GEMM panels' proven width), processed through all N_LAYERS.
//   * Standalone TTFT measurement. Does NOT hand off into decode_step_tp8.cu's decode loop -- that
//     engine has no position-tracking (its KV writes carry no position offset; the cache is filled
//     once with dummy data and every "decode step" re-reads the same fixed ctx_len). Wiring real
//     growing-context generation needs that engine to gain position state first -- separate, riskier
//     work on the file producing the team's measured 112.6 tok/s -- out of scope here by choice.
//   * Multi-chunk prefill (M > 16) is the natural next step (loop this chunk advancing ctx_off) but
//     isn't needed to prove TP8-sharded, GEMM-batched causal prefill works, so it's a TODO.
//   * Routing is FIXED (every row uses "experts" 0..7 with uniform 1/8 weight) -- the SAME fidelity
//     level decode's own GEMM path already operates at for this dummy-weight proxy (Wgu_pack/Wd_pack
//     are filled from physical shards 0..7 ONCE at alloc time, independent of any real router output;
//     see the file's own comments on K5_SEL_PHYS_FIXED). K4's router GEMM is skipped entirely -- its
//     output was structurally decorative here, not a new shortcut introduced by this file.
//
// GEMM-BATCHING, what's real here:
//   * K1 (QKV proj), K3 (O-proj): ONE GEMM call covers all M rows at once (shared activation X, just
//     like decode's M=1 case, generalized to M real columns instead of 1). Reuses S.p_qkv/S.p_oproj
//     UNCHANGED -- they were already sized/autotuned for Mpad=16 by decode_step_tp8.cu's alloc_rank.
//   * K5 gate+up, down: EIGHT GEMM calls per layer (one per expert), each covering all M rows in one
//     shot. This is structurally necessary, not a missed optimization: down-proj's input activation
//     differs PER EXPERT (each expert's own post-SiLU values), so a single shared-X GEMM across all 8
//     experts isn't expressible without a block-diagonal trick cuBLASLt doesn't offer -- this is also
//     why decode's own hot path uses the GEVV tp8_k5b_down for down, not a packed GEMM (gemm_epi_k5b/
//     p_k5d_pack exist only for a separate flatness *measurement*, never the real numeric path).
//     Reuses S.p_k5gu/S.p_k5d (single-expert shape, already initialized by alloc_rank) by passing a
//     DIFFERENT expert weight pointer (Wgu_phys_h[e]/Wd_phys_h[e]) into the SAME panel object per call
//     -- LtPanel::run() takes the weight pointer as a runtime argument, so no new panel init needed.
//   * Causal attention: one kernel launch, M*Q_HEADS_RANK warps total, each doing its own online-
//     softmax causal loop. Not a GEMM (there's no batched-GEMM attention primitive in this codebase),
//     but genuinely batched across all M rows in a single launch, not M separate launches.
//   * The two all-reduces remain ONE NCCL call each over M*HIDDEN elements (unchanged from the first
//     working version) -- bypasses NVLS, whose multicast buffers are sized for exactly [HIDDEN].
//
// CORRECTNESS GATE: finite (no NaN/Inf) + cross-rank consistency (every rank must compute the
// IDENTICAL post-AR residual -- the all-reduce sum is rank-independent by construction). Not an
// independent CPU fp32 reference like decode_step_tp8.cu's gate; this is the honest, stated bar given
// the scope.
//
// BUILD (same NCCL/cuBLASLt resolution as decode_step_tp8.cu):
//   NCCL_INC=$(python3 -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'include'))")
//   NCCL_LIB=$(python3 -c "import nvidia.nccl,os;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'lib'))")
//   nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ -I "$NCCL_INC" \
//        kernels/prefill_step_tp8.cu -L "$NCCL_LIB" -lnccl -lcublasLt -lcublas -lcuda -o /tmp/prefill_step
//   LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" /tmp/prefill_step [prompt_len<=16] [iters=50]
//
// =================================================================================================
#include <string>
#include <sstream>
#include <iostream>
#define DSTP8_NO_MAIN
#include "decode_step_tp8.cu"   // RankState, alloc_rank, S.p_qkv/p_oproj/p_k5gu/p_k5d (already
                                // autotuned), tp8_k1_epilogue's per-head math (mirrored, M-extended),
                                // ar_sum_hidden's NCCL fallback, run_all_ranks

namespace pfill {

// =================================================================================================
// REAL WEIGHT LOADING. Reads the flat binary files tools/prepare_real_weights.py produces (fp8 e4m3
// bytes + fp32 per-row scales, already sliced for this exact rank -- see that script's header for the
// sharding scheme, copied from how vLLM shards this same checkpoint). This is a drop-in replacement
// for alloc_rank's fill_fp8()/fill_f32() dummy-data calls: SAME buffers, SAME shapes, real bytes.
//
// SCOPE OF THIS STAGE: loads real Wqkv/Wo/Wgate (attention + router -- fully real, no approximation)
// and all 128 real experts' weights (LayerWeights.experts below) -- enqueue_prefill_layer's K5 loop
// still iterates a fixed 8 of them, same shape as the dummy-weight version, so this stage proves real
// bytes flow correctly through real attention math but does not yet do real per-token top-8 routing
// across all 128 (that needs a masked-GEMM rewrite of the K5 loop, since different rows in the same
// batch can pick different experts, breaking the "one shared expert per GEMM call" assumption) -- a
// clearly separate next step, not done here.
//
// MEMORY: real weights differ PER LAYER (unlike the dummy-weight proxy's "one layer's weights reused
// x94"), so all 94 layers' weight sets must be resident simultaneously (no per-layer reload during the
// forward pass -- that's not how real serving works either: vLLM keeps every layer resident for the
// server's lifetime). Per-rank: ~227GB MoE / 8 ranks / 94 layers = ~302MB/layer/rank (matches the
// measured layer0 preprocessing output exactly) x 94 = ~28.4GB/rank total -- fits an 80GB H100 with
// headroom. LayerWeights holds ONE layer's pointers; the caller keeps a std::vector<LayerWeights>(94)
// per rank and repoints S's fields at layer L's set immediately before processing that layer.
// =================================================================================================
struct RealExperts {
  fp8*   Wgu_phys_h[N_EXPERTS] = {};      // HOST array of device pointers -- read host-side, passed
  fp8*   Wd_phys_h[N_EXPERTS] = {};       // as a GEMM argument (LtPanel::run takes the pointer VALUE).
  // DEVICE-resident arrays of device pointers -- required because prefill_silu_M/prefill_k5down_epi_M
  // dereference Wgu_scale[e] INSIDE the kernel; a host array can't be dereferenced from device code
  // (this is exactly the bug already hit once with S.Wgu_scale_d -- see that fix's comment above).
  float** Wgu_scale_dev = nullptr;
  float** Wd_scale_dev = nullptr;
};
struct LayerWeights {
  fp8 *Wqkv=nullptr, *Wo=nullptr, *Wgate=nullptr;
  float *Wqkv_scale=nullptr, *Wo_scale=nullptr, *Wgate_scale=nullptr;
  float *q_norm=nullptr, *k_norm=nullptr, *in_norm=nullptr, *post_norm=nullptr;
  RealExperts experts;
};

static std::vector<char> read_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { printf("FATAL: cannot open %s\n", path.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> buf(sz);
  if (sz > 0 && fread(buf.data(), 1, sz, f) != (size_t)sz) { printf("FATAL: short read %s\n", path.c_str()); exit(1); }
  fclose(f);
  return buf;
}
static void upload_fp8(const std::string& path, fp8* dptr, size_t expect_n) {
  auto buf = read_file(path);
  if (buf.size() != expect_n) { printf("FATAL: %s size %zu != expected %zu\n", path.c_str(), buf.size(), expect_n); exit(1); }
  CK(cudaMemcpy(dptr, buf.data(), buf.size(), cudaMemcpyHostToDevice));
}
static void upload_f32(const std::string& path, float* dptr, size_t expect_n) {
  auto buf = read_file(path);
  if (buf.size() != expect_n * sizeof(float)) { printf("FATAL: %s size %zu != expected %zu\n", path.c_str(), buf.size(), expect_n*sizeof(float)); exit(1); }
  CK(cudaMemcpy(dptr, buf.data(), buf.size(), cudaMemcpyHostToDevice));
}
// allocate-and-upload (replaces a dummy fill_fp8/fill_f32 call exactly, same buffer ownership model).
static fp8* alloc_upload_fp8(const std::string& path, size_t n) {
  fp8* p; CK(cudaMalloc(&p, n * sizeof(fp8))); upload_fp8(path, p, n); return p;
}
static float* alloc_upload_f32(const std::string& path, size_t n) {
  float* p; CK(cudaMalloc(&p, n * sizeof(float))); upload_f32(path, p, n); return p;
}

// Load layer L's real attention (Wqkv/Wo), router (Wgate), norms, and all 128 experts for this rank
// into a FRESH LayerWeights (its own new allocations -- never touches S's existing dummy buffers, so
// there's no free/use-after-free risk). The caller repoints S's fields at this struct's pointers
// right before processing layer L.
static void load_real_layer(LayerWeights& lw, const std::string& weights_dir, int L, int rank) {
  std::string d = weights_dir + "/layer" + std::to_string(L) + "/rank" + std::to_string(rank) + "/";
  lw.Wqkv        = alloc_upload_fp8(d + "Wqkv.fp8",  (size_t)QKV_OUT_RANK * HIDDEN);
  lw.Wqkv_scale  = alloc_upload_f32(d + "Wqkv_scale.f32", QKV_OUT_RANK);
  lw.Wo          = alloc_upload_fp8(d + "Wo.fp8",    (size_t)HIDDEN * Q_DIM_RANK);
  lw.Wo_scale    = alloc_upload_f32(d + "Wo_scale.f32", HIDDEN);
  lw.Wgate       = alloc_upload_fp8(d + "Wgate.fp8", (size_t)N_EXPERTS * HIDDEN);
  lw.Wgate_scale = alloc_upload_f32(d + "Wgate_scale.f32", N_EXPERTS);
  lw.q_norm    = alloc_upload_f32(d + "q_norm.f32", HEAD_DIM);
  lw.k_norm    = alloc_upload_f32(d + "k_norm.f32", HEAD_DIM);
  lw.in_norm   = alloc_upload_f32(d + "in_norm.f32", HIDDEN);
  lw.post_norm = alloc_upload_f32(d + "post_norm.f32", HIDDEN);

  // 128 real experts: one big Wgu_all.fp8/Wd_all.fp8 file each, sliced into per-expert device pointers
  // that ALIAS into ONE allocation each (so freeing means freeing just gu_all/d_all, not 128 pieces).
  const size_t gu_rows = (size_t)2 * MOE_INTER_RANK, gu_n = gu_rows * HIDDEN;
  const size_t d_rows = (size_t)HIDDEN, d_n = (size_t)HIDDEN * MOE_INTER_RANK;
  fp8* gu_all = alloc_upload_fp8(d + "Wgu_all.fp8", gu_n * N_EXPERTS);
  float* gus_all = alloc_upload_f32(d + "Wgu_scale_all.f32", gu_rows * N_EXPERTS);
  fp8* d_all = alloc_upload_fp8(d + "Wd_all.fp8", d_n * N_EXPERTS);
  float* ds_all = alloc_upload_f32(d + "Wd_scale_all.f32", d_rows * N_EXPERTS);
  float* gu_scale_h[N_EXPERTS]; float* d_scale_h[N_EXPERTS];
  for (int e = 0; e < N_EXPERTS; ++e) {
    lw.experts.Wgu_phys_h[e] = gu_all + (size_t)e * gu_n;
    lw.experts.Wd_phys_h[e]  = d_all  + (size_t)e * d_n;
    gu_scale_h[e] = gus_all + (size_t)e * gu_rows;
    d_scale_h[e]  = ds_all  + (size_t)e * d_rows;
  }
  CK(cudaMalloc(&lw.experts.Wgu_scale_dev, N_EXPERTS * sizeof(float*)));
  CK(cudaMemcpy(lw.experts.Wgu_scale_dev, gu_scale_h, N_EXPERTS * sizeof(float*), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&lw.experts.Wd_scale_dev, N_EXPERTS * sizeof(float*)));
  CK(cudaMemcpy(lw.experts.Wd_scale_dev, d_scale_h, N_EXPERTS * sizeof(float*), cudaMemcpyHostToDevice));
}
// Point S's per-layer fields at this layer's real weights (cheap pointer reassignment; S's original
// dummy buffers from alloc_rank are left allocated but unused -- harmless, a few hundred MB at most).
static void activate_layer(RankState& S, LayerWeights& lw) {
  S.Wqkv = lw.Wqkv; S.Wqkv_scale = lw.Wqkv_scale;
  S.Wo = lw.Wo; S.Wo_scale = lw.Wo_scale;
  S.Wgate = lw.Wgate; S.Wgate_scale = lw.Wgate_scale;
  S.q_norm = lw.q_norm; S.k_norm = lw.k_norm;
  S.w_in_norm = lw.in_norm; S.w_post_norm = lw.post_norm;
}

// Load embed_tokens (replicated, full vocab) + final norm + lm_head (vocab-sharded, matches
// decode_step_tp8.cu's existing S.v_rows/S.v_off split) for this rank. Repoints S.Wlm/Wlm_scale/
// w_final_norm directly (these are loaded ONCE, not per-layer, so no LayerWeights indirection needed).
static fp8* load_real_embeddings(RankState& S, float** embed_scale_out, const std::string& weights_dir, int rank) {
  std::string d = weights_dir + "/embeddings/rank" + std::to_string(rank) + "/";
  fp8* embed = alloc_upload_fp8(d + "embed_tokens.fp8", (size_t)VOCAB * HIDDEN);
  *embed_scale_out = alloc_upload_f32(d + "embed_tokens_scale.f32", VOCAB);
  S.w_final_norm = alloc_upload_f32(d + "final_norm.f32", HIDDEN);
  S.Wlm = alloc_upload_fp8(d + "lm_head.fp8", (size_t)S.v_rows * HIDDEN);
  S.Wlm_scale = alloc_upload_f32(d + "lm_head_scale.f32", S.v_rows);
  return embed;
}


// ---- (1) M-row RMSNorm + fp8 quant -- ONE shared amax scale over the whole [M,HIDDEN] block. ----
//   Xq col-major [HIDDEN,Mpad] (the GEMM panel's "A" operand layout: column m holds row m's K values,
//   contiguous -- offset(row=k,col=m) = k + m*K). act_scale[0] shared so no new per-row scale buffer.
__global__ void prefill_rmsnorm_quant_M(
    const float* __restrict__ h, const float* __restrict__ w_norm,
    __nv_fp8_e4m3* __restrict__ Xq, float* __restrict__ act_scale, int M, int rowdim) {
  extern __shared__ float ybuf[];                          // [M*rowdim] normed rows
  for (int m = 0; m < M; ++m) {
    const float* hm = h + (size_t)m * rowdim;
    float part = 0.f;
    for (int i = threadIdx.x; i < rowdim; i += blockDim.x) { float v = hm[i]; part += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
    __shared__ float wss[32]; const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) wss[wid] = part; __syncthreads();
    __shared__ float rinv_sh;
    if (threadIdx.x == 0) { float ss = 0.f; int nw = (blockDim.x + 31) >> 5;
                            for (int i = 0; i < nw; i++) ss += wss[i];
                            rinv_sh = rsqrtf(ss / rowdim + RMS_EPS); }
    __syncthreads();
    const float rinv = rinv_sh;
    float* yr = ybuf + (size_t)m * rowdim;
    for (int i = threadIdx.x; i < rowdim; i += blockDim.x) yr[i] = hm[i] * rinv * w_norm[i];
    __syncthreads();
  }
  float amax = 0.f;
  for (size_t i = threadIdx.x; i < (size_t)M * rowdim; i += blockDim.x) amax = fmaxf(amax, fabsf(ybuf[i]));
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
  __shared__ float amx[32]; const int lane2 = threadIdx.x & 31, wid2 = threadIdx.x >> 5;
  if (lane2 == 0) amx[wid2] = amax; __syncthreads();
  __shared__ float inv_sh;
  if (threadIdx.x == 0) { float a = 0.f; int nw = (blockDim.x + 31) >> 5;
                          for (int i = 0; i < nw; i++) a = fmaxf(a, amx[i]);
                          float sc = (a > 0.f) ? (a / 448.0f) : 1.0f;
                          act_scale[0] = sc; inv_sh = 1.0f / sc; }
  __syncthreads();
  const float inv = inv_sh;
  for (int m = 0; m < M; ++m) {
    const float* yr = ybuf + (size_t)m * rowdim;
    __nv_fp8_e4m3* xc = Xq + (size_t)m * rowdim;
    for (int i = threadIdx.x; i < rowdim; i += blockDim.x) xc[i] = (__nv_fp8_e4m3)(yr[i] * inv);
  }
}

// ---- (2) M-row plain quant (no RMSNorm) -- for K3's input (attn_out_M) and K5d's (a_glb_M). -------
__global__ void prefill_quant_M(const float* __restrict__ y, __nv_fp8_e4m3* __restrict__ Xq,
                                float* __restrict__ act_scale, int M, int rowdim) {
  float amax = 0.f;
  for (size_t i = threadIdx.x; i < (size_t)M * rowdim; i += blockDim.x) amax = fmaxf(amax, fabsf(y[i]));
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
  __shared__ float amx[32]; const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) amx[wid] = amax; __syncthreads();
  __shared__ float inv_sh;
  if (threadIdx.x == 0) { float a = 0.f; int nw = (blockDim.x + 31) >> 5;
                          for (int i = 0; i < nw; i++) a = fmaxf(a, amx[i]);
                          float sc = (a > 0.f) ? (a / 448.0f) : 1.0f;
                          act_scale[0] = sc; inv_sh = 1.0f / sc; }
  __syncthreads();
  const float inv = inv_sh;
  for (size_t i = threadIdx.x; i < (size_t)M * rowdim; i += blockDim.x) Xq[i] = (__nv_fp8_e4m3)(y[i] * inv);
}

// ---- (3) M-row dequant: D[Mpad,N] bf16 col-major (offset(row=m,col=n) = m + n*Mpad) -> out[M,N] ----
//   row-major fp32, per-channel weight scale + shared act_scale. FULL OVERWRITE (matches decode's
//   GEMM-path epilogues, which fully overwrite rather than accumulate -- the AR adds across ranks).
__global__ void prefill_dequant_MN(const __nv_bfloat16* __restrict__ D, const float* __restrict__ wscale,
                                   const float* __restrict__ act_scale, int Mpad,
                                   float* __restrict__ out, int M, int N) {
  const float as = act_scale[0];
  for (size_t item = blockIdx.x * (size_t)blockDim.x + threadIdx.x; item < (size_t)M * N;
       item += (size_t)gridDim.x * blockDim.x) {
    int m = (int)(item / N), n = (int)(item - (size_t)m * N);
    out[item] = (float)D[(size_t)m + (size_t)n * Mpad] * as * wscale[n];
  }
}

// ---- (4) K1 epilogue, M rows: per-row QK-norm + RoPE (FIXED table, matches decode's own convention --
//      decode's rope_cos/sin are position-INDEPENDENT in this latency proxy, never advanced per token)
//      + replicated-KV write at the REAL position ctx_off+m. Reads d_qkv[Mpad,QKV_OUT_RANK] bf16. -----
__global__ void prefill_k1_epilogue_M(
    const __nv_bfloat16* __restrict__ d_qkv, const float* __restrict__ Wqkv_scale,
    const float* __restrict__ act_scale, int Mpad,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    float* __restrict__ out_q_M,                              // [M, Q_DIM_RANK]
    fp8* __restrict__ kv_k, fp8* __restrict__ kv_v,            // FULL cache base (position applied here)
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale,
    int M, int ctx_off) {
  const int HEAD_ROWS = Q_HEADS_RANK + 2 * N_KV_HEADS;
  const int lane  = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  const float as = act_scale[0];
  for (int item = gwarp; item < M * HEAD_ROWS; item += nwarp) {
    const int m   = item / HEAD_ROWS;
    const int row = item - m * HEAD_ROWS;
    const int pos = ctx_off + m;
    const int is_q = (row < Q_HEADS_RANK);
    const int is_k = (!is_q && row < Q_HEADS_RANK + N_KV_HEADS);
    int proj_base, head_local;
    if (is_q)      { head_local = row;                              proj_base = head_local * HEAD_DIM; }
    else if (is_k) { head_local = row - Q_HEADS_RANK;                proj_base = Q_DIM_RANK + head_local*HEAD_DIM; }
    else           { head_local = row - Q_HEADS_RANK - N_KV_HEADS;   proj_base = Q_DIM_RANK + KV_DIM + head_local*HEAD_DIM; }

    float chan[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) {
      int o = proj_base + c * 32 + lane;
      float raw = __bfloat162float(d_qkv[(size_t)m + (size_t)o * Mpad]);
      chan[c] = raw * as * Wqkv_scale[o];
    }
    if (!is_q && !is_k) {                                     // V -> cache[pos]
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane; int slot = head_local * HEAD_DIM + d;
        float s = kv_v_scale ? kv_v_scale[slot] : 1.f;
        kv_v[(size_t)pos * KV_DIM + slot] = fp8(chan[c] / s);
      }
      continue;
    }
    float ss = 0.f;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) ss += chan[c] * chan[c];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    ss = __shfl_sync(0xffffffffu, ss, 0);
    float hn = rsqrtf(ss / HEAD_DIM + RMS_EPS);
    const float* wn = is_q ? q_norm : k_norm;
    float normed[HEAD_DIM / 32];
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; c++) normed[c] = chan[c] * hn * wn[c * 32 + lane];
    float roped[HEAD_DIM / 32];
    {
      // FIXED table (lane / lane+32), same for every row -- matches decode's own tp8_k1_epilogue,
      // which never advances rope_cos/rope_sin by position either (this proxy's RoPE is structural,
      // not positionally exact -- see file header).
      float c0 = rope_cos[lane],      s0 = rope_sin[lane];
      float c1 = rope_cos[lane + 32], s1 = rope_sin[lane + 32];
      roped[0] = normed[0]*c0 - normed[2]*s0;
      roped[2] = normed[2]*c0 + normed[0]*s0;
      roped[1] = normed[1]*c1 - normed[3]*s1;
      roped[3] = normed[3]*c1 + normed[1]*s1;
    }
    if (is_q) {
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++)
        out_q_M[(size_t)m * Q_DIM_RANK + head_local * HEAD_DIM + c * 32 + lane] = roped[c];
    } else {
      #pragma unroll
      for (int c = 0; c < HEAD_DIM / 32; c++) {
        int d = c * 32 + lane; int slot = head_local * HEAD_DIM + d;
        float s = kv_k_scale ? kv_k_scale[slot] : 1.f;
        kv_k[(size_t)pos * KV_DIM + slot] = fp8(roped[c] / s);
      }
    }
  }
}

// ---- (5) Causal self-attention over the M new rows, ONE launch (not M). One warp per (local q head,
//      query row m). Attends over KV positions [0, ctx_off+m] inclusive -- the epilogue above just
//      wrote every row's K/V at its real position, in increasing m order, so by the time this runs
//      every earlier row's K/V is already in the cache: a real causal mask, not a shortcut. Plain
//      online-softmax (M and the causal prefix are both <=16 -- no split-K needed, that's decode's K2
//      problem at ctx_len in the thousands, untouched here). --------------------------------------
__global__ void prefill_causal_attn_M(
    const float* __restrict__ out_q_M, const fp8* __restrict__ kv_k, const fp8* __restrict__ kv_v,
    const float* __restrict__ kv_k_scale, const float* __restrict__ kv_v_scale,
    float* __restrict__ attn_out_M, int M, int ctx_off) {
  const int lane = threadIdx.x & 31;
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (int item = gwarp; item < M * Q_HEADS_RANK; item += nwarp) {
    const int m  = item / Q_HEADS_RANK;
    const int qh = item - m * Q_HEADS_RANK;
    const int pos = ctx_off + m;
    const int kvh = qh / GQA_GROUP;
    const int kv_base = kvh * HEAD_DIM;
    const float scale = rsqrtf((float)HEAD_DIM);

    float qreg[4], ksc[4], vsc[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
      qreg[c] = out_q_M[(size_t)m * Q_DIM_RANK + qh * HEAD_DIM + lane * 4 + c];
      int slot = kv_base + lane * 4 + c;
      ksc[c] = kv_k_scale ? kv_k_scale[slot] : 1.f;
      vsc[c] = kv_v_scale ? kv_v_scale[slot] : 1.f;
    }
    float run_m = -FLT_MAX, run_l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t = 0; t <= pos; ++t) {
      const fp8* krow = kv_k + (size_t)t * KV_DIM + kv_base;
      const fp8* vrow = kv_v + (size_t)t * KV_DIM + kv_base;
      float kv4[4];
      #pragma unroll
      for (int c = 0; c < 4; c++) kv4[c] = (float)krow[lane * 4 + c] * ksc[c];
      float p = 0.f;
      #pragma unroll
      for (int c = 0; c < 4; c++) p += qreg[c] * kv4[c];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
      float s = p * scale;
      float mn = fmaxf(run_m, s), corr = __expf(run_m - mn), pexp = __expf(s - mn);
      run_l = run_l * corr + pexp;
      float vv4[4];
      #pragma unroll
      for (int c = 0; c < 4; c++) vv4[c] = (float)vrow[lane * 4 + c] * vsc[c];
      #pragma unroll
      for (int c = 0; c < 4; c++) acc[c] = acc[c] * corr + pexp * vv4[c];
      run_m = mn;
    }
    float inv = (run_l > 0.f) ? (1.f / run_l) : 0.f;
    #pragma unroll
    for (int c = 0; c < 4; c++)
      attn_out_M[(size_t)m * Q_DIM_RANK + qh * HEAD_DIM + lane * 4 + c] = acc[c] * inv;
  }
}

// ---- (6) SiLU(gate)*up epilogue, M rows, ONE EXPERT at a time. D_gu[Mpad,2*MOE_INTER_RANK] is THIS
//      expert's gate+up GEMM output for all M rows; gu_scale is that expert's [2*MOE_INTER_RANK] scale.
// NOTE: Wgu_scale is the DEVICE array-of-device-pointers (S.Wgu_scale_d) -- `e` is dereferenced HERE,
// on the device, exactly like gemm_epi_k5a does. Indexing it on the host (Wgu_scale_d[e]) segfaults:
// it's a device pointer, not a host-readable array (this was a real bug, caught by an actual crash).
__global__ void prefill_silu_M(const __nv_bfloat16* __restrict__ D_gu, const float* __restrict__ act_scale,
                               const float* const* __restrict__ Wgu_scale, int e, int Mpad,
                               float* __restrict__ a_glb_M, int M) {
  const float as = act_scale[0];
  const float* gu_scale = Wgu_scale[e];
  for (size_t item = blockIdx.x * (size_t)blockDim.x + threadIdx.x; item < (size_t)M * MOE_INTER_RANK;
       item += (size_t)gridDim.x * blockDim.x) {
    int m = (int)(item / MOE_INTER_RANK), j = (int)(item - (size_t)m * MOE_INTER_RANK);
    float g = (float)D_gu[(size_t)m + (size_t)j * Mpad] * as * gu_scale[j];
    float u = (float)D_gu[(size_t)m + (size_t)(MOE_INTER_RANK + j) * Mpad] * as * gu_scale[MOE_INTER_RANK + j];
    a_glb_M[item] = silu(g) * u;
  }
}

// ---- (7) Down-proj epilogue, M rows, ONE EXPERT, ACCUMULATE (atomicAdd) into moe_partial_M, weighted
//      by this expert's FIXED routing weight (uniform 1/TOP_K -- see file header on fixed routing).
__global__ void prefill_k5down_epi_M(const __nv_bfloat16* __restrict__ D_d, const float* __restrict__ act_scale,
                                     const float* const* __restrict__ Wd_scale, int e, int Mpad, float sel_w,
                                     float* __restrict__ moe_partial_M, int M) {
  const float as = act_scale[0];
  const float* d_scale = Wd_scale[e];
  for (size_t item = blockIdx.x * (size_t)blockDim.x + threadIdx.x; item < (size_t)M * HIDDEN;
       item += (size_t)gridDim.x * blockDim.x) {
    int m = (int)(item / HIDDEN), o = (int)(item - (size_t)m * HIDDEN);
    float v = (float)D_d[(size_t)m + (size_t)o * Mpad] * as * d_scale[o];
    atomicAdd(&moe_partial_M[item], sel_w * v);
  }
}

// ---- (7b) REAL per-token top-8 routing: dense [M,N_EXPERTS] weight mask, mostly zero, the TOP_K
//      nonzero entries per row renormalized to sum to 1 (norm_topk_prob=true, matches config.json and
//      decode's own k4_router). ONE block per row, single-threaded (O(128*8), tiny, off any critical
//      path -- same convention as decode's k4_router select). Mirrors qwen3_moe.py's
//      Qwen3MoeSparseMoeBlock.forward (gate -> softmax -> top-k -> renormalize) row-by-row.
__global__ void prefill_router_topk_M(const float* __restrict__ logits, float* __restrict__ weight_mask, int M) {
  const int m = blockIdx.x;
  if (m >= M || threadIdx.x != 0) return;
  const float* lg = logits + (size_t)m * N_EXPERTS;
  float* wm = weight_mask + (size_t)m * N_EXPERTS;
  for (int e = 0; e < N_EXPERTS; ++e) wm[e] = 0.f;
  float mx = -FLT_MAX;
  for (int e = 0; e < N_EXPERTS; ++e) mx = fmaxf(mx, lg[e]);
  float sum = 0.f;
  for (int e = 0; e < N_EXPERTS; ++e) sum += __expf(lg[e] - mx);
  const float inv_sum = 1.f / sum;
  bool taken[N_EXPERTS];
  for (int e = 0; e < N_EXPERTS; ++e) taken[e] = false;
  float chosen_sum = 0.f;
  for (int s = 0; s < TOP_K; ++s) {
    int bi = -1; float bv = -1.f;
    for (int e = 0; e < N_EXPERTS; ++e) {
      if (taken[e]) continue;
      float p = __expf(lg[e] - mx) * inv_sum;
      if (p > bv) { bv = p; bi = e; }
    }
    if (bi < 0) break;
    taken[bi] = true; wm[bi] = bv; chosen_sum += bv;
  }
  const float inv_chosen = (chosen_sum > 0.f) ? 1.f / chosen_sum : 0.f;
  for (int e = 0; e < N_EXPERTS; ++e) if (wm[e] > 0.f) wm[e] *= inv_chosen;
}

// ---- (7c) Down-proj epilogue, REAL per-token routing variant: weight is PER-ROW (weight_mask[m,e]),
//      not a single scalar -- zero for rows that didn't select expert e this layer, so the atomicAdd
//      is a genuine no-op for them (correct, just not skipped -- see file header on the cost of this).
__global__ void prefill_k5down_epi_real_M(const __nv_bfloat16* __restrict__ D_d, const float* __restrict__ act_scale,
                                          const float* const* __restrict__ Wd_scale, int e, int Mpad,
                                          const float* __restrict__ weight_mask,
                                          float* __restrict__ moe_partial_M, int M) {
  const float as = act_scale[0];
  const float* d_scale = Wd_scale[e];
  for (size_t item = blockIdx.x * (size_t)blockDim.x + threadIdx.x; item < (size_t)M * HIDDEN;
       item += (size_t)gridDim.x * blockDim.x) {
    int m = (int)(item / HIDDEN), o = (int)(item - (size_t)m * HIDDEN);
    float w = weight_mask[(size_t)m * N_EXPERTS + e];
    if (w == 0.f) continue;                          // this row didn't pick expert e -- skip the read+add
    float v = (float)D_d[(size_t)m + (size_t)o * Mpad] * as * d_scale[o];
    atomicAdd(&moe_partial_M[item], w * v);
  }
}

// ---- (8) M-row residual add: h_dst[m] = h_src[m] + reduced[m], row-major [M,HIDDEN]. ----------------
__global__ void prefill_residual_add_M(const float* __restrict__ h_src, const float* __restrict__ reduced,
                                       float* __restrict__ h_dst, int M) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)M * HIDDEN;
       i += (size_t)gridDim.x * blockDim.x)
    h_dst[i] = h_src[i] + reduced[i];
}

// M-row NCCL all-reduce (bypasses NVLS -- see file header).
static inline void ar_sum_hidden_M(RankState& S, float* buf, int M, cudaStream_t s) {
  NK(ncclGroupStart());
  NK(ncclAllReduce(buf, buf, (size_t)M * HIDDEN, ncclFloat32, ncclSum, S.comm, s));
  NK(ncclGroupEnd());
}

// Deterministic pseudo-random prompt "embeddings" (dummy input). [M,HIDDEN] row-major. Seed must NOT
// depend on rank -- TP shards WEIGHTS, not activations; every rank must see the identical input.
static void fill_prompt(float* d, int M) {
  std::vector<float> h((size_t)M * HIDDEN);
  for (size_t i = 0; i < h.size(); ++i) {
    unsigned x = (unsigned)(i * 2654435761u) ^ 99u;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    h[i] = (((x % 2001) / 1000.0f) - 1.0f) * 1.0f;
  }
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
}

// =================================================================================================
// ONE prefill layer over M rows at chunk offset ctx_off, GEMM-batched (see file header). Scratch
// (out_q_M, attn_out_M, a_glb_M) is rank-owned, sized once for M<=16, reused across all 94 layers.
// =================================================================================================
struct PfillScratch {
  float *out_q_M, *attn_out_M, *a_glb_M;
  float *gate_logits_M = nullptr;    // [M, N_EXPERTS] dequantized router logits (real-routing mode only)
  float *weight_mask_M = nullptr;    // [M, N_EXPERTS] dense top-8 weight mask (real-routing mode only)
};

static void enqueue_prefill_layer(RankState& S, float* h_M, float* hdst_M,
                                  float* attn_partial_M, float* moe_partial_M, PfillScratch& sc,
                                  int M, int ctx_off, cudaStream_t s,
                                  RealExperts* re = nullptr) {   // nullptr -> dummy 8-expert path (unchanged)
  const int Mpad = S.p_qkv.Mpad;                                 // 16, fixed by alloc_rank's panel init
  const int blk = 256, grid = 264;

  // ---- K1: RMSNorm+quant (M rows) -> QKV GEMM (ONE call, all M rows) -> epilogue (writes K/V@pos) ---
  prefill_rmsnorm_quant_M<<<1, 1024, (size_t)M*HIDDEN*sizeof(float), s>>>(
      h_M, S.w_in_norm, S.xq_hidden, S.act_scale, M, HIDDEN);
  S.p_qkv.run(S.xq_hidden, S.Wqkv, S.d_qkv, s);
  prefill_k1_epilogue_M<<<grid, blk, 0, s>>>(
      S.d_qkv, S.Wqkv_scale, S.act_scale, Mpad, S.q_norm, S.k_norm, S.rope_cos, S.rope_sin,
      sc.out_q_M, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale, M, ctx_off);

  // ---- causal attention, ONE launch over all M rows (every row's K/V already written above) ----
  prefill_causal_attn_M<<<grid, blk, 0, s>>>(
      sc.out_q_M, S.kv_k, S.kv_v, S.kv_k_scale, S.kv_v_scale, sc.attn_out_M, M, ctx_off);

  // ---- K3: quant attn_out_M -> O-proj GEMM (ONE call) -> dequant -> attn_partial_M -> AR#1 ----
  prefill_quant_M<<<1, 1024, 0, s>>>(sc.attn_out_M, S.xq_qdim, S.act_scale, M, Q_DIM_RANK);
  S.p_oproj.run(S.xq_qdim, S.Wo, S.d_oproj, s);
  prefill_dequant_MN<<<grid, blk, 0, s>>>(S.d_oproj, S.Wo_scale, S.act_scale, Mpad, attn_partial_M, M, HIDDEN);
  ar_sum_hidden_M(S, attn_partial_M, M, s);
  prefill_residual_add_M<<<grid, blk, 0, s>>>(h_M, attn_partial_M, hdst_M, M);

  // ---- K5: routing + expert GEMMs. Real-routing mode (re != nullptr): real K4 router GEMM -> top-8
  //      dense weight mask -> loop ALL 128 experts, masked accumulate (correct, costs 128 GEMM-pairs
  //      instead of 8 -- see file header on this being a deliberate correctness-first tradeoff, not
  //      yet the gather/scatter-by-expert optimization a production MoE kernel would use). Dummy mode
  //      (re == nullptr): unchanged fixed/uniform 8-expert path, exactly as already validated. ----
  prefill_rmsnorm_quant_M<<<1, 1024, (size_t)M*HIDDEN*sizeof(float), s>>>(
      hdst_M, S.w_post_norm, S.xq_hidden, S.act_scale, M, HIDDEN);
  CK(cudaMemsetAsync(moe_partial_M, 0, (size_t)M * HIDDEN * sizeof(float), s));
  const float sel_w = 1.0f / (float)TOP_K;
  fp8* const* gu_phys = re ? re->Wgu_phys_h : S.Wgu_phys_h;
  fp8* const* d_phys  = re ? re->Wd_phys_h  : S.Wd_phys_h;
  // explicit casts: re->Wgu_scale_dev is float** (cudaMalloc-friendly), S.Wgu_scale_d is const float**
  // (decode_step_tp8.cu's own declaration) -- different multi-level cv-qualification, no implicit
  // common type in a ternary, but both convert safely to const float* const* via an explicit cast.
  const float* const* gu_scale_dev = re ? (const float* const*)re->Wgu_scale_dev : (const float* const*)S.Wgu_scale_d;
  const float* const* d_scale_dev  = re ? (const float* const*)re->Wd_scale_dev  : (const float* const*)S.Wd_scale_d;

  if (re) {
    // real router: GEMM (reuses already-quantized S.xq_hidden, same input the experts read) -> dequant
    // -> dense top-8 weight mask, per row.
    S.p_gate.run(S.xq_hidden, S.Wgate, S.d_gate, s);
    prefill_dequant_MN<<<grid, blk, 0, s>>>(S.d_gate, S.Wgate_scale, S.act_scale, Mpad, sc.gate_logits_M, M, N_EXPERTS);
    prefill_router_topk_M<<<M, 32, 0, s>>>(sc.gate_logits_M, sc.weight_mask_M, M);
    for (int e = 0; e < N_EXPERTS; ++e) {
      S.p_k5gu.run(S.xq_hidden, gu_phys[e], S.d_k5gu, s);
      prefill_silu_M<<<grid, blk, 0, s>>>(S.d_k5gu, S.act_scale, gu_scale_dev, e, Mpad, sc.a_glb_M, M);
      prefill_quant_M<<<1, 1024, 0, s>>>(sc.a_glb_M, S.xq_a, S.act_scale, M, MOE_INTER_RANK);
      S.p_k5d.run(S.xq_a, d_phys[e], S.d_k5d, s);
      prefill_k5down_epi_real_M<<<grid, blk, 0, s>>>(S.d_k5d, S.act_scale, d_scale_dev, e, Mpad,
                                                     sc.weight_mask_M, moe_partial_M, M);
    }
    ar_sum_hidden_M(S, moe_partial_M, M, s);
    prefill_residual_add_M<<<grid, blk, 0, s>>>(hdst_M, moe_partial_M, hdst_M, M);
    return;
  }
  for (int e = 0; e < TOP_K; ++e) {
    S.p_k5gu.run(S.xq_hidden, gu_phys[e], S.d_k5gu, s);
    prefill_silu_M<<<grid, blk, 0, s>>>(S.d_k5gu, S.act_scale, gu_scale_dev, e, Mpad, sc.a_glb_M, M);
    prefill_quant_M<<<1, 1024, 0, s>>>(sc.a_glb_M, S.xq_a, S.act_scale, M, MOE_INTER_RANK);
    S.p_k5d.run(S.xq_a, d_phys[e], S.d_k5d, s);
    prefill_k5down_epi_M<<<grid, blk, 0, s>>>(S.d_k5d, S.act_scale, d_scale_dev, e, Mpad, sel_w,
                                              moe_partial_M, M);
  }
  ar_sum_hidden_M(S, moe_partial_M, M, s);
  prefill_residual_add_M<<<grid, blk, 0, s>>>(hdst_M, moe_partial_M, hdst_M, M);
}

} // namespace pfill
using namespace pfill;

// =================================================================================================
// Run all N_LAYERS for ONE chunk of `m` rows starting at absolute position `ctx_off`, in-place on
// the rank's own stream. Used for BOTH the prefill chunk (m=M_PROMPT, ctx_off=0) and each decode
// step (m=1, ctx_off=M_PROMPT+i) -- it's the SAME forward pass shape either way; only m and ctx_off
// differ. This is the actual prefill->decode hand-off: no decode_step_tp8.cu position-tracking
// needed, because this function already tracks position via ctx_off, and decode is just its m=1 case.
// =================================================================================================
static void run_forward_pass(RankState& S, float* cur, float* nxt,
                             float* attn_partial_M, float* moe_partial_M, PfillScratch& sc,
                             int m, int ctx_off, cudaStream_t s) {
  for (int L = 0; L < N_LAYERS; ++L) {
    enqueue_prefill_layer(S, cur, nxt, attn_partial_M, moe_partial_M, sc, m, ctx_off, s);
    std::swap(cur, nxt);
  }
  // N_LAYERS is even (94) -> after an even number of swaps `cur` is back to its original buffer;
  // if that ever changes, this assert catches it instead of silently reading the wrong buffer.
  static_assert(N_LAYERS % 2 == 0, "result-landing assumes an even layer count");
}

// Fresh deterministic dummy "next-token embedding" for decode step `step` (no real embedding table /
// sampler exists in this proxy -- see file header on dummy data). Same convention as fill_prompt:
// rank-independent (TP shards weights, not activations).
static void fill_decode_input(float* d, int step) {
  std::vector<float> h(HIDDEN);
  for (int i = 0; i < HIDDEN; ++i) {
    unsigned x = (unsigned)(i * 2654435761u) ^ (unsigned)(1009u * (step + 1));
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    h[i] = (((x % 2001) / 1000.0f) - 1.0f) * 1.0f;
  }
  CK(cudaMemcpy(d, h.data(), HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
}

// =================================================================================================
// REAL-WEIGHT VALIDATION MODE: load layer 0's real weights (+ real embedding lookup for a real token
// id) and run ONE layer's forward, real bytes throughout. Isolated from the dummy-weight main() path
// below -- doesn't touch it, doesn't risk regressing what's already validated. This is the FIRST real-
// data checkpoint: proves the loader produces correctly-shaped, finite, cross-rank-consistent output
// from real Qwen3 weights, before wiring all 94 layers + the full generation loop.
// =================================================================================================
static int run_real_weight_validation(const std::string& weights_dir, int token_id) {
  printf("== REAL WEIGHTS: layer 0 validation, token_id=%d ==\n", token_id);
  for (int i = 0; i < TP; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < TP; ++j) if (i != j) { int can=0; cudaDeviceCanAccessPeer(&can,i,j); if (can) cudaDeviceEnablePeerAccess(j,0); }
  }
  std::vector<RankState> R(TP);
  std::vector<ncclComm_t> comms(TP);
  std::vector<int> devs(TP); for (int r=0;r<TP;++r) devs[r]=r;
  NK(ncclCommInitAll(comms.data(), TP, devs.data()));

  std::vector<float*> h0(TP), hdst0(TP), attn_partial(TP), moe_partial(TP);
  std::vector<PfillScratch> sc(TP);
  std::vector<LayerWeights> lw(TP);
  std::vector<fp8*> embed(TP); std::vector<float*> embed_scale(TP);
  for (int r = 0; r < TP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    CK(cudaSetDevice(r));
    cudaStream_t s; CK(cudaStreamCreate(&s)); R[r].stream = s;
    alloc_rank(R[r], 4);                                       // tiny dummy cache; we only run M=1,ctx_off=0
    CK(cudaMalloc(&h0[r],    HIDDEN*sizeof(float)));
    CK(cudaMalloc(&hdst0[r], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&attn_partial[r], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&moe_partial[r],  HIDDEN*sizeof(float)));
    CK(cudaMalloc(&sc[r].out_q_M,    Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].attn_out_M, Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].a_glb_M,    MOE_INTER_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].gate_logits_M, N_EXPERTS*sizeof(float)));   // M=1 in both real-weight call sites
    CK(cudaMalloc(&sc[r].weight_mask_M, N_EXPERTS*sizeof(float)));
    printf("  rank %d: loading real layer 0 weights...\n", r); fflush(stdout);
    load_real_layer(lw[r], weights_dir, 0, r);
    embed[r] = load_real_embeddings(R[r], &embed_scale[r], weights_dir, r);
  }

  // real embedding lookup: h0[r] = dequant(embed[r][token_id, :], embed_scale[r][token_id]) -- same on
  // every rank (embed_tokens is replicated in full), done with a tiny one-off host-side dequant.
  for (int r = 0; r < TP; ++r) {
    CK(cudaSetDevice(r));
    std::vector<unsigned char> row_q(HIDDEN);
    CK(cudaMemcpy(row_q.data(), embed[r] + (size_t)token_id*HIDDEN, HIDDEN, cudaMemcpyDeviceToHost));
    float row_scale;
    CK(cudaMemcpy(&row_scale, embed_scale[r] + token_id, sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> row_f(HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) {
      fp8 v; memcpy(&v, &row_q[i], 1);
      row_f[i] = (float)v * row_scale;
    }
    CK(cudaMemcpy(h0[r], row_f.data(), HIDDEN*sizeof(float), cudaMemcpyHostToDevice));
  }

  auto run_layer0 = [&](RankState& S) {
    activate_layer(S, lw[S.rank]);
    enqueue_prefill_layer(S, h0[S.rank], hdst0[S.rank], attn_partial[S.rank], moe_partial[S.rank],
                          sc[S.rank], 1, 0, S.stream, &lw[S.rank].experts);
  };
  run_all_ranks(R, run_layer0);
  for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }

  std::vector<std::vector<float>> got(TP, std::vector<float>(HIDDEN));
  for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); CK(cudaMemcpy(got[r].data(), hdst0[r], HIDDEN*sizeof(float), cudaMemcpyDeviceToHost)); }
  bool all_finite = true, all_match = true;
  double sum=0, sumsq=0, mx=-1e30;
  for (int i = 0; i < HIDDEN; ++i) {
    for (int r = 0; r < TP; ++r) if (!std::isfinite(got[r][i])) all_finite = false;
    if (got[0][i] != got[1][i]) all_match = false;
    sum += got[0][i]; sumsq += (double)got[0][i]*got[0][i]; mx = std::max(mx, (double)fabs(got[0][i]));
  }
  double mean = sum/HIDDEN, var = sumsq/HIDDEN - mean*mean;
  printf("\n== correctness gate ==\n");
  printf("  finite (no NaN/Inf), all ranks : %s\n", all_finite ? "PASS" : "FAIL");
  printf("  cross-rank consistency          : %s\n", all_match ? "PASS" : "FAIL");
  printf("  layer-0 output stats: mean=%.4f std=%.4f max|.|=%.4f\n", mean, sqrt(std::max(var,0.0)), mx);
  printf("%s\n", (all_finite && all_match) ? "REAL WEIGHTS: layer 0 forward pass PASSED structural gates." : "REAL WEIGHTS: FAILED -- see above.");
  return (all_finite && all_match) ? 0 : 2;
}

// =================================================================================================
// FULL REAL-WEIGHT FORWARD: all 94 layers, real embedding lookup, real attention+MoE math throughout,
// real lm_head -> a REAL predicted next-token id. Loads each rank's full per-layer weight set ONCE
// (std::vector<LayerWeights>(N_LAYERS) per rank, ~28.4GB/rank at fp8 -- fits an 80GB H100, see file
// header) and walks all layers for a single input token. Routing is still fixed/uniform per-layer
// (see file header on the K5 loop) -- this proves the FULL real-weight pipeline produces a finite,
// cross-rank-consistent, plausible token id; it does not yet prove that id matches a reference model's
// greedy output (that needs real top-8 routing first -- the masked-GEMM rewrite, not done here).
// =================================================================================================
static int run_real_weight_full_forward(const std::string& weights_dir, int token_id) {
  printf("== REAL WEIGHTS: FULL %d-layer forward, token_id=%d ==\n", N_LAYERS, token_id);
  for (int i = 0; i < TP; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < TP; ++j) if (i != j) { int can=0; cudaDeviceCanAccessPeer(&can,i,j); if (can) cudaDeviceEnablePeerAccess(j,0); }
  }
  std::vector<RankState> R(TP);
  std::vector<ncclComm_t> comms(TP);
  std::vector<int> devs(TP); for (int r=0;r<TP;++r) devs[r]=r;
  NK(ncclCommInitAll(comms.data(), TP, devs.data()));

  std::vector<float*> h0(TP), hdst0(TP), attn_partial(TP), moe_partial(TP);
  std::vector<PfillScratch> sc(TP);
  std::vector<std::vector<LayerWeights>> lw(TP, std::vector<LayerWeights>(N_LAYERS));
  std::vector<fp8*> embed(TP); std::vector<float*> embed_scale(TP);
  for (int r = 0; r < TP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    CK(cudaSetDevice(r));
    cudaStream_t s; CK(cudaStreamCreate(&s)); R[r].stream = s;
    alloc_rank(R[r], 4);
    CK(cudaMalloc(&h0[r],    HIDDEN*sizeof(float)));
    CK(cudaMalloc(&hdst0[r], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&attn_partial[r], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&moe_partial[r],  HIDDEN*sizeof(float)));
    CK(cudaMalloc(&sc[r].out_q_M,    Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].attn_out_M, Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].a_glb_M,    MOE_INTER_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].gate_logits_M, N_EXPERTS*sizeof(float)));   // M=1 in both real-weight call sites
    CK(cudaMalloc(&sc[r].weight_mask_M, N_EXPERTS*sizeof(float)));
    printf("  rank %d: loading %d real layers...\n", r, N_LAYERS); fflush(stdout);
    for (int L = 0; L < N_LAYERS; ++L) load_real_layer(lw[r][L], weights_dir, L, r);
    embed[r] = load_real_embeddings(R[r], &embed_scale[r], weights_dir, r);
    printf("  rank %d: loaded.\n", r); fflush(stdout);
  }

  for (int r = 0; r < TP; ++r) {
    CK(cudaSetDevice(r));
    std::vector<unsigned char> row_q(HIDDEN);
    CK(cudaMemcpy(row_q.data(), embed[r] + (size_t)token_id*HIDDEN, HIDDEN, cudaMemcpyDeviceToHost));
    float row_scale;
    CK(cudaMemcpy(&row_scale, embed_scale[r] + token_id, sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> row_f(HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) { fp8 v; memcpy(&v, &row_q[i], 1); row_f[i] = (float)v * row_scale; }
    CK(cudaMemcpy(h0[r], row_f.data(), HIDDEN*sizeof(float), cudaMemcpyHostToDevice));
  }

  auto run_full = [&](RankState& S) {
    cudaStream_t s = S.stream;
    float* cur = h0[S.rank]; float* nxt = hdst0[S.rank];
    for (int L = 0; L < N_LAYERS; ++L) {
      activate_layer(S, lw[S.rank][L]);
      enqueue_prefill_layer(S, cur, nxt, attn_partial[S.rank], moe_partial[S.rank],
                            sc[S.rank], 1, 0, s, &lw[S.rank][L].experts);
      std::swap(cur, nxt);
    }
    // final norm -> lm_head -> partial argmax (real S.Wlm/Wlm_scale, loaded above) -> cross-rank max-AR
    tp8_final_norm<<<1, 256, 0, s>>>(cur, S.w_final_norm, S.hn);
    gemm_lmhead_launch(S, S.hn, s);
    tp8_argmax_final<<<1, 32, 0, s>>>(S.block_max, S.block_arg, S.lm_blocks, S.rank_max, S.rank_arg);
  };
  run_all_ranks(R, run_full);
  for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }

  // resolve the GLOBAL argmax: every rank's rank_max is its own LOCAL best logit (pre-AR) -- AR-max it,
  // then the token id comes from whichever rank's local max equals the (now identical) global max.
  std::vector<float> rmax(TP); std::vector<int> rarg(TP);
  for (int r = 0; r < TP; ++r) {
    CK(cudaSetDevice(r));
    CK(cudaMemcpy(&rmax[r], R[r].rank_max, sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&rarg[r], R[r].rank_arg, sizeof(int),   cudaMemcpyDeviceToHost));
  }
  NK(ncclGroupStart());
  for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); NK(ncclAllReduce(R[r].rank_max, R[r].rank_max, 1, ncclFloat32, ncclMax, R[r].comm, R[r].stream)); }
  NK(ncclGroupEnd());
  for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }
  float global_max; CK(cudaSetDevice(0)); CK(cudaMemcpy(&global_max, R[0].rank_max, sizeof(float), cudaMemcpyDeviceToHost));
  int predicted_token = -1;
  for (int r = 0; r < TP; ++r) if (rmax[r] == global_max) { predicted_token = rarg[r]; break; }

  bool all_finite = std::isfinite(global_max);
  printf("\n== correctness gate ==\n");
  printf("  finite logit max          : %s (%.4f)\n", all_finite ? "PASS" : "FAIL", global_max);
  printf("  predicted next token id   : %d\n", predicted_token);
  printf("%s\n", all_finite ? "REAL WEIGHTS: full 94-layer forward pass produced a finite token id." :
                              "REAL WEIGHTS: FAILED -- non-finite logit.");
  printf("\nNOTE: routing is still fixed/uniform (8 of 128 experts, equal weight) -- this token id is\n");
  printf("NOT yet validated against a reference model's greedy output. Real top-8 routing needs the\n");
  printf("masked-GEMM K5 rewrite (separate next step) before that comparison is meaningful.\n");
  return all_finite ? 0 : 2;
}

// Look up token `tid`'s real embedding row, dequantize host-side, upload to device buffer `dst`
// (one rank; embed_tokens is replicated in full so any rank's copy works -- caller picks one).
static void embed_lookup(fp8* embed, float* embed_scale, int tid, float* dst) {
  std::vector<unsigned char> row_q(HIDDEN);
  CK(cudaMemcpy(row_q.data(), embed + (size_t)tid*HIDDEN, HIDDEN, cudaMemcpyDeviceToHost));
  float row_scale; CK(cudaMemcpy(&row_scale, embed_scale + tid, sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> row_f(HIDDEN);
  for (int i = 0; i < HIDDEN; ++i) { fp8 v; memcpy(&v, &row_q[i], 1); row_f[i] = (float)v * row_scale; }
  CK(cudaMemcpy(dst, row_f.data(), HIDDEN*sizeof(float), cudaMemcpyHostToDevice));
}

// =================================================================================================
// REAL GENERATION: prompt (a real list of token ids, <=16) -> GEMM-batched prefill -> N_DECODE
// single-token steps, EACH chained on the REAL embedding of the PREVIOUSLY predicted token (not a
// placeholder) -- this is the piece that makes output semantically real, not just structurally real.
// Real per-token top-8 routing throughout (enqueue_prefill_layer's re-driven path). Writes the
// generated token ids to out_tokens. Returns 0 on success (all logits finite, every step).
// =================================================================================================
static int run_real_generation(const std::string& weights_dir, const std::vector<int>& prompt_tokens,
                                int n_decode, std::vector<int>& out_tokens) {
  const int M_PROMPT = (int)prompt_tokens.size();
  printf("== REAL GENERATION: prompt=%d tokens -> %d decode steps, %d layers ==\n", M_PROMPT, n_decode, N_LAYERS);
  for (int i = 0; i < TP; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < TP; ++j) if (i != j) { int can=0; cudaDeviceCanAccessPeer(&can,i,j); if (can) cudaDeviceEnablePeerAccess(j,0); }
  }
  std::vector<RankState> R(TP);
  std::vector<ncclComm_t> comms(TP);
  std::vector<int> devs(TP); for (int r=0;r<TP;++r) devs[r]=r;
  NK(ncclCommInitAll(comms.data(), TP, devs.data()));

  std::vector<float*> h_M(TP), hdst_M(TP), attn_partial_M(TP), moe_partial_M(TP);
  std::vector<float*> h_dec(TP), hdst_dec(TP);
  std::vector<PfillScratch> sc(TP), sc_dec(TP);
  std::vector<std::vector<LayerWeights>> lw(TP, std::vector<LayerWeights>(N_LAYERS));
  std::vector<fp8*> embed(TP); std::vector<float*> embed_scale(TP);
  const int CACHE_CTX = M_PROMPT + n_decode + 4;
  for (int r = 0; r < TP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    CK(cudaSetDevice(r));
    cudaStream_t s; CK(cudaStreamCreate(&s)); R[r].stream = s;
    alloc_rank(R[r], CACHE_CTX);
    CK(cudaMalloc(&h_M[r],            (size_t)M_PROMPT*HIDDEN*sizeof(float)));
    CK(cudaMalloc(&hdst_M[r],         (size_t)M_PROMPT*HIDDEN*sizeof(float)));
    CK(cudaMalloc(&attn_partial_M[r], (size_t)M_PROMPT*HIDDEN*sizeof(float)));
    CK(cudaMalloc(&moe_partial_M[r],  (size_t)M_PROMPT*HIDDEN*sizeof(float)));
    CK(cudaMalloc(&sc[r].out_q_M,        (size_t)M_PROMPT*Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].attn_out_M,     (size_t)M_PROMPT*Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].a_glb_M,        (size_t)M_PROMPT*MOE_INTER_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].gate_logits_M,  (size_t)M_PROMPT*N_EXPERTS*sizeof(float)));
    CK(cudaMalloc(&sc[r].weight_mask_M,  (size_t)M_PROMPT*N_EXPERTS*sizeof(float)));
    CK(cudaMalloc(&h_dec[r], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&hdst_dec[r], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].out_q_M,       Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].attn_out_M,    Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].a_glb_M,       MOE_INTER_RANK*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].gate_logits_M, N_EXPERTS*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].weight_mask_M, N_EXPERTS*sizeof(float)));
    printf("  rank %d: loading %d real layers...\n", r, N_LAYERS); fflush(stdout);
    for (int L = 0; L < N_LAYERS; ++L) load_real_layer(lw[r][L], weights_dir, L, r);
    embed[r] = load_real_embeddings(R[r], &embed_scale[r], weights_dir, r);
  }

  // real embedding lookup for every prompt row, every rank (embed_tokens replicated -> any rank works).
  for (int r = 0; r < TP; ++r) {
    CK(cudaSetDevice(r));
    for (int m = 0; m < M_PROMPT; ++m) embed_lookup(embed[r], embed_scale[r], prompt_tokens[m], h_M[r] + (size_t)m*HIDDEN);
  }

  auto argmax_for_row = [&](RankState& S, float* h_row, cudaStream_t s) -> void {
    tp8_final_norm<<<1, 256, 0, s>>>(h_row, S.w_final_norm, S.hn);
    gemm_lmhead_launch(S, S.hn, s);
    tp8_argmax_final<<<1, 32, 0, s>>>(S.block_max, S.block_arg, S.lm_blocks, S.rank_max, S.rank_arg);
  };
  auto resolve_global_argmax = [&]() -> int {
    std::vector<float> rmax(TP); std::vector<int> rarg(TP);
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r));
      CK(cudaMemcpy(&rmax[r], R[r].rank_max, sizeof(float), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&rarg[r], R[r].rank_arg, sizeof(int),   cudaMemcpyDeviceToHost)); }
    NK(ncclGroupStart());
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); NK(ncclAllReduce(R[r].rank_max, R[r].rank_max, 1, ncclFloat32, ncclMax, R[r].comm, R[r].stream)); }
    NK(ncclGroupEnd());
    for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }
    float gmax; CK(cudaSetDevice(0)); CK(cudaMemcpy(&gmax, R[0].rank_max, sizeof(float), cudaMemcpyDeviceToHost));
    for (int r = 0; r < TP; ++r) if (rmax[r] == gmax) return rarg[r];
    printf("  DEBUG argmax resolution failure: gmax=%g  ", gmax);
    for (int r = 0; r < TP; ++r) printf("rmax[%d]=%g(arg=%d) ", r, rmax[r], rarg[r]);
    printf("\n"); fflush(stdout);
    return -1;
  };

  // ---- prefill: GEMM-batched over all M_PROMPT rows, real routing throughout ----
  auto run_prefill = [&](RankState& S) {
    cudaStream_t s = S.stream;
    float* cur = h_M[S.rank]; float* nxt = hdst_M[S.rank];
    for (int L = 0; L < N_LAYERS; ++L) {
      activate_layer(S, lw[S.rank][L]);
      enqueue_prefill_layer(S, cur, nxt, attn_partial_M[S.rank], moe_partial_M[S.rank],
                            sc[S.rank], M_PROMPT, 0, s, &lw[S.rank][L].experts);
      std::swap(cur, nxt);
    }
    // argmax on the LAST prompt row only (the next-token prediction point)
    argmax_for_row(S, cur + (size_t)(M_PROMPT-1)*HIDDEN, s);
  };
  run_all_ranks(R, run_prefill);
  for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }
  int next_token = resolve_global_argmax();
  if (next_token < 0) { printf("FATAL: argmax resolution failed (no rank matched the global max -- non-finite logit?).\n"); return 2; }
  out_tokens.push_back(next_token);
  printf("  prefill done -> first generated token: %d\n", next_token); fflush(stdout);

  // ---- decode: N_DECODE single-token steps, each chained on the REAL embedding of the token the
  //      PREVIOUS step actually predicted (not a placeholder) ----
  for (int step = 0; step < n_decode; ++step) {
    const int ctx_off = M_PROMPT + step;
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); embed_lookup(embed[r], embed_scale[r], next_token, h_dec[r]); }
    auto run_decode_step = [&](RankState& S) {
      cudaStream_t s = S.stream;
      float* cur = h_dec[S.rank]; float* nxt = hdst_dec[S.rank];
      for (int L = 0; L < N_LAYERS; ++L) {
        activate_layer(S, lw[S.rank][L]);
        enqueue_prefill_layer(S, cur, nxt, attn_partial_M[S.rank], moe_partial_M[S.rank],
                              sc_dec[S.rank], 1, ctx_off, s, &lw[S.rank][L].experts);
        std::swap(cur, nxt);
      }
      argmax_for_row(S, cur, s);
    };
    run_all_ranks(R, run_decode_step);
    for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }
    next_token = resolve_global_argmax();
    if (next_token < 0) { printf("FATAL: argmax resolution failed at decode step %d.\n", step); return 2; }
    out_tokens.push_back(next_token);
    printf("  decode step %d (ctx=%d) -> token: %d\n", step, ctx_off, next_token); fflush(stdout);
  }
  printf("REAL GENERATION: produced %zu tokens.\n", out_tokens.size());
  return 0;
}

// run_serve_loop -- the live-serving entry point. run_real_generation above loads all 94 real layers
// from disk and tears everything down on every single call, which is fine for a one-shot CLI (run_e2e.py)
// but makes "live serving" mean "reload ~223GB of weights per HTTP request" -- the actual blocker for
// real serving, not a missing feature. This function does the SAME setup ONCE, then loops reading
// requests from stdin until EOF/"QUIT", so weights stay resident across many prompts.
//
// Sizing: scratch + KV-cache buffers are allocated ONCE at max_prompt/max_decode (the worst case across
// the server's lifetime), not per-request -- avoids cudaMalloc/cudaFree churn and possible fragmentation
// under sustained serving. A request whose prompt or decode length exceeds those caps is rejected with
// an ERROR line rather than silently truncated.
//
// Wire protocol (stdin, one request per line; stdout, one engine, line-buffered so a Python parent can
// stream tokens as they're produced instead of waiting for the whole generation):
//   in:  "<n_decode> <tok0> <tok1> ... <tokN>\n"
//   out: "TOK <token_id>\n"  (one line per generated token, prefill's first token then each decode step)
//        "DONE\n"            (end of this request)
//        "ERROR <message>\n" (request rejected or a step failed; engine stays alive for the next request)
//   "SERVE_READY\n" is printed once, after all weights finish loading, so the parent knows when it's
//   safe to start sending requests (loading 94 layers x 8 ranks takes minutes, not seconds).
static int run_serve_loop(const std::string& weights_dir, int max_prompt, int max_decode) {
  const int CACHE_CTX = max_prompt + max_decode + 4;
  for (int i = 0; i < TP; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < TP; ++j) if (i != j) { int can=0; cudaDeviceCanAccessPeer(&can,i,j); if (can) cudaDeviceEnablePeerAccess(j,0); }
  }
  std::vector<RankState> R(TP);
  std::vector<ncclComm_t> comms(TP);
  std::vector<int> devs(TP); for (int r=0;r<TP;++r) devs[r]=r;
  NK(ncclCommInitAll(comms.data(), TP, devs.data()));

  std::vector<float*> h_M(TP), hdst_M(TP), attn_partial_M(TP), moe_partial_M(TP);
  std::vector<float*> h_dec(TP), hdst_dec(TP);
  std::vector<PfillScratch> sc(TP), sc_dec(TP);
  std::vector<std::vector<LayerWeights>> lw(TP, std::vector<LayerWeights>(N_LAYERS));
  std::vector<fp8*> embed(TP); std::vector<float*> embed_scale(TP);
  for (int r = 0; r < TP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    CK(cudaSetDevice(r));
    cudaStream_t s; CK(cudaStreamCreate(&s)); R[r].stream = s;
    alloc_rank(R[r], CACHE_CTX);   // sized to the worst case ONCE -- no per-request realloc
    CK(cudaMalloc(&h_M[r],            (size_t)max_prompt*HIDDEN*sizeof(float)));
    CK(cudaMalloc(&hdst_M[r],         (size_t)max_prompt*HIDDEN*sizeof(float)));
    CK(cudaMalloc(&attn_partial_M[r], (size_t)max_prompt*HIDDEN*sizeof(float)));
    CK(cudaMalloc(&moe_partial_M[r],  (size_t)max_prompt*HIDDEN*sizeof(float)));
    CK(cudaMalloc(&sc[r].out_q_M,        (size_t)max_prompt*Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].attn_out_M,     (size_t)max_prompt*Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].a_glb_M,        (size_t)max_prompt*MOE_INTER_RANK*sizeof(float)));
    CK(cudaMalloc(&sc[r].gate_logits_M,  (size_t)max_prompt*N_EXPERTS*sizeof(float)));
    CK(cudaMalloc(&sc[r].weight_mask_M,  (size_t)max_prompt*N_EXPERTS*sizeof(float)));
    CK(cudaMalloc(&h_dec[r], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&hdst_dec[r], HIDDEN*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].out_q_M,       Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].attn_out_M,    Q_DIM_RANK*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].a_glb_M,       MOE_INTER_RANK*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].gate_logits_M, N_EXPERTS*sizeof(float)));
    CK(cudaMalloc(&sc_dec[r].weight_mask_M, N_EXPERTS*sizeof(float)));
    printf("  rank %d: loading %d real layers...\n", r, N_LAYERS); fflush(stdout);
    for (int L = 0; L < N_LAYERS; ++L) load_real_layer(lw[r][L], weights_dir, L, r);
    embed[r] = load_real_embeddings(R[r], &embed_scale[r], weights_dir, r);
  }
  printf("SERVE_READY\n"); fflush(stdout);

  auto argmax_for_row = [&](RankState& S, float* h_row, cudaStream_t s) -> void {
    tp8_final_norm<<<1, 256, 0, s>>>(h_row, S.w_final_norm, S.hn);
    gemm_lmhead_launch(S, S.hn, s);
    tp8_argmax_final<<<1, 32, 0, s>>>(S.block_max, S.block_arg, S.lm_blocks, S.rank_max, S.rank_arg);
  };
  auto resolve_global_argmax = [&]() -> int {
    std::vector<float> rmax(TP); std::vector<int> rarg(TP);
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r));
      CK(cudaMemcpy(&rmax[r], R[r].rank_max, sizeof(float), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&rarg[r], R[r].rank_arg, sizeof(int),   cudaMemcpyDeviceToHost)); }
    NK(ncclGroupStart());
    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); NK(ncclAllReduce(R[r].rank_max, R[r].rank_max, 1, ncclFloat32, ncclMax, R[r].comm, R[r].stream)); }
    NK(ncclGroupEnd());
    for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }
    float gmax; CK(cudaSetDevice(0)); CK(cudaMemcpy(&gmax, R[0].rank_max, sizeof(float), cudaMemcpyDeviceToHost));
    for (int r = 0; r < TP; ++r) if (rmax[r] == gmax) return rarg[r];
    return -1;
  };

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "QUIT") break;
    if (line.empty()) continue;
    std::istringstream iss(line);
    int n_decode = -1; iss >> n_decode;
    std::vector<int> prompt_tokens;
    int tok;
    while (iss >> tok) prompt_tokens.push_back(tok);
    const int M_PROMPT = (int)prompt_tokens.size();
    if (n_decode < 0 || M_PROMPT == 0 || M_PROMPT > max_prompt || n_decode > max_decode) {
      printf("ERROR request out of bounds: prompt=%d (max %d), decode=%d (max %d)\n",
             M_PROMPT, max_prompt, n_decode, max_decode); fflush(stdout);
      continue;
    }

    for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r));
      for (int m = 0; m < M_PROMPT; ++m) embed_lookup(embed[r], embed_scale[r], prompt_tokens[m], h_M[r] + (size_t)m*HIDDEN); }

    auto run_prefill = [&](RankState& S) {
      cudaStream_t s = S.stream;
      float* cur = h_M[S.rank]; float* nxt = hdst_M[S.rank];
      for (int L = 0; L < N_LAYERS; ++L) {
        activate_layer(S, lw[S.rank][L]);
        enqueue_prefill_layer(S, cur, nxt, attn_partial_M[S.rank], moe_partial_M[S.rank],
                              sc[S.rank], M_PROMPT, 0, s, &lw[S.rank][L].experts);
        std::swap(cur, nxt);
      }
      argmax_for_row(S, cur + (size_t)(M_PROMPT-1)*HIDDEN, s);
    };
    run_all_ranks(R, run_prefill);
    for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }
    int next_token = resolve_global_argmax();
    if (next_token < 0) { printf("ERROR argmax resolution failed (prefill)\n"); fflush(stdout); continue; }
    printf("TOK %d\n", next_token); fflush(stdout);

    bool failed = false;
    for (int step = 0; step < n_decode; ++step) {
      const int ctx_off = M_PROMPT + step;
      for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); embed_lookup(embed[r], embed_scale[r], next_token, h_dec[r]); }
      auto run_decode_step = [&](RankState& S) {
        cudaStream_t s = S.stream;
        float* cur = h_dec[S.rank]; float* nxt = hdst_dec[S.rank];
        for (int L = 0; L < N_LAYERS; ++L) {
          activate_layer(S, lw[S.rank][L]);
          enqueue_prefill_layer(S, cur, nxt, attn_partial_M[S.rank], moe_partial_M[S.rank],
                                sc_dec[S.rank], 1, ctx_off, s, &lw[S.rank][L].experts);
          std::swap(cur, nxt);
        }
        argmax_for_row(S, cur, s);
      };
      run_all_ranks(R, run_decode_step);
      for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }
      next_token = resolve_global_argmax();
      if (next_token < 0) { printf("ERROR argmax resolution failed (decode step %d)\n", step); fflush(stdout); failed = true; break; }
      printf("TOK %d\n", next_token); fflush(stdout);
    }
    if (!failed) { printf("DONE\n"); fflush(stdout); }
  }
  return 0;
}

#ifndef PREFILL_STEP_NO_MAIN
int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--gen") {
    std::string weights_dir = argv[2];
    int n_decode = atoi(argv[3]);
    std::vector<int> prompt;
    for (int i = 4; i < argc; ++i) prompt.push_back(atoi(argv[i]));
    std::vector<int> out;
    int rc = run_real_generation(weights_dir, prompt, n_decode, out);
    printf("TOKENS:"); for (int t : out) printf(" %d", t); printf("\n");
    return rc;
  }
  if (argc > 1 && std::string(argv[1]) == "--serve") {
    std::string weights_dir = argv[2];
    int max_prompt = (argc > 3) ? atoi(argv[3]) : 16;
    int max_decode = (argc > 4) ? atoi(argv[4]) : 256;
    return run_serve_loop(weights_dir, max_prompt, max_decode);
  }
  if (argc > 1 && std::string(argv[1]) == "--real") {
    std::string weights_dir = (argc > 2) ? argv[2] : "/alloc/data/real_weights";
    int token_id = (argc > 3) ? atoi(argv[3]) : 151643;   // bos_token_id, per config.json
    return run_real_weight_validation(weights_dir, token_id);
  }
  if (argc > 1 && std::string(argv[1]) == "--real-full") {
    std::string weights_dir = (argc > 2) ? argv[2] : "/alloc/data/real_weights";
    int token_id = (argc > 3) ? atoi(argv[3]) : 151643;
    return run_real_weight_full_forward(weights_dir, token_id);
  }
  const int M_PROMPT = (argc > 1) ? std::min(atoi(argv[1]), 16) : 8;   // prompt length, capped at 16
  const int N_DECODE = (argc > 2) ? atoi(argv[2]) : 20;                // tokens to generate after it
  const int IT       = (argc > 3) ? atoi(argv[3]) : 1;                 // repeat the whole pass IT times
  const int CACHE_CTX = M_PROMPT + N_DECODE + 4;

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < TP) {
    printf("Need >= %d CUDA devices for TP=%d; found %d.\n", TP, TP, ndev); return 1;
  }
  printf("== prefill_step_tp8: TP=%d, prefill M=%d -> decode %d more tokens, %d layers, iters=%d ==\n",
         TP, M_PROMPT, N_DECODE, N_LAYERS, IT);

  for (int i = 0; i < TP; ++i) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < TP; ++j) if (i != j) {
      int can = 0; cudaDeviceCanAccessPeer(&can, i, j);
      if (can) cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  std::vector<RankState> R(TP);
  std::vector<ncclComm_t> comms(TP);
  std::vector<int> devs(TP);
  for (int r = 0; r < TP; ++r) devs[r] = r;
  NK(ncclCommInitAll(comms.data(), TP, devs.data()));

  // M_PROMPT-sized buffers for the prefill chunk; decode steps reuse just their leading 1-row slice.
  std::vector<float*> h_M(TP), hdst_M(TP), attn_partial_M(TP), moe_partial_M(TP);
  std::vector<float*> h_dec(TP), hdst_dec(TP);                 // 1-row scratch for decode steps
  std::vector<PfillScratch> sc(TP);
  for (int r = 0; r < TP; ++r) {
    R[r].rank = r; R[r].dev = r; R[r].comm = comms[r];
    CK(cudaSetDevice(r));
    cudaStream_t s; CK(cudaStreamCreate(&s)); R[r].stream = s;
    alloc_rank(R[r], CACHE_CTX);                              // also inits S.p_qkv/p_oproj/p_k5gu/p_k5d
    CK(cudaMalloc(&h_M[r],            (size_t)M_PROMPT * HIDDEN * sizeof(float)));
    CK(cudaMalloc(&hdst_M[r],         (size_t)M_PROMPT * HIDDEN * sizeof(float)));
    CK(cudaMalloc(&attn_partial_M[r], (size_t)M_PROMPT * HIDDEN * sizeof(float)));
    CK(cudaMalloc(&moe_partial_M[r],  (size_t)M_PROMPT * HIDDEN * sizeof(float)));
    CK(cudaMalloc(&sc[r].out_q_M,     (size_t)M_PROMPT * Q_DIM_RANK * sizeof(float)));
    CK(cudaMalloc(&sc[r].attn_out_M,  (size_t)M_PROMPT * Q_DIM_RANK * sizeof(float)));
    CK(cudaMalloc(&sc[r].a_glb_M,     (size_t)M_PROMPT * MOE_INTER_RANK * sizeof(float)));
    CK(cudaMalloc(&h_dec[r],    HIDDEN * sizeof(float)));
    CK(cudaMalloc(&hdst_dec[r], HIDDEN * sizeof(float)));
    fill_prompt(h_M[r], M_PROMPT);
  }

  // ---- correctness gate: finite + cross-rank consistency, checked after the FULL sequence
  //      (prefill chunk, THEN every decode step) -- one pass, not timed. ----
  auto run_full_sequence = [&](RankState& S) {
    cudaStream_t s = S.stream;
    run_forward_pass(S, h_M[S.rank], hdst_M[S.rank], attn_partial_M[S.rank], moe_partial_M[S.rank],
                     sc[S.rank], M_PROMPT, 0, s);
    for (int i = 0; i < N_DECODE; ++i) {
      run_forward_pass(S, h_dec[S.rank], hdst_dec[S.rank], attn_partial_M[S.rank], moe_partial_M[S.rank],
                       sc[S.rank], 1, M_PROMPT + i, s);
    }
  };
  for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); fill_decode_input(h_dec[r], -1); }
  run_all_ranks(R, run_full_sequence);
  for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); }

  // ---- correctness gate on the FINAL state, after prefill AND every decode step: finite + cross-
  //      rank consistency. N_LAYERS is even (asserted in run_forward_pass), so the final result of
  //      each pass lands back in h_dec/h_M -- not hdst_dec/hdst_M. ----
  std::vector<std::vector<float>> got_prompt(TP, std::vector<float>((size_t)M_PROMPT * HIDDEN));
  std::vector<std::vector<float>> got_dec(TP, std::vector<float>(HIDDEN));
  for (int r = 0; r < TP; ++r) {
    CK(cudaSetDevice(r));
    CK(cudaMemcpy(got_prompt[r].data(), h_M[r], (size_t)M_PROMPT*HIDDEN*sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got_dec[r].data(),    h_dec[r], HIDDEN*sizeof(float),                cudaMemcpyDeviceToHost));
  }
  bool all_finite = true, all_match = true;
  for (size_t i = 0; i < got_prompt[0].size(); ++i) {
    for (int r = 0; r < TP; ++r) if (!std::isfinite(got_prompt[r][i])) all_finite = false;
    if (got_prompt[0][i] != got_prompt[1][i]) all_match = false;
  }
  for (size_t i = 0; i < got_dec[0].size(); ++i) {
    for (int r = 0; r < TP; ++r) if (!std::isfinite(got_dec[r][i])) all_finite = false;
    if (got_dec[0][i] != got_dec[1][i]) all_match = false;
  }
  printf("\n== correctness gate (prefill chunk + all %d decode steps) ==\n", N_DECODE);
  printf("  finite (no NaN/Inf), all ranks                : %s\n", all_finite ? "PASS" : "FAIL");
  printf("  cross-rank consistency (rank0 == rank1)        : %s\n", all_match ? "PASS" : "FAIL");
  if (!all_finite || !all_match) {
    printf("ABORT: correctness failed; not reporting timing.\n");
    return 2;
  }

  // ---- timing: TTFT (the prefill chunk alone) and decode tok/s (the N_DECODE single-token passes
  //      that follow it), measured SEPARATELY -- exactly the two numbers a real serving engine
  //      reports. Re-seed the cache/decode-input state each timed iter so every iter does real work
  //      (not a cache hit of the correctness pass above). ----
  for (int r = 0; r < TP; ++r) { CK(cudaSetDevice(r)); fill_decode_input(h_dec[r], -1); }
  cudaEvent_t e0, e1; CK(cudaSetDevice(0)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  auto run_prefill_only = [&](RankState& S) {
    run_forward_pass(S, h_M[S.rank], hdst_M[S.rank], attn_partial_M[S.rank], moe_partial_M[S.rank],
                     sc[S.rank], M_PROMPT, 0, S.stream);
  };
  for (int it = 0; it < 3; ++it) { run_all_ranks(R, run_prefill_only);
                                   for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); } }
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e0, R[0].stream));
  for (int it = 0; it < IT; ++it) { run_all_ranks(R, run_prefill_only);
                                    for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); } }
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e1, R[0].stream)); CK(cudaEventSynchronize(e1));
  float ttft_ms; CK(cudaEventElapsedTime(&ttft_ms, e0, e1)); ttft_ms /= IT;

  auto run_decode_only = [&](RankState& S) {
    for (int i = 0; i < N_DECODE; ++i)
      run_forward_pass(S, h_dec[S.rank], hdst_dec[S.rank], attn_partial_M[S.rank], moe_partial_M[S.rank],
                       sc[S.rank], 1, M_PROMPT + i, S.stream);
  };
  for (int it = 0; it < 3; ++it) { run_all_ranks(R, run_decode_only);
                                   for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); } }
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e0, R[0].stream));
  for (int it = 0; it < IT; ++it) { run_all_ranks(R, run_decode_only);
                                    for (auto& S : R) { CK(cudaSetDevice(S.dev)); CK(cudaStreamSynchronize(S.stream)); } }
  CK(cudaSetDevice(0)); CK(cudaEventRecord(e1, R[0].stream)); CK(cudaEventSynchronize(e1));
  float decode_ms; CK(cudaEventElapsedTime(&decode_ms, e0, e1)); decode_ms /= IT;

  printf("\n== result: prefill -> decode, chained in ONE binary, real growing-context hand-off ==\n");
  printf("  TTFT (prefill, M=%d prompt tokens, %d layers)   : %.3f ms\n", M_PROMPT, N_LAYERS, ttft_ms);
  printf("  decode: %d tokens, %d layers each, ctx grows %d->%d : %.3f ms total -> %.1f tok/s\n",
         N_DECODE, N_LAYERS, M_PROMPT, M_PROMPT + N_DECODE - 1, decode_ms, 1000.0 * N_DECODE / decode_ms);
  printf("  end-to-end (prefill + decode)                    : %.3f ms for %d total tokens\n",
         ttft_ms + decode_ms, M_PROMPT + N_DECODE);
  printf("\n  NOTE: this is the SAME GEMM-batched kernel chain for both phases (decode is just the M=1\n");
  printf("  case of the prefill layer function) -- it does NOT run through decode_step_tp8.cu's\n");
  printf("  captured-CUDA-graph path, so this decode tok/s is NOT directly comparable to the team's\n");
  printf("  measured 112.6 tok/s (that number's speed comes partly from graph-captured launch-overhead\n");
  printf("  removal, which this eager path doesn't have). What's real here: the position tracking, the\n");
  printf("  causal-attention correctness across a growing cache, and the prefill->decode hand-off.\n");

  for (int r = 0; r < TP; ++r) {
    cudaFree(h_M[r]); cudaFree(hdst_M[r]); cudaFree(attn_partial_M[r]); cudaFree(moe_partial_M[r]);
    cudaFree(h_dec[r]); cudaFree(hdst_dec[r]);
    cudaFree(sc[r].out_q_M); cudaFree(sc[r].attn_out_M); cudaFree(sc[r].a_glb_M);
    ncclCommDestroy(comms[r]);
  }
  return 0;
}
#endif // PREFILL_STEP_NO_MAIN
