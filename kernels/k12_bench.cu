// k12_bench.cu — correctness (vs CPU fp32 reference) + cudaEvent microbench for K1 (attention
// prologue) and K2 (split-KV flash-decode), Qwen3-235B-A22B B=1 decode, sm_90a / H100.
//
//   build:  nvcc -arch=sm_90a -O3 --use_fast_math -I kernels/ kernels/k12_bench.cu -o /tmp/k12
//   run:    CUDA_VISIBLE_DEVICES=0 /tmp/k12 [ctx_len] [n_splits]      (defaults: ctx=4096, auto)
//
// Reports: K1 max-abs-err vs naive prologue; K2 max-abs-err vs naive attention; us/token for both;
//          and for K2 the KV-read GB/s + % of H100 HBM peak (3.35 TB/s).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

#define Q3_K1_LAUNCH_HELPER
#define Q3_K2_LAUNCH_HELPER
#include "k1_attn_prologue.cu"
#include "k2_flash_decode.cu"

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)

// ---- seeded deterministic host RNG (no <random> dependency in device code path) ----
static float rnd(unsigned& s, float lo, float hi) {
  s = s * 1664525u + 1013904223u;
  float u = (float)((s >> 8) & 0xFFFFFF) / (float)0xFFFFFF;     // [0,1)
  return lo + u * (hi - lo);
}

// fp8 e4m3 round-trip (the cache/weight storage error is part of what we validate against).
static float q_fp8(float v) { return (float)(fp8)v; }

// =================================================================================================
// K1 CPU fp32 reference (naive prologue): input-RMSNorm -> QKV GEMV (dequant fp8 W) -> per-head
// QK-norm(q,k) -> RoPE -> store k,v (fp8 round-trip on the cache write).
// =================================================================================================
static void k1_ref(const std::vector<float>& h, const std::vector<float>& w_in_norm,
                   const std::vector<fp8>& Wqkv, const std::vector<float>& Wscale,
                   const std::vector<float>& q_norm, const std::vector<float>& k_norm,
                   const std::vector<float>& rope_cos, const std::vector<float>& rope_sin,
                   const std::vector<float>& kv_k_scale, const std::vector<float>& kv_v_scale,
                   std::vector<float>& out_q, std::vector<float>& k_cache_f, std::vector<float>& v_cache_f) {
  // 1) RMSNorm
  double ss = 0; for (int i = 0; i < HIDDEN; i++) ss += (double)h[i]*h[i];
  float rinv = 1.f / std::sqrt((float)(ss / HIDDEN) + RMS_EPS);
  std::vector<float> x(HIDDEN);
  for (int i = 0; i < HIDDEN; i++) x[i] = h[i] * rinv * w_in_norm[i];

  // 2) QKV GEMV
  std::vector<float> proj(QKV_OUT);
  for (int o = 0; o < QKV_OUT; o++) {
    double a = 0;
    const fp8* wr = &Wqkv[(size_t)o * HIDDEN];
    for (int k = 0; k < HIDDEN; k++) a += (double)((float)wr[k]) * x[k];
    proj[o] = (float)a * Wscale[o];
  }

  // helper: per-head RMSNorm + RoPE on a HEAD_DIM vector.
  auto headnorm_rope = [&](float* v, const std::vector<float>& wn) {
    double s2 = 0; for (int d = 0; d < HEAD_DIM; d++) s2 += (double)v[d]*v[d];
    float hn = 1.f / std::sqrt((float)(s2 / HEAD_DIM) + RMS_EPS);
    float nm[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) nm[d] = v[d] * hn * wn[d];
    int half = HEAD_DIM / 2;
    for (int i = 0; i < half; i++) {
      float c = rope_cos[i], sn = rope_sin[i];
      v[i]        = nm[i]      * c - nm[i+half] * sn;
      v[i+half]   = nm[i+half] * c + nm[i]      * sn;
    }
  };

  // 3) Q heads
  for (int hd = 0; hd < N_Q_HEADS; hd++) {
    float buf[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) buf[d] = proj[hd*HEAD_DIM + d];
    headnorm_rope(buf, q_norm);
    for (int d = 0; d < HEAD_DIM; d++) out_q[hd*HEAD_DIM + d] = buf[d];
  }
  // 4) K heads (QK-norm + RoPE) -> fp8 cache (round-trip)
  for (int hd = 0; hd < N_KV_HEADS; hd++) {
    float buf[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) buf[d] = proj[Q_DIM + hd*HEAD_DIM + d];
    headnorm_rope(buf, k_norm);
    for (int d = 0; d < HEAD_DIM; d++) {
      int slot = hd*HEAD_DIM + d; float s = kv_k_scale[slot];
      k_cache_f[slot] = q_fp8(buf[d] / s) * s;              // store quantized, read back dequantized
    }
  }
  // 5) V heads (no norm/rope) -> fp8 cache
  for (int hd = 0; hd < N_KV_HEADS; hd++) {
    for (int d = 0; d < HEAD_DIM; d++) {
      int slot = hd*HEAD_DIM + d; float s = kv_v_scale[slot];
      float val = proj[Q_DIM + KV_DIM + hd*HEAD_DIM + d];
      v_cache_f[slot] = q_fp8(val / s) * s;
    }
  }
}

// =================================================================================================
// K2 CPU fp32 reference (naive single-query GQA attention over a dequantized fp8 cache).
// =================================================================================================
static void k2_ref(const std::vector<float>& q,
                   const std::vector<float>& kc, const std::vector<float>& vc, // dequantized [ctx,KV_DIM]
                   int ctx_len, std::vector<float>& out) {
  const float scale = 1.f / std::sqrt((float)HEAD_DIM);
  for (int qh = 0; qh < N_Q_HEADS; qh++) {
    int kvh = qh / GQA_GROUP, kb = kvh * HEAD_DIM;
    std::vector<float> logit(ctx_len);
    float mx = -1e30f;
    for (int t = 0; t < ctx_len; t++) {
      double d = 0;
      for (int i = 0; i < HEAD_DIM; i++) d += (double)q[qh*HEAD_DIM+i] * kc[(size_t)t*KV_DIM + kb + i];
      logit[t] = (float)d * scale; mx = std::max(mx, logit[t]);
    }
    double denom = 0; std::vector<float> p(ctx_len);
    for (int t = 0; t < ctx_len; t++) { p[t] = std::exp(logit[t] - mx); denom += p[t]; }
    for (int i = 0; i < HEAD_DIM; i++) {
      double a = 0;
      for (int t = 0; t < ctx_len; t++) a += (double)p[t] * vc[(size_t)t*KV_DIM + kb + i];
      out[qh*HEAD_DIM + i] = (float)(a / denom);
    }
  }
}

int main(int argc, char** argv) {
  const int ctx_len  = (argc > 1) ? atoi(argv[1]) : 4096;
  const int n_splits = (argc > 2) ? atoi(argv[2]) : -1;
  const double PEAK  = (argc > 3) ? atof(argv[3]) : 3350.0;   // GB/s, H100
  unsigned seed = 0x1234abcdu;
  printf("== K1/K2 bench  ctx_len=%d  PEAK=%.0f GB/s ==\n", ctx_len, PEAK);

  // ---------------- K1 host inputs ----------------
  std::vector<float> h(HIDDEN), w_in_norm(HIDDEN), Wscale(QKV_OUT);
  std::vector<float> q_norm(HEAD_DIM), k_norm(HEAD_DIM);
  std::vector<float> rope_cos(HEAD_DIM/2), rope_sin(HEAD_DIM/2);
  std::vector<float> kv_k_scale(KV_DIM), kv_v_scale(KV_DIM);
  std::vector<fp8>   Wqkv((size_t)QKV_OUT * HIDDEN);
  for (auto& v : h)         v = rnd(seed, -1.f, 1.f);
  for (auto& v : w_in_norm) v = rnd(seed, 0.5f, 1.5f);
  for (auto& v : q_norm)    v = rnd(seed, 0.5f, 1.5f);
  for (auto& v : k_norm)    v = rnd(seed, 0.5f, 1.5f);
  for (auto& v : Wscale)    v = rnd(seed, 0.01f, 0.03f);
  for (auto& v : kv_k_scale)v = rnd(seed, 0.02f, 0.05f);
  for (auto& v : kv_v_scale)v = rnd(seed, 0.02f, 0.05f);
  for (int i = 0; i < HEAD_DIM/2; i++) {
    float freq = std::pow(ROPE_THETA, -2.f*i/HEAD_DIM);
    float ang  = freq * 7.f;                                 // pretend position = 7
    rope_cos[i] = std::cos(ang); rope_sin[i] = std::sin(ang);
  }
  for (auto& v : Wqkv) v = (fp8)rnd(seed, -1.f, 1.f);

  // K1 reference
  std::vector<float> out_q_ref(Q_DIM), kc_ref(KV_DIM), vc_ref(KV_DIM);
  k1_ref(h, w_in_norm, Wqkv, Wscale, q_norm, k_norm, rope_cos, rope_sin,
         kv_k_scale, kv_v_scale, out_q_ref, kc_ref, vc_ref);

  // K1 device buffers
  float *d_h,*d_win,*d_Ws,*d_qn,*d_kn,*d_rc,*d_rs,*d_oq,*d_kks,*d_kvs; fp8 *d_W,*d_kk,*d_kv;
  CK(cudaMalloc(&d_h,HIDDEN*4)); CK(cudaMalloc(&d_win,HIDDEN*4)); CK(cudaMalloc(&d_Ws,QKV_OUT*4));
  CK(cudaMalloc(&d_qn,HEAD_DIM*4)); CK(cudaMalloc(&d_kn,HEAD_DIM*4));
  CK(cudaMalloc(&d_rc,HEAD_DIM/2*4)); CK(cudaMalloc(&d_rs,HEAD_DIM/2*4));
  CK(cudaMalloc(&d_kks,KV_DIM*4)); CK(cudaMalloc(&d_kvs,KV_DIM*4));
  CK(cudaMalloc(&d_W,(size_t)QKV_OUT*HIDDEN*sizeof(fp8)));
  CK(cudaMalloc(&d_oq,Q_DIM*4)); CK(cudaMalloc(&d_kk,KV_DIM*sizeof(fp8))); CK(cudaMalloc(&d_kv,KV_DIM*sizeof(fp8)));
  CK(cudaMemcpy(d_h,h.data(),HIDDEN*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_win,w_in_norm.data(),HIDDEN*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_Ws,Wscale.data(),QKV_OUT*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_qn,q_norm.data(),HEAD_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_kn,k_norm.data(),HEAD_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_rc,rope_cos.data(),HEAD_DIM/2*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_rs,rope_sin.data(),HEAD_DIM/2*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_kks,kv_k_scale.data(),KV_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_kvs,kv_v_scale.data(),KV_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_W,Wqkv.data(),(size_t)QKV_OUT*HIDDEN*sizeof(fp8),cudaMemcpyHostToDevice));

  k1_launch(d_h,d_win,d_W,d_Ws,d_qn,d_kn,d_rc,d_rs,d_oq,d_kk,d_kv,d_kks,d_kvs);
  CK(cudaDeviceSynchronize());

  // K1 correctness: compare out_q, and the dequantized k/v cache writes.
  std::vector<float> oq(Q_DIM); std::vector<fp8> kk(KV_DIM), kv(KV_DIM);
  CK(cudaMemcpy(oq.data(),d_oq,Q_DIM*4,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(kk.data(),d_kk,KV_DIM*sizeof(fp8),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(kv.data(),d_kv,KV_DIM*sizeof(fp8),cudaMemcpyDeviceToHost));
  double e_oq=0,e_k=0,e_v=0;
  for (int i=0;i<Q_DIM;i++)  e_oq=std::max(e_oq,(double)std::fabs(oq[i]-out_q_ref[i]));
  for (int i=0;i<KV_DIM;i++){ float gk=(float)kk[i]*kv_k_scale[i], gv=(float)kv[i]*kv_v_scale[i];
    e_k=std::max(e_k,(double)std::fabs(gk-kc_ref[i])); e_v=std::max(e_v,(double)std::fabs(gv-vc_ref[i])); }
  printf("K1  max-abs-err:  out_q=%.3e  k_cache=%.3e  v_cache=%.3e   (target < 1e-2)\n", e_oq,e_k,e_v);

  // ---------------- K2 setup: build a full ctx-length fp8 KV cache ----------------
  std::vector<fp8> KC((size_t)ctx_len*KV_DIM), VC((size_t)ctx_len*KV_DIM);
  std::vector<float> KCf((size_t)ctx_len*KV_DIM), VCf((size_t)ctx_len*KV_DIM);     // dequantized ref
  for (int t=0;t<ctx_len;t++) for (int c=0;c<KV_DIM;c++){
    float vk=rnd(seed,-1.f,1.f)/ (1.f/kv_k_scale[c]);   // scale-aware magnitude so fp8 has range
    float vv=rnd(seed,-1.f,1.f)/ (1.f/kv_v_scale[c]);
    size_t idx=(size_t)t*KV_DIM+c;
    KC[idx]=(fp8)(vk/kv_k_scale[c]); VC[idx]=(fp8)(vv/kv_v_scale[c]);
    KCf[idx]=(float)KC[idx]*kv_k_scale[c]; VCf[idx]=(float)VC[idx]*kv_v_scale[c];
  }
  // query for K2: use a fresh random normed-roped-like q (independent of K1 to isolate K2 math).
  std::vector<float> q2(Q_DIM); for (auto& v:q2) v=rnd(seed,-1.f,1.f);

  std::vector<float> out_ref(Q_DIM);
  k2_ref(q2, KCf, VCf, ctx_len, out_ref);

  // device buffers
  float *d_q2,*d_attn,*d_pm,*d_pl,*d_pacc; fp8 *d_KC,*d_VC;
  int S = (n_splits>0)? n_splits : k2_pick_splits(ctx_len);
  CK(cudaMalloc(&d_q2,Q_DIM*4)); CK(cudaMalloc(&d_attn,Q_DIM*4));
  CK(cudaMalloc(&d_KC,(size_t)ctx_len*KV_DIM*sizeof(fp8))); CK(cudaMalloc(&d_VC,(size_t)ctx_len*KV_DIM*sizeof(fp8)));
  CK(cudaMalloc(&d_pm, k2_partials_elems_m(S)*4));
  CK(cudaMalloc(&d_pl, k2_partials_elems_m(S)*4));
  CK(cudaMalloc(&d_pacc, k2_partials_elems_acc(S)*4));
  CK(cudaMemcpy(d_q2,q2.data(),Q_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_KC,KC.data(),(size_t)ctx_len*KV_DIM*sizeof(fp8),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_VC,VC.data(),(size_t)ctx_len*KV_DIM*sizeof(fp8),cudaMemcpyHostToDevice));

  S = k2_launch(d_q2,d_KC,d_VC,d_kks,d_kvs,ctx_len,d_pm,d_pl,d_pacc,d_attn,n_splits);
  CK(cudaDeviceSynchronize());
  std::vector<float> attn(Q_DIM); CK(cudaMemcpy(attn.data(),d_attn,Q_DIM*4,cudaMemcpyDeviceToHost));
  double e_at=0; for (int i=0;i<Q_DIM;i++) e_at=std::max(e_at,(double)std::fabs(attn[i]-out_ref[i]));
  printf("K2  max-abs-err:  attn_out=%.3e   (splits=%d, target < 1e-2)\n", e_at, S);

  // ---------------- microbench (cudaEvent) ----------------
  cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e)); const int WARM=30,IT=300;
  // K1
  for(int i=0;i<WARM;i++) k1_launch(d_h,d_win,d_W,d_Ws,d_qn,d_kn,d_rc,d_rs,d_oq,d_kk,d_kv,d_kks,d_kvs);
  CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
  for(int i=0;i<IT;i++)   k1_launch(d_h,d_win,d_W,d_Ws,d_qn,d_kn,d_rc,d_rs,d_oq,d_kk,d_kv,d_kks,d_kvs);
  CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e)); float ms1; CK(cudaEventElapsedTime(&ms1,s,e)); ms1/=IT;
  double k1_bytes=(double)QKV_OUT*HIDDEN*sizeof(fp8);       // dominated by reading Wqkv
  printf("K1  %.2f us/token   (Wqkv read %.1f MB -> %.0f GB/s, %.1f%% peak)\n",
         ms1*1e3, k1_bytes/1e6, k1_bytes/1e6/ms1, k1_bytes/1e6/ms1/PEAK*100.0);

  // K2
  for(int i=0;i<WARM;i++) k2_launch(d_q2,d_KC,d_VC,d_kks,d_kvs,ctx_len,d_pm,d_pl,d_pacc,d_attn,n_splits);
  CK(cudaDeviceSynchronize()); CK(cudaEventRecord(s));
  for(int i=0;i<IT;i++)   k2_launch(d_q2,d_KC,d_VC,d_kks,d_kvs,ctx_len,d_pm,d_pl,d_pacc,d_attn,n_splits);
  CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e)); float ms2; CK(cudaEventElapsedTime(&ms2,s,e)); ms2/=IT;
  double k2_bytes=2.0*(double)ctx_len*KV_DIM*sizeof(fp8);   // read K + V cache once each
  printf("K2  %.2f us/token   KV read %.1f MB -> %.0f GB/s, %.1f%% of H100 peak (%.0f GB/s)\n",
         ms2*1e3, k2_bytes/1e6, k2_bytes/1e6/ms2, k2_bytes/1e6/ms2/PEAK*100.0, PEAK);

  printf("== done ==\n");
  return 0;
}
