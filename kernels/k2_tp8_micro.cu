// k2_tp8_micro.cu — microbench K2 variants for the TP=8 shard (8 Q heads/rank, ctx 4096).
// Compares: (A) current 2-kernel split+serial-reduce, (B) fused 1-CTA/head, (C) new parallel reduce.
// Build: nvcc -arch=sm_90a -O3 --use_fast_math -I . k2_tp8_micro.cu -o /tmp/k2mt
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"
using namespace q3;

constexpr int Q_HEADS_RANK = N_Q_HEADS / 8;   // 8
constexpr int Q_DIM_RANK   = Q_HEADS_RANK * HEAD_DIM; // 1024
constexpr int K2_VPL = HEAD_DIM / 32;

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

static __device__ __forceinline__ float wsum(float v){
  #pragma unroll
  for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o);
  return v;
}
static __device__ __forceinline__ void load4(const unsigned* b,int lane,const float* s,float* o){
  unsigned w=b[lane]; __nv_fp8x2_e4m3 lo,hi; lo.__x=(unsigned short)(w&0xffffu); hi.__x=(unsigned short)(w>>16);
  float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi);
  o[0]=fl.x*s[0]; o[1]=fl.y*s[1]; o[2]=fh.x*s[2]; o[3]=fh.y*s[3];
}

// ---------- (A) current path ----------
__global__ void A_partial(const float* q,const fp8* kk,const fp8* vv,const float* ks,const float* vs,
    int ctx,int nsp,int rank,float* pm,float* pl,float* pa){
  const int lane=threadIdx.x&31, wid=threadIdx.x>>5;
  const int lqh=blockIdx.y*(blockDim.x>>5)+wid; if(lqh>=Q_HEADS_RANK) return;
  const int gqh=rank*Q_HEADS_RANK+lqh, split=blockIdx.x, kvh=gqh/GQA_GROUP;
  const int chunk=(ctx+nsp-1)/nsp, t0=split*chunk, t1=min(t0+chunk,ctx);
  const float scale=rsqrtf((float)HEAD_DIM); const int kvb=kvh*HEAD_DIM, c0=kvb+lane*K2_VPL;
  float qr[K2_VPL],kc[K2_VPL],vc[K2_VPL];
  #pragma unroll
  for(int c=0;c<K2_VPL;c++){qr[c]=q[lqh*HEAD_DIM+lane*K2_VPL+c]; kc[c]=ks[c0+c]; vc[c]=vs[c0+c];}
  float m=-FLT_MAX,l=0.f,acc[K2_VPL]; for(int c=0;c<K2_VPL;c++)acc[c]=0.f;
  const unsigned* k32=(const unsigned*)kk; const unsigned* v32=(const unsigned*)vv;
  const int rw=KV_DIM/4, bw=kvb/4;
  for(int t=t0;t<t1;t++){
    float kv[K2_VPL]; load4(k32+(size_t)t*rw+bw,lane,kc,kv);
    float p=0.f; for(int c=0;c<K2_VPL;c++)p+=qr[c]*kv[c];
    float s=wsum(p)*scale; float mn=fmaxf(m,s),co=__expf(m-mn),pe=__expf(s-mn); l=l*co+pe;
    float vr[K2_VPL]; load4(v32+(size_t)t*rw+bw,lane,vc,vr);
    for(int c=0;c<K2_VPL;c++)acc[c]=acc[c]*co+pe*vr[c]; m=mn;
  }
  const size_t pi=(size_t)lqh*nsp+split; if(lane==0){pm[pi]=m;pl[pi]=l;}
  float* ao=pa+pi*HEAD_DIM+lane*K2_VPL; for(int c=0;c<K2_VPL;c++)ao[c]=acc[c];
}
__global__ void A_reduce(const float* pm,const float* pl,const float* pa,int nsp,float* out){
  const int lane=threadIdx.x&31, lqh=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
  if(lqh>=Q_HEADS_RANK) return;
  float m=-FLT_MAX,l=0.f,acc[K2_VPL]; for(int c=0;c<K2_VPL;c++)acc[c]=0.f;
  for(int sp=0;sp<nsp;sp++){ const size_t pi=(size_t)lqh*nsp+sp; float ms=pm[pi],ls=pl[pi];
    if(ls<=0.f)continue; const float* ai=pa+pi*HEAD_DIM+lane*K2_VPL;
    float mn=fmaxf(m,ms),co=__expf(m-mn),cs=__expf(ms-mn); l=l*co+ls*cs;
    for(int c=0;c<K2_VPL;c++)acc[c]=acc[c]*co+ai[c]*cs; m=mn; }
  float inv=(l>0.f)?1.f/l:0.f; float* o=out+lqh*HEAD_DIM+lane*K2_VPL;
  for(int c=0;c<K2_VPL;c++)o[c]=acc[c]*inv;
}

// ---------- (C) NEW parallel reduce: 1 CTA per head, W warps tree-combine the nsp splits ----------
// Each of the W warps handles a strided subset of splits into a per-warp (m,l,acc), then warp 0
// combines the W partials from smem.  W*HEAD_DIM smem.  grid = Q_HEADS_RANK CTAs.
template<int W>
__global__ void C_reduce(const float* pm,const float* pl,const float* pa,int nsp,float* out){
  const int lane=threadIdx.x&31, wid=threadIdx.x>>5;
  const int lqh=blockIdx.x;
  float m=-FLT_MAX,l=0.f,acc[K2_VPL]; for(int c=0;c<K2_VPL;c++)acc[c]=0.f;
  for(int sp=wid;sp<nsp;sp+=W){ const size_t pi=(size_t)lqh*nsp+sp; float ms=pm[pi],ls=pl[pi];
    if(ls<=0.f)continue; const float* ai=pa+pi*HEAD_DIM+lane*K2_VPL;
    float mn=fmaxf(m,ms),co=__expf(m-mn),cs=__expf(ms-mn); l=l*co+ls*cs;
    for(int c=0;c<K2_VPL;c++)acc[c]=acc[c]*co+ai[c]*cs; m=mn; }
  __shared__ float sm[W], sl[W], sa[W*HEAD_DIM];
  if(lane==0){sm[wid]=m; sl[wid]=l;}
  float* sao=sa+wid*HEAD_DIM+lane*K2_VPL; for(int c=0;c<K2_VPL;c++)sao[c]=acc[c];
  __syncthreads();
  if(wid==0){ float rm=-FLT_MAX,rl=0.f,ra[K2_VPL]; for(int c=0;c<K2_VPL;c++)ra[c]=0.f;
    for(int w=0;w<W;w++){ float ms=sm[w],ls=sl[w]; if(ls<=0.f)continue;
      const float* ai=sa+w*HEAD_DIM+lane*K2_VPL; float mn=fmaxf(rm,ms),co=__expf(rm-mn),cs=__expf(ms-mn);
      rl=rl*co+ls*cs; for(int c=0;c<K2_VPL;c++)ra[c]=ra[c]*co+ai[c]*cs; rm=mn; }
    float inv=(rl>0.f)?1.f/rl:0.f; float* o=out+lqh*HEAD_DIM+lane*K2_VPL;
    for(int c=0;c<K2_VPL;c++)o[c]=ra[c]*inv; }
}

// ---------- (B) fully fused: 1 CTA per head, W warps split KV time, smem combine ----------
template<int W>
__global__ void B_fused(const float* q,const fp8* kk,const fp8* vv,const float* ks,const float* vs,
    int ctx,int rank,float* out){
  const int lane=threadIdx.x&31, wid=threadIdx.x>>5;
  const int lqh=blockIdx.x, gqh=rank*Q_HEADS_RANK+lqh, kvh=gqh/GQA_GROUP;
  const float scale=rsqrtf((float)HEAD_DIM); const int kvb=kvh*HEAD_DIM, c0=kvb+lane*K2_VPL;
  float qr[K2_VPL],kc[K2_VPL],vc[K2_VPL];
  #pragma unroll
  for(int c=0;c<K2_VPL;c++){qr[c]=q[lqh*HEAD_DIM+lane*K2_VPL+c]; kc[c]=ks[c0+c]; vc[c]=vs[c0+c];}
  const int chunk=(ctx+W-1)/W, t0=wid*chunk, t1=min(t0+chunk,ctx);
  float m=-FLT_MAX,l=0.f,acc[K2_VPL]; for(int c=0;c<K2_VPL;c++)acc[c]=0.f;
  const unsigned* k32=(const unsigned*)kk; const unsigned* v32=(const unsigned*)vv;
  const int rw=KV_DIM/4, bw=kvb/4;
  int t=t0;
  for(;t+1<t1;t+=2){
    float k0[K2_VPL],k1[K2_VPL]; load4(k32+(size_t)t*rw+bw,lane,kc,k0); load4(k32+(size_t)(t+1)*rw+bw,lane,kc,k1);
    float p0=0.f,p1=0.f; for(int c=0;c<K2_VPL;c++){p0+=qr[c]*k0[c];p1+=qr[c]*k1[c];}
    float s0=wsum(p0)*scale,s1=wsum(p1)*scale;
    float v0[K2_VPL],v1[K2_VPL]; load4(v32+(size_t)t*rw+bw,lane,vc,v0); load4(v32+(size_t)(t+1)*rw+bw,lane,vc,v1);
    float mn=fmaxf(m,s0),co=__expf(m-mn),pe=__expf(s0-mn); l=l*co+pe;
    for(int c=0;c<K2_VPL;c++)acc[c]=acc[c]*co+pe*v0[c]; m=mn;
    mn=fmaxf(m,s1);co=__expf(m-mn);pe=__expf(s1-mn);l=l*co+pe;
    for(int c=0;c<K2_VPL;c++)acc[c]=acc[c]*co+pe*v1[c]; m=mn;
  }
  for(;t<t1;t++){ float kv[K2_VPL]; load4(k32+(size_t)t*rw+bw,lane,kc,kv);
    float p=0.f; for(int c=0;c<K2_VPL;c++)p+=qr[c]*kv[c]; float s=wsum(p)*scale;
    float vr[K2_VPL]; load4(v32+(size_t)t*rw+bw,lane,vc,vr);
    float mn=fmaxf(m,s),co=__expf(m-mn),pe=__expf(s-mn); l=l*co+pe;
    for(int c=0;c<K2_VPL;c++)acc[c]=acc[c]*co+pe*vr[c]; m=mn; }
  __shared__ float sm[W],sl[W],sa[W*HEAD_DIM];
  if(lane==0){sm[wid]=m;sl[wid]=l;}
  float* sao=sa+wid*HEAD_DIM+lane*K2_VPL; for(int c=0;c<K2_VPL;c++)sao[c]=acc[c];
  __syncthreads();
  if(wid==0){ float rm=-FLT_MAX,rl=0.f,ra[K2_VPL]; for(int c=0;c<K2_VPL;c++)ra[c]=0.f;
    for(int w=0;w<W;w++){ float ms=sm[w],ls=sl[w]; if(ls<=0.f)continue;
      const float* ai=sa+w*HEAD_DIM+lane*K2_VPL; float mn=fmaxf(rm,ms),co=__expf(rm-mn),cs=__expf(ms-mn);
      rl=rl*co+ls*cs; for(int c=0;c<K2_VPL;c++)ra[c]=ra[c]*co+ai[c]*cs; rm=mn; }
    float inv=(rl>0.f)?1.f/rl:0.f; float* o=out+lqh*HEAD_DIM+lane*K2_VPL;
    for(int c=0;c<K2_VPL;c++)o[c]=ra[c]*inv; }
}

static float frnd(unsigned s,size_t i){ unsigned h=s*2654435761u+(unsigned)i*40503u; h^=h>>13; h*=2246822519u; h^=h>>16; return (((h%2001)/1000.f)-1.f); }

int main(int argc,char**argv){
  int ctx=argc>1?atoi(argv[1]):4096; int reps=argc>2?atoi(argv[2]):2000; int rank=0;
  int nsp=argc>3?atoi(argv[3]):64;
  std::vector<float> hq(Q_DIM_RANK); for(int i=0;i<Q_DIM_RANK;i++)hq[i]=frnd(7u,i)*0.5f;
  std::vector<fp8> hk((size_t)ctx*KV_DIM),hv((size_t)ctx*KV_DIM);
  for(size_t i=0;i<hk.size();i++){hk[i]=(fp8)(frnd(20u,i)*0.25f); hv[i]=(fp8)(frnd(21u,i)*0.25f);}
  std::vector<float> hks(KV_DIM),hvs(KV_DIM); for(int i=0;i<KV_DIM;i++){hks[i]=fabsf(frnd(22u,i)*0.04f)+1e-3f;hvs[i]=fabsf(frnd(23u,i)*0.04f)+1e-3f;}
  float *q,*ks,*vs,*pm,*pl,*pa,*o1,*o2,*o3; fp8 *kk,*vv;
  CK(cudaMalloc(&q,Q_DIM_RANK*4)); CK(cudaMemcpy(q,hq.data(),Q_DIM_RANK*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&kk,hk.size())); CK(cudaMemcpy(kk,hk.data(),hk.size(),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&vv,hv.size())); CK(cudaMemcpy(vv,hv.data(),hv.size(),cudaMemcpyHostToDevice));
  CK(cudaMalloc(&ks,KV_DIM*4)); CK(cudaMemcpy(ks,hks.data(),KV_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&vs,KV_DIM*4)); CK(cudaMemcpy(vs,hvs.data(),KV_DIM*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&pm,(size_t)Q_HEADS_RANK*nsp*4)); CK(cudaMalloc(&pl,(size_t)Q_HEADS_RANK*nsp*4));
  CK(cudaMalloc(&pa,(size_t)Q_HEADS_RANK*nsp*HEAD_DIM*4));
  CK(cudaMalloc(&o1,Q_DIM_RANK*4)); CK(cudaMalloc(&o2,Q_DIM_RANK*4)); CK(cudaMalloc(&o3,Q_DIM_RANK*4));

  const int wpc=4, blk=wpc*32; dim3 gP(nsp,(Q_HEADS_RANK+wpc-1)/wpc); dim3 gR((Q_HEADS_RANK+wpc-1)/wpc);
  auto run_A=[&](cudaStream_t s){ A_partial<<<gP,blk,0,s>>>(q,kk,vv,ks,vs,ctx,nsp,rank,pm,pl,pa);
    A_reduce<<<gR,blk,0,s>>>(pm,pl,pa,nsp,o1); };
  auto run_AC=[&](cudaStream_t s){ A_partial<<<gP,blk,0,s>>>(q,kk,vv,ks,vs,ctx,nsp,rank,pm,pl,pa);
    C_reduce<8><<<Q_HEADS_RANK,8*32,0,s>>>(pm,pl,pa,nsp,o2); };
  auto run_AC16=[&](cudaStream_t s){ A_partial<<<gP,blk,0,s>>>(q,kk,vv,ks,vs,ctx,nsp,rank,pm,pl,pa);
    C_reduce<16><<<Q_HEADS_RANK,16*32,0,s>>>(pm,pl,pa,nsp,o2); };
  auto run_Ponly=[&](cudaStream_t s){ A_partial<<<gP,blk,0,s>>>(q,kk,vv,ks,vs,ctx,nsp,rank,pm,pl,pa); };
  auto run_B=[&](cudaStream_t s){ B_fused<16><<<Q_HEADS_RANK,16*32,0,s>>>(q,kk,vv,ks,vs,ctx,rank,o3); };

  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto bench=[&](const char* nm,auto fn)->double{
    for(int i=0;i<50;i++)fn(s); CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(e0,s)); for(int i=0;i<reps;i++)fn(s); CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); double us=ms*1e3/reps; printf("  %-30s %8.3f us/token\n",nm,us); return us; };

  printf("== K2 TP=8 micro (ctx=%d, 8 heads/rank, nsp=%d, reps=%d) ==\n",ctx,nsp,reps);
  double pp=bench("  partial-only",run_Ponly);
  double a=bench("A: partial + serial reduce",run_A);
  double ac=bench("A-partial + C parallel reduce(W8)",run_AC);
  double ac16=bench("A-partial + C parallel reduce(W16)",run_AC16);
  double b=bench("B: fully fused (W=16)",run_B);
  (void)pp;(void)ac16;

  // correctness: compare A vs AC vs B
  std::vector<float> r1(Q_DIM_RANK),r2(Q_DIM_RANK),r3(Q_DIM_RANK);
  run_A(s);CK(cudaStreamSynchronize(s));CK(cudaMemcpy(r1.data(),o1,Q_DIM_RANK*4,cudaMemcpyDeviceToHost));
  run_AC(s);CK(cudaStreamSynchronize(s));CK(cudaMemcpy(r2.data(),o2,Q_DIM_RANK*4,cudaMemcpyDeviceToHost));
  run_B(s);CK(cudaStreamSynchronize(s));CK(cudaMemcpy(r3.data(),o3,Q_DIM_RANK*4,cudaMemcpyDeviceToHost));
  float e_ac=0,e_b=0; for(int i=0;i<Q_DIM_RANK;i++){e_ac=fmaxf(e_ac,fabsf(r1[i]-r2[i]));e_b=fmaxf(e_b,fabsf(r1[i]-r3[i]));}
  printf("  max|A-AC|=%.2e  max|A-B|=%.2e\n",e_ac,e_b);
  printf("  speedup AC vs A=%.2fx  B vs A=%.2fx\n",a/ac,a/b);
  return 0;
}
