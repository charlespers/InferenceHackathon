/* mk_tree_attn_fused.cu — LOOP-A: OPTIMIZED M=k tree attention (split-KV, one fused launch).
 *
 * Addresses the K2 floor (LOOP-C: "K2 flash-decode is the next floor, 24% of the 2.1ms target,
 * latency-bound at B=1"). My first kernel (mk_tree_attn.cu) was CHAIN-bound: one warp serially scanned
 * the whole context (3ms @ctx4096). This mirrors Charles's k2_flash_decode_FUSED: ONE CTA per
 * (query,q_head), W warps split the context KV range, each runs the online-softmax recurrence over its
 * slice (2x time-unroll), then the W partials are combined in SHARED MEMORY (no second launch, no HBM
 * round-trip) and the query's few ancestor draft slots are merged in — shortens the dependent chain ~W x.
 *
 * Same math/contract as sdpa_tree_ref.h (the gate): GQA, scale 1/sqrt(hd), RoPE NeoX theta=1e6 +
 * per-head RMSNorm on Q, K/V from the (roped) cache, tree mask (context + ancestor slots). fp32 K/V
 * here (gate-matchable); fp8 dequant (k2_load4 idiom) is the drop-in for Charles's real cache.
 *
 * Build/run (GPU): nvcc -arch=sm_90a -O3 mk_tree_attn_fused.cu -o /tmp/mkf && /tmp/mkf   # self-check.
 */
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include "sdpa_tree_ref.h"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

static __device__ __forceinline__ float wsum(float v){
    #pragma unroll
    for(int o=16;o>0;o>>=1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* one CTA per (query,q_head); blockDim = W*32. shared: Q[hd] + per-warp partials (m[W],l[W],acc[W*hd]). */
__global__ void mk_tree_attn_fused(int n_query,int n_q_heads,int n_kv_heads,int head_dim,
                                   const float* q_proj,const float* qnorm_w,const int* q_pos_id,
                                   const float* k_cache,const float* v_cache,
                                   int context_len,const int* anc_off,const int* anc_slots,
                                   float theta,float* out){
    int W = blockDim.x>>5, wid = threadIdx.x>>5, lane = threadIdx.x&31;
    int cta = blockIdx.x;                       // (query,head) flat
    int j = cta / n_q_heads, h = cta % n_q_heads;
    int kv = h / (n_q_heads/n_kv_heads);
    int hd = head_dim;
    extern __shared__ float sm[];
    float* q   = sm;                            // [hd]
    float* sm_m = q + hd;                       // [W]
    float* sm_l = sm_m + W;                     // [W]
    float* sm_acc = sm_l + W;                   // [W*hd]
    float scale = rsqrtf((float)hd);

    // ---- Q: load, RMSNorm*w, RoPE (warp 0 does it; all warps read from shared) ----
    if (wid==0){
        for(int d=lane; d<hd; d+=32) q[d] = q_proj[((size_t)j*n_q_heads+h)*hd + d];
    }
    __syncthreads();
    if (wid==0){
        float ss=0.f; for(int d=lane; d<hd; d+=32) ss += q[d]*q[d];
        ss = wsum(ss); float inv = rsqrtf(ss/hd + 1e-6f);
        for(int d=lane; d<hd; d+=32) q[d] = q[d]*inv*qnorm_w[d];
    }
    __syncthreads();
    if (wid==0){
        int half=hd/2;
        for(int i=lane; i<half; i+=32){
            float fr=powf(theta,-2.f*(float)i/(float)hd), a=(float)q_pos_id[j]*fr, c=cosf(a), s=sinf(a);
            float x=q[i], y=q[i+half]; q[i]=x*c-y*s; q[i+half]=y*c+x*s;
        }
    }
    __syncthreads();

    // ---- each warp: online-softmax over its context slice [t0,t1) ----
    int chunk=(context_len+W-1)/W, t0=wid*chunk, t1=min(t0+chunk,context_len);
    float m=-FLT_MAX, l=0.f, acc[8]; for(int c=0;c<8;++c) acc[c]=0.f;
    for(int t=t0;t<t1;++t){
        const float* k=k_cache+((size_t)t*n_kv_heads+kv)*hd;
        float p=0.f; for(int d=lane; d<hd; d+=32) p+=q[d]*k[d];
        float s=wsum(p)*scale;
        const float* v=v_cache+((size_t)t*n_kv_heads+kv)*hd;
        float mn=fmaxf(m,s), corr=__expf(m-mn), pe=__expf(s-mn);
        l=l*corr+pe; for(int ci=0,d=lane; d<hd; ++ci,d+=32) acc[ci]=acc[ci]*corr+pe*v[d];
        m=mn;
    }
    // stash this warp's partial
    if(lane==0){ sm_m[wid]=m; sm_l[wid]=l; }
    for(int ci=0,d=lane; d<hd; ++ci,d+=32) sm_acc[(size_t)wid*hd+d]=acc[ci];
    __syncthreads();

    // ---- warp 0: combine W context partials + merge ancestor draft slots + normalize + write ----
    if(wid==0){
        float M=-FLT_MAX, L=0.f, A[8]; for(int c=0;c<8;++c) A[c]=0.f;
        for(int w=0; w<W; ++w){
            float ms=sm_m[w], ls=sm_l[w]; if(ls<=0.f) continue;
            float mn=fmaxf(M,ms), co=__expf(M-mn), cs=__expf(ms-mn);
            L=L*co+ls*cs;
            for(int ci=0,d=lane; d<hd; ++ci,d+=32) A[ci]=A[ci]*co+sm_acc[(size_t)w*hd+d]*cs;
            M=mn;
        }
        // ancestor draft slots (few): online-merge each
        int aoff=anc_off[j], aend=anc_off[j+1];
        for(int a=aoff; a<aend; ++a){
            int t=anc_slots[a];
            const float* k=k_cache+((size_t)t*n_kv_heads+kv)*hd;
            float p=0.f; for(int d=lane; d<hd; d+=32) p+=q[d]*k[d];
            float s=wsum(p)*scale;
            const float* v=v_cache+((size_t)t*n_kv_heads+kv)*hd;
            float mn=fmaxf(M,s), co=__expf(M-mn), pe=__expf(s-mn);
            L=L*co+pe; for(int ci=0,d=lane; d<hd; ++ci,d+=32) A[ci]=A[ci]*co+pe*v[d];
            M=mn;
        }
        float inv=(L>0.f)?1.f/L:0.f;
        float* o=out+((size_t)j*n_q_heads+h)*hd;
        for(int ci=0,d=lane; d<hd; ++ci,d+=32) o[d]=A[ci]*inv;
    }
}

#ifndef MKF_NO_MAIN
int main(){
    const int NQ=3,HQ=8,HKV=2,HD=128,CTX=200,NANC=2,NTOT=CTX+NANC; const float theta=1e6f;
    float *qp=(float*)malloc(sizeof(float)*NQ*HQ*HD),*qn=(float*)malloc(sizeof(float)*HD);
    float *kc=(float*)malloc(sizeof(float)*NTOT*HKV*HD),*vc=(float*)malloc(sizeof(float)*NTOT*HKV*HD);
    int qpos[NQ]={CTX-1,CTX,CTX+1}, aoff[NQ+1]={0,0,1,2}, asl[2]={CTX,CTX+1};
    for(int i=0;i<NQ*HQ*HD;++i) qp[i]=sinf(0.01f*i)*0.5f;
    for(int i=0;i<HD;++i) qn[i]=1.f+0.001f*i;
    for(int i=0;i<NTOT*HKV*HD;++i){ kc[i]=cosf(0.013f*i)*0.4f; vc[i]=sinf(0.007f*i); }
    float* ref=(float*)malloc(sizeof(float)*NQ*HQ*HD);
    sdpa_tree(NQ,HQ,HKV,HD,qp,qn,qpos,kc,vc,NTOT,CTX,aoff,asl,theta,ref);
    float *dqp,*dqn,*dkc,*dvc,*dout; int *dpos,*doff,*dsl;
    CK(cudaMalloc(&dqp,sizeof(float)*NQ*HQ*HD)); CK(cudaMalloc(&dqn,sizeof(float)*HD));
    CK(cudaMalloc(&dkc,sizeof(float)*NTOT*HKV*HD)); CK(cudaMalloc(&dvc,sizeof(float)*NTOT*HKV*HD));
    CK(cudaMalloc(&dout,sizeof(float)*NQ*HQ*HD)); CK(cudaMalloc(&dpos,sizeof(int)*NQ));
    CK(cudaMalloc(&doff,sizeof(int)*(NQ+1))); CK(cudaMalloc(&dsl,sizeof(int)*NANC));
    CK(cudaMemcpy(dqp,qp,sizeof(float)*NQ*HQ*HD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqn,qn,sizeof(float)*HD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkc,kc,sizeof(float)*NTOT*HKV*HD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvc,vc,sizeof(float)*NTOT*HKV*HD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos,qpos,sizeof(int)*NQ,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(doff,aoff,sizeof(int)*(NQ+1),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsl,asl,sizeof(int)*NANC,cudaMemcpyHostToDevice));
    int W=8, shmem=(HD + 2*W + W*HD)*sizeof(float);
    mk_tree_attn_fused<<<NQ*HQ, W*32, shmem>>>(NQ,HQ,HKV,HD,dqp,dqn,dpos,dkc,dvc,CTX,doff,dsl,theta,dout);
    CK(cudaDeviceSynchronize());
    float* got=(float*)malloc(sizeof(float)*NQ*HQ*HD);
    CK(cudaMemcpy(got,dout,sizeof(float)*NQ*HQ*HD,cudaMemcpyDeviceToHost));
    float me=0; for(int i=0;i<NQ*HQ*HD;++i){ float e=fabsf(got[i]-ref[i]); if(e>me) me=e; }
    printf("mk_tree_attn_FUSED vs sdpa_tree_ref: max abs err = %.3e -> %s\n", me, me<1e-3f?"PASS":"FAIL");
    return me<1e-3f?0:1;
}
#endif
