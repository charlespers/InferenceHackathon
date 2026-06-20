/* mk_tree_attn_fp8.cu — LOOP-A: fp8-K/V fused M=k tree attention (the K2-floor win; matches Charles's cache).
 *
 * = mk_tree_attn_fused (split-KV, one fused launch) but K/V read from the fp8 e4m3 cache with per-channel
 * dequant scales (Charles's decode_step_tp8 / k2 cache format), using his k2_load4 idiom: each lane loads
 * its 4 CONTIGUOUS channels [4L,4L+4) as one 32-bit word and dequants via fp8x2->half2. fp8 halves the HBM
 * bytes vs fp32 -> ~2x faster -> ~180us@M=1/ctx4096 (beats the k2 ~500us placeholder) AND handles M=k trees.
 *
 * Math/contract = sdpa_tree_ref.h (the gate): GQA, scale 1/sqrt(hd), RoPE NeoX theta=1e6 + per-head
 * RMSNorm on Q, K/V from the roped cache, tree mask. Validated vs the fp32 gate at FP8 TOLERANCE (~few %,
 * matching Charles's M1 fp8 validation mean_rel~4%). Drop-in for the real cache: pass kv_k/kv_v (fp8) +
 * kv_*_scale (per-channel, length n_kv_heads*head_dim).
 *
 * Build/run (GPU): nvcc -arch=sm_90a -O3 mk_tree_attn_fp8.cu -o /tmp/mk8 && /tmp/mk8
 */
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include "sdpa_tree_ref.h"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)
typedef __nv_fp8_e4m3 fp8;

static __device__ __forceinline__ float wsum(float v){
    #pragma unroll
    for(int o=16;o>0;o>>=1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
/* k2_load4: lane reads its 4 contiguous fp8 channels (one 32-bit word) + dequant with 4 per-chan scales. */
static __device__ __forceinline__ void load4(const unsigned* base32, int lane, const float* s, float* out){
    unsigned w = base32[lane];
    __nv_fp8x2_e4m3 lo, hi; lo.__x=(unsigned short)(w&0xffffu); hi.__x=(unsigned short)(w>>16);
    float2 fl=__half22float2((__half2)lo), fh=__half22float2((__half2)hi);
    out[0]=fl.x*s[0]; out[1]=fl.y*s[1]; out[2]=fh.x*s[2]; out[3]=fh.y*s[3];
}

/* one CTA per (query,head); blockDim=W*32; CONTIGUOUS channel layout (lane L owns [4L,4L+4)). */
__global__ void mk_tree_attn_fp8(int n_query,int n_q_heads,int n_kv_heads,int head_dim,
                                 const float* q_proj,const float* qnorm_w,const int* q_pos_id,
                                 const fp8* kv_k,const fp8* kv_v,const float* kv_k_scale,const float* kv_v_scale,
                                 int context_len,const int* anc_off,const int* anc_slots,
                                 float theta,float* out){
    const int VPL=4;                            // head_dim/32 (=4 for hd=128)
    int W=blockDim.x>>5, wid=threadIdx.x>>5, lane=threadIdx.x&31;
    int cta=blockIdx.x, j=cta/n_q_heads, h=cta%n_q_heads, kv=h/(n_q_heads/n_kv_heads), hd=head_dim;
    int kv_dim=n_kv_heads*hd, row_words=kv_dim/4, kv_base=kv*hd, base_words=kv_base/4, c0=kv_base+lane*VPL;
    float scale=rsqrtf((float)hd);
    extern __shared__ float sm[];
    float* q=sm; float* sm_m=q+hd; float* sm_l=sm_m+W; float* sm_acc=sm_l+W;
    // Q -> shared, RMSNorm*w, RoPE (warp 0)
    if(wid==0) for(int d=lane; d<hd; d+=32) q[d]=q_proj[((size_t)j*n_q_heads+h)*hd+d];
    __syncthreads();
    if(wid==0){ float ss=0; for(int d=lane;d<hd;d+=32) ss+=q[d]*q[d]; ss=wsum(ss); float inv=rsqrtf(ss/hd+1e-6f);
        for(int d=lane;d<hd;d+=32) q[d]=q[d]*inv*qnorm_w[d]; }
    __syncthreads();
    if(wid==0){ int half=hd/2; for(int i=lane;i<half;i+=32){ float fr=powf(theta,-2.f*(float)i/(float)hd),a=(float)q_pos_id[j]*fr,c=cosf(a),s=sinf(a); float x=q[i],y=q[i+half]; q[i]=x*c-y*s; q[i+half]=y*c+x*s; } }
    __syncthreads();
    // per-lane: q's 4 contiguous channels [4L,4L+4) + the 4 dequant scales (constant in t)
    float qreg[4], ksc[4], vsc[4];
    #pragma unroll
    for(int c=0;c<4;++c){ qreg[c]=q[lane*VPL+c]; ksc[c]=kv_k_scale?kv_k_scale[c0+c]:1.f; vsc[c]=kv_v_scale?kv_v_scale[c0+c]:1.f; }
    const unsigned* k32=reinterpret_cast<const unsigned*>(kv_k); const unsigned* v32=reinterpret_cast<const unsigned*>(kv_v);
    // warp's context slice
    int chunk=(context_len+W-1)/W, t0=wid*chunk, t1=min(t0+chunk,context_len);
    float m=-FLT_MAX,l=0.f,acc[4]={0,0,0,0};
    for(int t=t0;t<t1;++t){
        float kk[4]; load4(k32+(size_t)t*row_words+base_words, lane, ksc, kk);
        float p=0.f; for(int c=0;c<4;++c) p+=qreg[c]*kk[c]; float s=wsum(p)*scale;
        float vv[4]; load4(v32+(size_t)t*row_words+base_words, lane, vsc, vv);
        float mn=fmaxf(m,s),corr=__expf(m-mn),pe=__expf(s-mn); l=l*corr+pe;
        for(int c=0;c<4;++c) acc[c]=acc[c]*corr+pe*vv[c]; m=mn;
    }
    if(lane==0){ sm_m[wid]=m; sm_l[wid]=l; }
    for(int c=0;c<4;++c) sm_acc[(size_t)wid*hd+lane*VPL+c]=acc[c];
    __syncthreads();
    if(wid==0){
        float M=-FLT_MAX,L=0.f,A[4]={0,0,0,0};
        for(int w=0;w<W;++w){ float ms=sm_m[w],ls=sm_l[w]; if(ls<=0.f) continue;
            float mn=fmaxf(M,ms),co=__expf(M-mn),cs=__expf(ms-mn); L=L*co+ls*cs;
            for(int c=0;c<4;++c) A[c]=A[c]*co+sm_acc[(size_t)w*hd+lane*VPL+c]*cs; M=mn; }
        for(int a=anc_off[j]; a<anc_off[j+1]; ++a){ int t=anc_slots[a];
            float kk[4]; load4(k32+(size_t)t*row_words+base_words, lane, ksc, kk);
            float p=0.f; for(int c=0;c<4;++c) p+=qreg[c]*kk[c]; float s=wsum(p)*scale;
            float vv[4]; load4(v32+(size_t)t*row_words+base_words, lane, vsc, vv);
            float mn=fmaxf(M,s),co=__expf(M-mn),pe=__expf(s-mn); L=L*co+pe;
            for(int c=0;c<4;++c) A[c]=A[c]*co+pe*vv[c]; M=mn; }
        float inv=(L>0.f)?1.f/L:0.f; float* o=out+((size_t)j*n_q_heads+h)*hd;
        for(int c=0;c<4;++c) o[lane*VPL+c]=A[c]*inv;
    }
}

#ifndef MK8_NO_MAIN
/* quantize fp32 [n_pos*kv_dim] -> fp8 e4m3 + per-channel scale (length kv_dim). */
static void quant_fp8(const float* src,int n_pos,int kv_dim,fp8* dst,float* scale){
    for(int c=0;c<kv_dim;++c){ float mx=1e-8f; for(int p=0;p<n_pos;++p){ float a=fabsf(src[(size_t)p*kv_dim+c]); if(a>mx)mx=a; } scale[c]=mx/448.f; }
    for(int p=0;p<n_pos;++p) for(int c=0;c<kv_dim;++c) dst[(size_t)p*kv_dim+c]=(fp8)(src[(size_t)p*kv_dim+c]/scale[c]);
}
int main(){
    const int NQ=3,HQ=8,HKV=2,HD=128,CTX=200,NANC=2,NTOT=CTX+NANC,KVD=HKV*HD; const float theta=1e6f;
    float *qp=(float*)malloc(sizeof(float)*NQ*HQ*HD),*qn=(float*)malloc(sizeof(float)*HD);
    float *kc=(float*)malloc(sizeof(float)*NTOT*KVD),*vc=(float*)malloc(sizeof(float)*NTOT*KVD);
    int qpos[NQ]={CTX-1,CTX,CTX+1},aoff[NQ+1]={0,0,1,2},asl[2]={CTX,CTX+1};
    for(int i=0;i<NQ*HQ*HD;++i) qp[i]=sinf(0.01f*i)*0.5f;
    for(int i=0;i<HD;++i) qn[i]=1.f+0.001f*i;
    for(int i=0;i<NTOT*KVD;++i){ kc[i]=cosf(0.013f*i)*0.4f; vc[i]=0.5f+0.4f*sinf(0.007f*i); } // V non-zero-mean (else outputs ~0 -> rel err meaningless)
    // fp32 reference (dequant-equivalent: gate uses fp32 K/V; fp8 introduces ~few% error)
    float* ref=(float*)malloc(sizeof(float)*NQ*HQ*HD);
    sdpa_tree(NQ,HQ,HKV,HD,qp,qn,qpos,kc,vc,NTOT,CTX,aoff,asl,theta,ref);
    // quantize K/V to fp8 + scales
    fp8 *hk=(fp8*)malloc(NTOT*KVD),*hv=(fp8*)malloc(NTOT*KVD); float *ks=(float*)malloc(sizeof(float)*KVD),*vs=(float*)malloc(sizeof(float)*KVD);
    quant_fp8(kc,NTOT,KVD,hk,ks); quant_fp8(vc,NTOT,KVD,hv,vs);
    float *dqp,*dqn,*dout,*dks,*dvs; fp8 *dk,*dv; int *dpos,*doff,*dsl;
    CK(cudaMalloc(&dqp,sizeof(float)*NQ*HQ*HD)); CK(cudaMalloc(&dqn,sizeof(float)*HD)); CK(cudaMalloc(&dout,sizeof(float)*NQ*HQ*HD));
    CK(cudaMalloc(&dk,NTOT*KVD)); CK(cudaMalloc(&dv,NTOT*KVD)); CK(cudaMalloc(&dks,sizeof(float)*KVD)); CK(cudaMalloc(&dvs,sizeof(float)*KVD));
    CK(cudaMalloc(&dpos,sizeof(int)*NQ)); CK(cudaMalloc(&doff,sizeof(int)*(NQ+1))); CK(cudaMalloc(&dsl,sizeof(int)*NANC));
    CK(cudaMemcpy(dqp,qp,sizeof(float)*NQ*HQ*HD,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dqn,qn,sizeof(float)*HD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk,hk,NTOT*KVD,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,NTOT*KVD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dks,ks,sizeof(float)*KVD,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dvs,vs,sizeof(float)*KVD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos,qpos,sizeof(int)*NQ,cudaMemcpyHostToDevice)); CK(cudaMemcpy(doff,aoff,sizeof(int)*(NQ+1),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsl,asl,sizeof(int)*NANC,cudaMemcpyHostToDevice));
    int W=8, shmem=(HD+2*W+W*HD)*sizeof(float);
    mk_tree_attn_fp8<<<NQ*HQ,W*32,shmem>>>(NQ,HQ,HKV,HD,dqp,dqn,dpos,dk,dv,dks,dvs,CTX,doff,dsl,theta,dout);
    CK(cudaDeviceSynchronize());
    float* got=(float*)malloc(sizeof(float)*NQ*HQ*HD); CK(cudaMemcpy(got,dout,sizeof(float)*NQ*HQ*HD,cudaMemcpyDeviceToHost));
    float me=0,mxref=0; for(int i=0;i<NQ*HQ*HD;++i){ float e=fabsf(got[i]-ref[i]); if(e>me)me=e; float ar=fabsf(ref[i]); if(ar>mxref)mxref=ar; }
    float rel=me/(mxref+1e-9f);  // global-scale relative (per-element rel is meaningless on near-zero outputs)
    printf("mk_tree_attn_FP8 vs fp32 gate: max abs err=%.3e = %.2f%% of max|ref|=%.3f -> %s (fp8 tol ~5%%)\n", me, rel*100, mxref, rel<0.05f?"PASS":"FAIL");
    return rel<0.05f?0:1;
}
#endif
