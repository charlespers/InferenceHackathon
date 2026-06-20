/* mk_tree_attn.cu — LOOP-A: the M=k TREE-masked attention KERNEL (the one new kernel for native spec).
 *
 * Correct-first implementation (matches sdpa_tree_ref.h exactly): one warp per (query, q_head), online
 * softmax over the attended set = context [0..context_len) + the query's ancestor draft slots
 * (tree_attn.h). GQA (kv = h/group), scale 1/sqrt(hd), per-head RMSNorm * qnorm_w on Q + RoPE theta=1e6
 * NeoX rotate-half; K/V read from the (already-roped) cache. This is the lossless tree verify's
 * attention; the proj/MoE around it are Charles's flat M=k GEMM path.
 *
 * Perf note: this is the CORRECT baseline (serial KV scan per warp). The OPTIMIZATION is to drop in
 * k2_flash_decode's split-KV + 2-pass online-softmax for the context part (the big read) and keep this
 * masked scan only for the few ancestor draft slots — a straightforward swap once correctness is banked.
 *
 * Build/test (needs a GPU to RUN; nvcc-COMPILES anywhere):
 *   nvcc -arch=sm_90a -O3 mk_tree_attn.cu -o /tmp/mkta && /tmp/mkta   # self-checks vs sdpa_tree_ref.h
 */
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "sdpa_tree_ref.h"  /* CPU gate (host-only fns) */

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

static __device__ __forceinline__ float warp_sum(float v){
    #pragma unroll
    for(int o=16;o>0;o>>=1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* one warp per (query, q_head). shared mem: head_dim floats per warp (Q). */
__global__ void mk_tree_attn(int n_query,int n_q_heads,int n_kv_heads,int head_dim,
                             const float* q_proj,const float* qnorm_w,const int* q_pos_id,
                             const float* k_cache,const float* v_cache,
                             int context_len,const int* anc_off,const int* anc_slots,
                             float theta,float* out){
    int wpb = blockDim.x/32;
    int warp = blockIdx.x*wpb + threadIdx.x/32;
    int lane = threadIdx.x & 31;
    if(warp >= n_query*n_q_heads) return;
    int j = warp / n_q_heads, h = warp % n_q_heads;
    int kv = h / (n_q_heads/n_kv_heads);

    extern __shared__ float sh[];
    float* q = sh + (threadIdx.x/32)*head_dim;
    for(int d=lane; d<head_dim; d+=32) q[d] = q_proj[((size_t)j*n_q_heads+h)*head_dim + d];
    __syncwarp();
    // RMSNorm * qnorm_w
    float ss=0.f; for(int d=lane; d<head_dim; d+=32) ss += q[d]*q[d];
    ss = warp_sum(ss); float inv = rsqrtf(ss/head_dim + 1e-6f);
    for(int d=lane; d<head_dim; d+=32) q[d] = q[d]*inv*qnorm_w[d];
    __syncwarp();
    // RoPE NeoX rotate-half (pairs i, i+half)
    int half=head_dim/2;
    for(int i=lane; i<half; i+=32){
        float fr = powf(theta, -2.f*(float)i/(float)head_dim);
        float a = (float)q_pos_id[j]*fr, c=cosf(a), s=sinf(a);
        float x=q[i], y=q[i+half];
        q[i]=x*c - y*s; q[i+half]=y*c + x*s;
    }
    __syncwarp();
    // online-softmax masked attention over context + ancestors
    int n_anc = anc_off[j+1]-anc_off[j];
    int n_att = context_len + n_anc;
    float scale = rsqrtf((float)head_dim);
    float m=-INFINITY, l=0.f;
    float acc[8]; for(int ci=0;ci<8;++ci) acc[ci]=0.f;  // up to head_dim/32 (<=8 for hd<=256)
    for(int a=0;a<n_att;++a){
        int t = (a<context_len)? a : anc_slots[anc_off[j] + (a-context_len)];
        const float* k = k_cache + ((size_t)t*n_kv_heads+kv)*head_dim;
        float dot=0.f; for(int d=lane; d<head_dim; d+=32) dot += q[d]*k[d];
        dot = warp_sum(dot)*scale;
        float mn = fmaxf(m,dot), corr=expf(m-mn), p=expf(dot-mn);
        l = l*corr + p;
        const float* v = v_cache + ((size_t)t*n_kv_heads+kv)*head_dim;
        for(int ci=0, d=lane; d<head_dim; ++ci, d+=32) acc[ci] = acc[ci]*corr + p*v[d];
        m = mn;
    }
    float* o = out + ((size_t)j*n_q_heads+h)*head_dim;
    for(int ci=0, d=lane; d<head_dim; ++ci, d+=32) o[d] = (l>0.f)? acc[ci]/l : 0.f;
}

#ifndef MK_TREE_ATTN_NO_MAIN
/* self-check on GPU vs the CPU gate (sdpa_tree_ref). Needs a GPU to run. */
int main(){
    const int NQ=3, HQ=8, HKV=2, HD=128, CTX=40, NANC=2, NTOT=CTX+NANC;
    const float theta=1000000.f;
    // host buffers
    float *qp=(float*)malloc(sizeof(float)*NQ*HQ*HD), *qn=(float*)malloc(sizeof(float)*HD);
    float *kc=(float*)malloc(sizeof(float)*NTOT*HKV*HD), *vc=(float*)malloc(sizeof(float)*NTOT*HKV*HD);
    int qpos[NQ]={CTX-1,CTX,CTX+1};
    int anc_off[NQ+1]={0,0,1,2}; int anc_slots[2]={CTX, CTX+1};  // q1->slot CTX, q2->slot CTX+1
    for(int i=0;i<NQ*HQ*HD;++i) qp[i]=sinf(0.01f*i)*0.5f;
    for(int i=0;i<HD;++i) qn[i]=1.0f+0.001f*i;
    for(int i=0;i<NTOT*HKV*HD;++i){ kc[i]=cosf(0.013f*i)*0.4f; vc[i]=sinf(0.007f*i); }
    // CPU reference
    float* ref=(float*)malloc(sizeof(float)*NQ*HQ*HD);
    sdpa_tree(NQ,HQ,HKV,HD, qp,qn,qpos, kc,vc,NTOT, CTX, anc_off,anc_slots, theta, ref);
    // GPU
    float *dqp,*dqn,*dkc,*dvc,*dout; int *dpos,*doff,*dslots;
    CK(cudaMalloc(&dqp,sizeof(float)*NQ*HQ*HD)); CK(cudaMalloc(&dqn,sizeof(float)*HD));
    CK(cudaMalloc(&dkc,sizeof(float)*NTOT*HKV*HD)); CK(cudaMalloc(&dvc,sizeof(float)*NTOT*HKV*HD));
    CK(cudaMalloc(&dout,sizeof(float)*NQ*HQ*HD)); CK(cudaMalloc(&dpos,sizeof(int)*NQ));
    CK(cudaMalloc(&doff,sizeof(int)*(NQ+1))); CK(cudaMalloc(&dslots,sizeof(int)*NANC));
    CK(cudaMemcpy(dqp,qp,sizeof(float)*NQ*HQ*HD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqn,qn,sizeof(float)*HD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkc,kc,sizeof(float)*NTOT*HKV*HD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvc,vc,sizeof(float)*NTOT*HKV*HD,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos,qpos,sizeof(int)*NQ,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(doff,anc_off,sizeof(int)*(NQ+1),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dslots,anc_slots,sizeof(int)*NANC,cudaMemcpyHostToDevice));
    int warps=NQ*HQ, wpb=4, blocks=(warps+wpb-1)/wpb, shmem=wpb*HD*sizeof(float);
    mk_tree_attn<<<blocks, wpb*32, shmem>>>(NQ,HQ,HKV,HD, dqp,dqn,dpos, dkc,dvc, CTX,doff,dslots, theta,dout);
    CK(cudaDeviceSynchronize());
    float* got=(float*)malloc(sizeof(float)*NQ*HQ*HD);
    CK(cudaMemcpy(got,dout,sizeof(float)*NQ*HQ*HD,cudaMemcpyDeviceToHost));
    float maxerr=0.f; for(int i=0;i<NQ*HQ*HD;++i){ float e=fabsf(got[i]-ref[i]); if(e>maxerr) maxerr=e; }
    printf("mk_tree_attn vs sdpa_tree_ref: max abs err = %.3e  -> %s\n", maxerr, maxerr<1e-3f?"PASS":"FAIL");
    return maxerr<1e-3f?0:1;
}
#endif
