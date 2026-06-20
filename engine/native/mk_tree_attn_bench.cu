/* mk_tree_attn_bench.cu — LOOP-A: real-scale latency of mk_tree_attn (M=k tree attention).
 * Measures how the attention cost scales with TREE WIDTH (n_query nodes) at the real config
 * (ctx 4096, 64 Q / 4 KV / head_dim 128) — the attention analog of Charles's flat M=k GEMM verify.
 * If a wider tree stays ~cheap, trees are "free" on attention too -> push tau via width.
 *
 * Run on a CLEAN GPU (exclusive slot) for trustworthy timing:
 *   nvcc -arch=sm_90a -O3 mk_tree_attn_bench.cu -o /tmp/mkb && /tmp/mkb [ctx=4096] [iters=200]
 */
#define MK_TREE_ATTN_NO_MAIN
#include "mk_tree_attn.cu"

int main(int argc, char** argv){
    const int HQ=64, HKV=4, HD=128;
    int CTX = argc>1 ? atoi(argv[1]) : 4096;
    int ITERS = argc>2 ? atoi(argv[2]) : 200;
    const float theta=1000000.f;
    int widths[5] = {1,4,8,16,32};   // tree sizes (n_query draft nodes)
    int maxNQ=32, NTOT=CTX+maxNQ;

    float *qp,*qn,*kc,*vc,*out; int *pos,*off,*slots;
    cudaMalloc(&qp,sizeof(float)*maxNQ*HQ*HD); cudaMalloc(&qn,sizeof(float)*HD);
    cudaMalloc(&kc,sizeof(float)*NTOT*HKV*HD); cudaMalloc(&vc,sizeof(float)*NTOT*HKV*HD);
    cudaMalloc(&out,sizeof(float)*maxNQ*HQ*HD); cudaMalloc(&pos,sizeof(int)*maxNQ);
    cudaMalloc(&off,sizeof(int)*(maxNQ+1)); cudaMalloc(&slots,sizeof(int)*maxNQ);
    // init device buffers (values don't matter for timing; just fill)
    cudaMemset(qp,1,sizeof(float)*maxNQ*HQ*HD); cudaMemset(kc,1,sizeof(float)*NTOT*HKV*HD);
    cudaMemset(vc,1,sizeof(float)*NTOT*HKV*HD);
    // qnorm=1, pos=ctx..., a CHAIN tree (node i attends ctx + ancestors 1..i): off[i]=i-1 slots etc.
    {
        float* hqn=(float*)malloc(sizeof(float)*HD); for(int i=0;i<HD;++i) hqn[i]=1.f;
        cudaMemcpy(qn,hqn,sizeof(float)*HD,cudaMemcpyHostToDevice); free(hqn);
        int* hpos=(int*)malloc(sizeof(int)*maxNQ); for(int i=0;i<maxNQ;++i) hpos[i]=CTX+i;
        cudaMemcpy(pos,hpos,sizeof(int)*maxNQ,cudaMemcpyHostToDevice); free(hpos);
        // chain ancestors: node j attends slots {CTX..CTX+j} (its path). off[j+1]=off[j]+(j+1).
        int* hoff=(int*)malloc(sizeof(int)*(maxNQ+1)); int* hsl=(int*)malloc(sizeof(int)*maxNQ*maxNQ);
        int w=0; hoff[0]=0; for(int j=0;j<maxNQ;++j){ for(int a=0;a<=j;++a) hsl[w++]=CTX+a; hoff[j+1]=w; }
        cudaFree(slots); cudaMalloc(&slots,sizeof(int)*w);
        cudaMemcpy(off,hoff,sizeof(int)*(maxNQ+1),cudaMemcpyHostToDevice);
        cudaMemcpy(slots,hsl,sizeof(int)*w,cudaMemcpyHostToDevice); free(hoff); free(hsl);
    }
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    printf("mk_tree_attn latency @ ctx=%d, %dQ/%dKV/hd%d (chain tree):\n", CTX,HQ,HKV,HD);
    for(int wi=0; wi<5; ++wi){
        int NQ=widths[wi];
        int warps=NQ*HQ, wpb=4, blocks=(warps+wpb-1)/wpb, shmem=wpb*HD*sizeof(float);
        // warmup
        for(int i=0;i<10;++i) mk_tree_attn<<<blocks,wpb*32,shmem>>>(NQ,HQ,HKV,HD,qp,qn,pos,kc,vc,CTX,off,slots,theta,out);
        cudaDeviceSynchronize();
        cudaEventRecord(s);
        for(int i=0;i<ITERS;++i) mk_tree_attn<<<blocks,wpb*32,shmem>>>(NQ,HQ,HKV,HD,qp,qn,pos,kc,vc,CTX,off,slots,theta,out);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms,s,e);
        double us = 1000.0*ms/ITERS;
        printf("  tree width %2d : %7.1f us/round  (%.2f us/node)\n", NQ, us, us/NQ);
    }
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess) printf("CUDA err: %s\n", cudaGetErrorString(err));
    return 0;
}
