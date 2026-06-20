/* mk_tree_attn_fp8_bench.cu — latency of the fp8-K/V fused M=k tree attention vs tree width. */
#define MK8_NO_MAIN
#include "mk_tree_attn_fp8.cu"
int main(int argc,char**argv){
  const int HQ=64,HKV=4,HD=128; int CTX=argc>1?atoi(argv[1]):4096, IT=argc>2?atoi(argv[2]):300, W=8;
  const float theta=1e6f; int widths[5]={1,4,8,16,32}, maxNQ=32, NTOT=CTX+maxNQ, KVD=HKV*HD;
  float *qp,*qn,*out,*ks,*vs; fp8 *kk,*vv; int *pos,*off,*sl;
  cudaMalloc(&qp,sizeof(float)*maxNQ*HQ*HD); cudaMalloc(&qn,sizeof(float)*HD); cudaMalloc(&out,sizeof(float)*maxNQ*HQ*HD);
  cudaMalloc(&kk,(size_t)NTOT*KVD); cudaMalloc(&vv,(size_t)NTOT*KVD); cudaMalloc(&ks,sizeof(float)*KVD); cudaMalloc(&vs,sizeof(float)*KVD);
  cudaMalloc(&pos,sizeof(int)*maxNQ); cudaMalloc(&off,sizeof(int)*(maxNQ+1));
  cudaMemset(qp,1,sizeof(float)*maxNQ*HQ*HD); cudaMemset(kk,1,(size_t)NTOT*KVD); cudaMemset(vv,1,(size_t)NTOT*KVD);
  float* h=(float*)malloc(sizeof(float)*KVD); for(int i=0;i<KVD;++i)h[i]=1.f; cudaMemcpy(ks,h,sizeof(float)*KVD,cudaMemcpyHostToDevice); cudaMemcpy(vs,h,sizeof(float)*KVD,cudaMemcpyHostToDevice);
  float* hq=(float*)malloc(sizeof(float)*HD); for(int i=0;i<HD;++i)hq[i]=1.f; cudaMemcpy(qn,hq,sizeof(float)*HD,cudaMemcpyHostToDevice);
  int* hp=(int*)malloc(sizeof(int)*maxNQ); for(int i=0;i<maxNQ;++i)hp[i]=CTX+i; cudaMemcpy(pos,hp,sizeof(int)*maxNQ,cudaMemcpyHostToDevice);
  int* ho=(int*)malloc(sizeof(int)*(maxNQ+1)); int* hs=(int*)malloc(sizeof(int)*maxNQ*maxNQ); int w=0; ho[0]=0;
  for(int j=0;j<maxNQ;++j){ for(int a=0;a<=j;++a) hs[w++]=CTX+a; ho[j+1]=w; }
  cudaMalloc(&sl,sizeof(int)*w); cudaMemcpy(off,ho,sizeof(int)*(maxNQ+1),cudaMemcpyHostToDevice); cudaMemcpy(sl,hs,sizeof(int)*w,cudaMemcpyHostToDevice);
  int shmem=(HD+2*W+W*HD)*sizeof(float); cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  printf("FP8 mk_tree_attn @ ctx=%d, 64Q/4KV/hd128, W=%d:\n", CTX, W);
  for(int wi=0;wi<5;++wi){ int NQ=widths[wi];
    for(int i=0;i<10;++i) mk_tree_attn_fp8<<<NQ*HQ,W*32,shmem>>>(NQ,HQ,HKV,HD,qp,qn,pos,kk,vv,ks,vs,CTX,off,sl,theta,out);
    cudaDeviceSynchronize(); cudaEventRecord(s);
    for(int i=0;i<IT;++i) mk_tree_attn_fp8<<<NQ*HQ,W*32,shmem>>>(NQ,HQ,HKV,HD,qp,qn,pos,kk,vv,ks,vs,CTX,off,sl,theta,out);
    cudaEventRecord(e); cudaEventSynchronize(e); float ms=0; cudaEventElapsedTime(&ms,s,e);
    printf("  width %2d : %7.1f us/round  (%.2f us/node)\n", NQ, 1000.0*ms/IT, 1000.0*ms/IT/NQ);
  }
  cudaError_t er=cudaGetLastError(); if(er!=cudaSuccess) printf("CUDA err: %s\n",cudaGetErrorString(er));
  return 0;
}
