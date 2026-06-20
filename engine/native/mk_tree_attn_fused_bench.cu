/* mk_tree_attn_fused_bench.cu — latency of the FUSED split-KV M=k tree attention vs tree width. */
#define MKF_NO_MAIN
#include "mk_tree_attn_fused.cu"
int main(int argc,char**argv){
  const int HQ=64,HKV=4,HD=128; int CTX=argc>1?atoi(argv[1]):4096, IT=argc>2?atoi(argv[2]):300, W=8;
  const float theta=1e6f; int widths[5]={1,4,8,16,32}, maxNQ=32, NTOT=CTX+maxNQ;
  float *qp,*qn,*kc,*vc,*out; int *pos,*off,*sl;
  cudaMalloc(&qp,sizeof(float)*maxNQ*HQ*HD); cudaMalloc(&qn,sizeof(float)*HD);
  cudaMalloc(&kc,sizeof(float)*NTOT*HKV*HD); cudaMalloc(&vc,sizeof(float)*NTOT*HKV*HD);
  cudaMalloc(&out,sizeof(float)*maxNQ*HQ*HD); cudaMalloc(&pos,sizeof(int)*maxNQ);
  cudaMalloc(&off,sizeof(int)*(maxNQ+1));
  cudaMemset(qp,1,sizeof(float)*maxNQ*HQ*HD); cudaMemset(kc,1,sizeof(float)*NTOT*HKV*HD); cudaMemset(vc,1,sizeof(float)*NTOT*HKV*HD);
  float* hqn=(float*)malloc(sizeof(float)*HD); for(int i=0;i<HD;++i)hqn[i]=1.f; cudaMemcpy(qn,hqn,sizeof(float)*HD,cudaMemcpyHostToDevice);
  int* hpos=(int*)malloc(sizeof(int)*maxNQ); for(int i=0;i<maxNQ;++i)hpos[i]=CTX+i; cudaMemcpy(pos,hpos,sizeof(int)*maxNQ,cudaMemcpyHostToDevice);
  int* hoff=(int*)malloc(sizeof(int)*(maxNQ+1)); int* hsl=(int*)malloc(sizeof(int)*maxNQ*maxNQ); int w=0; hoff[0]=0;
  for(int j=0;j<maxNQ;++j){ for(int a=0;a<=j;++a) hsl[w++]=CTX+a; hoff[j+1]=w; }
  cudaMalloc(&sl,sizeof(int)*w); cudaMemcpy(off,hoff,sizeof(int)*(maxNQ+1),cudaMemcpyHostToDevice); cudaMemcpy(sl,hsl,sizeof(int)*w,cudaMemcpyHostToDevice);
  int shmem=(HD+2*W+W*HD)*sizeof(float);
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  printf("FUSED mk_tree_attn @ ctx=%d, 64Q/4KV/hd128, W=%d:\n", CTX, W);
  for(int wi=0;wi<5;++wi){ int NQ=widths[wi];
    for(int i=0;i<10;++i) mk_tree_attn_fused<<<NQ*HQ,W*32,shmem>>>(NQ,HQ,HKV,HD,qp,qn,pos,kc,vc,CTX,off,sl,theta,out);
    cudaDeviceSynchronize(); cudaEventRecord(s);
    for(int i=0;i<IT;++i) mk_tree_attn_fused<<<NQ*HQ,W*32,shmem>>>(NQ,HQ,HKV,HD,qp,qn,pos,kc,vc,CTX,off,sl,theta,out);
    cudaEventRecord(e); cudaEventSynchronize(e); float ms=0; cudaEventElapsedTime(&ms,s,e);
    printf("  width %2d : %7.1f us/round  (%.2f us/node)\n", NQ, 1000.0*ms/IT, 1000.0*ms/IT/NQ);
  }
  cudaError_t er=cudaGetLastError(); if(er!=cudaSuccess) printf("CUDA err: %s\n",cudaGetErrorString(er));
  return 0;
}
