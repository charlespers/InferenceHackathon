// k2b_splits.cu — sweep n_splits at fixed M to find the real SM-fill point (does >4096 warps help?).
#define K2B_NO_MAIN
#include "k2_batched_decode.cu"
#include <cstdio>
#include <vector>
using namespace q3;
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
static void fill(std::vector<fp8>&h){ unsigned s=12345; for(size_t i=0;i<h.size();i++){ s=s*1664525u+1013904223u; float v=((int)(s>>8&0xffff)-32768)/32768.f*0.5f; __nv_fp8_e4m3 q(v); h[i]=*reinterpret_cast<fp8*>(&q);} }
int main(int argc,char**argv){
  int ctx=(argc>1)?atoi(argv[1]):4096; int iters=(argc>2)?atoi(argv[2]):300;
  int M=(argc>3)?atoi(argv[3]):8;
  std::vector<fp8> hK((size_t)ctx*KV_DIM),hV((size_t)ctx*KV_DIM); fill(hK); fill(hV);
  fp8 *dK,*dV; CK(cudaMalloc(&dK,hK.size())); CK(cudaMalloc(&dV,hV.size()));
  CK(cudaMemcpy(dK,hK.data(),hK.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dV,hV.data(),hV.size(),cudaMemcpyHostToDevice));
  int MMAX=16; std::vector<float> hQ((size_t)MMAX*N_Q_HEADS*HEAD_DIM);
  for(size_t i=0;i<hQ.size();i++) hQ[i]=(float)(((i*2654435761u)>>12)&0x3ff)/1024.f-0.5f;
  float*dQ; CK(cudaMalloc(&dQ,hQ.size()*4)); CK(cudaMemcpy(dQ,hQ.data(),hQ.size()*4,cudaMemcpyHostToDevice));
  int smax=256; float *dpm,*dpl,*dpa,*dout;
  CK(cudaMalloc(&dpm,k2b_part_m_elems(MMAX,smax)*4)); CK(cudaMalloc(&dpl,k2b_part_m_elems(MMAX,smax)*4));
  CK(cudaMalloc(&dpa,k2b_part_acc_elems(MMAX,smax)*4)); CK(cudaMalloc(&dout,(size_t)MMAX*N_Q_HEADS*HEAD_DIM*4));
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  printf("== splits sweep  ctx=%d  M=%d  (N_Q_HEADS=%d) ==\n",ctx,M,N_Q_HEADS);
  printf("  n_splits  warps_in_flight  chunk_len   us/K2-fwd\n");
  int splits[]={4,8,16,32,48,64,96,128,192,256};
  for(int ns: splits){
    int maxc=(ctx+31)/32; if(ns>maxc) continue;
    long warps=(long)M*N_Q_HEADS*ns; int chunk=(ctx+ns-1)/ns;
    for(int w=0;w<30;w++) k2b_launch(dQ,dK,dV,nullptr,nullptr,ctx,M,dpm,dpl,dpa,dout,ns);
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
    for(int i=0;i<iters;i++) k2b_launch(dQ,dK,dV,nullptr,nullptr,ctx,M,dpm,dpl,dpa,dout,ns);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1));
    printf("  %-9d %-16ld %-11d %8.2f\n",ns,warps,chunk,ms*1000.0/iters);
  }
  return 0;
}
