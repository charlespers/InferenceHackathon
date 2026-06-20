// gridsync_microbench.cu — isolate the COST of cg::grid_group::sync() on this box.
//
// The megakernel is ~472us/layer = ~12 grid.sync()/layer + tiny compute + NVLS AR.  The whole question
// for 500 tok/s is: is that ~450us/layer the grid.sync PRIMITIVE (~33us each), or load-IMBALANCE the
// barrier waits on?  This kernel does ONLY grid.sync() in a loop, ZERO work between — so the measured
// us/sync is the pure primitive cost.  Single GPU (grid.sync is intra-device), cooperative launch,
// matched to the megakernel's launch config (256 threads, 32KB dyn smem -> same occupancy).
//
// BUILD: nvcc -arch=sm_90a -O3 -rdc=true kernels/gridsync_microbench.cu -lcuda -o /tmp/gsync
// RUN:   CUDA_VISIBLE_DEVICES=0 /tmp/gsync
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// Pure barrier loop: N grid.sync()s, nothing else.  dummy write so the compiler can't elide.
extern "C" __global__ void gsync_only(int iters, int* sink) {
  cg::grid_group grid = cg::this_grid();
  int acc = 0;
  for (int i = 0; i < iters; ++i) { grid.sync(); acc += i; }
  if (threadIdx.x == 0 && blockIdx.x == 0) *sink = acc;
}

// Barrier + a SMALL balanced touch (1 fma/thread) to mimic perfectly-balanced work between syncs.
extern "C" __global__ void gsync_balanced(int iters, int* sink, float* buf, int n) {
  cg::grid_group grid = cg::this_grid();
  const int g = blockIdx.x*blockDim.x+threadIdx.x, stride = gridDim.x*blockDim.x;
  float acc = 0.f;
  for (int i = 0; i < iters; ++i) {
    for (int j = g; j < n; j += stride) acc += buf[j]*1.0001f;   // balanced grid-stride touch
    grid.sync();
  }
  if (g == 0) *sink = (int)acc;
}

int main(int argc, char** argv) {
  const int ITERS = (argc>1)?atoi(argv[1]):2000;
  int dev=0; CK(cudaSetDevice(dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,dev));
  const int block = 256;
  const size_t smem = 32*1024;   // match megakernel dyn smem (drives occupancy)
  CK(cudaFuncSetAttribute(gsync_only, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CK(cudaFuncSetAttribute(gsync_balanced, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int bpsm=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, gsync_only, block, smem));
  printf("device %s  SMs=%d  max blocks/SM @256thr,32KBsmem = %d\n", prop.name, prop.multiProcessorCount, bpsm);

  int* sink; CK(cudaMalloc(&sink,sizeof(int)));
  const int N = 4096; float* buf; CK(cudaMalloc(&buf,N*sizeof(float))); CK(cudaMemset(buf,0,N*sizeof(float)));
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

  printf("\n  %-28s %8s %12s %10s\n","config","blocks","us/sync","(iters=K)");
  for (int bp = 1; bp <= bpsm && bp <= 4; ++bp) {
    int blocks = bp * prop.multiProcessorCount;
    // ---- pure grid.sync ----
    void* args[] = { (void*)&ITERS, (void*)&sink };
    dim3 g(blocks), bl(block);
    // warmup
    int warm = 200; void* wargs[]={(void*)&warm,(void*)&sink};
    CK(cudaLaunchCooperativeKernel((void*)gsync_only, g, bl, wargs, smem, 0)); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    CK(cudaLaunchCooperativeKernel((void*)gsync_only, g, bl, args, smem, 0));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
    printf("  %-28s %8d %12.3f %10d\n","pure grid.sync()", blocks, ms*1e3/ITERS, ITERS);
  }
  // ---- balanced-work variant at the megakernel's block count (1/SM) ----
  {
    int blocks = prop.multiProcessorCount;
    void* args[] = { (void*)&ITERS, (void*)&sink, (void*)&buf, (void*)&N };
    dim3 g(blocks), bl(block);
    int warm=200; void* wargs[]={(void*)&warm,(void*)&sink,(void*)&buf,(void*)&N};
    CK(cudaLaunchCooperativeKernel((void*)gsync_balanced, g, bl, wargs, smem, 0)); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    CK(cudaLaunchCooperativeKernel((void*)gsync_balanced, g, bl, args, smem, 0));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
    printf("  %-28s %8d %12.3f %10d\n","grid.sync()+balanced touch", blocks, ms*1e3/ITERS, ITERS);
  }
  printf("\nINTERPRETATION:\n");
  printf("  if pure grid.sync ~= 33us  -> the BARRIER PRIMITIVE is the wall (need head-locality restructure)\n");
  printf("  if pure grid.sync ~= 1-5us -> the megakernel's 450us/layer is LOAD IMBALANCE (tractable: balance work)\n");
  printf("  the megakernel needs ~12 syncs/layer; 500 tok/s needs <21us/layer TOTAL -> grid.sync must be ~1us.\n");
  return 0;
}
