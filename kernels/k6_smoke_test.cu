// k6_smoke_test.cu — does k6_overlap_decode.cu's cooperative-launch design even RUN, single-GPU,
// without hanging? This is a LIVENESS smoke test, not a correctness validation:
//   - mc_buf comes from nvls_allreduce_singleproc.cu's mc_setup() -- that file's multi-DEVICE
//     correctness check still fails (maxerr=inf, an unresolved race), but a SINGLE-device launch
//     (this test) never has a second concurrent caller racing the same address, so that specific
//     bug doesn't apply here. Don't read this test as "the AR is correct" -- only as "did the grid
//     launch and complete."
//   - k6_overlap_decode.cu's own comments mark the ATTENTION and MoE-down-proj compute as "omitted"
//     -- this kernel does NOT implement a full transformer layer, only the AR-A/AR-M scheduling
//     skeleton + the expert_gemv call site. So even a clean run here does not validate end-to-end
//     decode correctness; it validates the cooperative-launch + grid.sync() + SM-specialization
//     MECHANISM in isolation.
// Real value: per k6_overlap_exactness_gate.md's "liveness caveat" -- if N_REDUCE_BLOCKS + compute
// blocks exceeds cooperative-launch occupancy, grid.sync() DEADLOCKS (hangs, not a wrong answer).
// This test is exactly the check that caveat calls for, run with the REAL kernel (not the proxy
// shape used for the earlier occupancy estimate).
//
// Build: nvcc -O3 -arch=sm_90a -rdc=true kernels/k6_smoke_test.cu -lcuda -o /tmp/k6_smoke
// Run:   timeout 30 /tmp/k6_smoke    (the timeout IS the hang detector)

#define NVLS_SP_NO_MAIN
#include "nvls_allreduce_singleproc.cu"   // MC, mc_setup(), CK/CKR -- reuse the (partially-working) multicast setup
#include "k6_device_functions.cu"          // multimem_allreduce_8kb, stream_weight_tile, expert_gemv (bodies)
#include "k6_overlap_decode.cu"            // the kernel itself (its own extern decls match the bodies above)

#define HIDDEN 4096
#define K6_INTER 1536

int main(int argc, char** argv) {
  int num_layers = (argc > 1) ? atoi(argv[1]) : 3;   // small on purpose for a first smoke test, not 94
  int compute_blocks = (argc > 2) ? atoi(argv[2]) : 32;

  CK(cuInit(0));
  CKR(cudaSetDevice(0));

  MC m;
  if (mc_setup(&m, HIDDEN * sizeof(half)) != 0) {
    fprintf(stderr, "mc_setup failed -- can't even get a multicast buffer; smoke test can't proceed.\n");
    return 1;
  }
  printf("== k6_smoke_test: num_layers=%d, compute_blocks=%d (+%d reduce) ==\n",
         num_layers, compute_blocks, 4 /*N_REDUCE_BLOCKS, hardcoded in k6_overlap_decode.cu*/);

  // ---- act: the residual stream, single device, just needs to be non-garbage for a liveness test ----
  half* act;
  CKR(cudaMalloc(&act, HIDDEN * sizeof(half)));
  std::vector<half> act_init(HIDDEN, __float2half(1.0f));
  CKR(cudaMemcpy(act, act_init.data(), HIDDEN * sizeof(half), cudaMemcpyHostToDevice));

  // ---- dummy per-layer weights + scales (real shapes, placeholder data -- not validating numerics here) ----
  std::vector<void*> h_weights(num_layers);
  std::vector<half*> h_scales(num_layers);
  std::vector<uint8_t> w_init(K6_INTER * HIDDEN, 1);          // fp8-sized placeholder bytes
  std::vector<half> s_init(K6_INTER, __float2half(1.0f));
  for (int L = 0; L < num_layers; ++L) {
    void* w; CKR(cudaMalloc(&w, K6_INTER * HIDDEN));
    CKR(cudaMemcpy(w, w_init.data(), K6_INTER * HIDDEN, cudaMemcpyHostToDevice));
    h_weights[L] = w;
    half* s; CKR(cudaMalloc(&s, K6_INTER * sizeof(half)));
    CKR(cudaMemcpy(s, s_init.data(), K6_INTER * sizeof(half), cudaMemcpyHostToDevice));
    h_scales[L] = s;
  }
  void** d_weights; CKR(cudaMalloc(&d_weights, num_layers * sizeof(void*)));
  CKR(cudaMemcpy(d_weights, h_weights.data(), num_layers * sizeof(void*), cudaMemcpyHostToDevice));
  half** d_scales; CKR(cudaMalloc(&d_scales, num_layers * sizeof(half*)));
  CKR(cudaMemcpy(d_scales, h_scales.data(), num_layers * sizeof(half*), cudaMemcpyHostToDevice));

  // ---- occupancy check against the REAL kernel (not the earlier proxy shape) before committing to launch ----
  int smem_bytes = 8 /*TILE_ROWS, matching k6_overlap_decode.cu's placeholder*/ * HIDDEN;  // 32KB
  int threads_per_block = 256;
  int max_blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, k6_overlap_decode, threads_per_block, smem_bytes);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  int max_grid = max_blocks_per_sm * prop.multiProcessorCount;
  int requested_grid = 4 + compute_blocks;
  printf("  occupancy: max co-resident blocks = %d, requesting %d (4 reduce + %d compute)\n",
         max_grid, requested_grid, compute_blocks);
  if (requested_grid > max_grid) {
    fprintf(stderr, "  REQUESTED GRID EXCEEDS COOPERATIVE-LAUNCH OCCUPANCY -- this WOULD deadlock on "
                     "grid.sync(). Reduce compute_blocks below %d. Not launching.\n", max_grid - 4);
    return 1;
  }

  void* args[] = {&act, &d_weights, &d_scales, &m.mc_va[0], &num_layers};
  dim3 grid(requested_grid), block(threads_per_block);

  printf("  launching cooperatively (timeout is the hang detector)...\n");
  cudaError_t launch_err = cudaLaunchCooperativeKernel((void*)k6_overlap_decode, grid, block, args, smem_bytes, 0);
  if (launch_err != cudaSuccess) {
    fprintf(stderr, "  cudaLaunchCooperativeKernel FAILED: %s\n", cudaGetErrorString(launch_err));
    return 1;
  }
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    fprintf(stderr, "  kernel ran but cudaDeviceSynchronize reported an ERROR (not a hang): %s\n",
            cudaGetErrorString(sync_err));
    return 1;
  }
  printf("  LIVENESS: completed without hanging or erroring (%d layers, grid=%d blocks).\n",
         num_layers, requested_grid);
  printf("  NOTE: this does not validate AR correctness (mc_buf's multi-device path is still buggy)\n");
  printf("  or full-layer numerics (attention/MoE-down are 'omitted' in k6_overlap_decode.cu itself).\n");
  return 0;
}
