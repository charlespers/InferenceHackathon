// bf16_gemm_ext.cu — minimal cuBLASLt bf16xbf16->bf16 GEMM, exposed as a PyTorch op.
//
// THE MIDPOINT COMPROMISE: rather than replacing vLLM's serving stack (tokenizer, scheduler,
// PagedAttention KV-cache, sampling -- everything this engine's native pipeline still lacks), this
// swaps in ONLY the team's GEMM kernel at vLLM's own sanctioned extension point: RowParallelLinear /
// QKVParallelLinear dispatch to `self.quant_method.apply(layer, x, bias)` (confirmed by reading
// vllm/model_executor/layers/linear.py directly -- this is not a guess). Everything else stays vLLM's.
//
// WHY BF16, NOT FP8: this live deployment runs `--dtype bfloat16` (no --quantization flag), so a fp8
// GEMM here would need a whole new on-the-fly quantization step just to plug in -- bf16xbf16->bf16 is
// the simpler, more honest match to what's actually running, and cuBLASLt supports it natively with no
// scale tracking at all (unlike kernels/gemm_engine.cuh's fp8 path, which exists because the NATIVE
// engine's own weights are stored fp8 -- vLLM's aren't, here).
//
// VALIDATION CONTRACT: this kernel's output is meant to be diffed against vLLM's own stock GEMM (same
// X, same W) before ever being trusted in the live serving path -- see tools/validate_bf16_gemm_swap.py.
// Don't skip that step; a GEMM that's wrong but fast is worse than not swapping anything.
//
// HONEST STATUS: this creates/destroys the cuBLASLt handle and re-runs the heuristic search on EVERY
// call -- correctness-first, not optimized (matches every other "validate first, tune later" step in
// this session). A real integration would cache the handle/algo per (M,N,K) shape, same as
// kernels/gemm_engine.cuh's LtPanel does for the native engine. Don't read any speed claim into this
// until that's done AND the numeric validation passes.
//
// BUILD: JIT-compiled via torch.utils.cpp_extension.load() at import time (tools/vllm_bf16_linear_patch.py)
// -- no separate build step, matches "queue it and validate" rather than a packaged release.
#include <torch/extension.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>

#define CL(x) do { cublasStatus_t st_ = (x); \
  TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, "cuBLASLt error ", (int)st_, " at ", __FILE__, ":", __LINE__); \
  } while (0)

// One-shot bf16 TN-GEMM: Y[M,N] = X[M,K] @ W[N,K]^T (W stored row-major [N,K], matching nn.Linear's
// weight layout exactly -- no repack needed vs PyTorch's own convention).
//   X: [M,K] bf16, row-major (torch default)
//   W: [N,K] bf16, row-major (torch default, nn.Linear.weight's natural shape)
//   Y: [M,N] bf16, row-major
// cuBLAS is column-major internally. X's row-major [M,K] bytes ARE, with zero transform, a valid
// column-major [K,M] matrix (= X^T) -- and likewise W's row-major [N,K] bytes ARE column-major [K,N]
// (= W^T). We want Y^T[N,M] (col-major) = W[N,K] @ X^T[K,M] = op(W^T_stored)[N,K] @ X^T_stored[K,M],
// i.e. A = W^T_stored[K,N] with opA=TRANSPOSE (recovers W), B = X^T_stored[K,M] with opB=NONE (already
// the right shape, no transform needed) -- NOT opB=TRANSPOSE (an earlier version of this file had that
// backwards: cuBLASLt requires the STORED pre-op shape to be [n,k] when opB=T, but [K,M] is [k,n], not
// [n,k] -- that mismatch is exactly what cuBLASLt's heuristic search rejected with INVALID_VALUE).
torch::Tensor bf16_gemm(torch::Tensor X, torch::Tensor W) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda(), "bf16_gemm: inputs must be CUDA tensors");
  TORCH_CHECK(X.dtype() == torch::kBFloat16 && W.dtype() == torch::kBFloat16, "bf16_gemm: expects bf16");
  TORCH_CHECK(X.dim() == 2 && W.dim() == 2, "bf16_gemm: expects 2D [M,K] and [N,K]");
  TORCH_CHECK(X.size(1) == W.size(1), "bf16_gemm: K mismatch");
  X = X.contiguous(); W = W.contiguous();
  const int M = X.size(0), K = X.size(1), N = W.size(0);

  auto Y = torch::empty({M, N}, X.options());

  cublasLtHandle_t lt;
  CL(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op;
  CL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
  CL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));

  cublasLtMatrixLayout_t aL, bL, dL;
  CL(cublasLtMatrixLayoutCreate(&aL, CUDA_R_16BF, K, N, K));   // W^T_stored [K,N], opA=TRANSPOSE -> W[N,K]
  CL(cublasLtMatrixLayoutCreate(&bL, CUDA_R_16BF, K, M, K));   // X^T_stored [K,M], opB=NONE (already right shape)
  CL(cublasLtMatrixLayoutCreate(&dL, CUDA_R_16BF, N, M, N));   // Y^T, col-major [N,M] == Y row-major [M,N]

  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsBytes = 32ull << 20;
  void* ws; cudaMalloc(&ws, wsBytes);
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes)));

  cublasLtMatmulHeuristicResult_t heur;
  int got = 0;
  CL(cublasLtMatmulAlgoGetHeuristic(lt, op, aL, bL, dL, dL, pref, 1, &heur, &got));
  TORCH_CHECK(got > 0, "bf16_gemm: no cuBLASLt heuristic found for this shape");

  const float alpha = 1.f, beta = 0.f;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CL(cublasLtMatmul(lt, op, &alpha, W.data_ptr(), aL, X.data_ptr(), bL, &beta,
                    Y.data_ptr(), dL, Y.data_ptr(), dL, &heur.algo, ws, wsBytes, stream));

  cudaFree(ws);
  cublasLtMatrixLayoutDestroy(aL); cublasLtMatrixLayoutDestroy(bL); cublasLtMatrixLayoutDestroy(dL);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatmulDescDestroy(op);
  cublasLtDestroy(lt);
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_gemm", &bf16_gemm, "bf16xbf16->bf16 GEMM via cuBLASLt (Y = X @ W^T)");
}
