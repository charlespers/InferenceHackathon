// K6 — host-side whole-step CUDA graph capture + replay, with on-device sampling.
// Captures K1->K2->K3->K4->K5 x 94 layers + final-norm + lm_head + argmax/sample as ONE graph,
// so the entire decode step (incl. EP all-to-all collectives) replays with ~zero launch cost.
// For speculative decode, capture the fixed (gamma+1)-position verify pass + masked KV commit
// (see docs/kernel-design/spec-decode-cuda-graph.md).
#include "common.cuh"
#include <cuda_runtime.h>

struct DecodeGraph { cudaGraph_t graph{}; cudaGraphExec_t exec{}; cudaStream_t stream{}; bool built=false; };

// Build once (first decode step): record all kernel launches on a capture stream.
// launch_step() must enqueue the full per-token kernel sequence onto `s` using only
// device-resident inputs (token id from a device ring; sel_idx/sel_w on device; etc.) so the
// graph is self-contained and needs no per-token host hand-off.
template <class LaunchStep>
inline void build_decode_graph(DecodeGraph& g, LaunchStep&& launch_step) {
  cudaStreamCreate(&g.stream);
  cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal);
  launch_step(g.stream);                  // K1..K5 x94 + final norm + lm_head + on-device argmax
  cudaStreamEndCapture(g.stream, &g.graph);
  cudaGraphInstantiate(&g.exec, g.graph, nullptr, nullptr, 0);
  g.built = true;
  // TODO(on-box): confirm capture-compat of the comms backend (NCCL graph-safe init / NVSHMEM);
  //   for spec-decode use conditional nodes (CUDA 12.4+) or the fixed-window masked-commit graph.
}

// Replay per token: device ring supplies the previous token id; output ring receives the new
// one. Host drains the output ring lazily (every N steps), never in the critical path.
inline void replay_decode_step(DecodeGraph& g) {
  cudaGraphLaunch(g.exec, g.stream);
  // NO cudaStreamSynchronize here per token — keep replays back-to-back; sync only when the
  // host wants to drain output ids. TODO(on-box): measure Nsight for any hidden D2H sync.
}
// TODO(on-box): on-device argmax (greedy) / categorical+spec-sampling (temp>0); device counter
//   n_accept + masked KV commit for the speculative path; keep a non-graph eager fallback.
