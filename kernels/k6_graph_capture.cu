// K6 — whole-step CUDA graph capture/replay helper for the fused B=1 decode step.
//
// decode_step.cu builds the per-token kernel chain (K1->K2->K3->K4->K5 x 94 layers + final-norm +
// lm_head + on-device argmax/sample) — ~660 launches/token. At B=1 each launch is a few us of
// CPU->GPU dispatch and the model runs at ~11% of the HBM roofline, i.e. launch/overhead-bound.
// K6 records that whole chain ONCE into a cudaGraph and replays it per token with a SINGLE launch,
// so per-token dispatch collapses from ~660 to 1.
//
// Header-style (no main, all inline): #include into the step translation unit (see decode_step.cu).
// Standard CUDA only; public model facts only.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

struct DecodeGraph {
  cudaGraph_t     graph       = nullptr;
  cudaGraphExec_t exec        = nullptr;
  cudaStream_t    stream      = nullptr;   // capture + replay stream
  bool            owns_stream = false;     // true if K6 created it (then K6 destroys it)
  bool            built       = false;
};

// Capture the full decode step into a graph.
//
// `launch_step(s)` MUST enqueue the ENTIRE per-token kernel sequence onto stream `s` using only
// device-resident inputs (token id from a device ring; sel_idx/sel_w on device; etc.) so the graph
// is self-contained and needs no per-token host hand-off. Do all cudaMalloc + cudaFuncSetAttribute
// BEFORE calling this (allocation is not stream-capturable); K6 warms the step up once outside the
// capture region so lazy module load / func-attr resolve before recording.
//
// Pass a stream to capture on, or 0 to have K6 create (and later destroy) one. Returns false on any
// CUDA error so the caller can fall back to eager replay.
template <class LaunchStep>
inline bool build_decode_graph(DecodeGraph& g, LaunchStep&& launch_step, cudaStream_t stream = 0) {
  if (stream) { g.stream = stream; g.owns_stream = false; }
  else {
    if (cudaStreamCreate(&g.stream) != cudaSuccess) return false;
    g.owns_stream = true;
  }
  // Warm up once OUTSIDE capture (module load, func-attr, lazy allocs in the launch helpers).
  launch_step(g.stream);
  if (cudaStreamSynchronize(g.stream) != cudaSuccess) return false;

  if (cudaStreamBeginCapture(g.stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
  launch_step(g.stream);                       // record K1..K5 x94 + final-norm + lm_head + argmax
  if (cudaStreamEndCapture(g.stream, &g.graph) != cudaSuccess) return false;
  if (cudaGraphInstantiate(&g.exec, g.graph, nullptr, nullptr, 0) != cudaSuccess) return false;
  g.built = true;
  // TODO(on-box): confirm capture-compat of the comms backend (NCCL graph-safe init / NVSHMEM);
  //   for spec-decode use conditional nodes (CUDA 12.4+) or a fixed-window masked-commit graph.
  return true;
}

// Number of nodes the captured graph instantiated (sanity / diagnostics).
inline size_t decode_graph_nodes(const DecodeGraph& g) {
  size_t n = 0;
  if (g.graph) cudaGraphGetNodes(g.graph, nullptr, &n);
  return n;
}

// Replay the whole step per token: one launch, no per-token host sync. The device ring supplies the
// previous token id and receives the new one; the host drains the output ring lazily (every N steps),
// never in the critical path. Keep replays back-to-back; sync only to drain.
inline void replay_decode_step(DecodeGraph& g) { cudaGraphLaunch(g.exec, g.stream); }

inline void destroy_decode_graph(DecodeGraph& g) {
  if (g.exec)  cudaGraphExecDestroy(g.exec);
  if (g.graph) cudaGraphDestroy(g.graph);
  if (g.owns_stream && g.stream) cudaStreamDestroy(g.stream);
  g = DecodeGraph{};
}

// Eager fallback (no graph): call `launch_step(stream)` directly per token. Keep it for
// capture-incompatible backends; decode_step.cu times graph vs eager to expose the launch-overhead
// delta. TODO(on-box): on-device argmax (greedy) / categorical+spec sampling (temp>0); device
// n_accept counter + masked KV commit for the speculative verify path.
