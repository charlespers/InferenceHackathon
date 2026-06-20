// k6_overlap_decode.cu — the deferred-overlap megakernel STRUCTURE (LOOP-C's schedule + the NVLS kernel).
// The lossless comms-hide for 1000 tok/s (results-reaction-05, research/exact_deferred_overlap.md §2b):
// stale/predicted-TP is DEAD (info barrier); the comms is hidden LOSSLESSLY by overlapping the EXACT NVLS
// all-reduce with the next op's HBM weight stream — different HW paths (NVLink vs HBM). This is the SM-
// specialization skeleton: a few blocks run the multimem reduce while the rest cp.async the next weight tiles,
// a grid barrier gates the dependent multiply on the reduced activation. ONE persistent kernel over 94 layers.
//
// STATUS: STRUCTURE/skeleton to integrate into the team's megakernel_decode.cu + tune on box (like K5/NVLS began).
// Build: nvcc -O3 -arch=sm_90a k6_overlap_decode.cu -lcuda  (Hopper: multimem + cooperative-groups grid sync).
// Make-or-break C: measure_collective.sh — C≤~4µs (≤ the ~4.3µs fp8 per-collective weight cover) ⇒ FULL hide ⇒
// comms→0 ⇒ ~roofline (~1280, no spec needed); C=16µs ⇒ partial (~half) ⇒ ~938 with spec. Both lossless.

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
namespace cg = cooperative_groups;

#define NRANKS 8
#define N_REDUCE_BLOCKS 4     // 8KB multimem needs only a few blocks; rest stream weights. TUNE {2,4,8}.

// (declared in nvls_allreduce.cu / k5_experts_pipelined.cu; shown here as the overlap call sites)
//
// FIX (research/k6_overlap_exactness_gate.md, C3 — confirmed required, not optional): expert_gemv's
// fp8 dequant (k5_experts_pipelined.cu's deq2()) needs a per-row scale; the original signature here had
// no parameter for it. fp8 weights with no scale is the WRONG MAGNITUDE, not a benign cast — it would
// compile and run without crashing but fail the bit-exact gate against the serial reference. Widened
// below, and the top-level kernel now takes `layer_scales` (mirrors `layer_weights`'s per-layer shape)
// to actually have a scales pointer to pass through.
__device__ void multimem_allreduce_8kb(half* mc_ptr, int n);                 // the NVLS reduce (few blocks)
// ALSO widened (same reason as expert_gemv above): cuda::memcpy_async needs a pipeline/barrier handle
// to know what to wait on later -- stream_weight_tile's own doc says the CALLER owns acquire/commit/
// wait/release around it, so the pipeline is threaded through here, not hidden inside the function.
__device__ void stream_weight_tile(const void* hbm_src, void* smem_dst, int bytes,
                                    cuda::pipeline<cuda::thread_scope_block>& pipe);  // cp.async (many blocks)
__device__ void expert_gemv(const half* x, const void* w_smem, const half* scales,
                             half* y, int rows, int k);

// TILE_ROWS: a PLACEHOLDER reduced width, not the real MOE_INTER_RANK=1536. The smoke test crashed
// with an illegal memory access because both stream_weight_tile call sites below hardcoded bytes=0
// (a literal unwired placeholder -- 0 bytes ever copied) while expert_gemv read as if the FULL
// [1536 x 4096] weight matrix (6.29 MB) were smem-resident, which can't fit (SMs have ~228KB). The
// real fix is the full STAGES/TILE_K multi-tile loop k5_experts_pipelined.cu already has for its
// __global__ kernel -- porting that loop structure to this call site is the next real task, not
// done here. This is a SMALLER, honest placeholder (8 rows, 8*4096=32KB smem) that actually fits and
// lets the scheduling/sync mechanism (grid.sync, pipeline acquire/wait) be smoke-tested without
// crashing -- it does NOT process the real MoE width.
#define TILE_ROWS 8

// The persistent megakernel. grid = N_REDUCE_BLOCKS + (many stream/compute blocks). One launch, loops 94 layers.
extern "C" __global__ void k6_overlap_decode(half* act,                 // residual stream (on-chip across layers)
                                             const void* const* layer_weights, // [94] per-layer weight bases (HBM)
                                             const half* const* layer_scales,  // [94] per-layer dequant scales (HBM)
                                             half* mc_buf,              // multicast buffer for the all-reduce
                                             int num_layers) {
    cg::grid_group grid = cg::this_grid();
    bool is_reduce = (blockIdx.x < N_REDUCE_BLOCKS);     // SM SPECIALIZATION: reduce blocks vs stream/compute blocks
    extern __shared__ unsigned char smem[];
    // one BLOCK-SCOPED pipeline per block (make_pipeline() alone defaults to thread-scope, which
    // doesn't typecheck against stream_weight_tile's thread_scope_block signature) -- needs an
    // explicit shared_state per the documented cuda::pipeline block-scope construction pattern.
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1> pipe_state;
    auto pipe = cuda::make_pipeline(cg::this_thread_block(), &pipe_state);

    for (int L = 0; L < num_layers; ++L) {
        // ---- ATTENTION (omitted): produces attn_out into `act` (TP-sharded partial) ----
        // ===== Collective AR-A (post-attn) || prefetch this layer's MoE gate/up weights (LOOP-C schedule) =====
        if (is_reduce) {
            multimem_allreduce_8kb(mc_buf, /*n=*/4096);          // EXACT in-switch reduce on a few SMs
        } else {
            pipe.producer_acquire();
            stream_weight_tile(layer_weights[L]/*MoE gate/up*/, smem, TILE_ROWS * 4096, pipe);  // HBM read in flight
            pipe.producer_commit();
        }
        grid.sync();                                              // gate the dependent multiply on the reduced act
        // now act holds the reduced attention output AND the MoE gate/up weights are smem-resident:
        if (!is_reduce) {
            pipe.consumer_wait();                                  // wait for this block's own prefetch to land
            expert_gemv(act, smem, layer_scales[L], act, /*rows*/TILE_ROWS, 4096);  // MoE gate/up (PLACEHOLDER width)
            pipe.consumer_release();
        }
        grid.sync();

        // ---- MoE down-proj (omitted): produces moe_out partial ----
        // ===== Collective AR-M (post-MoE) || prefetch NEXT layer's QKV weights (LOOP-C schedule) =====
        if (is_reduce) {
            multimem_allreduce_8kb(mc_buf, 4096);
        } else if (L + 1 < num_layers) {
            pipe.producer_acquire();
            stream_weight_tile(layer_weights[L + 1]/*next QKV*/, smem, TILE_ROWS * 4096, pipe);  // next-layer QKV (PLACEHOLDER width)
            pipe.producer_commit();
        }
        grid.sync();
        // act = residual + reduced moe_out ; next layer's QKV already resident -> loop continues with no stall.
    }
    // final norm + lm_head + argmax (greedy fast-path) -> sampled token stays on-device for the next step.
}

// Notes for the box / the team's megakernel_decode.cu integration:
//  * SM specialization via blockIdx is the simplest split; a work-queue is more flexible if expert routing is
//    data-dependent (the reduce blocks could also help stream once their tiny AR is done).
//  * The hide is `min(C, weight_cover)` per collective; cover ≈ 4.3µs fp8 (MoE gate/up ~3.7, next-QKV ~1.75).
//    So C≤4µs fully hides; keep N_REDUCE_BLOCKS small so the stream/compute blocks stay resident (occupancy).
//  * grid.sync() requires cooperative launch (cudaLaunchCooperativeKernel) — verify the occupancy fits one wave.
//  * Validate: parity vs the bf16 reference (LOSSLESS — the exact AR still runs); then the e2e tok/s vs 85.7.
//  * This subsumes K5 (the expert_gemv = the fp8 cp.async kernel) + the NVLS reduce + the fast-path loop.
