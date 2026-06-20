// k5_experts_pipelined.cu — cp.async double-buffered fp8 MoE-expert GEMV (the e->1 path).
// The SECOND make-or-break kernel for 1000 tok/s (docs/megakernel-build-plan.md Stage 2, k5-tuning-roadmap.md):
// take the measured K5 (warp-per-row, e=0.459 = 46% of HBM peak) toward e->1 by hiding HBM latency with
// cp.async multi-stage prefetch. At B=1 the expert FFN is pure memory streaming (AI~1); e=0.46 means the warps
// stall on long-scoreboard (HBM latency) — the fix is MORE in-flight loads, i.e. prefetch tile k+1 while
// computing tile k. This is the structure; TUNE stages/threads on the box against k5_microbench's `e`.
//
// Build: nvcc -O3 -arch=sm_90a k5_experts_pipelined.cu -o k5p   (Hopper cp.async.cg)
// Validate: vs a bf16 reference GEMV, max_rel < 1e-3 (fp8 dequant). Report e = achieved DRAM BW / peak.
//
// Expert FFN (Qwen3): SwiGLU. y = down( silu(gate(x)) * up(x) ). At B=1, x is one vector (hidden=4096).
// gate,up: [1536 x 4096] fp8 ; down: [4096 x 1536] fp8. This kernel shows the down/gate GEMV core (M=1).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cstdint>
#include <cstdio>

#define HIDDEN 4096
#define INTER  1536
#define STAGES 3            // cp.async pipeline depth — TUNE {2,3,4} vs smem budget on box
#define TILE_K 512          // contraction tile (elements of the reduction dim) per step

// fp8 (e4m3) -> half2 dequant of 2 packed fp8 values (per-channel scale hoisted by caller).
__device__ __forceinline__ half2 deq2(uint16_t packed, half scale) {
    // __nv_fp8x2_e4m3 -> half2 ; multiply by the (single) channel scale. (Box: use __nv_cvt_fp8x2_to_halfraw2.)
    __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3);
    half2 h = *reinterpret_cast<half2*>(&r);
    return __hmul2(h, __halves2half2(scale, scale));
}

// One warp computes one OUTPUT row (a dot product of length K). Weights streamed via cp.async double-buffer.
// w_row: fp8 weights for this output row [K] ; x: activation [K] in smem (shared across rows) ; scale: per-row.
__global__ void expert_gemv_pipelined(const uint8_t* __restrict__ W,   // [ROWS x K] fp8, row-major
                                      const half* __restrict__ x_g,     // [K] activation
                                      const half* __restrict__ scales,  // [ROWS] per-row dequant scale
                                      half* __restrict__ y,             // [ROWS] output
                                      int ROWS, int K) {
    extern __shared__ uint8_t smem[];                       // STAGES weight tiles + the x tile
    uint8_t* wbuf = smem;                                   // STAGES * TILE_K bytes (fp8)
    half*    xbuf = reinterpret_cast<half*>(smem + STAGES * TILE_K);  // TILE_K halves

    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (row >= ROWS) return;
    const uint8_t* w_row = W + (size_t)row * K;
    half scale = scales[row];

    auto pipe = cuda::make_pipeline();
    float acc = 0.f;
    int n_tiles = (K + TILE_K - 1) / TILE_K;

    // prime the pipeline: issue the first STAGES-1 cp.async loads of weight (and x) tiles.
    for (int s = 0; s < STAGES - 1 && s < n_tiles; ++s) {
        pipe.producer_acquire();
        // coalesced 128-bit cp.async of this tile's fp8 weights into wbuf[s] (warp-strided).
        for (int i = lane * 16; i < TILE_K; i += 32 * 16)
            cuda::memcpy_async(&wbuf[s*TILE_K + i], &w_row[s*TILE_K + i], cuda::aligned_size_t<16>(16), pipe);
        pipe.producer_commit();
    }
    for (int t = 0; t < n_tiles; ++t) {
        // prefetch tile t+STAGES-1 while we compute tile t (the overlap that raises e).
        int pf = t + STAGES - 1;
        if (pf < n_tiles) {
            pipe.producer_acquire();
            int s = pf % STAGES;
            for (int i = lane * 16; i < TILE_K; i += 32 * 16)
                cuda::memcpy_async(&wbuf[s*TILE_K + i], &w_row[pf*TILE_K + i], cuda::aligned_size_t<16>(16), pipe);
            pipe.producer_commit();
        }
        pipe.consumer_wait();                               // wait for tile t to land
        int s = t % STAGES;
        // compute: dot(deq(wbuf[s]), x[t*TILE_K:]) over the warp; fp8x2->half2, 2 macs/iter.
        for (int i = lane * 2; i < TILE_K; i += 64) {
            uint16_t packed = *reinterpret_cast<const uint16_t*>(&wbuf[s*TILE_K + i]);
            half2 w2 = deq2(packed, scale);
            half2 x2 = *reinterpret_cast<const half2*>(&x_g[t*TILE_K + i]);  // (or xbuf if staged)
            acc += __half2float(__hmul(w2.x, x2.x)) + __half2float(__hmul(w2.y, x2.y));
        }
        pipe.consumer_release();
    }
    // warp-reduce acc across lanes
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[row] = __float2half(acc);
}

int main() {
    printf("k5_experts_pipelined: cp.async double-buffered fp8 expert GEMV (the e->1 path).\n");
    printf("  On box: wire W/x/scales (one expert, M=1, K=4096 or 1536), launch ROWS warps, STAGES=3.\n");
    printf("  Measure e = achieved DRAM BW / 3.35TB/s vs k5_experts_warp.cu (0.46). TUNE: STAGES{2,3,4},\n");
    printf("  threads/CTA{256,512}, TILE_K{256,512}; watch Nsight stall_long_scoreboard CRASH as overlap kicks in.\n");
    printf("  TARGET: e -> 0.75-0.85 (k5-tuning-roadmap.md). At e=0.85 the fp8 weight read is 0.92ms ->\n");
    printf("  ladder_to_1000.py still clears 1000 (1040) at NVLS@2us + small-spec. So e need not be 1.0.\n");
    return 0;
}
