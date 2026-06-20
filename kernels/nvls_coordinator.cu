// nvls_coordinator.cu — Long-lived process that owns the CUmulticastObject.
// Creates the MC object for N_GPUS devices, exports one POSIX FD per GPU,
// writes /tmp/nvls_mc.json with coordinates, then blocks until SIGTERM.
//
// vLLM workers read the JSON, open /proc/<coord_pid>/fd/<fd_for_rank>
// and call cuMemImportFromShareableHandle to join the multicast group.
//
// BUILD:
//   nvcc -arch=sm_90a -O3 -I kernels/ kernels/nvls_coordinator.cu -lcuda -o /tmp/nvls_coord
// RUN (in background before vLLM):
//   /tmp/nvls_coord &
//   python3 tools/start_vllm.py   # with patch applied

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/stat.h>

#define DK(x) do { \
    CUresult _r=(x); \
    if(_r!=CUDA_SUCCESS){ const char*_e=nullptr; cuGetErrorString(_r,&_e); \
      fprintf(stderr,"CU ERR %s:%d [%d] %s\n",__FILE__,__LINE__,(int)_r,_e?_e:"?"); exit(1); } \
} while(0)
#define CK(x) do { \
    cudaError_t _e=(x); \
    if(_e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } \
} while(0)

static volatile int g_running = 1;
static void on_sig(int) { g_running = 0; }

static const char* COORD_JSON = "/tmp/nvls_mc.json";
static const int N_GPUS = 8;
static const size_t NBYTES_ATTN  = 4096  * sizeof(float);  // HIDDEN * fp32
static const size_t NBYTES_MOE   = 4096  * sizeof(float);  // same
static const size_t NBYTES_FLAG  = 2 * 4096 * sizeof(unsigned);  // 2 flag arrays

static size_t align_up(size_t n, size_t g) { return ((n+g-1)/g)*g; }

struct McHandle {
    CUmemGenericAllocationHandle h;
    int fds[N_GPUS];   // exported POSIX FD per GPU (each worker opens its own copy)
    size_t size;
};

static void make_mc(int n_gpus, size_t nbytes, McHandle& out) {
    CUmulticastObjectProp mcp; memset(&mcp, 0, sizeof(mcp));
    mcp.numDevices = n_gpus;
    mcp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    mcp.size = nbytes;

    size_t g = 0;
    DK(cuMulticastGetGranularity(&g, &mcp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    out.size = align_up(nbytes, g); mcp.size = out.size;

    DK(cuMulticastCreate(&out.h, &mcp));
    fprintf(stderr, "[coord] created MC object (size=%zu gran=%zu)\n", out.size, g);

    // Export one FD per GPU. Each FD is a separate file descriptor pointing to the
    // same multicast object — each worker gets its own to avoid FD lifetime issues.
    for (int d = 0; d < n_gpus; d++) {
        int fd = -1;
        CUresult er = cuMemExportToShareableHandle(&fd, out.h,
                                                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
        if (er != CUDA_SUCCESS) {
            const char* e = nullptr; cuGetErrorString(er, &e);
            fprintf(stderr, "[coord] FAIL: cuMemExportToShareableHandle [%d] %s\n", (int)er, e?e:"?");
            fprintf(stderr, "  => Cross-process multicast requires IMEX on this setup.\n");
            fprintf(stderr, "  => Run tools/patch_vllm_nvls.py --mode=ipc instead.\n");
            exit(2);  // exit code 2 = fall back to IPC
        }
        out.fds[d] = fd;
    }
    fprintf(stderr, "[coord] exported %d FDs: [%d..%d]\n", n_gpus, out.fds[0], out.fds[n_gpus-1]);
}

static void write_json(pid_t pid, const McHandle& attn, const McHandle& moe, const McHandle& flag) {
    FILE* f = fopen(COORD_JSON, "w");
    if (!f) { perror(COORD_JSON); exit(1); }
    fprintf(f, "{\n");
    fprintf(f, "  \"pid\": %d,\n", (int)pid);
    fprintf(f, "  \"n_gpus\": %d,\n", N_GPUS);

    auto dump = [&](const char* name, const McHandle& mc) {
        fprintf(f, "  \"%s_size\": %zu,\n", name, mc.size);
        fprintf(f, "  \"%s_fds\": [", name);
        for (int d = 0; d < N_GPUS; d++) fprintf(f, "%s%d", d?",":"", mc.fds[d]);
        fprintf(f, "],\n");
    };
    dump("attn", attn);
    dump("moe", moe);
    dump("flag", flag);
    fprintf(f, "  \"ready\": 1\n}\n");
    fclose(f);
    fprintf(stderr, "[coord] wrote %s\n", COORD_JSON);
}

int main() {
    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);

    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    if (ndev < N_GPUS) {
        fprintf(stderr, "[coord] need %d GPUs, have %d\n", N_GPUS, ndev);
        return 1;
    }

    // Initialize CUDA driver API.
    DK(cuInit(0));

    fprintf(stderr, "[coord] creating multicast objects for TP=%d...\n", N_GPUS);

    McHandle attn_mc, moe_mc, flag_mc;
    make_mc(N_GPUS, NBYTES_ATTN, attn_mc);
    make_mc(N_GPUS, NBYTES_MOE, moe_mc);
    make_mc(N_GPUS, NBYTES_FLAG, flag_mc);

    write_json(getpid(), attn_mc, moe_mc, flag_mc);
    fprintf(stderr, "[coord] READY — waiting for SIGTERM\n");
    fflush(stderr);

    // Keep running while vLLM uses the FDs.
    while (g_running) sleep(1);

    fprintf(stderr, "[coord] exiting\n");
    unlink(COORD_JSON);
    return 0;
}
