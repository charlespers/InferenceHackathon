// nvls_xproc_test.cu — POSIX-FD cross-process multicast viability test.
// Tests if cuMemExportToShareableHandle works for a CUmulticastObject handle
// (created by cuMulticastCreate) without IMEX / CAP_SYS_ADMIN.
//
// If this prints "PASS: cross-process NVLS all-reduce verified" we can wire it
// into vLLM's workers.  If it prints "FAIL: export blocked" we fall back to the
// CUDA IPC reduction kernel.
//
// BUILD: nvcc -arch=sm_90a -O3 -I kernels/ kernels/nvls_xproc_test.cu \
//         -lcuda -o /tmp/nvls_xproc_test
// RUN:   /tmp/nvls_xproc_test
//
// Uses exactly 2 GPUs (devices 0+1). Extend to 8 for full TP=8 validation.

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define DK(x) do { \
    CUresult _r=(x); \
    if(_r!=CUDA_SUCCESS){ const char*_e=nullptr; cuGetErrorString(_r,&_e); \
      printf("CU ERR %s:%d [%d] %s\n",__FILE__,__LINE__,(int)_r,_e?_e:"?"); exit(1); } \
} while(0)
#define CK(x) do { \
    cudaError_t _e=(x); \
    if(_e!=cudaSuccess){ printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } \
} while(0)

static const int NDEV = 2;
static const size_t NELEMS = 4096;  // fp32 floats = 16 KB (matches HIDDEN)
static const size_t NBYTES = NELEMS * sizeof(float);

// Send an open file descriptor over a unix socket using SCM_RIGHTS.
static void send_fd(int sock, int fd) {
    char buf[1] = {0};
    struct iovec iov = {buf, 1};
    char cmsg_buf[CMSG_SPACE(sizeof(int))];
    memset(cmsg_buf, 0, sizeof(cmsg_buf));
    struct msghdr msg = {};
    msg.msg_iov = &iov; msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf; msg.msg_controllen = sizeof(cmsg_buf);
    struct cmsghdr* cm = CMSG_FIRSTHDR(&msg);
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS;
    cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &fd, sizeof(int));
    if (sendmsg(sock, &msg, 0) < 0) { perror("sendmsg"); exit(1); }
}

static int recv_fd(int sock) {
    char buf[1];
    struct iovec iov = {buf, 1};
    char cmsg_buf[CMSG_SPACE(sizeof(int))];
    memset(cmsg_buf, 0, sizeof(cmsg_buf));
    struct msghdr msg = {};
    msg.msg_iov = &iov; msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf; msg.msg_controllen = sizeof(cmsg_buf);
    if (recvmsg(sock, &msg, 0) < 0) { perror("recvmsg"); exit(1); }
    struct cmsghdr* cm = CMSG_FIRSTHDR(&msg);
    if (!cm || cm->cmsg_type != SCM_RIGHTS) { printf("no FD in msg\n"); exit(1); }
    int fd; memcpy(&fd, CMSG_DATA(cm), sizeof(int)); return fd;
}

// Kernel: each device writes rank_id+1 to the multicast VA, then we verify reduce.
__global__ void write_mc(float* mc, float val, int n) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += gridDim.x*blockDim.x) mc[i] = val;
}

__global__ void nvls_reduce(float* mc, int n, int rank, int npes, float* out) {
    // disjoint-slice AR: each rank reduces its own slice
    int chunk = ((n/4)+npes-1)/npes*4;
    int lo = rank*chunk, hi = min(n, lo+chunk);
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = lo+tid*4; i < hi; i += gridDim.x*blockDim.x*4) {
        float a,b,c,d;
        asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3},[%4];"
            :"=f"(a),"=f"(b),"=f"(c),"=f"(d):"l"(mc+i):"memory");
        asm volatile("multimem.st.global.v4.f32 [%0],{%1,%2,%3,%4};"
            ::"l"(mc+i),"f"(a),"f"(b),"f"(c),"f"(d):"memory");
    }
    // copy result to out (for rank 0 to verify)
    if (rank == 0) {
        __threadfence_system();
        for (int i = tid; i < n; i += gridDim.x*blockDim.x) out[i] = mc[i];
    }
}

// Per-process setup: initialise CUDA on `dev`, import MC handle, bind mem, map MC VA.
struct RankCtx {
    CUdeviceptr mc_va;   // multicast VA (shared view)
    CUdeviceptr uc_va;   // unicast VA  (this rank's physical backing)
    float* out_host;     // host result buffer (rank 0 only)
    cudaStream_t s;
    int dev, rank;
};

static size_t align_up(size_t n, size_t g) { return ((n+g-1)/g)*g; }

static void setup_rank(int dev, int rank, CUmemGenericAllocationHandle mc_h,
                        size_t mc_gran, size_t mc_size, RankCtx& ctx) {
    ctx.dev = dev; ctx.rank = rank;
    CK(cudaSetDevice(dev));
    DK(cuCtxSetCurrent(nullptr));  // ensure we're in the right context
    CK(cudaStreamCreate(&ctx.s));

    // Each rank binds its own physical memory to the multicast object.
    CUmemAllocationProp p; memset(&p, 0, sizeof(p));
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = dev;
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t mg = 0; DK(cuMemGetAllocationGranularity(&mg, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    size_t ms = align_up(mc_size, mg);

    CUmemGenericAllocationHandle ph; DK(cuMemCreate(&ph, ms, &p, 0));
    DK(cuMulticastBindMem(mc_h, 0, ph, 0, mc_size, 0));

    // Unicast VA for this rank's physical backing.
    DK(cuMemAddressReserve(&ctx.uc_va, ms, 0, 0, 0));
    DK(cuMemMap(ctx.uc_va, ms, 0, ph, 0));
    CUmemAccessDesc ad; memset(&ad, 0, sizeof(ad));
    ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id = dev;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DK(cuMemSetAccess(ctx.uc_va, ms, &ad, 1));
    CK(cudaMemset((void*)ctx.uc_va, 0, mc_size));

    // Multicast VA (visible by all ranks simultaneously via NVSwitch).
    // Only ONE process maps this — we'll re-use the parent's mc_va passed via pipe.
    // (mc_va is set by parent and written to the shared pipe)
    ctx.mc_va = 0;  // will be set after parent creates it
    ctx.out_host = nullptr;
}

// ======================== PARENT (rank 0) ========================
static int run_parent(int sock, int dev0, size_t* out_mc_gran, size_t* out_mc_size,
                       CUmemGenericAllocationHandle* out_mc_h) {
    CK(cudaSetDevice(dev0)); DK(cuInit(0));

    // Query multicast granularity for NDEV devices.
    CUmulticastObjectProp mcp; memset(&mcp, 0, sizeof(mcp));
    mcp.numDevices = NDEV;
    mcp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    mcp.size = NBYTES;
    size_t g = 0;
    CUresult gr = cuMulticastGetGranularity(&g, &mcp, CU_MULTICAST_GRANULARITY_RECOMMENDED);
    if (gr != CUDA_SUCCESS) {
        const char* e=nullptr; cuGetErrorString(gr, &e);
        printf("FAIL: cuMulticastGetGranularity [%d] %s\n", (int)gr, e?e:"?");
        return 1;
    }
    size_t mc_size = align_up(NBYTES, g); mcp.size = mc_size;
    *out_mc_gran = g; *out_mc_size = mc_size;

    // Create multicast object.
    CUmemGenericAllocationHandle mc;
    CUresult cr = cuMulticastCreate(&mc, &mcp);
    if (cr != CUDA_SUCCESS) {
        const char* e=nullptr; cuGetErrorString(cr, &e);
        printf("FAIL: cuMulticastCreate [%d] %s\n", (int)cr, e?e:"?");
        return 1;
    }
    *out_mc_h = mc;

    // Try to export as POSIX FD — THE KEY TEST.
    int mc_fd = -1;
    CUresult er = cuMemExportToShareableHandle(&mc_fd, mc, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    if (er != CUDA_SUCCESS) {
        const char* e=nullptr; cuGetErrorString(er, &e);
        printf("FAIL: export blocked [%d] %s\n"
               "  (this means IMEX/fabric is required for cross-process multicast on this setup)\n",
               (int)er, e?e:"?");
        return 1;
    }
    printf("  export OK: mc_fd=%d\n", mc_fd);

    // Add dev0 to the multicast group before sending the FD.
    DK(cuMulticastAddDevice(mc, dev0));

    // Send the FD to the child process.
    send_fd(sock, mc_fd);
    printf("  parent: sent FD to child\n");

    // Wait for child to signal it has imported + added dev1.
    char ready; recv(sock, &ready, 1, 0);

    // Now add dev1 too (child called cuMulticastAddDevice for dev1 already).
    // Map the multicast VA with access for both devices.
    CUdeviceptr mc_va;
    DK(cuMemAddressReserve(&mc_va, mc_size, g, 0, 0));
    DK(cuMemMap(mc_va, mc_size, 0, mc, 0));
    CUmemAccessDesc ads[NDEV];
    for (int d = 0; d < NDEV; d++) {
        memset(&ads[d], 0, sizeof(ads[d]));
        ads[d].location.type = CU_MEM_LOCATION_TYPE_DEVICE; ads[d].location.id = d;
        ads[d].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    DK(cuMemSetAccess(mc_va, mc_size, ads, NDEV));

    // Send mc_va to child.
    send(sock, &mc_va, sizeof(mc_va), 0);
    printf("  parent: sent mc_va=0x%lx\n", (unsigned long)mc_va);

    // Wait for child's unicast VA so parent can bind its own physical mem.
    // (Parent sets up its physical mem and binds it)
    RankCtx rctx{}; setup_rank(dev0, 0, mc, g, mc_size, rctx);
    rctx.mc_va = mc_va;

    // Wait for child's ready signal.
    recv(sock, &ready, 1, 0);

    // Write rank+1 = 1.0 to our unicast VA.
    float v = 1.0f;
    write_mc<<<1,256,0,rctx.s>>>((float*)rctx.uc_va, v, (int)NELEMS);
    CK(cudaStreamSynchronize(rctx.s));

    // Signal child we wrote.
    send(sock, &ready, 1, 0);

    // Wait for child to write + signal AR done.
    recv(sock, &ready, 1, 0);

    // Read result from mc_va.
    float* h = new float[NELEMS];
    CK(cudaMemcpy(h, (void*)mc_va, NBYTES, cudaMemcpyDeviceToHost));
    float expected = 1.0f + 2.0f;  // rank0 writes 1.0, rank1 writes 2.0 → sum=3.0
    int ok = 1;
    for (size_t i = 0; i < NELEMS; i++) if (fabsf(h[i] - expected) > 0.01f) { ok=0; break; }
    if (ok) printf("PASS: cross-process NVLS all-reduce verified (expected=%.1f)\n", expected);
    else    printf("FAIL: reduction incorrect (got[0]=%.4f expected=%.1f)\n", h[0], expected);
    delete[] h;
    return ok ? 0 : 1;
}

// ======================== CHILD (rank 1) ========================
static int run_child(int sock, int dev1) {
    CK(cudaSetDevice(dev1)); DK(cuInit(0));

    // Receive the multicast FD from parent.
    int mc_fd = recv_fd(sock);
    printf("  child: received mc_fd=%d\n", mc_fd);

    // Import the multicast handle.
    CUmemGenericAllocationHandle mc;
    CUresult ir = cuMemImportFromShareableHandle(&mc, &mc_fd,
                                                  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (ir != CUDA_SUCCESS) {
        const char* e=nullptr; cuGetErrorString(ir, &e);
        printf("FAIL: import [%d] %s\n", (int)ir, e?e:"?");
        return 1;
    }
    printf("  child: imported OK\n");

    // Add dev1 to the multicast group.
    DK(cuMulticastAddDevice(mc, dev1));

    // Signal parent we're done adding our device.
    char ready = 1; send(sock, &ready, 1, 0);

    // Receive mc_va from parent.
    CUdeviceptr mc_va;
    recv(sock, &mc_va, sizeof(mc_va), 0);
    printf("  child: got mc_va=0x%lx\n", (unsigned long)mc_va);

    // Get mc size.
    CUmulticastObjectProp mcp; memset(&mcp, 0, sizeof(mcp));
    mcp.numDevices = NDEV; mcp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; mcp.size = NBYTES;
    size_t g=0; DK(cuMulticastGetGranularity(&g, &mcp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    size_t mc_size = align_up(NBYTES, g);

    // Bind child's physical memory.
    RankCtx rctx{}; setup_rank(dev1, 1, mc, g, mc_size, rctx);
    rctx.mc_va = mc_va;

    // Signal parent we're ready.
    send(sock, &ready, 1, 0);

    // Wait for parent to write.
    recv(sock, &ready, 1, 0);

    // Write rank+1 = 2.0 to our unicast VA.
    float v = 2.0f;
    write_mc<<<1,256,0,rctx.s>>>((float*)rctx.uc_va, v, (int)NELEMS);
    CK(cudaStreamSynchronize(rctx.s));

    // Run the NVLS all-reduce (disjoint-slice).
    nvls_reduce<<<1,256,0,rctx.s>>>((float*)mc_va, (int)NELEMS, 1, NDEV, nullptr);
    CK(cudaStreamSynchronize(rctx.s));

    // Signal parent AR done.
    send(sock, &ready, 1, 0);
    return 0;
}

int main() {
    // Check GPU count.
    int ndev=0; CK(cudaGetDeviceCount(&ndev));
    if (ndev < NDEV) { printf("Need %d GPUs, have %d\n", NDEV, ndev); return 1; }
    printf("== nvls_xproc_test: POSIX-FD cross-process multicast viability ==\n");
    printf("   Testing if cuMemExportToShareableHandle works for multicast WITHOUT IMEX\n\n");

    // Create a unix socket pair for communication.
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) < 0) { perror("socketpair"); return 1; }

    pid_t child = fork();
    if (child < 0) { perror("fork"); return 1; }

    if (child == 0) {
        // Child process: rank 1, GPU 1.
        close(fds[0]);
        int r = run_child(fds[1], 1);
        close(fds[1]);
        exit(r);
    } else {
        // Parent process: rank 0, GPU 0.
        close(fds[1]);
        size_t mc_gran=0, mc_size=0;
        CUmemGenericAllocationHandle mc_h;
        int r = run_parent(fds[0], 0, &mc_gran, &mc_size, &mc_h);
        close(fds[0]);
        int status=0; waitpid(child, &status, 0);
        if (WEXITSTATUS(status) != 0) r = 1;
        return r;
    }
}
