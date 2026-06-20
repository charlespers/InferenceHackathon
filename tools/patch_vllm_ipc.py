"""
patch_vllm_ipc.py — Replace vLLM's tensor_model_parallel_all_reduce with a
direct CUDA IPC reduction kernel for B=1 decode.

No IMEX / no CAP_SYS_ADMIN required.  Uses cudaIpcGetMemHandle to share
buffers across the 8 TP workers, then ipc_ar_f32 kernel sums all 8 in one pass.

Usage (called inside each vLLM worker after torch.distributed.init):
    import tools.patch_vllm_ipc as ipc_patch
    ipc_patch.patch()
"""
from __future__ import annotations
import ctypes, os, subprocess, tempfile, torch
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_SO_PATH = "/tmp/ipc_ar.so"
_lib: ctypes.CDLL | None = None
_buf: torch.Tensor | None = None      # pinned symmetric buffer
_buf_ptr: int = 0
_rank  = -1
_world = -1
_ready = False
_original_ar = None

HIDDEN = 4096  # only patch tensors of this size (HIDDEN fp32 = 16 KB)

# ---------------------------------------------------------------------------
# Build the .so once (cached by mtime)
# ---------------------------------------------------------------------------
def _build_so() -> bool:
    src = str(REPO / "kernels" / "ipc_allreduce.cu")
    if not os.path.exists(src):
        print(f"[ipc_patch] kernel source not found: {src}")
        return False
    if os.path.exists(_SO_PATH):
        if os.path.getmtime(_SO_PATH) > os.path.getmtime(src):
            return True  # already up to date

    print(f"[ipc_patch] compiling {src} -> {_SO_PATH} ...", flush=True)
    r = subprocess.run([
        "nvcc", "-arch=sm_90a", "-O3", "--use_fast_math",
        "--shared", "-Xcompiler", "-fPIC",
        src, "-lcuda", "-lcudart", "-o", _SO_PATH
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ipc_patch] nvcc failed:\n{r.stderr}")
        return False
    print(f"[ipc_patch] compiled OK")
    return True


# ---------------------------------------------------------------------------
# Per-worker init
# ---------------------------------------------------------------------------
def _worker_init(rank: int, world: int):
    global _lib, _buf, _buf_ptr, _rank, _world, _ready

    if not _build_so():
        return

    try:
        _lib = ctypes.CDLL(_SO_PATH)
    except OSError as e:
        print(f"[ipc_patch] cannot load {_SO_PATH}: {e}")
        return

    # Allocate a pinned, IPC-exportable buffer on this GPU.
    # Size: HIDDEN fp32 = 16 KB — same as the TP all-reduce tensor for B=1.
    _buf = torch.zeros(HIDDEN, dtype=torch.float32, device=f"cuda:{rank}").pin_memory()
    # pin_memory() returns CPU pinned; we need GPU pinned for IPC.
    # Use cudaMalloc + register instead:
    _buf = torch.zeros(HIDDEN, dtype=torch.float32, device=f"cuda:{rank}")
    # cudaHostRegister is not needed — CUDA IPC works on any device allocation.
    _buf_ptr = _buf.data_ptr()

    _rank  = rank
    _world = world

    _lib.ipc_ar_init.restype  = ctypes.c_int
    _lib.ipc_ar_reduce.restype = ctypes.c_int

    ret = _lib.ipc_ar_init(
        ctypes.c_int(rank),
        ctypes.c_int(world),
        ctypes.c_void_p(_buf_ptr),
        ctypes.c_size_t(HIDDEN * 4))

    if ret != 0:
        print(f"[ipc_patch rank={rank}] ipc_ar_init failed")
        return

    _ready = True
    print(f"[ipc_patch rank={rank}] IPC all-reduce ACTIVE ✓  (fallback; NVLS needs IMEX)")


# ---------------------------------------------------------------------------
# Patched all-reduce
# ---------------------------------------------------------------------------
def _ipc_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    if not _ready or tensor.numel() != HIDDEN or tensor.dtype != torch.float32:
        return _original_ar(tensor)

    # Copy partial result into our symmetric IPC buffer.
    _buf.copy_(tensor)
    torch.cuda.synchronize()  # ensure write is visible before reduction

    # Barrier: all ranks must have written before any rank reads peers.
    # Simple host-side barrier via torch.distributed.barrier.
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()

    # Run the IPC reduction kernel (reads all 8 buffers, writes sum to _buf).
    stream = torch.cuda.current_stream().cuda_stream
    _lib.ipc_ar_reduce(
        ctypes.c_void_p(_buf_ptr),
        ctypes.c_int(HIDDEN),
        ctypes.c_void_p(stream))

    torch.cuda.synchronize()  # wait for kernel

    # Barrier: all ranks see the result before proceeding.
    if dist.is_initialized():
        dist.barrier()

    # Copy reduced result back into the original tensor.
    tensor.copy_(_buf)
    return tensor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def patch():
    """Call once per vLLM worker after torch.distributed is initialized."""
    global _original_ar

    rank  = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", "8"))

    _worker_init(rank, world)

    if not _ready:
        print(f"[ipc_patch rank={rank}] IPC patch inactive — using default all-reduce")
        return

    try:
        from vllm.distributed import parallel_state as ps
        _original_ar = ps.tensor_model_parallel_all_reduce
        ps.tensor_model_parallel_all_reduce = _ipc_all_reduce
        print(f"[ipc_patch rank={rank}] patched tensor_model_parallel_all_reduce ✓")
    except Exception as e:
        print(f"[ipc_patch rank={rank}] failed to patch: {e}")
