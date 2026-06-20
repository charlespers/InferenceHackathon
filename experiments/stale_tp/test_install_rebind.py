"""Offline integration test for stale_tp.install() — the ONE path the scheduler
unit tests don't cover. Simulates vLLM's module layout in sys.modules (fake torch,
fake vllm.distributed.communication_op, fake consumer module that imported the
symbol by name like RowParallelLinear does) and asserts:
  1. install() rebinds tensor_model_parallel_all_reduce in BOTH the defining module
     and every consumer module that held a reference to the original.
  2. the installed wrapper routes through the scheduler (real on refresh layers,
     stale clone on non-refresh) — i.e. the end-to-end hook works without a GPU.

Run: python experiments/stale_tp/test_install_rebind.py   (no torch / no vLLM / no GPU)
"""
import sys
import os
import types

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)


def _install_fakes():
    """Inject fake torch + vllm module tree so stale_tp.install() runs offline."""
    # fake torch (install just `import torch`)
    sys.modules.setdefault("torch", types.ModuleType("torch"))

    # the original all-reduce: returns a uniquely-stamped fake tensor per call,
    # ignoring its input (we only care about routing/caching identity).
    state = {"n": 0}

    class FakeTensor:
        def __init__(self, tag):
            self.tag = tag
            self.shape = (1, 4096)   # B=1 decode: first dim 1

        def dim(self):
            return 2

        def clone(self):
            return FakeTensor(self.tag)  # preserve tag so we can assert reuse

    def orig(input_):
        t = FakeTensor(f"R{state['n']}")
        state["n"] += 1
        return t

    # vllm.distributed.communication_op holds the canonical symbol
    vllm = types.ModuleType("vllm")
    dist = types.ModuleType("vllm.distributed")
    cop = types.ModuleType("vllm.distributed.communication_op")
    cop.tensor_model_parallel_all_reduce = orig
    dist.communication_op = cop
    vllm.distributed = dist
    sys.modules["vllm"] = vllm
    sys.modules["vllm.distributed"] = dist
    sys.modules["vllm.distributed.communication_op"] = cop

    # a CONSUMER module that imported the symbol by name (like linear.py)
    consumer = types.ModuleType("vllm.model_executor.layers.linear")
    consumer.tensor_model_parallel_all_reduce = orig
    sys.modules["vllm.model_executor.layers.linear"] = consumer

    return orig, cop, consumer, FakeTensor


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    print(f"  ok: {name}")


def main():
    orig, cop, consumer, FakeTensor = _install_fakes()

    import stale_tp

    # configure the scheduler for a tiny 2-layer pass, layer/proxy K=2, enabled
    s = stale_tp.get_scheduler()
    s.enable = True
    s.K = 2
    s.mode = "layer"
    s.policy = "proxy"
    s.decode_only = True
    s.period = 4              # 2 layers x 2 slots
    s.cpl = 2
    s.ctl_path = ""           # no file IO
    s._call = 0
    s._last_by_slot.clear()
    s.stats = {k: 0 for k in s.stats}

    n = stale_tp.install()
    check("install rebound >=2 modules (defining + consumer)", n >= 2)
    check("communication_op symbol replaced", cop.tensor_model_parallel_all_reduce is not orig)
    check("consumer symbol replaced", consumer.tensor_model_parallel_all_reduce is not orig)
    wrapper = cop.tensor_model_parallel_all_reduce
    check("wrapper is marked", getattr(wrapper, "_stale_tp_wrapped", False) is True)
    check("idempotent: second install is a no-op", stale_tp.install() == 0)

    # drive one decode pass through the wrapper; input is the local partial.
    tags = []
    for i in range(4):
        out = wrapper(FakeTensor(f"L{i}"))
        tags.append(out.tag)
    # layer0 (idx0,1) refresh -> real R0,R1 ; layer1 (idx2,3) non-refresh proxy ->
    # reuse cached per-slot real values -> R0,R1
    check("wrapper routes real-then-stale per slot", tags == ["R0", "R1", "R0", "R1"])

    snap = s.snapshot()
    check("scheduler saw 2 real + 2 stale", snap[stale_tp.REAL] == 2 and snap[stale_tp.STALE] == 2)

    print("\nALL INSTALL-REBIND CHECKS PASSED (offline, no GPU)")


if __name__ == "__main__":
    main()
