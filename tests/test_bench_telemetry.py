import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inferutil.bench.telemetry import (
    GpuSample, NullTelemetry, FakeTelemetrySource, NvmlTelemetry,
)


def _s(gpu, t, sm, power, temp=60.0):
    return GpuSample(gpu_index=gpu, t_seconds=t, temp_c=temp, sm_util_pct=sm,
                     mem_util_pct=sm, power_w=power, sm_clock_mhz=1500.0,
                     mem_clock_mhz=2600.0, mem_used_bytes=40_000_000_000)


def test_null_source_unavailable_and_empty():
    n = NullTelemetry()
    n.start()
    assert n.available is False and n.stop() == []


def test_fake_source_returns_canned():
    samples = [_s(0, 0.0, 80.0, 600.0), _s(1, 0.0, 40.0, 500.0)]
    f = FakeTelemetrySource(samples)
    f.start()
    assert f.available is True and f.stop() == samples


def test_nvml_available_is_bool_without_gpu():
    # No GPU / no pynvml on the dev box: must not raise; available is False-ish bool.
    t = NvmlTelemetry()
    assert isinstance(t.available, bool)
    t.start()
    assert isinstance(t.stop(), list)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("ok ", k)
