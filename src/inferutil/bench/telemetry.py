from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GpuSample:
    gpu_index: int
    t_seconds: float
    temp_c: float
    sm_util_pct: float
    mem_util_pct: float
    power_w: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    mem_used_bytes: int


class TelemetrySource(Protocol):
    available: bool
    def start(self) -> None: ...
    def stop(self) -> list: ...     # list[GpuSample]


class NullTelemetry:
    """Used when no NVML is present (e.g. laptop). Reports nothing."""
    available = False
    def start(self) -> None: pass
    def stop(self) -> list: return []


class FakeTelemetrySource:
    """Canned samples for tests."""
    def __init__(self, samples):
        self._samples = list(samples)
        self.available = True
    def start(self) -> None: pass
    def stop(self) -> list: return self._samples


class NvmlTelemetry:
    """Background NVML sampler. pynvml is imported lazily; if it (or the driver)
    is unavailable, `available` is False and start/stop are no-ops."""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            self._p = pynvml
            self._n = pynvml.nvmlDeviceGetCount()
            self.available = True
        except Exception:
            self._p = None
            self._n = 0
            self.available = False

    def start(self) -> None:
        if not self.available:
            return
        self._stop.clear()
        self._samples = []
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        p = self._p
        handles = [p.nvmlDeviceGetHandleByIndex(i) for i in range(self._n)]
        while not self._stop.is_set():
            t = time.monotonic() - self._t0
            for i, h in enumerate(handles):
                u = p.nvmlDeviceGetUtilizationRates(h)
                self._samples.append(GpuSample(
                    gpu_index=i, t_seconds=t,
                    temp_c=float(p.nvmlDeviceGetTemperature(h, p.NVML_TEMPERATURE_GPU)),
                    sm_util_pct=float(u.gpu), mem_util_pct=float(u.memory),
                    power_w=p.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    sm_clock_mhz=float(p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_SM)),
                    mem_clock_mhz=float(p.nvmlDeviceGetClockInfo(h, p.NVML_CLOCK_MEM)),
                    mem_used_bytes=int(p.nvmlDeviceGetMemoryInfo(h).used)))
            self._stop.wait(self.interval_s)

    def stop(self) -> list:
        if not self.available:
            return []
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return list(self._samples)
