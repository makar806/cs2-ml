"""Read-only, low-rate memory/page-in telemetry. No OS configuration changes."""

import ctypes as C
import time

import psutil


class MemorySampler:
    def __init__(self, worker_pid):
        self.process = psutil.Process()
        self.worker_pid = worker_pid
        self.last = 0
        self.query = C.c_void_p()
        self.counter = C.c_void_p()
        self.pdh = None
        try:
            self.pdh = C.WinDLL("pdh")
            self.pdh.PdhOpenQueryW.argtypes = [C.c_wchar_p, C.c_size_t, C.POINTER(C.c_void_p)]
            self.pdh.PdhAddEnglishCounterW.argtypes = [
                C.c_void_p,
                C.c_wchar_p,
                C.c_size_t,
                C.POINTER(C.c_void_p),
            ]
            self.pdh.PdhCollectQueryData.argtypes = [C.c_void_p]
            self.pdh.PdhCloseQuery.argtypes = [C.c_void_p]
            if self.pdh.PdhOpenQueryW(None, 0, C.byref(self.query)):
                raise RuntimeError("PDH open")
            if self.pdh.PdhAddEnglishCounterW(
                self.query, r"\Memory\Pages Input/sec", 0, C.byref(self.counter)
            ):
                raise RuntimeError("PDH add")
            self.pdh.PdhCollectQueryData(self.query)
        except Exception:
            self.close()

    def sample(self):
        memory = psutil.virtual_memory()
        cores = psutil.cpu_percent(percpu=True)
        result = dict(
            ram_percent=memory.percent,
            available_mb=memory.available / 1048576,
            cpu_peak_core_percent=max(cores, default=0),
        )
        for name, pid in [("panel", self.process.pid), ("worker", self.worker_pid)]:
            try:
                m = psutil.Process(pid).memory_info()
                result[name + "_rss_mb"] = m.rss / 1048576
                result[name + "_private_mb"] = getattr(m, "private", m.vms) / 1048576
            except psutil.Error:
                pass
        if self.pdh and self.query:

            class Value(C.Structure):
                _fields_ = [("status", C.c_ulong), ("value", C.c_double)]

            value = Value()
            self.pdh.PdhGetFormattedCounterValue.argtypes = [
                C.c_void_p,
                C.c_ulong,
                C.c_void_p,
                C.POINTER(Value),
            ]
            if (
                not self.pdh.PdhCollectQueryData(self.query)
                and not self.pdh.PdhGetFormattedCounterValue(
                    self.counter, 0x200, None, C.byref(value)
                )
                and value.status in (0, 1)
            ):
                result["pages_input_per_sec"] = value.value
        return result

    def close(self):
        if self.pdh and self.query:
            self.pdh.PdhCloseQuery(self.query)
        self.query = C.c_void_p()
        self.pdh = None


def memory_monitor(stop, state, output, worker_pid):
    import json

    sampler = MemorySampler(worker_pid)
    try:
        with (output / "memory.jsonl").open("w", encoding="utf-8") as log:
            while not stop.is_set():
                sample = sampler.sample()
                state.update(sample)
                log.write(json.dumps(dict(time=time.time(), **sample)) + "\n")
                log.flush()
                stop.wait(1)
    finally:
        sampler.close()
