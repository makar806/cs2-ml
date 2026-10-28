"""Capture lifetime is independent of F8/foreground; only Stop releases it."""

import time


class CaptureSession:
    def __init__(self, factory, window_bound=False, warmup=10, clock=time.perf_counter):
        self.factory = factory
        self.window_bound = window_bound
        self.warmup = warmup
        self.clock = clock
        self.capture = None
        self.hwnd = None
        self.created_at = None
        self.frames = 0
        self.creations = 0

    def bind(self, hwnd):
        if hwnd == self.hwnd:
            return
        if self.window_bound and self.capture is not None:
            self.close()
        self.hwnd = hwnd
        if self.capture is not None and hasattr(self.capture, "bind"):
            self.capture.bind(hwnd)

    def ensure(self):
        if self.capture is None and (self.hwnd is not None or not self.window_bound):
            self.capture = self.factory(self.hwnd)
            self.created_at = self.clock()
            self.frames = 0
            self.creations += 1
        return self.capture

    def note_frame(self):
        self.frames += 1

    @property
    def ready(self):
        return (
            self.capture is not None
            and self.frames > 0
            and self.clock() - self.created_at >= self.warmup
        )

    @property
    def remaining(self):
        return (
            self.warmup
            if self.created_at is None
            else max(0, self.warmup - (self.clock() - self.created_at))
        )

    def close(self):
        if self.capture is not None:
            self.capture.close()
        self.capture = None
        self.created_at = None
        self.frames = 0
