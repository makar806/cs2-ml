"""Explicit capture selection. DXGI never silently falls back to slow MSS."""

from mvp_windows import USER, Capture, client_rect, foreground


class DesktopCapture:
    def __init__(self, hwnd, hz):
        self.hwnd = hwnd
        self.inner = Capture(hz, "dxcam")
        self.warm_region = (
            0,
            0,
            min(1280, USER.GetSystemMetrics(0)),
            min(960, USER.GetSystemMetrics(1)),
        )
        self.inner.set_region(self.warm_region)

    def bind(self, hwnd):
        self.hwnd = hwnd

    def warm_frame(self):
        if foreground(self.hwnd):
            rect = client_rect(self.hwnd)
            if rect is not None:
                self.inner.set_region(rect)
        return self.inner.get()

    @property
    def last_ms(self):
        return self.inner.last_ms

    @property
    def error(self):
        return self.inner.error

    def get(self):
        if not foreground(self.hwnd):
            return None
        rect = client_rect(self.hwnd)
        self.inner.set_region(rect)
        frame = self.inner.get()
        return frame if frame is not None and frame[3] == rect else None

    def close(self):
        self.inner.close()


def create_capture(hwnd, hz, backend):
    if backend == "dxgi":
        return DesktopCapture(hwnd, hz)
    if backend == "wgc":
        from mvp_wgc import WindowCapture

        return WindowCapture(hwnd, hz)
    raise ValueError(f"Unknown capture backend: {backend}")
