import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mvp_capture_session import CaptureSession


class FakeCapture:
    def __init__(self):
        self.closes = 0
        self.bindings = []

    def bind(self, hwnd):
        self.bindings.append(hwnd)

    def close(self):
        self.closes += 1


def test_desktop_capture_created_once_across_unfocused_and_f8_idle_periods():
    made = []

    def factory(hwnd):
        c = FakeCapture()
        made.append(c)
        return c

    s = CaptureSession(factory)
    for hwnd in (None, 12, 12, None, 12, 12, None, 12):
        s.bind(hwnd)
        s.ensure()
    assert s.creations == 1 and len(made) == 1 and made[0].closes == 0
    s.close()
    assert made[0].closes == 1


def test_ready_requires_real_capture_frame_and_ten_second_preparation():
    now = [0.0]
    s = CaptureSession(lambda _: FakeCapture(), clock=lambda: now[0])
    s.ensure()
    assert not s.ready
    s.note_frame()
    now[0] = 9.9
    assert not s.ready
    now[0] = 10.1
    assert s.ready


def test_window_capture_recreated_only_for_a_different_target_window():
    made = []

    def factory(hwnd):
        c = FakeCapture()
        made.append(c)
        return c

    s = CaptureSession(factory, window_bound=True)
    assert s.ensure() is None
    s.bind(12)
    s.ensure()
    s.bind(12)
    s.ensure()
    assert s.creations == 1 and not made[0].closes
    s.bind(13)
    s.ensure()
    assert s.creations == 2 and made[0].closes == 1
