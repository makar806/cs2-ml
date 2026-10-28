"""Actual engine scheduler wiring with fake mouse; never inject into Windows."""

import io
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_mvp as live
from mvp_core import Controller, Settings
from mvp_motion import MotionHistory, MotionPlan


class Mouse:
    def __init__(self):
        self.events = []
        self.released = False

    def act(self, dx, dy, shoot):
        self.events.append((dx, dy, shoot))

    def release(self):
        self.released = True


def engine(monkeypatch, tmp_path):
    e = live.Engine.__new__(live.Engine)
    e.lock = threading.RLock()
    e.stop = threading.Event()
    e.armed = True
    e.hwnd = 123
    e.settings = Settings()
    e.controller = Controller(e.settings)
    e.mouse = Mouse()
    e.motion = MotionPlan()
    e.history = MotionHistory()
    e.last_snap_at = 0
    e.last_motion_at = 0
    e.state = dict(moves=0, shots=0, input_error="")
    e.output = tmp_path
    monkeypatch.setattr(live, "foreground", lambda _: True)
    monkeypatch.setattr(live, "client_rect", lambda _: (0, 0, 1280, 720))
    return e


def schedule(e):
    now = live.time.perf_counter()
    a = dict(
        dx=10, dy=0, shoot=True, tracking=False, target=object(), reason="snap_fire", error=(2, 0)
    )
    e.motion.begin(a, (1280, 720), (0, 0, 1280, 720), now, now, 0.1)


@pytest.mark.parametrize("condition", ["focus", "stopped", "disarmed", "geometry"])
def test_engine_cancels_pending_before_any_mouse_input(monkeypatch, tmp_path, condition):
    e = engine(monkeypatch, tmp_path)
    schedule(e)
    if condition == "focus":
        monkeypatch.setattr(live, "foreground", lambda _: False)
    if condition == "stopped":
        e.stop.set()
    if condition == "disarmed":
        e.disarm()
    if condition == "geometry":
        monkeypatch.setattr(live, "client_rect", lambda _: (0, 0, 1920, 1080))
    e.send_pending(io.StringIO(), 1)
    assert not e.mouse.events and e.motion.pending is None


def test_emission_acknowledges_only_sent_counts_and_actual_shot_time(monkeypatch, tmp_path):
    e = engine(monkeypatch, tmp_path)
    schedule(e)
    log = io.StringIO()
    e.send_pending(log, 1)
    assert e.mouse.events == [(10, 0, True)]
    assert e.state["shots"] == 1 and e.state["moves"] == 1
    assert len(e.history.events) == 1 and e.controller.camera_delta[0] > 0
    assert e.controller.last_shot == e.last_motion_at


def test_rejected_input_cancels_plan_releases_and_records_error(monkeypatch, tmp_path):
    e = engine(monkeypatch, tmp_path)
    schedule(e)

    def reject(*args):
        raise OSError("fake input rejection")

    e.mouse.act = reject
    e.send_pending(io.StringIO(), 1)
    assert not e.armed and e.mouse.released and e.motion.pending is None
    assert e.state["shots"] == e.state["moves"] == 0
    assert (tmp_path / "input-error.txt").exists()
