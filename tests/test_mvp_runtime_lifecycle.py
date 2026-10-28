"""Actual worker loop with fake capture/model/window APIs. No Windows input."""

import sys
import threading
import time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvp_capture
import mvp_runtime as runtime
import mvp_windows as win
import psutil


class LocalQueue(Queue):
    def cancel_join_thread(self):
        pass


class Capture:
    def __init__(self):
        self.error = ""
        self.last_ms = 0.0
        self.closes = 0
        self.counter = 0

    def bind(self, hwnd):
        pass

    def warm_frame(self):
        self.counter += 1
        return self.counter, time.perf_counter(), np.zeros((32, 32, 3), np.uint8), (0, 0, 32, 32)

    def close(self):
        self.closes += 1


def test_real_worker_keeps_one_capture_and_pauses_prediction_across_f8_and_focus(
    monkeypatch, tmp_path
):
    made = []
    calls = []
    focused = [False]

    def factory(*args):
        c = Capture()
        made.append(c)
        return c

    monkeypatch.setattr(mvp_capture, "create_capture", factory)
    monkeypatch.setattr(win, "find_window", lambda *a, **kw: 12)
    monkeypatch.setattr(win, "foreground", lambda _: focused[0])
    monkeypatch.setattr(win, "client_rect", lambda _: (0, 0, 32, 32))
    monkeypatch.setattr(win.USER, "IsWindow", lambda _: True)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=2 * 1024**3))

    def predict(image):
        calls.append(time.perf_counter())
        return []

    monkeypatch.setattr(
        runtime, "load_model", lambda *a: (SimpleNamespace(predict=predict), "Native fake")
    )
    stop = threading.Event()
    active = threading.Event()
    packets = LocalQueue(1)
    messages = LocalQueue(4)
    config = dict(
        mode="local-bots",
        pipeline="full",
        model_backend="native",
        capture_backend="dxgi",
        imgsz=960,
        hz=1000,
        capture_warmup_seconds=0,
        output=str(tmp_path),
    )
    thread = threading.Thread(
        target=runtime.inference_process, args=(config, stop, packets, messages, active)
    )
    thread.start()
    try:
        deadline = time.perf_counter() + 2
        ready = False
        while time.perf_counter() < deadline:
            message = messages.get(timeout=1)
            if message.get("ready"):
                ready = True
                break
        assert ready and len(calls) == 3
        for _ in range(2):
            focused[0] = True
            active.set()
            packets.get(timeout=1)
            active.clear()
            focused[0] = False
            time.sleep(0.02)
            count = len(calls)
            time.sleep(0.02)
            assert len(calls) == count and made[0].closes == 0
        assert len(made) == 1
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not thread.is_alive() and made[0].closes == 1
    assert not (tmp_path / "error.txt").exists()


def test_worker_waits_without_creating_desktop_capture_when_game_is_closed(monkeypatch, tmp_path):
    made = []
    monkeypatch.setattr(mvp_capture, "create_capture", lambda *args: made.append(args))
    monkeypatch.setattr(win, "find_window", lambda *a, **kw: None)
    monkeypatch.setattr(
        runtime, "load_model", lambda *a: (SimpleNamespace(predict=lambda image: []), "Native fake")
    )
    stop = threading.Event()
    active = threading.Event()
    packets = LocalQueue(1)
    messages = LocalQueue(4)
    config = dict(
        mode="local-bots",
        pipeline="full",
        model_backend="native",
        capture_backend="dxgi",
        imgsz=960,
        hz=30,
        capture_warmup_seconds=0,
        output=str(tmp_path),
    )
    thread = threading.Thread(
        target=runtime.inference_process, args=(config, stop, packets, messages, active)
    )
    thread.start()
    try:
        deadline = time.perf_counter() + 2
        waiting = False
        while time.perf_counter() < deadline:
            message = messages.get(timeout=1)
            if message.get("status") == "Ожидание окна CS2":
                assert not message["ready"]
                waiting = True
                break
        assert waiting and not made and packets.empty()
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not thread.is_alive() and not (tmp_path / "error.txt").exists()
