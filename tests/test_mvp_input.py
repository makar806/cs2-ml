"""Verify event ordering/release using a fake SendInput; never inject into Windows."""

import os
import sys
from pathlib import Path

import pytest

if os.name != "nt":
    pytest.skip("Windows input structures", allow_module_level=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mvp_windows as win


def test_move_and_button_down_are_one_ordered_batch(monkeypatch):
    calls = []
    released = []

    def send(count, events, size):
        calls.append(
            [(events[i].mi.dx, events[i].mi.dy, events[i].mi.dwFlags) for i in range(count)]
        )
        return count

    monkeypatch.setattr(win.USER, "SendInput", send)
    monkeypatch.setattr(win, "mouse_event", lambda **kw: released.append(kw["flags"]))
    mouse = win.Mouse()
    mouse.act(300, -100, True)
    assert calls == [[(300, -100, 1), (0, 0, 2)]] and mouse.down
    assert mouse.last_call["kind"] == "move+down" and mouse.last_call["api_ms"] >= 0
    mouse.release_at = 0
    mouse.tick()
    assert released == [4] and not mouse.down
    assert mouse.timings[-1]["kind"] == "up"


def test_partial_batch_failure_can_always_release_button_state(monkeypatch):
    monkeypatch.setattr(win.USER, "SendInput", lambda count, events, size: 1)
    released = []
    monkeypatch.setattr(win, "mouse_event", lambda **kw: released.append(kw["flags"]))
    mouse = win.Mouse()
    with pytest.raises(OSError, match="SendInput=1/2"):
        mouse.act(100, 10, True)
    mouse.release()
    assert released == [4] and not mouse.down
