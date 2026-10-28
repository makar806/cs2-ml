import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mvp_core import Controller, Detection, Settings
from mvp_motion import MotionHistory, MotionPlan

SIZE = (1280, 720)
RECT = (0, 0, 1280, 720)


def action(dx=100, dy=-37, shoot=True, tracking=True):
    return dict(
        dx=dx,
        dy=dy,
        shoot=shoot,
        tracking=tracking,
        target=object(),
        error=(20, 0),
        reason="snap_fire",
    )


def begin(plan, a, now=1, source=0.98):
    plan.begin(a, SIZE, RECT, source, now, 0.1)


def test_first_acquisition_and_shot_remain_immediate():
    p = MotionPlan()
    begin(p, action(tracking=False))
    e = p.poll(1)
    assert (e["dx"], e["dy"], e["shoot"], e["done"]) == (100, -37, True, True)
    assert p.poll(1.004) is None


def test_tracking_distributed_into_short_steps_with_exact_total_and_one_shot():
    p = MotionPlan()
    begin(p, action())
    assert p.poll(1) is None
    events = [p.poll(1 + t) for t in (0.0041, 0.0082, 0.0123, 0.0164, 0.0205)]
    events = [e for e in events if e]
    assert len(events) == 5 and max(abs(e["dx"]) for e in events) < 30
    assert sum(e["dx"] for e in events) == 100 and sum(e["dy"] for e in events) == -37
    assert not any(e["shoot"] for e in events[:-1]) and events[-1]["shoot"]


def test_cancel_expired_missing_and_render_wait_cannot_emit_later():
    p = MotionPlan()
    begin(p, action())
    p.cancel()
    assert p.poll(1.01) is None
    begin(p, action())
    assert p.poll(1.081) is None
    for invalid in (dict(action(), target=None), dict(action(), reason="await_render_after_snap")):
        begin(p, action())
        begin(p, invalid, now=1.003)
        assert p.poll(1.02) is None


def test_replacement_uses_residual_not_old_remaining_and_does_not_starve_shot():
    p = MotionPlan()
    begin(p, action(dx=100, dy=0))
    first = p.poll(1.0081)
    assert first["dx"] == 45
    begin(p, action(dx=55, dy=0, shoot=False), now=1.010)
    e = p.poll(1.019)
    assert e["dx"] == 55 and e["shoot"] and e["done"]
    assert first["dx"] + e["dx"] == 100 and p.poll(1.03) is None


def test_aim_only_and_integer_rounding_do_not_add_drift():
    for dx, dy in [(1, -1), (0, 0), (-101, 53)]:
        p = MotionPlan()
        begin(p, action(dx=dx, dy=dy, shoot=False))
        events = [p.poll(1 + t) for t in (0.0041, 0.0082, 0.0123, 0.0164, 0.0205)]
        events = [e for e in events if e]
        assert sum(e["dx"] for e in events) == dx and sum(e["dy"] for e in events) == dy
        assert not any(e["shoot"] for e in events)


def test_delayed_scheduler_emits_one_residual_without_burst_or_repeated_fire():
    p = MotionPlan()
    begin(p, action())
    assert p.poll(1.04)["dx"] == 100
    assert p.poll(1.041) is None


def target(x):
    return Detection(0, 0.99, (x - 20, 350, x + 20, 450), (x, 360), 0.99)


def test_inflight_camera_motion_is_rebased_without_double_correction_or_false_velocity():
    c = Controller()
    history = MotionHistory()
    a = c.step([target(700)], *SIZE, 1, 1, enabled=True, foreground=True, fire=False)
    c.acknowledge(a["dx"], 0, *SIZE)
    history.record(1.001, a["dx"], 0, SIZE, c.settings)
    rendered = 640 + 480 * math.tan(math.atan2(60, 480) - math.radians(a["dx"] * 0.022))

    c.acknowledge(25, 0, *SIZE)
    history.record(1.03, 25, 0, SIZE, c.settings)
    image = [target(rendered)]
    rebased = history.rebase(image, 1.02, SIZE)
    b = c.step(rebased, *SIZE, 1.06, 1.02, enabled=True, foreground=True, fire=False)
    assert b["tracking"] and abs(b["velocity"][0]) < 1e-6
    assert abs(b["error"][0] + 480 * math.tan(math.radians(25 * 0.022))) < 0.2


def test_history_ignores_commands_before_source_and_clears_on_disarm():
    h = MotionHistory()
    s = Settings()
    d = target(640)
    h.record(1, 100, 0, SIZE, s)
    assert h.rebase([d], 1.01, SIZE)[0].head == d.head
    assert h.rebase([d], 0.99, SIZE)[0].head[0] < 640
    h.clear()
    assert h.rebase([d], 0.99, SIZE)[0].head == d.head
