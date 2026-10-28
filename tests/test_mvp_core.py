import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mvp_core import Controller, Detection, Settings


def target(cls=0, x=640, y=360, score=0.95, head_score=0.99):
    return Detection(cls, score, (x - 20, y - 10, x + 20, y + 100), (x, y), head_score)


def call(controller, detections, now=1, age=0, enabled=True, focused=True, fire=True):
    return controller.step(
        detections, 1280, 720, now, now - age, enabled=enabled, foreground=focused, fire=fire
    )


def test_invalid_or_wrong_team_never_fires():
    for d in [target(cls=1), target(head_score=0.1), target(x=float("nan")), target(score=0.1)]:
        a = call(Controller(), [d])
        assert a["target"] is None and not a["shoot"] and a["dx"] == a["dy"] == 0


def test_focus_stale_disabled_and_future_fail_closed():
    for kwargs in [dict(enabled=False), dict(focused=False), dict(age=0.2), dict(age=-0.01)]:
        a = call(Controller(), [target()], **kwargs)
        assert a["target"] is None and not a["shoot"]


def test_first_frame_full_correction_and_fire_without_alignment_wait():
    c = Controller()
    a = call(c, [target(x=820, y=420)])
    expected = round(math.degrees(math.atan2(180, 480)) / 0.022)
    assert a["dx"] == expected and a["dy"] > 0 and a["shoot"]
    assert a["reason"] == "snap_fire"


def test_immediate_center_shot_cooldown_and_aim_only():
    c = Controller()
    assert call(c, [target()])["shoot"]
    assert not call(c, [target()], now=1.1)["shoot"]
    assert not call(c, [target()], now=1.3)["shoot"]
    assert call(c, [target()], now=1.41)["shoot"]
    assert not call(Controller(), [target(x=740)], fire=False)["shoot"]


def test_clamped_correction_never_fires_and_outside_radius_is_ignored():
    a = call(Controller(Settings(max_step=5)), [target(x=820)])
    assert a["dx"] == 5 and not a["shoot"]
    assert call(Controller(Settings(radius=260)), [target(x=1000)])["target"] is None


def test_distant_point_corrected_immediately_rather_than_waiting_to_align():
    c = Controller()
    d = Detection(0, 0.99, (638, 350, 660, 390), (644, 360), 0.99)
    a = call(c, [d])
    assert a["shoot"] and a["dx"] > 0
    remaining = 4 - math.tan(math.radians(a["dx"] * 0.022)) * 480
    assert abs(remaining) < 0.2


def test_moving_target_prediction_compensates_actual_frame_age():
    c = Controller()
    call(c, [target(x=680)], now=1, fire=False)

    a = call(c, [target(x=690)], now=1.07, age=0.02, fire=False)
    assert abs(a["velocity"][0] - 200) < 1e-6
    assert abs(a["target"].head[0] - 694) < 1e-6


def test_acknowledged_camera_motion_is_not_mistaken_for_target_velocity():
    c = Controller()
    a = call(c, [target(x=690)], now=1, fire=False)
    c.acknowledge(a["dx"], a["dy"], 1280, 720)

    rendered = 640 + 480 * math.tan(math.atan2(50, 480) - math.radians(a["dx"] * 0.022))
    b = call(c, [target(x=rendered)], now=1.05, fire=False)
    assert abs(b["velocity"][0]) < 1e-5


def test_first_shot_is_immediate_but_unchanged_render_does_not_repeat_flick():
    c = Controller()
    a = call(c, [target(x=820, y=420)])
    assert a["shoot"] and a["dx"] > 0
    c.acknowledge(a["dx"], a["dy"], 1280, 720)
    b = call(c, [target(x=820, y=420)], now=1.04)
    assert not b["shoot"] and b["dx"] == b["dy"] == 0


def test_new_closer_enemy_does_not_steal_lock_even_if_detection_order_changes():
    c = Controller()
    call(c, [target(x=700)], fire=False)
    for index in range(1, 8):
        locked = target(x=700 + index)
        a = call(c, [target(x=640), locked], now=1 + index * 0.02, fire=False)
        assert a["target"].box == locked.box


def test_missing_locked_head_pauses_instead_of_switching_then_reacquires():
    c = Controller()
    call(c, [target(x=740)], fire=False)
    a = call(c, [target(x=640)], now=1.1)
    assert a["target"] is None and not a["shoot"] and a["dx"] == 0
    b = call(c, [target(x=640), target(x=744)], now=1.2, fire=False)
    assert b["target"].box == target(x=744).box
    assert call(c, [target(x=640)], now=1.51)["target"] is None
    c.reset()
    assert call(c, [target(x=640)], now=1.55)["target"].box == target(x=640).box


def test_freeze_does_not_release_old_identity_and_acquire_new_neighbor():
    c = Controller()
    call(c, [target(x=740)], fire=False)
    assert call(c, [target(x=640)], now=1.5, age=0.3)["target"] is None
    assert call(c, [target(x=640)], now=2)["target"] is None
    assert call(c, [target(x=744)], now=2.04, fire=False)["target"] is not None


def test_larger_closer_body_cannot_replace_held_target():
    c = Controller()
    call(c, [target()], fire=False)
    front = Detection(0, 0.99, (600, 340, 680, 540), (640, 360), 0.99)
    assert call(c, [front], now=1.1, fire=False)["target"] is None


def test_ambiguous_overlap_pauses_instead_of_switching():
    c = Controller()
    call(c, [target()], fire=False)
    a = call(c, [target(x=638), target(x=642)], now=1.1)
    assert a["target"] is None and not a["shoot"]


def test_camera_shift_keeps_original_target_when_another_is_now_closer():
    c = Controller()
    a = call(c, [target(x=800)], fire=False)
    c.acknowledge(a["dx"], a["dy"], 1280, 720)
    shift = c.camera_delta[0]
    tracked = target(x=800 - shift + 8)
    b = call(c, [target(x=620), tracked], now=1.04, fire=False)
    assert b["target"].box == tracked.box


def test_lock_survives_outside_acquisition_radius_and_explicit_reset_releases_it():
    c = Controller()
    call(c, [target(x=890)], fire=False)
    assert call(c, [target(x=905)], now=1.04, fire=False)["target"] is not None
    c.reset()
    assert (
        call(Controller(Settings(radius=260)), [target(x=905)], now=1.08, fire=False)["target"]
        is None
    )


def test_default_acquisition_covers_left_right_top_and_bottom_of_frame():
    for x, y in [(30, 360), (1250, 360), (640, 12), (640, 610), (30, 12), (1250, 610)]:
        a = call(Controller(), [target(x=x, y=y)], fire=False)
        assert a["target"] is not None and (a["dx"] or a["dy"])
        assert a["reason"] != "limited_step"


def test_full_frame_acquisition_does_not_break_existing_target_lock():
    c = Controller()
    call(c, [target(x=1100)], fire=False)
    a = call(c, [target(x=640), target(x=1102)], now=1.04, fire=False)
    assert a["target"].head[0] == 1102


def test_duplicate_t_ct_box_does_not_pause_tracking_of_one_body():
    c = Controller()
    call(c, [target()], fire=False)
    a = call(c, [target(), target(cls=1, x=641)], now=1.1, fire=False)
    assert a["target"] is not None and a["target"].cls == 0


def test_confirmed_kill_releases_identity_without_aim_cooldown():
    c = Controller()
    call(c, [target()], fire=False)
    assert call(c, [target(x=1000)], now=1.1, fire=False)["target"] is None
    assert c.release_after_confirmed_kill(1.11)
    a = call(c, [target(), target(x=1000)], now=1.12, fire=False)
    assert a["target"].head[0] == 1000 and not a["tracking"] and a["dx"] > 0


def test_kill_release_preserves_weapon_cooldown_across_new_target():
    c = Controller()
    assert call(c, [target()])["shoot"]
    c.release_after_confirmed_kill(1.1)
    a = call(c, [target(x=1000)], now=1.11)
    assert a["target"] is not None and a["dx"] > 0 and not a["shoot"]


def test_default_shot_spacing_remains_slow_across_target_loss():
    c = Controller()
    assert call(c, [target()])["shoot"]
    call(c, [], now=1.05)
    assert not call(c, [target()], now=1.2)["shoot"]
    assert call(c, [target()], now=1.41)["shoot"]
