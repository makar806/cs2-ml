import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_recorded_aim import run
from mvp_core import Detection
from mvp_recorded_aim import RecordedController, RecordedMotionPlan, minimum_jerk

PROFILE = dict(duration_seconds=0.2492, reference_angle_degrees=34.459)


def target(x=640, cls=0, confidence=0.99):
    return Detection(cls, 0.99, (x - 20, 350, x + 20, 450), (x, 360), confidence)


def step(c, source, *, now=None, x=640, fire=True, detections=None):
    return c.step(
        [target(x)] if detections is None else detections,
        1280,
        720,
        source if now is None else now,
        source,
        enabled=True,
        foreground=True,
        fire=fire,
    )


def test_first_off_center_observation_never_includes_a_shot():
    c = RecordedController(profile=PROFILE)
    action = step(c, 1, x=950)
    assert not action["shoot"] and action["recorded"]
    plan = RecordedMotionPlan()
    plan.begin(action, (1280, 720), (0, 0, 1280, 720), 1, 1, 0.1)
    assert plan.poll(1) is None
    first = plan.poll(1.004)
    assert first is None or abs(first["dx"]) <= 3


def test_shot_requires_distinct_observations_and_is_committed_only_on_emission():
    c = RecordedController(profile=PROFILE)
    assert not step(c, 1)["shoot"]
    assert not step(c, 1)["shoot"]
    assert not step(c, 1.033)["shoot"]
    action = step(c, 1.066)
    assert action["shoot"] and action["dx"] == action["dy"] == 0
    assert not c.shot_sent and c.can_emit_shot(1.066, 1.07)
    c.shot_emitted(1.07)
    assert c.shot_sent and not c.can_emit_shot(1.066, 1.071)
    for index in range(1, 100):
        assert not step(c, 1.07 + index * 0.033)["shoot"]


def test_large_input_requires_new_images_and_uncertain_heads_do_not_fire():
    c = RecordedController(profile=PROFILE)
    step(c, 1)
    c.movement_sent(1.01)
    assert not step(c, 1.02, now=1.04)["shoot"]
    assert not step(c, 1.04, detections=[target(confidence=0.6)])["shoot"]
    assert not step(c, 1.05, now=1.14)["shoot"]
    assert not c.can_emit_shot(1.05, 1.14)


def test_missing_observation_cancels_verification_and_preserves_identity():
    c = RecordedController(profile=PROFILE)
    step(c, 1)
    assert step(c, 1.033, detections=[])["target"] is None
    assert c.locked is not None
    assert not step(c, 1.066)["shoot"]
    assert step(c, 1.099)["shoot"]


def test_fire_switch_does_not_consume_budget_or_copy_recorded_clicks():
    c = RecordedController(profile=PROFILE)
    for source in (1, 1.033, 1.066):
        assert not step(c, source, fire=False)["shoot"]
    assert not c.shot_sent
    action = step(c, 1.1)
    assert action["shoot"]
    c.shot_emitted(1.1)
    for source, fire in ((1.6, False), (1.633, True)):
        assert not step(c, source, fire=fire)["shoot"]


def test_only_confirmed_kill_or_manual_reset_allows_a_new_target_shot():
    c = RecordedController(profile=PROFILE)
    step(c, 1)
    assert c.locked is not None and not c.shot_sent
    step(c, 1.033)
    c.shot_emitted(1.033)
    assert c.release_after_confirmed_kill(1.1)
    assert not c.shot_sent and c.locked is None and c.last_shot == 1.033
    assert step(c, 1.2)["target"] is None


def test_motion_never_carries_a_superseded_shot_into_a_new_action():
    p = RecordedMotionPlan()
    args = ((1280, 720), (0, 0, 1280, 720), 1, 1, 0.1)
    action = dict(
        target=target(),
        reason="verified",
        dx=0,
        dy=0,
        shoot=True,
        recorded=True,
        movement_duration=0,
        shot_token=1,
    )
    p.begin(action, *args)
    p.begin(dict(action, shoot=False, shot_token=None), *args)
    assert p.poll(1.001) is None
    p.begin(action, *args)
    assert p.poll(1.2) is None


def test_minimum_jerk_substeps_conserve_counts_without_a_start_jump():
    p = RecordedMotionPlan()
    action = dict(
        target=target(),
        reason="recorded_aim",
        dx=100,
        dy=-30,
        shoot=False,
        recorded=True,
        movement_duration=0.08,
        curve=(0, 1),
    )
    p.begin(action, (1280, 720), (0, 0, 1280, 720), 1, 1, 0.1)
    assert p.poll(1) is None
    events = [p.poll(1 + i * 0.001) for i in range(1, 91)]
    events = [event for event in events if event]
    assert sum(e["dx"] for e in events) == 100
    assert sum(e["dy"] for e in events) == -30
    assert not any(e["shoot"] for e in events)
    assert abs(events[0]["dx"]) < 5
    assert minimum_jerk(0) == 0 and minimum_jerk(1) == 1


@pytest.mark.parametrize("offset", [-250, 80, 300])
@pytest.mark.parametrize("speed", [0, 60, 180])
@pytest.mark.parametrize("delay", [0.03, 0.07])
def test_delayed_noisy_synthetic_camera_emits_only_one_precise_shot(
    offset, speed, delay
):
    report = run(PROFILE, offset=offset, speed=speed, delay=delay, noise=0.5)
    assert len(report["shots"]) == 1
    assert report["shots"][0]["actual_error_px"] < 3
    assert report["shots"][0]["at"] > 0.15
    assert report["movement_events"] > 20


def test_frames_older_than_allowed_never_generate_a_shot():
    c = RecordedController(profile=PROFILE)
    for source in (1, 1.033, 1.066):
        assert not step(c, source, now=source + 0.2)["shoot"]
    assert c.locked is None


@pytest.mark.parametrize("x,y", [(-250, -120), (250, 160), (0, -180), (0, 200)])
def test_vertical_and_diagonal_moves_are_verified_in_two_axes(x, y):
    report = run(PROFILE, offset=x, offset_y=y, delay=0.03, noise=0.5)
    assert len(report["shots"]) == 1
    assert report["shots"][0]["actual_error_px"] < 3


def test_delayed_kill_banner_releases_budget_and_acquires_next_enemy():
    from mvp_recorded_aim import visual_kill_matches

    c = RecordedController(profile=PROFILE)
    step(c, 1)
    c.shot_emitted(1.1)
    assert visual_kill_matches(c, 4.0, 4.1)
    assert c.release_after_confirmed_kill(4.1)
    action = step(c, 4.15, detections=[target(), target(850)])
    assert action["target"].head[0] == 850
    assert not c.shot_sent and c.course is not None
    assert not visual_kill_matches(c, 4.0, 4.16)


def test_old_or_stale_feed_cannot_release_a_living_target():
    from mvp_recorded_aim import visual_kill_matches

    c = RecordedController(profile=PROFILE)
    step(c, 1)
    assert not visual_kill_matches(c, 0.95, 1.1)
    c.shot_emitted(1.2)
    assert not visual_kill_matches(c, 1.1, 1.3)
    assert not visual_kill_matches(c, 1.3, 3)
    assert not visual_kill_matches(c, None, 2)
    assert c.locked is not None


def test_motion_bridges_detector_gap_but_expires_when_frames_stop():
    c = RecordedController(profile=PROFILE)
    step(c, 1, now=1.07, x=950)
    action = step(c, 1.06, now=1.13, x=950)
    p = RecordedMotionPlan()
    p.begin(action, (1280, 720), (0, 0, 1280, 720), 1.06, 1.13, 0.1)
    # The old source+100 ms limit stopped this movement before the next frame.
    assert p.poll(1.17) is not None
    assert p.poll(1.19) is not None
    assert p.poll(1.241) is None and p.pending is None


def test_refresh_preserves_movement_slope_and_never_carries_a_shot():
    p = RecordedMotionPlan()
    action = dict(
        target=target(),
        reason="recorded_aim",
        recorded=True,
        dx=100,
        dy=0,
        shoot=False,
        movement_duration=0.09,
        motion_max_age=0.18,
    )
    p.begin(action, (1280, 720), (0, 0, 1280, 720), 1, 1.07, 0.1)
    p.poll(1.10)
    old_velocity = p.pending["velocity"][0]
    p.begin(dict(action, dx=70), (1280, 720), (0, 0, 1280, 720), 1.033, 1.103, 0.1)
    assert p.pending["start_velocity"][0] == old_velocity
    assert p.poll(1.107)["dx"] > 0
    assert not p.pending["shoot"]


def test_banner_delivered_after_input_stall_still_releases_held_target():
    from mvp_recorded_aim import visual_kill_matches

    c = RecordedController(profile=PROFILE)
    step(c, 1)
    c.shot_emitted(1.1)
    # Kill observed at 1.45; a blocked input call delays dispatch to 2.05.
    assert visual_kill_matches(c, 1.45, 2.05)
    assert c.release_after_confirmed_kill(2.05)
    assert c.locked is None and not c.shot_sent


def test_manual_finishing_shot_banner_also_unlocks_next_target():
    from mvp_recorded_aim import visual_kill_matches

    c = RecordedController(profile=PROFILE)
    step(c, 1, x=850, fire=False)
    assert not c.shot_sent and visual_kill_matches(c, 1.3, 1.4)
    assert c.release_after_confirmed_kill(1.4)
    a = step(c, 1.45, detections=[target(850), target(420)])
    assert a["target"].head[0] == 420


def test_head_filter_reduces_alternating_keypoint_noise():
    from mvp_recorded_aim import HeadFilter

    f = HeadFilter()
    raw = []
    estimated = []
    for i in range(30):
        x = 640 + (1.5 if i % 2 else -1.5)
        point, velocity = f.observe((x, 360), 1 + i / 30, 5)
        raw.append(x)
        estimated.append(point[0])
    assert max(estimated[4:]) - min(estimated[4:]) < 0.6 * (max(raw) - min(raw))
    assert abs(velocity[0]) < 1


def test_filter_does_not_treat_our_camera_correction_as_target_velocity():
    from mvp_recorded_aim import HeadFilter
    from mvp_core import Settings
    from mvp_geometry import rotate_geometry

    f = HeadFilter()
    settings = Settings()
    point = (850, 360)
    for i in range(3):
        f.observe(point, 1 + i / 30, 5)
    f.acknowledge(200, 0, (1280, 720), settings)
    _, new = rotate_geometry((830, 350, 870, 450), point, 200, 0, (1280, 720), 1, 90)
    _, velocity = f.observe(new, 1.1, 5)
    assert abs(velocity[0]) < 1


def test_manual_camera_move_uses_two_bodies_to_keep_original_identity():
    c = RecordedController(profile=PROFILE)
    old = [target(640), target(400), target(970)]
    step(c, 1, detections=old, fire=False)
    a = step(c, 1.033, detections=[target(740), target(500), target(1070)], fire=False)
    assert a["target"].head[0] == 740
    assert a["scene_camera_shift"] == (100, 0)


def test_single_new_closer_enemy_cannot_steal_lock_using_camera_registration():
    c = RecordedController(profile=PROFILE)
    step(c, 1, x=850, fire=False)
    a = step(c, 1.033, detections=[target(640)], fire=False)
    assert a["target"] is None and c.locked.head[0] == 850


def test_recorded_course_is_twenty_four_percent_faster_without_artificial_overshoot():
    c = RecordedController(profile=PROFILE)
    step(c, 1, x=1100, fire=False)
    import math

    angle = math.degrees(math.atan2(460, 480))
    expected = (
        PROFILE["duration_seconds"]
        / 1.24
        * (angle / PROFILE["reference_angle_degrees"]) ** 0.35
    )
    assert abs(c.course["duration"] - expected) < 1e-8


def test_uncertain_three_frame_prediction_does_not_authorize_a_shot():
    c = RecordedController(profile=PROFILE)
    step(c, 1, x=637)
    step(c, 1.033, x=643)
    a = step(c, 1.066, x=640)
    assert a["prediction_uncertainty_px"] > a["head_radius_px"] * 0.5
    assert not a["shoot"]


def test_repeated_scene_layout_does_not_guess_camera_identity():
    from mvp_recorded_aim import scene_shift

    # Two equally supported translations in a regularly repeated row.
    old = [target(300), target(500)]
    new = [target(400), target(600), target(800)]
    assert scene_shift(old, new, None, 1) is None


def test_single_body_continues_across_moderate_manual_motion_without_firing():
    c = RecordedController(profile=PROFILE)
    step(c, 1, x=750)
    a = step(c, 1.033, x=650)
    assert a["target"] is not None
    assert not a["shoot"]
    assert c.locked.head[0] == 650


def test_single_body_does_not_reassociate_after_observation_gap():
    c = RecordedController(profile=PROFILE)
    step(c, 1, x=750)
    a = step(c, 1.2, x=650)
    assert a["target"] is None


def test_single_body_fallback_does_not_choose_between_two_people():
    c = RecordedController(profile=PROFILE)
    step(c, 1, x=750)
    a = step(c, 1.033, detections=[target(650), target(850)])
    assert a["target"] is None


def test_new_distant_group_reacquires_without_f8_after_lost_identity():
    c=RecordedController(profile=PROFILE)
    step(c,1,x=850)
    assert step(c,1.033,x=400)["target"] is None
    a=step(c,1.26,x=400)
    assert a["target"] is not None
    assert a["reacquired_after_scene_change"]
    assert not a["shoot"]


def test_nearby_occlusion_does_not_release_held_identity():
    c=RecordedController(profile=PROFILE)
    step(c,1,x=850)
    a=step(c,1.3,x=750)
    assert a["target"] is None
    assert c.locked.head[0]==850


def test_center_hold_suppresses_corrections_during_fire_verification():
    c=RecordedController(profile=PROFILE)
    for i,x in enumerate([638.8,639.4,640.1,640.7,639.8,640.2]):
        a=step(c,1+i*.033,x=x,fire=False)
        assert a['center_hold']
        assert (a['dx'],a['dy'])==(0,0)


def test_center_hold_releases_when_head_moves_outside_center():
    c=RecordedController(profile=PROFILE)
    step(c,1,x=640,fire=False)
    a=step(c,1.033,x=655,fire=False)
    assert not a['center_hold']
    assert a['dx']>0


def test_established_precise_center_does_not_wait_extra_alignment_frame():
    c=RecordedController(profile=PROFILE)
    step(c,1,x=640,fire=False)
    step(c,1.033,x=640,fire=False)
    c.verified_samples=0
    a=step(c,1.066,x=640,fire=True)
    assert a['verified_samples']==1
    assert a['immediate_center'] and a['shoot']

def test_cold_center_does_not_use_immediate_path():
    c=RecordedController(profile=PROFILE)
    a=step(c,1,x=640,fire=True)
    assert not a['immediate_center'] and not a['shoot']
