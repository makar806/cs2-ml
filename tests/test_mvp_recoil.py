import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mvp_core import Controller, Detection


def target(x=640, y=360):
    return Detection(0, 0.99, (x - 20, y - 10, x + 20, y + 100), (x, y), 0.99)


def step(c, d, now=1, age=0, fire=True):
    return c.step(d, 1280, 720, now, now - age, enabled=True, foreground=True, fire=fire)


def test_fast_noisy_observation_cannot_predict_several_heads_ahead():
    c = Controller()
    step(c, [target()], fire=False)
    a = step(c, [target(x=660)], now=1.06, age=0.05, fire=False)
    assert abs(a["lead"][0]) <= 6
    assert a["target"].head[0] <= 666


def test_shot_camera_kick_is_not_extrapolated_as_enemy_motion():
    c = Controller()
    assert step(c, [target()])["shoot"]
    a = step(c, [target(x=650)], now=1.16, age=0.02)
    assert a["velocity"][0] != 0 and a["lead"] == (0, 0)
    assert not a["shoot"]


def test_followup_requires_two_observed_aligned_heads():
    c = Controller()
    assert step(c, [target()])["shoot"]
    assert not step(c, [target(x=650)], now=1.5)["shoot"]
    assert not step(c, [target()], now=1.54)["shoot"]
    assert step(c, [target()], now=1.58)["shoot"]


def test_torso_keypoint_does_not_receive_input_or_shot():
    c = Controller()
    d = Detection(0, 0.99, (620, 300, 660, 440), (640, 400), 0.99)
    a = step(c, [d])
    assert a["target"] is None and not a["shoot"]


def test_misaligned_followup_never_fires_just_because_cooldown_expired():
    c = Controller()
    assert step(c, [target()])["shoot"]
    for i in range(12):
        a = step(c, [target(x=650)], now=1.5 + i * 0.04)
        assert not a["shoot"]


def test_initial_recoil_impulse_does_not_receive_counter_flick():
    c = Controller()
    assert step(c, [target()])["shoot"]
    a = step(c, [target(x=650)], now=1.02)
    assert a["reason"] == "recoil_settle" and a["dx"] == a["dy"] == 0


def test_held_identity_keeps_team_when_classifier_flips_after_camera_moves():
    c = Controller()
    step(c, [target()], fire=False)
    for i in range(1, 10):
        d = target(x=642)
        d.cls = 1
        a = step(c, [d], now=1 + i * 0.04, fire=False)
        assert a["target"] is not None and a["target"].cls == 0 and a["raw_class"] == 1


def test_friendly_cannot_be_acquired_after_old_identity_disappears():
    c = Controller()
    step(c, [target()], fire=False)
    d = target(x=800)
    d.cls = 1
    assert step(c, [d], now=1.1, fire=False)["target"] is None
    assert step(c, [d], now=1.5, fire=False)["target"] is None
