from pathlib import Path
from queue import Queue
import sys
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import mvp_capture
import mvp_windows
from mvp_core import Detection, Settings
from mvp_runtime import inference_process
from mvp_zone_aim import ZoneController


def step(controller, at, detections):
    return controller.step(detections, 1440, 1080, at, at,
                           enabled=True, foreground=True, fire=True)


def test_recovers_lone_candidate_from_recorded_lockout_without_f8():
    controller = ZoneController(Settings(fire_policy='zone'))
    old = Detection(0, .9, (590, 517, 614, 584), (598, 525), .97)
    visible = Detection(0, .9, (638, 508, 664, 573), (650, 514), .97)
    step(controller, 10, [old])
    assert step(controller, 10.10, [visible])['target'] is None
    assert step(controller, 10.14, [visible])['target'] is None
    recovered = step(controller, 10.18, [visible])
    assert recovered['recovered_lone_target']
    assert recovered['target'].head == visible.head
    assert controller.acquired_at == 10
    assert not recovered['shot_ready']


def test_capture_is_released_on_leaving_game_foreground(monkeypatch, tmp_path):
    stop = Event()
    focused = True
    calls = []

    class Capture:
        error = ''

        def get(self):
            nonlocal focused
            calls.append('get')
            focused = False
            return None

        def close(self):
            calls.append('close')
            stop.set()

    class WorkerQueue(Queue):
        def cancel_join_thread(self):
            pass

    monkeypatch.setattr(mvp_windows, 'find_window', lambda *args, **kwargs: 42)
    monkeypatch.setattr(mvp_windows, 'foreground', lambda hwnd: focused)
    monkeypatch.setattr(mvp_capture, 'create_capture', lambda *args: Capture())
    config = dict(mode='local-bots', pipeline='capture', hz=30, output=str(tmp_path), foreground_only=True)
    inference_process(config, stop, WorkerQueue(), WorkerQueue())
    assert calls == ['get', 'close']
    assert 'release_unfocused' in (tmp_path / 'capture-events.jsonl').read_text()
    assert not (tmp_path / 'error.txt').exists()


def test_persistent_loss_and_distant_candidates_do_not_release_old_lock():
    controller = ZoneController(Settings(fire_policy='zone'))
    old = Detection(0, .9, (899, 528, 932, 602), (916, 536), .99)
    visible = Detection(0, .9, (625, 531, 656, 604), (640, 539), .99)
    other = Detection(0, .9, (385, 531, 418, 605), (400, 539), .99)
    step(controller, 10, [old])
    for at in (10.25, 10.29, 10.33, 10.37, 11., 20.):
        assert step(controller, at, [visible, other])['target'] is None
    assert controller.acquired_at == 10
    assert controller.locked == old


def test_nearby_bot_of_different_size_does_not_block_reassociation():
    controller = ZoneController(Settings(fire_policy='zone'))
    old = Detection(0, .9, (733, 617, 786, 761), (753, 630), .99)
    visible = Detection(0, .9, (777, 686, 835, 852), (802, 693), .99)
    small = Detection(0, .9, (679, 692, 706, 769), (692, 699), .99)
    step(controller, 10, [old])
    assert step(controller, 10.10, [visible, small])['target'] is None
    assert step(controller, 10.14, [visible, small])['target'] is None
    recovered = step(controller, 10.18, [visible, small])
    assert recovered['recovery_mode'] == 'reassociate'
    assert recovered['target'].head == visible.head


def test_body_box_shrinks_without_losing_same_head():
    controller = ZoneController(Settings(fire_policy='zone'))
    old = Detection(0, .9, (737, 509, 804, 818), (764, 536), .99)
    cropped = Detection(0, .9, (738, 510, 802, 688), (764, 530), .99)
    other = Detection(0, .9, (835, 509, 902, 818), (862, 536), .99)
    step(controller, 10, [old])
    action = step(controller, 10.04, [cropped, other])
    assert action['target'] is not None
    assert controller.locked.head == cropped.head
    assert controller.acquired_at == 10
