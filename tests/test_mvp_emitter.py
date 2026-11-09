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
        self.holds = []
        self.last_call = {}

    def act(self, dx, dy, shoot=False, *, hold=False):
        self.events.append((dx, dy, shoot))
        if hold:
            self.holds.append((dx, dy, shoot))

    def release(self):
        self.released = True


def engine(monkeypatch, tmp_path):
    e = live.Engine.__new__(live.Engine)
    from mvp_stop import StopBeforeShot
    e.stop_before_shot=StopBeforeShot()
    monkeypatch.setattr(live,"key_down",lambda code:False)
    monkeypatch.setattr(live,"release_movement_keys",lambda keys:None)
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


def test_recorded_shot_is_committed_only_after_actual_mouse_submission(monkeypatch, tmp_path):
    from mvp_core import Detection
    from mvp_recorded_aim import RecordedController, RecordedMotionPlan

    e = engine(monkeypatch, tmp_path)
    e.fire = True
    e.controller = RecordedController()
    e.motion = RecordedMotionPlan()
    now = live.time.perf_counter()
    d = Detection(0, 0.99, (620, 350, 660, 450), (640, 360), 0.99)
    e.controller.step(
        [d], 1280, 720, now - 0.033, now - 0.033, enabled=True, foreground=True, fire=True
    )
    a = e.controller.step([d], 1280, 720, now, now, enabled=True, foreground=True, fire=True)
    assert a["shoot"] and not e.controller.shot_sent
    e.motion.begin(a, (1280, 720), (0, 0, 1280, 720), now, now, 0.1)
    e.send_pending(io.StringIO(), 1)
    assert e.controller.shot_sent and e.mouse.events == [(0, 0, True)]
    e.motion.begin(a, (1280, 720), (0, 0, 1280, 720), now, now, 0.1)
    e.send_pending(io.StringIO(), 2)
    assert e.mouse.events == [(0, 0, True)]


def test_recorded_fire_disabled_at_emission_drops_prepared_shot(monkeypatch, tmp_path):
    from mvp_core import Detection
    from mvp_recorded_aim import RecordedController, RecordedMotionPlan

    e = engine(monkeypatch, tmp_path)
    e.fire = False
    e.controller = RecordedController()
    e.motion = RecordedMotionPlan()
    now = live.time.perf_counter()
    d = Detection(0, 0.99, (620, 350, 660, 450), (640, 360), 0.99)
    e.controller.step(
        [d], 1280, 720, now - 0.033, now - 0.033, enabled=True, foreground=True, fire=True
    )
    a = e.controller.step([d], 1280, 720, now, now, enabled=True, foreground=True, fire=True)
    e.motion.begin(a, (1280, 720), (0, 0, 1280, 720), now, now, 0.1)
    e.send_pending(io.StringIO(), 1)
    assert not e.mouse.events and not e.controller.shot_sent


def test_movement_release_defers_shot_without_consuming_it(monkeypatch,tmp_path):
    e=engine(monkeypatch,tmp_path)
    released=[]
    monkeypatch.setattr(live,'key_down',lambda code:code==87)
    monkeypatch.setattr(live,'release_movement_keys',released.append)
    schedule(e)
    e.send_pending(io.StringIO(),1)
    assert released==[[87]]
    assert not e.mouse.events
    assert e.state['shots']==0
    assert e.motion.pending is None


def test_switch_filter_cancels_pending_input_and_records_selection(monkeypatch,tmp_path):
    import json
    from mvp_recorded_aim import RecordedController,OneEuroHeadFilter
    e=engine(monkeypatch,tmp_path)
    e.controller=RecordedController(e.settings)
    e.state['one_euro_parameters']=dict(min_cutoff=.5,beta=.3,derivative_cutoff=1.)
    (tmp_path/'settings.json').write_text(json.dumps({'controller':{}}),encoding='utf-8')
    schedule(e)
    e.set_head_filter('one_euro')
    assert e.motion.pending is None and e.mouse.released
    assert isinstance(e.controller.head_filter,OneEuroHeadFilter)
    assert e.state['head_filter']=='one_euro'
    assert json.loads((tmp_path/'settings.json').read_text())['controller']['head_filter']=='one_euro'
    assert json.loads((tmp_path/'filter-events.jsonl').read_text())['head_filter']=='one_euro'



def test_zone_timer_emits_at_250ms_without_waiting_next_model_frame(monkeypatch,tmp_path):
    from mvp_zone_aim import ZoneController
    from mvp_core import Detection,Settings
    from mvp_recorded_aim import RecordedMotionPlan
    e=engine(monkeypatch,tmp_path);e.fire=True
    e.settings=Settings(fire_policy='zone',head_filter='one_euro',shot_interval=.25)
    e.controller=ZoneController(e.settings);e.motion=RecordedMotionPlan()
    e.zone_context=dict(size=(1280,720),rect=(0,0,1280,720))
    d=Detection(0,.99,(620,350,660,450),(640,360),.99)
    clock=[1.]
    monkeypatch.setattr(live.time,'perf_counter',lambda:clock[0])
    e.controller.step([d],1280,720,1,1,enabled=True,foreground=True,fire=True)
    log=io.StringIO();e.send_pending(log,1)
    assert e.state['shots']==1
    clock[0]=1.24
    e.controller.step([d],1280,720,1.24,1.24,enabled=True,foreground=True,fire=True)
    e.send_pending(log,2);assert e.state['shots']==1
    clock[0]=1.25;e.send_pending(log,2);assert e.state['shots']==2
    clock[0]=1.26;e.send_pending(log,2);assert e.state['shots']==2
    # F9 cancels firing even with a valid cache and a due shot.
    clock[0]=1.5;e.fire=False
    e.controller.step([d],1280,720,1.5,1.5,enabled=True,foreground=True,fire=True)
    e.send_pending(log,3);assert e.state['shots']==2
    # Stale source never continues the old burst.
    clock[0]=1.7;e.fire=True;e.send_pending(log,3);assert e.state['shots']==2


def test_stop_before_shot_can_be_disabled_explicitly(monkeypatch,tmp_path):
    e=engine(monkeypatch,tmp_path);e.settings.stop_before_shot=False
    released=[]
    monkeypatch.setattr(live,'key_down',lambda code:True)
    monkeypatch.setattr(live,'release_movement_keys',released.append)
    schedule(e);e.send_pending(io.StringIO(),1)
    assert not released
    assert e.state['shots']==1


def test_side_button_toggles_spray_and_releases_owned_ctrl(monkeypatch, tmp_path):
    from mvp_zone_aim import ZoneController

    e = engine(monkeypatch, tmp_path)
    e.fire = True
    e.controller = ZoneController(e.settings)
    e.spray = live.AKSpray(e.settings.sensitivity)
    e.spray_button = 6
    clock = [10.0]
    held = {6}
    keyboard = []
    monkeypatch.setattr(live.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(live, 'key_down', held.__contains__)
    monkeypatch.setattr(live, 'set_key', lambda code, down: keyboard.append((code, down)))
    log = io.StringIO()

    e.send_pending(log, 1)
    assert e.spray.manual and e.mouse.holds == [(0, 0, True)]
    held.clear()
    clock[0] = 10.9
    e.send_pending(log, 2)
    assert e.spray.started is not None and not keyboard
    clock[0] = 11.01
    e.send_pending(log, 3)
    assert keyboard == [(0xA2, True)]
    held.add(6)
    e.send_pending(log, 4)
    assert e.mouse.released and e.spray.started is None
    assert keyboard == [(0xA2, True), (0xA2, False)]
    assert len(e.mouse.holds) == 1


def test_zone_starts_one_burst_and_focus_loss_releases_it(monkeypatch, tmp_path):
    from mvp_core import Detection
    from mvp_zone_aim import ZoneController

    e = engine(monkeypatch, tmp_path)
    e.fire = True
    e.settings = Settings(fire_policy='zone')
    e.controller = ZoneController(e.settings)
    e.spray = live.AKSpray(e.settings.sensitivity)
    e.spray_button = 5
    e.zone_context = dict(size=(1280, 720), rect=(0, 0, 1280, 720))
    clock = [1.0]
    monkeypatch.setattr(live.time, 'perf_counter', lambda: clock[0])
    target = Detection(0, .99, (620, 350, 660, 450), (640, 360), .99)
    e.controller.step([target], 1280, 720, 1, 1, enabled=True, foreground=True, fire=True)
    log = io.StringIO()
    e.send_pending(log, 1)
    e.send_pending(log, 1)
    assert e.mouse.holds == [(0, 0, True)] and e.state['shots'] == 1
    e.controller.zone['velocity'] = (180., 0.)
    clock[0] += .012
    e.send_pending(log, 2)
    assert e.mouse.events[-1][0] > 0
    assert not e.mouse.events[-1][2]
    assert e.mouse.holds == [(0, 0, True)]
    monkeypatch.setattr(live, 'foreground', lambda _: False)
    e.send_pending(log, 2)
    assert e.mouse.released and e.spray.started is None


def test_middle_distance_emits_two_taps_then_holds(monkeypatch, tmp_path):
    from mvp_core import Detection
    from mvp_zone_aim import ZoneController

    e = engine(monkeypatch, tmp_path)
    e.fire = True
    e.settings = Settings(fire_policy='zone', stop_before_shot=False)
    e.controller = ZoneController(e.settings)
    e.controller.distance_fire = True
    e.spray = live.AKSpray(e.settings.sensitivity)
    e.spray_button = 5
    e.zone_context = dict(size=(1280, 720), rect=(0, 0, 1280, 720))
    clock = [1.0]
    monkeypatch.setattr(live.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr('mvp_zone_aim.random', lambda: .9)
    log = io.StringIO()

    def run(at, y):
        clock[0] = at
        target = Detection(0, .99, (620, y-10, 660, y+40), (640, y), .99)
        e.controller.step([target], 1280, 720, at, at, enabled=True, foreground=True, fire=True)
        e.send_pending(log, 1)

    run(1., 360)
    run(1.19, 360)
    assert e.mouse.events == [(0, 0, True)]
    run(1.2, 360)
    assert len(e.mouse.events) == 2 and not e.mouse.holds
    run(1.41, 348.5)
    assert e.mouse.holds == [(0, 0, True)]
    acquired_at = e.controller.acquired_at
    clock[0] += e.spray.max_duration + .01
    end = clock[0]
    # The scheduler also requires a fresh observation at the end of a magazine.
    e.spray.last_visible = end
    e.send_pending(log, 2)
    assert e.spray.started is None
    assert e.state['spray_stop_reason'] == 'profile_complete'
    run(end + .50, 348.5)
    assert len(e.mouse.holds) == 1
    run(end + e.spray.reload_wait + .01, 348.5)
    assert e.controller.acquired_at == acquired_at
    assert len(e.mouse.holds) == 2
