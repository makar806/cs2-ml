import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mvp_core import Detection,Settings,Controller
from mvp_zone_aim import ZoneController
from mvp_recorded_aim import RecordedMotionPlan,visual_kill_matches
import live_mvp as live


def d(x=640,cls=0,head=None):
    return Detection(cls,.99,(x-20,350,x+20,450),head or (x,360),.99)

def step(c,t,items):
    return c.step(items,1280,720,t,t,enabled=True,foreground=True,fire=False)

def controller():return ZoneController(Settings(fire_policy='zone',head_filter='one_euro'))

def test_class_duplicate_with_noisy_heads_is_not_two_competing_people():
    c=Controller();step(c,1,[d(645)])
    items=[d(645,0,(639,360)),d(645,1,(651,360))]
    assert step(c,1.033,items)['target'] is not None


def test_valid_match_is_not_shifted_by_scene_registration(monkeypatch):
    import mvp_recorded_aim as module
    c=controller();step(c,1,[d(700),d(400),d(950)])
    def bad_registration(*args):raise AssertionError('Unnecessary scene registration')
    monkeypatch.setattr(module,'scene_shift',bad_registration)
    a=step(c,1.033,[d(700),d(400),d(950)])
    assert a['target'].head[0]==700


def test_live_lock_is_not_abandoned_for_distant_target_on_timeout():
    c=controller();step(c,1,[d(850)])
    a=step(c,1.5,[d(400)])
    assert a['target'] is None
    assert c.locked.head[0]==850


def test_kill_observed_before_latest_repeat_shot_still_releases_target():
    c=controller();step(c,1,[d()]);c.shot_emitted(1.5)
    assert visual_kill_matches(c,1.4,1.6)
    assert not visual_kill_matches(c,.9,1.6)
    assert c.release_after_confirmed_kill(1.6)


def test_fresh_movement_bridges_next_frame_but_stale_frame_cannot_fire():
    c=controller()
    a=c.step([d(850)],1280,720,1.09,1,enabled=True,foreground=True,fire=True)
    plan=RecordedMotionPlan();plan.begin(a,(1280,720),(0,0,1280,720),1,1.09,.1)
    assert plan.poll(1.12) is not None
    assert a['motion_max_age']==.18
    c=controller();c.step([d()],1280,720,1.09,1,enabled=True,foreground=True,fire=True)
    assert c.zone_fire_ready(1.09)
    assert not c.zone_fire_ready(1.11)


def test_panel_snapshot_never_waits_for_input_lock():
    e=live.Engine.__new__(live.Engine)
    class BusyLock:
        def acquire(self,blocking):
            assert blocking is False
            return False
    e.lock=BusyLock();e.last_snapshot=dict(status='previous',aim_zone={'center':[640,360]})
    s=e.snapshot()
    assert s['input_busy'] and s['aim_zone'] is None
    assert s['status']=='previous'
