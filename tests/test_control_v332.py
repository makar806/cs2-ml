"""Real scheduler/controller entry points, synthetic frames and fake OS input."""
import io
from types import SimpleNamespace
import pytest
from test_mvp_emitter import engine, authorize_weapon
import live_mvp as live
from mvp_core import Detection, Settings
from mvp_zone_aim import ZoneController
from mvp_recorded_aim import RecordedMotionPlan


def rig(monkeypatch,tmp_path):
    e=engine(monkeypatch,tmp_path)
    e.fire=True
    e.settings=Settings(fire_policy='zone',stop_before_shot=False,head_filter='one_euro')
    e.controller=ZoneController(e.settings)
    e.controller.distance_fire=True
    e.controller.weapon_aim_point='body'
    e.motion=RecordedMotionPlan()
    e.spray=live.AKSpray(e.settings.sensitivity)
    e.spray_button=5
    e.physical_keys=SimpleNamespace(ctrl_generation=0,held=False)
    e.physical_keys.ctrl_down=lambda:e.physical_keys.held
    e.physical_keys.left_ctrl_down=lambda:e.physical_keys.held
    e.clock=[1.]
    e.keys=set()
    e.keyboard=[]
    e.log=io.StringIO()
    monkeypatch.setattr(live.time,'perf_counter',lambda:e.clock[0])
    monkeypatch.setattr(live,'key_down',e.keys.__contains__)
    monkeypatch.setattr(live,'set_key',lambda key,down:e.keyboard.append((key,down)))
    authorize_weapon(e,1.)
    return e


def test_real_input_loop_hotkeys_focus_return_and_cleanup(monkeypatch,tmp_path):
    from collections import deque
    from queue import Queue
    from threading import Event
    import mvp_windows
    e=rig(monkeypatch,tmp_path)
    e.armed=False
    e.ready=True
    e.async_logs=False
    e.pipeline='full'
    e.messages=Queue()
    e.packets=Queue()
    e.worker=SimpleNamespace(is_alive=lambda:True)
    e.worker_stop=Event()
    e.worker_active=Event()
    e.mouse.timings=deque()
    observer=e.physical_keys
    closed=[]
    observer.close=lambda:closed.append(True)
    monkeypatch.setattr(mvp_windows,'PhysicalKeys',lambda:observer)
    # F8, F9 off/on, F7, Alt+Tab, return without F8, F8, F6.
    keys=[0x77,0x78,0x78,0x76,None,None,0x77,0x75]
    expected=[(True,True),(True,False),(True,True),(True,True),
              (False,True),(False,True),(True,True),(False,True)]
    index=[0]
    monkeypatch.setattr(live.USER,'GetAsyncKeyState',lambda key:1 if key==keys[index[0]] else 0)
    monkeypatch.setattr(live,'foreground',lambda hwnd:index[0]!=4)
    def tick():
        assert (e.armed,e.fire)==expected[index[0]]
        if index[0]==3:
            assert e.controller.locked is None
        index[0]+=1
        e.clock[0]+=.05
    e.mouse.tick=tick
    e.inputs()
    assert index[0]==len(keys) and e.stop.is_set() and e.worker_stop.is_set()
    assert not e.state['input_error'] and not e.armed and e.mouse.released
    assert closed==[True] and e.physical_keys is None


def observe(e,at,x=640,y=360,height=100,ammo=30,head_score=.99,visible=True):
    e.clock[0]=at
    authorize_weapon(e,at,ammo)
    d=Detection(0,.99,(x-20,y-height*.43,x+20,y+height*.57),
                (x,y-height*.33),head_score)
    a=e.controller.step([d] if visible else [],1280,720,at,at,
                        enabled=True,foreground=True,fire=e.fire)
    e.zone_context=dict(size=(1280,720),rect=(0,0,1280,720)) if a.get('target') else None
    return a


@pytest.mark.parametrize('draw,count',[(0.,1),(.499,1),(.5,2),(.999,2)])
def test_sample_once_per_series_and_never_hold_automatically(monkeypatch,tmp_path,draw,count):
    e=rig(monkeypatch,tmp_path)
    draws=[]
    monkeypatch.setattr('mvp_zone_aim.random',lambda:draws.append(draw) or draw)
    for i in range(20):
        observe(e,1+i*.05)
        if i==0:
            assert e.controller.opening_taps==count
        e.send_pending(e.log,i)
    assert len(draws)==1
    assert len([v for v in e.mouse.events if v[2]])>=3
    assert not e.mouse.holds


def test_pair_cadence_uses_successful_input_time(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    monkeypatch.setattr('mvp_zone_aim.random',lambda:.9)
    original=e.mouse.act
    def delayed(*args,**kwargs):
        original(*args,**kwargs)
        e.clock[0]+=.04
    e.mouse.act=delayed
    observe(e,1.)
    e.send_pending(e.log,1)
    assert e.controller.last_shot==pytest.approx(1.04)
    observe(e,1.2)
    e.send_pending(e.log,2)
    assert e.state['shots']==1
    observe(e,1.241)
    e.send_pending(e.log,3)
    assert e.state['shots']==2 and not e.mouse.holds


def test_kill_cancels_second_and_stale_frame_cannot_fire(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    monkeypatch.setattr('mvp_zone_aim.random',lambda:.9)
    observe(e,1.)
    e.send_pending(e.log,1)
    assert e.controller.release_after_confirmed_kill(1.05)
    e.release_fire(automatic_only=True,reason='visual_kill')
    e.clock[0]=1.21
    e.send_pending(e.log,2)
    assert e.state['shots']==1
    e.controller.retired=[]
    observe(e,1.3)
    e.clock[0]=1.41
    e.send_pending(e.log,3)
    assert e.state['shots']==1


def test_mouse4_rising_edge_only_and_wall_spray(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    e.keys.add(5)
    for at in (1.,1.01,1.02):
        e.clock[0]=at
        e.send_pending(e.log,1)
    assert e.spray.enabled and len(e.mouse.holds)==1 and e.spray.wall_mode
    e.keys.clear()
    e.send_pending(e.log,1)
    e.keys.add(5)
    e.send_pending(e.log,1)
    assert not e.spray.enabled and e.spray.started is None
    e.send_pending(e.log,1)
    assert len(e.mouse.holds)==1


@pytest.mark.parametrize('axis,delta',[(0,30),(1,30),(0,-30),(1,-30)])
def test_manual_spray_follows_current_body_in_both_directions(monkeypatch,tmp_path,axis,delta):
    e=rig(monkeypatch,tmp_path)
    observe(e,1.)
    e.keys.add(5)
    e.send_pending(e.log,1)
    # Isolate following from the unchanged wall recoil profile.
    e.spray.correction=lambda *args:(0,0)
    center=[640,360]
    center[axis]+=delta
    observe(e,1.033,*center)
    e.send_pending(e.log,2)
    assert e.mouse.events[-1][axis]*delta>0


def test_weak_head_and_enemy_crouch_move_body_down(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    observe(e,1.,head_score=.1)
    e.keys.add(5)
    e.send_pending(e.log,1)
    e.spray.correction=lambda *args:(0,0)
    a=observe(e,1.033,y=380,height=85,head_score=.1)
    assert a['target'] is not None and a['aim_point']=='body'
    e.send_pending(e.log,2)
    assert e.mouse.events[-1][1]>0


def test_ctrl_transfer_does_not_release_users_key(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    e.keys.add(5)
    e.send_pending(e.log,1)
    e.clock[0]=2.01
    authorize_weapon(e,e.clock[0])
    e.send_pending(e.log,2)
    assert e.keyboard==[(0xA2,True)]
    e.physical_keys.held=True
    e.physical_keys.ctrl_generation+=1
    e.clock[0]=2.02
    e.send_pending(e.log,3)
    e.disarm()
    assert e.keyboard==[(0xA2,True)]
    assert e.physical_keys.held


def test_user_ctrl_release_is_not_undone_by_auto_crouch(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    e.physical_keys.held=True
    e.keys.add(5)
    e.send_pending(e.log,1)
    e.physical_keys.held=False
    e.physical_keys.ctrl_generation+=1
    e.clock[0]=2.1
    authorize_weapon(e,e.clock[0])
    e.send_pending(e.log,2)
    assert not e.keyboard and e.spray.started==1.


def test_stance_preserves_phase_and_wall_reference():
    profile=live.AKSpray(.97)
    reference=live.AKSpray(.97,crouch_scale=1.)
    profile.start(1.,True,None)
    reference.start(1.,True,None)
    for i in range(1,201):
        t=i*.005
        assert profile.correction(t)==reference.correction(t)
    assert profile.correction(1.,True)==(0,0)
    assert profile.correction(1.,False)==(0,0)
    assert profile.started==1. and profile.elapsed==1.
    for i in range(201,401):
        profile.correction(i*.005,True)
        reference.correction(i*.005)
    assert profile.sent[1]<reference.sent[1]


def test_manual_repeat_needs_observed_reload_and_fresh_body(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    observe(e,1.)
    e.keys.add(5)
    e.send_pending(e.log,1)
    observe(e,4.06,ammo=5)
    e.send_pending(e.log,2)
    assert e.spray.awaiting_reload and len(e.mouse.holds)==1
    observe(e,8.,ammo=5)
    e.send_pending(e.log,3)
    assert len(e.mouse.holds)==1
    observe(e,8.1,ammo=0)
    e.send_pending(e.log,4)
    observe(e,8.6,ammo=30,visible=False)
    e.send_pending(e.log,5)
    assert len(e.mouse.holds)==1
    observe(e,8.7,ammo=30)
    e.send_pending(e.log,6)
    assert len(e.mouse.holds)==2


@pytest.mark.parametrize('field,value',[
    ('weapon_name',None),('weapon_name','weapon_deagle'),('weapon_name','weapon_knife'),
    ('weapon_name','weapon_hegrenade'),('weapon_zoom',1),('weapon_zoom',2),
    ('weapon_ammo',None),('weapon_ammo',0),('weapon_reloading',True),
])
def test_unknown_weapon_reload_and_zoom_cannot_use_ak(monkeypatch,tmp_path,field,value):
    e=rig(monkeypatch,tmp_path)
    observe(e,1.)
    e.weapons.sample[field]=value
    e.send_pending(e.log,1)
    e.keys.add(5)
    e.send_pending(e.log,2)
    assert not e.mouse.events


def test_manual_input_error_disarms_and_releases(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    def fail(*args,**kwargs):
        raise OSError('simulated input rejection')
    e.mouse.act=fail
    e.keys.add(5)
    e.send_pending(e.log,1)
    assert not e.armed and not e.spray.enabled and e.mouse.released


def test_right_ctrl_does_not_leave_synthetic_left_ctrl_stuck(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    e.keys.add(5)
    e.send_pending(e.log,1)
    e.clock[0]=2.01
    authorize_weapon(e,e.clock[0])
    e.send_pending(e.log,2)
    e.physical_keys.held=True
    e.physical_keys.left_ctrl_down=lambda:False
    e.physical_keys.ctrl_generation+=1
    e.clock[0]=2.02
    e.send_pending(e.log,3)
    assert e.keyboard==[(0xA2,True),(0xA2,False)]
    e.disarm()
    assert e.physical_keys.held


def test_stopping_spray_discards_queued_aim_and_requires_fresh_frame(monkeypatch,tmp_path):
    e=rig(monkeypatch,tmp_path)
    observe(e,1.)
    e.keys.add(5)
    e.send_pending(e.log,1)
    action=observe(e,1.03,x=670)
    e.motion.begin(action,(1280,720),(0,0,1280,720),1.03,1.03,.1)
    assert e.motion.pending
    e.release_fire(reason='manual_toggle')
    assert e.motion.pending is None and e.controller.zone is None
    assert e.controller.last_shot==1.03
    count=len(e.mouse.events)
    e.keys.clear()
    e.clock[0]=1.04
    e.send_pending(e.log,2)
    assert len(e.mouse.events)==count
