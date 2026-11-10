import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from mvp_core import Detection,Settings
from mvp_zone_aim import ZoneController
from benchmark_recorded_aim import run

def body(x=640,y=360,width=40,height=100):
    return Detection(0,.99,(x-width/2,y-10,x+width/2,y-10+height),(x,y),.99)

def step(c,t,d=None,fire=True,age=0):
    return c.step([d or body()],1280,720,t+age,t,enabled=True,foreground=True,fire=fire)

def controller():return ZoneController(Settings(fire_policy='zone',head_filter='one_euro',shot_interval=.25))

def test_weak_head_uses_visible_torso_without_lowering_box_threshold():
    c=controller()
    c.distance_fire=True
    weak=Detection(0,.95,(620,338.5,660,388.5),(-1,-1),.30)
    action=step(c,1,weak)
    assert action['body_fallback'] and action['fire_mode']=='opening'
    assert action['zone_center']==pytest.approx((640,360))
    assert c.zone_fire_ready(1)
    c.reset()
    assert step(c,2,Detection(0,.20,weak.box,weak.head,.30))['target'] is None

def test_every_distance_uses_sampled_opening_and_never_automatic_spray(monkeypatch):
    c=controller()
    c.distance_fire=True
    assert step(c,1,body(height=360))['fire_mode']=='opening'
    c.reset()
    far=step(c,2,body(height=30))
    assert far['fire_mode']=='opening'
    assert far['zone_center']==pytest.approx((640,360))
    c.reset()
    draws=iter((.1,.9))
    monkeypatch.setattr('mvp_zone_aim.random',lambda:next(draws))
    assert step(c,3,body(height=50))['fire_mode']=='opening'
    c.shot_emitted(3)
    c.tap_emitted()
    assert c.zone is None
    assert step(c,3.04,body(height=50))['fire_mode']=='tap'
    c.reset()
    assert step(c,4,body(height=50))['fire_mode']=='opening'
    c.shot_emitted(4)
    c.tap_emitted()
    assert step(c,4.19,body(height=50))['fire_mode']=='opening'
    assert not c.zone_fire_ready(4.199)
    assert c.zone_fire_ready(4.2)
    c.shot_emitted(4.2)
    c.tap_emitted()
    assert step(c,4.24,body(height=50))['fire_mode']=='tap'

def test_first_aligned_source_can_fire_without_confirmation_wait():
    c=controller();a=step(c,1)
    assert a['zone_inside'] and (a['dx'],a['dy'])==(0,0)
    assert c.zone_fire_ready(1)
    c.shot_emitted(1)
    step(c,1.24)
    assert not c.zone_fire_ready(1.249)
    assert c.zone_fire_ready(1.25)
    c.shot_emitted(1.25)
    step(c,1.49)
    assert not c.zone_fire_ready(1.499)
    assert c.zone_fire_ready(1.5)

def test_zone_firing_requires_visible_valid_fresh_target_and_enabled_fire():
    c=controller();step(c,1)
    assert not c.zone_fire_ready(1.101)
    step(c,1.2,fire=False)
    assert not c.zone_fire_ready(1.2)
    c.step([],1280,720,1.233,1.233,enabled=True,foreground=True,fire=True)
    assert not c.zone_fire_ready(1.233)

def test_dynamic_radius_shrinks_with_body_size():
    a=step(controller(),1,body(width=80,height=200))
    b=step(controller(),1,body(width=20,height=50))
    assert a['zone_radius_px']==pytest.approx(b['zone_radius_px']*4)

def test_global_cadence_survives_confirmed_kill():
    c=controller();step(c,1);c.shot_emitted(1)
    assert c.release_after_confirmed_kill(1.05)
    # New body outside the retired geometry, then camera centers it.
    step(c,1.06,body(800));c.retired=[]
    c.reset();step(c,1.1)
    assert not c.zone_fire_ready(1.1)

def test_prediction_includes_frame_age_and_future_correction():
    c=controller()
    for i,x in enumerate([500,507,514]):a=step(c,1+i*.033,body(x),age=.05)
    assert a['prediction_time_ms']>50
    assert a['prediction_point'][0]>514

@pytest.mark.parametrize('speed',[0,60,180])
@pytest.mark.parametrize('delay',[.03,.07])
def test_zone_closed_loop_tracks_and_repeats(speed,delay):
    r=run(dict(duration_seconds=.2492,reference_angle_degrees=34.459),offset=250,speed=speed,delay=delay,noise=.5,head_filter='one_euro',fire_policy='zone')
    assert len(r['shots'])>=3
    assert min(b['at']-a['at'] for a,b in zip(r['shots'],r['shots'][1:]))>=.249
    assert max(x['actual_error_px'] for x in r['shots'])<8


@pytest.mark.parametrize('offset',[80,250,450])
@pytest.mark.parametrize('delay',[.03,.07])
def test_stationary_target_does_not_cross_center_from_smooth_plan(offset,delay):
    r=run(dict(duration_seconds=.2492,reference_angle_degrees=34.459),offset=offset,delay=delay,noise=0,head_filter='one_euro',fire_policy='zone')
    assert r['min_signed_error_x']>=-.5
    assert len(r['shots'])>=3


def test_disabled_ram_guard_allows_near_exhausted_ram():
    from mvp_runtime import memory_paused
    assert not memory_paused(20,True,enabled=False)
    assert memory_paused(20,False,enabled=True)
