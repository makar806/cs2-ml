import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from mvp_core import Settings,Detection
from mvp_recorded_aim import RecordedController,OneEuroHeadFilter,HeadFilter
from mvp_one_euro import OneEuro
from mvp_geometry import rotate_geometry
from benchmark_recorded_aim import run


def test_euro_is_single_position_filter_not_cascaded_with_head_filter():
    f=OneEuroHeadFilter();reference=OneEuro(**f.parameters)
    for i in range(30):
        t=1+i/30;x=640+(-1)**i*1.5
        assert f.observe((x,360),t,5)[0][0]==reference(x,t)


def test_camera_input_moves_internal_euro_state_without_fake_velocity():
    f=OneEuroHeadFilter();s=Settings()
    for i in range(3):f.observe((850,360),1+i/30,5)
    f.acknowledge(200,10,(1280,720),s)
    _,point=rotate_geometry((830,350,870,450),(850,360),200,10,(1280,720),1,90)
    result,velocity=f.observe(point,1.1,5)
    assert math.dist(result,point)<1e-7
    assert math.hypot(*velocity)<1e-6


def test_scene_translation_and_explicit_clear_also_update_euro():
    f=OneEuroHeadFilter()
    for i in range(3):f.observe((640,360),1+i/30,5)
    f.translate(100,20)
    assert f.observe((740,380),1.1,5)[0]==(740,380)
    f.clear()
    assert f.observe((400,300),1.2,5)[0]==(400,300)
    assert len(f.samples)==1


def test_gap_reinitializes_euro_without_old_position_pull():
    f=OneEuroHeadFilter()
    f.observe((640,360),1,5)
    assert f.observe((800,360),1.3,5)[0]==(800,360)


def test_filter_switch_preserves_identity_and_spent_shot_but_revokes_token():
    c=RecordedController()
    d=Detection(0,.99,(620,350,660,450),(640,360),.99)
    for i in range(3):c.step([d],1280,720,1+i/30,1+i/30,enabled=True,foreground=True,fire=True)
    held=c.locked;c.shot_emitted(1.08)
    c.set_head_filter('one_euro')
    assert c.locked is held and c.shot_sent
    assert isinstance(c.head_filter,OneEuroHeadFilter)
    assert not c.can_emit_shot(1.1,1.1)
    c.reset()
    assert isinstance(c.head_filter,OneEuroHeadFilter)
    c.set_head_filter('current')
    assert type(c.head_filter) is HeadFilter


@pytest.mark.parametrize('offset',[-250,80,300])
@pytest.mark.parametrize('speed',[0,60,180])
@pytest.mark.parametrize('delay',[.03,.07])
def test_euro_closed_loop_on_delayed_moving_target(offset,speed,delay):
    r=run(dict(duration_seconds=.2492,reference_angle_degrees=34.459),offset=offset,speed=speed,delay=delay,noise=.5,head_filter='one_euro')
    assert len(r['shots'])==1
    assert r['shots'][0]['actual_error_px']<3
