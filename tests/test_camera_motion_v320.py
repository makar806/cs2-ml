import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cv2
import numpy as np
import pytest
from mvp_camera_motion import CameraMotion,CameraLock
from mvp_core import Detection
from profile_stutter import wait_for_live_state
from mvp_core import Settings
from mvp_zone_aim import ZoneController
from mvp_motion import MotionHistory


def target(x=640):
    return Detection(0,.99,(x-20,300,x+20,440),(x,310),.99)


def scene(seed=42):
    image=np.random.default_rng(seed).integers(0,256,(144,256,3),dtype=np.uint8)
    return cv2.resize(image,(1280,720),interpolation=cv2.INTER_NEAREST)


def test_background_tracks_camera_with_only_one_person():
    tracker=CameraMotion();image=scene()
    tracker.update(image,[target()],1)
    moved=cv2.warpAffine(image,np.float32([[1,0,-120],[0,1,15]]),(1280,720))
    steps=tracker.update(moved,[target(520)],1.033)
    lock=CameraLock();lock.anchor(target(),1)
    d,reason=lock.advance(steps,1.033,(1280,720))
    assert reason is None
    assert d.head[0]==pytest.approx(520,abs=4)
    assert d.head[1]==pytest.approx(325,abs=4)


def test_leaving_view_releases_without_kill_but_visible_target_stays():
    lock=CameraLock();lock.anchor(target(100),1)
    steps=[dict(source=1,at=1.03,matrix=[[1,0,-80],[0,1,0],[0,0,1]],cut=False)]
    d,reason=lock.advance(steps,1.03,(1280,720))
    assert d is not None and reason is None
    steps.append(dict(source=1.03,at=1.06,matrix=[[1,0,-100],[0,1,0],[0,0,1]],cut=False))
    assert lock.advance(steps,1.06,(1280,720))==(None,'camera_target_left_view')


def test_skipped_packets_compose_camera_motion():
    lock=CameraLock();lock.anchor(target(),1)
    steps=[dict(source=1,at=1.03,matrix=[[1,0,-80],[0,1,0],[0,0,1]],cut=False),
           dict(source=1.03,at=1.06,matrix=[[1,0,-100],[0,1,0],[0,0,1]],cut=False)]
    d,reason=lock.advance(steps,1.06,(1280,720))
    assert reason is None and d.head[0]==460
    assert lock.advance(steps,1.06,(1280,720))[0].head[0]==460


def test_missing_detections_and_time_alone_do_not_release_identity():
    lock=CameraLock();lock.anchor(target(),1)
    assert lock.advance([dict(source=1,at=1.03,matrix=None,cut=False)],1.03,(1280,720))==(None,None)
    assert lock.target is not None
    assert lock.advance([],5,(1280,720))[1] is None


def test_large_camera_cut_is_confirmed_on_next_stable_scene():
    tracker=CameraMotion();image=scene();other=scene(100)
    tracker.update(image,[target()],1)
    steps=tracker.update(other,[],1.03)
    assert not steps[-1]['cut']
    steps=tracker.update(other,[],1.06)
    assert steps[-1]['cut']
    lock=CameraLock();lock.anchor(target(),1)
    assert lock.advance(steps,1.06,(1280,720))==(None,'camera_scene_cut')


def test_brightness_change_does_not_release_as_camera_cut():
    tracker=CameraMotion();image=scene()
    tracker.update(image,[],1)
    brighter=np.minimum(image.astype(np.uint16)+40,255).astype(np.uint8)
    tracker.update(brighter,[],1.03)
    steps=tracker.update(image,[],1.06)
    assert not any(s['cut'] for s in steps)


def test_recorder_waits_until_model_is_ready_and_f8_active():
    states=iter([dict(ready=False,armed=False),dict(ready=True,armed=False),dict(ready=True,armed=True)])
    times=iter([0,0,.2,.4])
    sleeps=[]
    result=wait_for_live_state(lambda:next(states),lambda:True,clock=lambda:next(times),sleep=sleeps.append)
    assert result['armed'] and len(sleeps)==2


def test_recorder_does_not_start_when_only_ready():
    times=iter([0,0,2])
    with pytest.raises(RuntimeError):
        wait_for_live_state(lambda:dict(ready=True,armed=False),lambda:True,
                            timeout=1,clock=lambda:next(times),sleep=lambda _:None)


def test_controller_acquires_new_visible_target_after_camera_departure_without_kill():
    controller=ZoneController(Settings(fire_policy='zone',head_filter='one_euro'))
    controller.step([target(100)],1280,720,1,1,enabled=True,foreground=True,fire=True)
    lock=CameraLock();lock.anchor(target(100),1)
    steps=[dict(source=1,at=1.03,matrix=[[1,0,-300],[0,1,0],[0,0,1]],cut=False)]
    reason=lock.update_controller(controller,MotionHistory(),steps,1.03,(1280,720))
    assert reason=='camera_target_left_view'
    a=controller.step([target(600)],1280,720,1.03,1.03,enabled=True,foreground=True,fire=True)
    assert a['target'] is not None and controller.locked.head[0]==600
    assert a['dx']!=0


def test_controller_follows_same_person_after_camera_shift_despite_other_closer_person():
    controller=ZoneController(Settings(fire_policy='zone',head_filter='one_euro'))
    controller.step([target(640)],1280,720,1,1,enabled=True,foreground=True,fire=True)
    lock=CameraLock();lock.anchor(target(),1)
    steps=[dict(source=1,at=1.03,matrix=[[1,0,-220],[0,1,0],[0,0,1]],cut=False)]
    assert lock.update_controller(controller,MotionHistory(),steps,1.03,(1280,720)) is None
    a=controller.step([target(420),target(650)],1280,720,1.03,1.03,enabled=True,foreground=True,fire=True)
    assert a['target'] is not None and controller.locked.head[0]==420


def test_own_movement_is_not_compensated_twice_by_background_motion():
    controller=ZoneController(Settings(fire_policy='zone',head_filter='one_euro'))
    controller.step([target()],1280,720,1,1,enabled=True,foreground=True,fire=True)
    controller.acknowledge(100,0,1280,720)
    expected=controller.projected_locked.head[0]
    lock=CameraLock();lock.anchor(target(),1)
    steps=[dict(source=1,at=1.03,matrix=[[1,0,expected-640],[0,1,0],[0,0,1]],cut=False)]
    assert lock.update_controller(controller,MotionHistory(),steps,1.03,(1280,720)) is None
    assert controller.projected_locked.head[0]==expected
