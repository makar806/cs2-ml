from pathlib import Path
import io
import json
import sys
import threading
import pytest
import numpy as np
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mvp_logging import AsyncLog
from mvp_hud import HudReader
from mvp_weapon_state import WeaponState
from mvp_wgc import WindowCapture
from mvp_capture_session import CaptureSession


def test_slow_diagnostic_sink_cannot_block_control_and_loss_is_reported(tmp_path):
    entered=threading.Event()
    release=threading.Event()
    class Slow(io.StringIO):
        def write(self,text):
            entered.set()
            release.wait(1)
            return super().write(text)
    log=AsyncLog(tmp_path/'log',capacity=2,opener=Slow)
    log.write('first\n')
    assert entered.wait(1)
    for i in range(100):
        log.write(f'{i}\n')
    assert log.stats()['pending']==2 and log.stats()['dropped']==98
    release.set()
    log.close()
    assert not log.stats()['error']


def test_normal_log_close_flushes_all_records(tmp_path):
    path=tmp_path/'log.jsonl'
    log=AsyncLog(path)
    for i in range(100):
        log.write(json.dumps({'i':i})+'\n')
    log.close()
    assert [json.loads(x)['i'] for x in path.read_text().splitlines()]==list(range(100))
    assert log.stats()==dict(pending=0,dropped=0,error='')


def test_failed_capture_close_does_not_allow_second_resource():
    class Busy:
        def close(self):
            raise TimeoutError('blocked grab')
    calls=[]
    session=CaptureSession(lambda h:calls.append(h) or Busy())
    first=session.ensure()
    with pytest.raises(TimeoutError):
        session.close()
    assert session.ensure() is first and len(calls)==1


def test_wgc_frame_owns_pixels_even_for_contiguous_rgb():
    capture=WindowCapture.__new__(WindowCapture)
    pixels=np.zeros((100,200,3),np.uint8)
    result=capture._client(pixels,(0,0,200,100))
    pixels[:]=255
    assert not result.any() and not np.shares_memory(result,pixels)


def test_missing_hud_does_not_guess_weapon_or_readiness(tmp_path):
    reader=HudReader(tmp_path)
    frame=np.zeros((720,1280,3),np.uint8)
    sample=reader.update(frame,1.)
    assert sample['weapon_name'] is None and sample['weapon_ammo'] is None
    state=WeaponState()
    state.update(sample)
    assert state.reason(1.)=='weapon_unknown_or_unsupported'


def test_hud_uses_existing_frame_and_throttles_without_changing_source_age(tmp_path,monkeypatch):
    reader=HudReader(tmp_path)
    calls=[]
    monkeypatch.setattr(reader.icons,'update',lambda image,at:calls.append((id(image),at)) or {'weapon_name':None})
    frame=np.zeros((720,1280,3),np.uint8)
    first=reader.update(frame,1.)
    second=reader.update(frame,1.02)
    assert calls==[(id(frame),1.)] and second==first
    reader.update(frame,1.10)
    assert len(calls)==2


def test_weapon_switch_and_backwards_packet_cannot_restore_old_profile():
    state=WeaponState()
    sample=dict(weapon_name='weapon_ak47',weapon_sample_at=1.,weapon_ammo=30,
                weapon_reloading=False,weapon_zoom=0)
    assert state.update(sample)
    assert state.reason(1.) is None
    assert state.update(dict(sample,weapon_name='weapon_knife',weapon_sample_at=1.1))
    assert not state.update(sample)
    assert state.profile(1.1) is None
    assert state.reason(1.6)=='weapon_stale'


def synthetic_hud(directory):
    directory.mkdir()
    icons=directory/'weapons'
    icons.mkdir()
    frame=np.zeros((720,1280,3),np.uint8)
    shape=np.array([[0,3],[70,3],[79,7],[60,10],[45,10],[40,20],[33,18],[35,10],[15,10],[10,14],[6,13],[7,9],[0,9]])
    cv2.fillPoly(frame,[shape+(1160,500)],(255,255,255))
    cv2.putText(frame,'AK',(1200,537),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    cv2.imwrite(str(icons/'ak47.png'),frame[500:522,1160:1240].max(axis=2))
    wrong=np.zeros((22,80),np.uint8)
    cv2.rectangle(wrong,(1,1),(50,8),255,-1)
    cv2.rectangle(wrong,(40,1),(55,21),255,-1)
    cv2.imwrite(str(icons/'unknown_pistol.png'),wrong)
    for digit in range(10):
        glyph=np.zeros((32,20),np.uint8)
        cv2.putText(glyph,str(digit),(1,27),cv2.FONT_HERSHEY_SIMPLEX,.8,255,2)
        cv2.imwrite(str(directory/f'digit-{digit}.png'),glyph)
        if digit in (3,0):
            x=950 if digit==3 else 970
            frame[680:712,x:x+20]=glyph[:,:,None]
    ready=np.zeros((32,64),np.uint8)
    reload=np.zeros_like(ready)
    cv2.rectangle(ready,(4,4),(60,12),255,-1)
    cv2.line(reload,(5,25),(60,4),255,5)
    for name,image in (('ready',ready),('reloading',reload)):
        cv2.imwrite(str(directory/f'{name}.png'),image)
    frame[680:712,1050:1114]=ready[:,:,None]
    config=dict(schema_version=1,provenance=['synthetic test only'],
                ammo_roi=[950/1280,680/720,990/1280,712/720],
                state_roi=[1050/1280,680/720,1114/1280,712/720])
    (directory/'manifest.json').write_text(json.dumps(config))
    return frame,reload


def test_same_frame_hud_to_actual_engine_gates_reload_and_hidden_hud(monkeypatch,tmp_path):
    from test_control_v332 import rig
    from mvp_core import Detection
    e=rig(monkeypatch,tmp_path)
    frame,reload=synthetic_hud(tmp_path/'hud')
    reader=HudReader(tmp_path/'hud')
    assert reader.update(frame,1.)['weapon_name'] is None
    signal=reader.update(frame,1.1)
    assert signal['weapon_name']=='weapon_ak47' and signal['weapon_ammo']==30
    assert signal['weapon_reloading'] is False
    e.clock[0]=1.1
    e.update_weapon(signal,1.1)
    d=Detection(0,.99,(620,317,660,417),(640,327),.99)
    e.controller.step([d],1280,720,1.1,1.1,enabled=True,foreground=True,fire=True)
    e.zone_context=dict(size=(1280,720),rect=(0,0,1280,720))
    e.keys.add(5)
    e.send_pending(e.log,1)
    assert len(e.mouse.holds)==1
    frame[680:712,1050:1114]=reload[:,:,None]
    e.clock[0]=1.2
    signal=reader.update(frame,1.2)
    assert signal['weapon_reloading'] is True
    e.update_weapon(signal,1.2)
    e.send_pending(e.log,2)
    assert e.mouse.released and e.spray.awaiting_reload
    e.clock[0]=1.3
    e.update_weapon(reader.update(np.zeros_like(frame),1.3),1.3)
    assert not e.spray.enabled and e.controller.locked is None
