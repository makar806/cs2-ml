import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mvp_core import Detection,Settings,aim_detection
from mvp_recorded_aim import RecordedController

def test_head_mode_preserves_original_keypoint():
    d=Detection(0,.95,(600,300,680,500),(625,310),.7)
    assert aim_detection(d,'head') is d

def test_body_mode_uses_box_and_does_not_require_detected_head():
    d=Detection(0,.95,(600,300,680,500),(0,0),0)
    torso=aim_detection(d,'body')
    assert torso.head==(640,370)
    assert torso.head_score==.95
    c=RecordedController(settings=Settings(aim_point='body'))
    a=c.step([torso],1280,720,1,1,enabled=True,foreground=True,fire=False)
    assert a['target'] is not None
    assert a['dy']>0


def test_body_followup_budget_reopens_but_head_budget_does_not():
    for mode,expected in [('body',False),('head',True)]:
        c=RecordedController(settings=Settings(aim_point=mode))
        d=aim_detection(Detection(0,.95,(600,300,680,500),(640,310),.95),mode)
        c.step([d],1280,720,1,1,enabled=True,foreground=True,fire=False)
        c.shot_emitted(1.1)
        c.step([d],1280,720,1.6,1.6,enabled=True,foreground=True,fire=False)
        assert c.shot_sent is expected
