import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mvp_stop import StopBeforeShot

def test_release_wait_and_new_movement_restarts_settle():
    gate=StopBeforeShot();released=[]
    assert not gate.ready(1,[87],released.append)
    assert released==[[87]]
    assert not gate.ready(1.1,[],released.append)
    assert gate.ready(1.19,[],released.append)
    assert not gate.ready(1.2,[65],released.append)
    assert not gate.ready(1.3,[],released.append)

def test_stationary_has_no_artificial_delay():
    assert StopBeforeShot().ready(1,[],lambda keys:None)
