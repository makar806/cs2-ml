import copy
import json
from pathlib import Path
import sys
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mvp_gsi import KillCounter,KillReceiver,config_text


def payload(kills=0):
    return dict(provider=dict(appid=730,steamid='self'),map=dict(name='local-bots'),
                player=dict(steamid='self',activity='playing',match_stats=dict(kills=kills)))


def test_kill_counter_initial_duplicate_reset_and_map_change():
    c=KillCounter();assert c.update(payload(50));assert c.snapshot()['sequence']==0
    c.update(payload(51));assert c.snapshot()['sequence']==1
    c.update(payload(51));c.update(payload(0));assert c.snapshot()['sequence']==1
    p=payload(70);p['map']['name']='other';c.update(p);assert c.snapshot()['sequence']==1
    p['player']['match_stats']['kills']=71;c.update(p);assert c.snapshot()['sequence']==2


@pytest.mark.parametrize('change',[
    lambda p:p['player'].update(steamid='spectated'),
    lambda p:p['provider'].update(appid=123),
    lambda p:p['player'].update(activity='menu'),
    lambda p:p['player'].update(match_stats=dict(kills=True)),
    lambda p:p.update(player=[]),
])
def test_nonself_or_invalid_payload_cannot_release_target(change):
    c=KillCounter();c.update(payload(0));p=payload(99);change(p)
    assert not c.update(p) and c.snapshot()['sequence']==0


def test_loopback_receiver_rejects_bad_auth_and_accepts_real_counter():
    token='a'*48;r=KillReceiver(token,port=0)
    url=f'http://127.0.0.1:{r.server.server_port}/'
    try:
        p=payload();p['auth']=dict(token='wrong')
        with pytest.raises(HTTPError) as error:urlopen(Request(url,data=json.dumps(p).encode()),timeout=2)
        assert error.value.code==403 and r.snapshot()['kills'] is None
        for kills in [0,1]:
            p=payload(kills);p['auth']=dict(token=token)
            with urlopen(Request(url,data=json.dumps(p).encode()),timeout=2) as response:assert response.status==200
        assert r.snapshot()['sequence']==1
    finally:r.close()


def test_config_is_local_and_does_not_subscribe_to_player_positions():
    text=config_text('a'*48)
    assert 'http://127.0.0.1:8765/' in text
    assert 'allplayers' not in text and 'position' not in text
