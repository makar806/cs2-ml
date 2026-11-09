import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mvp_weapon_policy import active_weapon_point

def test_snipers_body_and_pistol_head():
    for name in ('awp','ssg08','scar20','g3sg1'):
        assert active_weapon_point({'a':{'name':'weapon_'+name,'state':'active'}},.1)=='body'
    assert active_weapon_point({'a':{'name':'weapon_deagle','state':'active'}},.1)=='head'

def test_held_primary_is_not_active_pistol():
    assert active_weapon_point({'a':{'name':'weapon_awp','state':'holstered'},'b':{'name':'weapon_glock','state':'active'}},.1)=='head'

def test_missing_stale_and_nonfirearm_do_not_guess():
    assert active_weapon_point({},.1) is None
    assert active_weapon_point({'a':{'name':'weapon_awp','state':'active'}},.6) is None
    assert active_weapon_point({'a':{'name':'weapon_knife','state':'active'}},.1) is None
