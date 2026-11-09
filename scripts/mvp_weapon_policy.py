"""Aim-point policy; None means no fresh, unambiguous active firearm."""
SNIPERS=frozenset(('weapon_awp','weapon_ssg08','weapon_scar20','weapon_g3sg1'))
FIREARMS=frozenset(('weapon_ak47','weapon_aug','weapon_famas','weapon_galilar','weapon_m4a1','weapon_m4a1_silencer','weapon_sg556','weapon_mac10','weapon_mp5sd','weapon_mp7','weapon_mp9','weapon_p90','weapon_bizon','weapon_ump45','weapon_mag7','weapon_nova','weapon_sawedoff','weapon_xm1014','weapon_m249','weapon_negev','weapon_deagle','weapon_elite','weapon_fiveseven','weapon_glock','weapon_hkp2000','weapon_p250','weapon_tec9','weapon_usp_silencer','weapon_cz75a','weapon_revolver'))

def aim_point_for_weapon(name):
    if name in SNIPERS:return 'body'
    if name in FIREARMS:return 'head'
    return None

def active_weapon_point(weapons,age):
    if age is None or not 0<=age<=.5 or not isinstance(weapons,dict):return None
    active=[w for w in weapons.values() if isinstance(w,dict) and w.get('state')=='active']
    if len(active)!=1:return None
    name=active[0].get('name')
    return aim_point_for_weapon(name) if isinstance(name,str) else None
