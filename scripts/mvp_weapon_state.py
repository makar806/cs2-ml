"""Weapon authorization uses observations from the existing video frame only."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WeaponProfile:
    name: str
    aim_point: str
    tap_interval: float
    recoil: str | None = None


# Only the supplied AK profile has a traceable source. Recognizing another
# icon does NOT authorize applying AK counts to it. Extend after calibration.
PROFILES = {
    'weapon_ak47': WeaponProfile('weapon_ak47', 'body', .25, 'ak47'),
}


class WeaponState:
    max_age = .35

    def __init__(self):
        self.sample = {}
        self.generation = 0

    def update(self, sample):
        previous = self.sample.get('weapon_name')
        source = sample.get('weapon_sample_at')
        old_source = self.sample.get('weapon_sample_at')
        if (source is None or not isinstance(source, (int, float))
                or not math.isfinite(source)):
            self.sample = {}
        elif old_source is None or source >= old_source:
            self.sample = dict(sample)
        if self.sample.get('weapon_name') != previous:
            self.generation += 1
            return True
        return False

    def profile(self, now):
        at = self.sample.get('weapon_sample_at')
        if at is None or not 0 <= now-at <= self.max_age:
            return None
        return PROFILES.get(self.sample.get('weapon_name'))

    def reason(self, now, *, spray=False):
        at = self.sample.get('weapon_sample_at')
        if at is None or not 0 <= now-at <= self.max_age:
            return 'weapon_stale'
        profile = self.profile(now)
        if profile is None:
            return 'weapon_unknown_or_unsupported'
        if self.sample.get('weapon_zoom') != 0:
            return 'zoom_unverified'
        if self.sample.get('weapon_reloading') is not False:
            return 'reload_unknown_or_active'
        ammo = self.sample.get('weapon_ammo')
        if type(ammo) is not int or not 0 <= ammo <= 30:
            return 'ammo_unknown'
        if ammo == 0:
            return 'magazine_empty'
        if spray and profile.recoil is None:
            return 'recoil_profile_unavailable'
        return None
