"""Calibrated CPU HUD reader; no second capture, OCR service or game API.

Templates are local screenshot crops with provenance. Missing/ambiguous data
stays unknown. This intentionally cannot guess readiness from a reload timer.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from mvp_weapon_visual import WeaponVisual, normalized


def crop(image, rect):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = rect
    return image[round(y1*h):round(y2*h), round(x1*w):round(x2*w)]


def white_mask(image):
    if image.size == 0:
        return np.zeros((1, 1), np.uint8)
    if image.ndim == 2:
        return (image > 190).astype(np.uint8)*255
    # Calibrated white/blue HUD; exclude bright yellow background lettering.
    b,g,r=cv2.split(image.astype(np.int16))
    return ((image.max(axis=2)>190)&(b>=r*.85)).astype(np.uint8)*255


def dice(a, b):
    return 2*np.logical_and(a, b).sum()/max(1, a.sum()+b.sum())


class HudReader:
    def __init__(self, directory, interval=.08):
        self.directory = Path(directory)
        self.interval = interval
        self.at = None
        self.result = {}
        self.config = {}
        self.digits = []
        self.states = []
        self.status = 'hud_calibration_missing'
        path = self.directory/'manifest.json'
        if path.exists():
            self.config = json.loads(path.read_text(encoding='utf-8'))
            if self.config.get('schema_version') != 1:
                raise ValueError('Unsupported HUD calibration')
            if not self.config.get('provenance'):
                raise ValueError('HUD calibration needs screenshot provenance')
            for key in ('ammo_roi','state_roi'):
                rect = self.config[key]
                if len(rect) != 4 or not (0 <= rect[0] < rect[2] <= 1
                                          and 0 <= rect[1] < rect[3] <= 1):
                    raise ValueError(f'Invalid HUD rectangle: {key}')
            for digit in range(10):
                path=self.directory/f'digit-{digit}.png'
                image = cv2.imread(str(path), 0) if path.exists() else None
                if image is not None:
                    self.digits.append((digit, normalized(image)))
            for state in ('ready','reloading'):
                path=self.directory/f'{state}.png'
                image=cv2.imread(str(path),0) if path.exists() else None
                if image is not None:
                    self.states.append((state,cv2.resize(image,(64,32))>127))
            self.status = 'hud_ready' if len(self.digits) == 10 and len(self.states)==2 else 'hud_calibration_incomplete'
        self.icons = WeaponVisual(self.directory/'weapons', interval=0.)

    def read_ammo(self, image):
        if len(self.digits) != 10:
            return None
        mask = white_mask(crop(image, self.config['ammo_roi']))
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            return None
        columns = np.any(mask, axis=0)
        spans = []
        start = None
        for x, present in enumerate(np.r_[columns, False]):
            if present and start is None:
                start = x
            elif not present and start is not None:
                spans.append((start, x))
                start = None
        if not 1 <= len(spans) <= 2:
            return None
        value = 0
        for left, right in spans:
            sample = normalized(mask[:, left:right])
            scores = sorted(((dice(sample, t), digit) for digit, t in self.digits), reverse=True)
            if scores[0][0] < .88 or scores[0][0]-scores[1][0] < .08:
                return None
            value = value*10+scores[0][1]
        return value if value <= 30 else None

    def update(self, image, at):
        if self.at is not None and 0 <= at-self.at < self.interval:
            return dict(self.result)
        self.at = at
        icon = self.icons.update(image, at)
        weapon = icon['weapon_name'] if self.config else None
        ammo = self.read_ammo(image) if self.config and weapon else None
        reloading = None
        if self.config and weapon and len(self.states) == 2:
            mask=white_mask(crop(image,self.config['state_roi']))
            sample=cv2.resize(mask,(64,32))>127
            scores=sorted(((dice(sample,t),state) for state,t in self.states),reverse=True)
            if scores[0][0]>=.94 and scores[0][0]-scores[1][0]>=.08:
                reloading=scores[0][1]=='reloading'
        # AK has no optical zoom. Unknown weapons, including both sniper zoom
        # levels, are blocked rather than applying unverified sensitivity.
        zoom = 0 if weapon == 'weapon_ak47' else None
        self.result = dict(weapon_name=weapon, weapon_sample_at=at,
                           weapon_ammo=ammo, weapon_zoom=zoom,
                           weapon_reloading=reloading,
                           weapon_hud_status=self.status)
        return dict(self.result)
