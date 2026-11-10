"""Time-based AK-47 profile for the controlled local experiment."""
import csv
import math
from pathlib import Path
from bisect import bisect_right


class AKSpray:
    max_duration = 3.05
    crouch_at = 1.0
    target_grace = .20
    cooldown = .45

    def __init__(self, sensitivity, crouch_scale=.75):
        if not .01 <= sensitivity <= 20 or not .25 <= crouch_scale <= 1.5:
            raise ValueError('Invalid sensitivity or crouch recoil scale')
        # Artanis-RCS, commit 7387d399. MIT notice in configs/recoil.
        # CSV uses relative counts, inverted vertical sign, and delays in ms.
        path = Path(__file__).resolve().parents[1] / 'configs/recoil/ak47.csv'
        self.times = [0.0]
        self.points = [(0.0, 0.0)]
        scale = 2.45 / sensitivity
        with path.open(encoding='utf-8-sig', newline='') as source:
            for row in csv.reader(source):
                if not row:
                    continue
                dx, dy, ms = map(float, row)
                x, y = self.points[-1]
                self.points.append((x + dx*scale, y - dy*scale))
                self.times.append(self.times[-1] + ms/1000)
        self.started = None
        self.enabled = False
        self.manual = False
        self.crouched = False
        self.ready_at = 0.0
        self.manual_pressed = False
        self.awaiting_reload = False
        self.reload_seen = False
        self.last_ammo = None
        self.crouch_scale = crouch_scale

    def start(self, now, manual, target):
        if not manual:
            raise ValueError('Spray requires an explicit Mouse 4 toggle')
        self.started = now
        self.last_visible = now
        self.manual = manual
        self.target = target
        self.sent = (0, 0)
        self.raw_point = (0., 0.)
        self.integral = (0., 0.)
        self.elapsed = 0.
        self.stance_scale = 1.
        self.follow_at = 0.
        self.follow_remainder = (0., 0.)
        self.wall_mode = target is None

    def stop(self, now):
        if self.started is not None:
            self.ready_at = now + self.cooldown
        self.started = None

    def correction(self, elapsed, crouched=False):
        elapsed = max(self.elapsed, min(elapsed, self.max_duration))
        index = min(bisect_right(self.times, elapsed)-1, len(self.times)-2)
        a, b = self.times[index:index+2]
        fraction = min(1.0, max(0.0, (elapsed-a)/(b-a)))
        raw = tuple(x + (y-x)*fraction
                      for x, y in zip(self.points[index], self.points[index+1]))
        dt = elapsed-self.elapsed
        desired = self.crouch_scale if crouched else 1.
        # Scale INCREMENTS, not the cumulative offset: changing stance cannot
        # replay/reset a second of recoil. .75 is a reversible hypothesis.
        self.stance_scale += (desired-self.stance_scale)*(1-math.exp(-dt/.12))
        self.integral = tuple(total+(new-old)*self.stance_scale
                              for total, new, old in zip(self.integral, raw, self.raw_point))
        self.raw_point = raw
        self.elapsed = elapsed
        point = tuple(round(x) for x in self.integral)
        delta = tuple(x-y for x, y in zip(point, self.sent))
        self.sent = point
        return delta

    def follow(self, elapsed, center, velocity, size, settings):
        """Bounded body feedback plus feed-forward, merged with recoil by Engine."""
        dt = min(.02, max(0., elapsed-self.follow_at))
        self.follow_at = elapsed
        width, height = size
        focal = height/(2*math.tan(math.radians(settings.fov_4_3)/2)*.75)
        factor = math.degrees(1/focal)/(settings.sensitivity*.022)
        errors = (center[0]-width/2, center[1]-height/2)
        steps, remainders = [], []
        for error, speed, remainder in zip(errors, velocity, self.follow_remainder):
            error = 0. if abs(error) < width/1280 else error
            rate = (error/.18 + speed)*factor
            counts = max(-950., min(950., rate))*dt+remainder
            step = round(counts)
            steps.append(step)
            remainders.append(counts-step)
        self.follow_remainder = tuple(remainders)
        return tuple(steps)
