"""Time-based AK-47 profile for the controlled local experiment."""
import csv
from pathlib import Path
from bisect import bisect_right


class AKSpray:
    max_duration = 3.05
    crouch_at = 1.0
    target_grace = .20
    cooldown = .45
    reload_wait = 2.5

    def __init__(self, sensitivity):
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
        self.manual = False
        self.crouched = False
        self.ready_at = 0.0
        self.manual_pressed = False
        self.reload_target = None
        self.reload_until = 0.

    def start(self, now, manual, target):
        self.started = now
        self.last_visible = now
        self.manual = manual
        self.target = target
        self.sent = (0, 0)
        self.follow_at = 0.
        self.follow_remainder = 0.

    def stop(self, now):
        if self.started is not None:
            self.ready_at = now + self.cooldown
        self.started = None

    def correction(self, elapsed):
        index = min(bisect_right(self.times, elapsed)-1, len(self.times)-2)
        a, b = self.times[index:index+2]
        fraction = min(1.0, max(0.0, (elapsed-a)/(b-a)))
        point = tuple(round(x + (y-x)*fraction)
                      for x, y in zip(self.points[index], self.points[index+1]))
        delta = tuple(x-y for x, y in zip(point, self.sent))
        self.sent = point
        return delta

    def follow_horizontal(self, elapsed, counts_per_second):
        dt = min(.02, max(0., elapsed-self.follow_at))
        self.follow_at = elapsed
        counts = max(-950., min(950., counts_per_second))*dt + self.follow_remainder
        step = round(counts)
        self.follow_remainder = counts-step
        return step
