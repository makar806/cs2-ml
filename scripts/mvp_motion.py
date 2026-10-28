"""Bounded mouse interpolation and camera history; no Windows input or ML imports."""

import math
from collections import deque

from mvp_core import Detection
from mvp_geometry import rotate_geometry


class MotionPlan:
    def __init__(self, duration=0.018, tick=0.004):
        self.duration = duration
        self.tick = tick
        self.cancel()

    def cancel(self):
        self.pending = None

    def begin(self, action, size, rect, captured_at, now, max_age):
        old = self.pending
        self.cancel()
        if action["target"] is None or action["reason"] == "await_render_after_snap":
            return
        dx, dy = action["dx"], action["dy"]

        glide = (
            action.get("tracking", False)
            and math.hypot(*action.get("error", (0, 0))) <= 80 * size[0] / 1280
        )
        duration = self.duration if glide and (dx or dy) else 0
        same_track = (
            old is not None
            and action.get("tracking", False)
            and old["size"] == size
            and old["rect"] == rect
            and now <= old["deadline"]
        )
        if duration and same_track:
            duration = min(duration, max(0, old["start"] + old["duration"] - now))
        shoot = action["shoot"] or (same_track and old["shoot"])
        self.pending = dict(
            dx=dx,
            dy=dy,
            shoot=shoot,
            sent=(0, 0),
            start=now,
            duration=duration,
            next_at=now if not duration else now + self.tick,
            size=size,
            rect=rect,
            source=captured_at,
            deadline=captured_at + max_age,
        )

    def poll(self, now):
        p = self.pending
        if p is None:
            return None
        if now > p["deadline"]:
            self.cancel()
            return None
        if now < p["next_at"]:
            return None
        fraction = min(1, max(0, (now - p["start"]) / p["duration"])) if p["duration"] else 1
        cumulative = (round(p["dx"] * fraction), round(p["dy"] * fraction))
        dx, dy = (cumulative[i] - p["sent"][i] for i in range(2))
        p["sent"] = cumulative
        p["next_at"] = now + self.tick
        done = fraction >= 1
        result = dict(
            dx=dx,
            dy=dy,
            shoot=bool(p["shoot"] and done),
            size=p["size"],
            rect=p["rect"],
            source=p["source"],
            done=done,
            glide=bool(p["duration"]),
        )
        if done:
            self.cancel()
        return result if dx or dy or result["shoot"] else None


class MotionHistory:
    """Rebase an in-flight image by commands issued AFTER its source timestamp.

    Without this, continuous input would either discard every model result or
    apply an old correction twice. Small-angle camera model matches Controller.
    """

    def __init__(self):
        self.events = deque()

    def clear(self):
        self.events.clear()

    def record(self, at, dx, dy, size, settings):
        if not (dx or dy):
            return
        self.events.append((at, (dx, dy), size, settings.sensitivity, settings.fov_4_3))
        while self.events and at - self.events[0][0] > 0.5:
            self.events.popleft()

    def rebase(self, detections, captured_at, size):
        commands = [
            (delta, sensitivity, fov)
            for at, delta, old_size, sensitivity, fov in self.events
            if at > captured_at and old_size == size
        ]
        width, height = size
        rebased = []
        for d in detections:
            box, head = d.box, d.head
            for (dx, dy), sensitivity, fov in commands:
                geometry = rotate_geometry(box, head, dx, dy, size, sensitivity, fov)
                if geometry is None:
                    break
                box, head = geometry
            else:
                hx, hy = head
                box = (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))
                if 0 <= hx < width and 0 <= hy < height and box[0] < box[2] and box[1] < box[3]:
                    rebased.append(Detection(d.cls, d.score, box, (hx, hy), d.head_score))
        return rebased
