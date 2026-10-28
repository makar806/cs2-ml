"""Deterministic multi-rate synthetic camera loop; does not control Windows."""

import json
import math
from pathlib import Path

from mvp_core import Controller, Detection
from mvp_motion import MotionHistory, MotionPlan


def run(hz=45, speed=180, smooth=True):
    c = Controller()
    p = MotionPlan(duration=0.018 if smooth else 0)
    h = MotionHistory()
    size = (1280, 720)
    rect = (0, 0, 1280, 720)
    camera = 0.0
    queue = []
    next_capture = 1.0
    last_snap = 0.0
    errors = []
    increments = []
    shots = 0
    for i in range(3001):
        now = 1 + i * 0.001
        world = 80 + speed * (now - 1)

        head = 640 + 480 * math.tan(math.atan2(world, 480) - camera)
        if now + 1e-9 >= next_capture:
            queue.append(
                (
                    now + 0.030,
                    now,
                    Detection(0, 0.99, (head - 20, 350, head + 20, 450), (head, 360), 0.99),
                )
            )
            next_capture += 1 / hz
        if queue and queue[0][0] <= now + 1e-9:
            _, source, d = queue.pop(0)
            if source > last_snap:
                ds = h.rebase([d], source, size)
                a = c.step(ds, *size, now, source, enabled=True, foreground=True, fire=True)
                p.begin(a, size, rect, source, now, c.settings.max_age)
        event = p.poll(now)
        if event:
            dx, dy = event["dx"], event["dy"]
            camera += math.radians(dx * 0.022)

            c.acknowledge(dx, dy, *size)
            h.record(now + 1e-7, dx, dy, size, c.settings)
            if not event["glide"] and (dx or dy):
                last_snap = now
            if event["shoot"]:
                shots += 1
                c.last_shot = now
            if now > 1.3:
                increments.append(abs(dx))
        if now > 1.3:
            errors.append(abs(480 * math.tan(math.atan2(world, 480) - camera)))
    return dict(
        hz=hz,
        speed_px_s=speed,
        smooth=smooth,
        shots=shots,
        mean_error_px=sum(errors) / len(errors),
        max_error_px=max(errors),
        largest_input_counts=max(increments, default=0),
        input_events=len(increments),
    )


if __name__ == "__main__":
    results = [
        run(hz, speed, smooth)
        for hz in (30, 45, 60)
        for speed in (0, 180, -180)
        for smooth in (False, True)
    ]
    output = Path(__file__).resolve().parents[1] / "runs/synthetic-tracking"
    output.mkdir(parents=True, exist_ok=True)
    (output / "synthetic-tracking.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
