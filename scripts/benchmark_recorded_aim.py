"""Closed-loop camera simulation with delayed/noisy detections, never Windows input."""

import argparse
import json
import math
from pathlib import Path

from mvp_core import Detection, Settings
from mvp_motion import MotionHistory
from mvp_recorded_aim import RecordedController, RecordedMotionPlan, load_profile

ROOT = Path(__file__).resolve().parents[1]


def run(
    profile, offset=200, speed=0, delay=0.03, hz=30, noise=0, duration=3, offset_y=0, speed_y=0, head_filter="current", fire_policy="verified", controller_type=None, turn_at=None
):
    if fire_policy=='zone':
        from mvp_zone_aim import ZoneController
        controller=(controller_type or ZoneController)(Settings(head_filter=head_filter,fire_policy='zone',shot_interval=.25),profile)
    else:controller = RecordedController(Settings(head_filter=head_filter), profile)
    motion = RecordedMotionPlan()
    history = MotionHistory()
    size = (1280, 720)
    rect = (0, 0, *size)
    camera = 0.0
    camera_pitch = 0.0

    def project(world_x, world_y):
        x = math.cos(camera) * world_x - math.sin(camera) * 480
        z = math.sin(camera) * world_x + math.cos(camera) * 480
        y = math.cos(camera_pitch) * world_y - math.sin(camera_pitch) * z
        z = math.sin(camera_pitch) * world_y + math.cos(camera_pitch) * z
        return 480 * x / max(0.01, z), 480 * y / max(0.01, z)

    pending = []
    next_capture = 1.0
    shots = []
    errors = []
    signed_errors = []
    corrections = []
    reasons = {}
    first_close=None
    close_since=None
    settled=None
    for index in range(int(duration * 1000) + 1):
        now = 1 + index * 0.001
        elapsed=now-1
        travel=elapsed if turn_at is None or elapsed<=turn_at else 2*turn_at-elapsed
        world = offset + speed * travel
        world_y = offset_y + speed_y * travel
        actual, actual_y = project(world, world_y)
        current_error=math.hypot(actual,actual_y)
        if current_error<=5:
            if first_close is None:first_close=elapsed
            if close_since is None:close_since=elapsed
            if settled is None and elapsed-close_since>=.2:settled=close_since
        else:close_since=None
        signed_errors.append(actual)
        if now + 1e-9 >= next_capture:
            head = 640 + actual + noise * math.sin(index * 0.57)
            head_y = 360 + actual_y
            if 22 < head < 1258 and 12 < head_y < 628:
                d = Detection(
                    0, 0.99, (head - 20, head_y - 10, head + 20, head_y + 90), (head, head_y), 0.99
                )
                pending.append((now + delay, now, d))
            next_capture += 1 / hz
        if pending and pending[0][0] <= now + 1e-9:
            _, source, detection = pending.pop(0)
            detections = history.rebase([detection], source, size)
            action = controller.step(
                detections, *size, now, source, enabled=True, foreground=True, fire=True
            )
            reasons[action["reason"]] = reasons.get(action["reason"], 0) + 1
            motion.begin(action, size, rect, source, now, controller.settings.max_age)
        if fire_policy=='zone' and controller.zone_fire_ready(now):
            motion.cancel()
            event=dict(dx=0,dy=0,shoot=True,source=controller.zone['source'])
        else:event = motion.poll(now)
        if event is not None:
            if event["shoot"] and not controller.can_emit_shot(event["source"], now):
                continue
            dx, dy = event["dx"], event["dy"]
            camera += math.radians(dx * 0.022)
            camera_pitch += math.radians(dy * 0.022)
            if dx or dy:
                controller.acknowledge(dx, dy, *size)
                controller.movement_sent(now, dx, dy)
                history.record(now + 1e-7, dx, dy, size, controller.settings)
                corrections.append(dict(t=now - 1, dx=dx, dy=dy))
            if event["shoot"]:
                error = math.hypot(*project(world, world_y))
                shots.append(dict(at=now - 1, actual_error_px=error))
                controller.shot_emitted(now)
        if now > 1.7:
            errors.append(math.hypot(actual, actual_y))
    ordered=sorted(errors)
    def percentile(p):
        at=(len(ordered)-1)*p;lo=math.floor(at);hi=math.ceil(at)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)
    return dict(
        min_signed_error_x=min(signed_errors),
        max_signed_error_x=max(signed_errors),
        offset=offset,
        speed=speed,
        delay=delay,
        hz=hz,
        noise=noise,
        turn_at=turn_at,
        first_within_5px_seconds=first_close,
        settled_within_5px_for_200ms_seconds=settled,
        tracking_error_p50_px=percentile(.5),tracking_error_p95_px=percentile(.95),
        tracking_error_p99_px=percentile(.99),
        offset_y=offset_y,
        speed_y=speed_y,
        shots=shots,
        mean_tracking_error_px=sum(errors) / len(errors),
        max_input_counts=max((abs(x["dx"]) for x in corrections), default=0),
        movement_events=len(corrections),
        reasons=reasons,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default=str(ROOT / "configs/aim-profile.json"))
    args = p.parse_args()
    profile = load_profile(args.profile)
    report = dict(
        note="Synthetic geometry and controlled detections, not CS2 hit rate, detector quality or frametime measurement",
        cases=[
            run(profile, offset=offset, speed=speed, delay=delay, noise=0.5)
            for offset in (-250, 80, 300)
            for speed in (0, 60, 180)
            for delay in (0.03, 0.07)
        ],
    )
    output = ROOT / "runs/recorded-aim/benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
