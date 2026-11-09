"""Calibrated smooth aiming and a separate one-shot authorization gate."""

import json
import math
from pathlib import Path
from collections import deque
from mvp_geometry import rotate_geometry

from mvp_core import Controller, Detection, box_iou
from mvp_motion import MotionPlan
from mvp_one_euro import OneEuro


def minimum_jerk(u):
    u = min(1.0, max(0.0, u))
    return u**3 * (10 - 15 * u + 6 * u**2)


def load_profile(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported aim profile")
    for key in ("duration_seconds", "reference_angle_degrees"):
        value = profile[key]
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid profile value: {key}")
    if not 0.12 <= profile["duration_seconds"] <= 0.9:
        raise ValueError("Recorded movement duration must be 0.12–0.9 seconds")
    return profile


def visual_kill_matches(controller, event_at, now):
    """New own-kill feed after acquisition, including manual finishing shots.

    The visual feed has no world victim ID. This remains a local-bot heuristic;
    sequence deduplication and active/focus checks are owned by Engine.
    """
    acquired = getattr(controller, "acquired_at", None)
    baseline = (
        acquired if acquired is not None and controller.settings.fire_policy == "zone" else
        max(acquired, controller.last_shot)
        if acquired is not None
        else controller.last_shot
    )
    return (
        event_at is not None
        and 0 <= now - event_at <= 1.5
        and math.isfinite(baseline)
        and baseline <= event_at
        and controller.locked is not None
    )


class HeadFilter:
    """Three source observations in the current camera coordinate system."""

    def __init__(self):
        self.samples = deque(maxlen=3)
        self.velocity = (0.0, 0.0)
        self.uncertainty = 0.0

    def clear(self):
        self.samples.clear()
        self.velocity = (0.0, 0.0)
        self.uncertainty = 0.0

    def translate(self, sx, sy):
        self.samples = deque(((t,(p[0]+sx,p[1]+sy)) for t,p in self.samples),maxlen=3)

    def acknowledge(self, dx, dy, size, settings):
        transformed = deque(maxlen=3)
        for at, point in self.samples:
            geometry = rotate_geometry(
                (point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1),
                point,
                dx,
                dy,
                size,
                settings.sensitivity,
                settings.fov_4_3,
            )
            if geometry is not None:
                transformed.append((at, geometry[1]))
        self.samples = transformed

    def observe(self, point, at, radius):
        if self.samples and (
            at <= self.samples[-1][0] or at - self.samples[-1][0] > 0.18
        ):
            self.clear()
        self.samples.append((at, point))
        if len(self.samples) < 3:
            self.uncertainty = 0.0
            return point, self.velocity
        first, middle, last = self.samples
        dt = last[0] - first[0]
        velocity = []
        for axis in (0, 1):
            a = middle[1][axis] - first[1][axis]
            b = last[1][axis] - middle[1][axis]
            # Opposite small displacements are head-keypoint noise, not a turn.
            coherent = a * b > 0 and abs(a) + abs(b) > radius * 0.35
            measured = (last[1][axis] - first[1][axis]) / dt if coherent else 0.0
            velocity.append(self.velocity[axis] * 0.35 + measured * 0.65)
        self.velocity = tuple(velocity)
        weights = (0.2, 0.3, 0.5)
        estimate = tuple(
            sum(
                w * (p[axis] + self.velocity[axis] * (at - t))
                for w, (t, p) in zip(weights, self.samples)
            )
            for axis in (0, 1)
        )
        residuals = [
            math.dist(
                tuple(p[i] + self.velocity[i] * (at - t) for i in (0, 1)), estimate
            )
            for t, p in self.samples
        ]
        self.uncertainty = max(residuals)
        # Never pull the estimated location more than half a head radius off observation.
        delta = tuple(estimate[i] - point[i] for i in (0, 1))
        gain = min(1, radius * 0.5 / max(1e-9, math.hypot(*delta)))
        return tuple(point[i] + delta[i] * gain for i in (0, 1)), self.velocity


class OneEuroHeadFilter(HeadFilter):
    """1-Euro position smoothing; shared velocity/residual checks, no second smoother."""
    parameters = dict(min_cutoff=0.5, beta=0.3, derivative_cutoff=1.0)

    def __init__(self):
        super().__init__()
        self.axes = [OneEuro(**self.parameters), OneEuro(**self.parameters)]

    def clear(self):
        super().clear()
        for axis in self.axes:
            axis.reset()

    def translate(self, sx, sy):
        super().translate(sx, sy)
        for axis,delta in zip(self.axes,(sx,sy)):
            if axis.at is not None:
                axis.raw += delta
                axis.value += delta

    def acknowledge(self, dx, dy, size, settings):
        def project(point):
            x,y=point
            geometry=rotate_geometry((x-1,y-1,x+1,y+1),point,dx,dy,size,settings.sensitivity,settings.fov_4_3)
            return None if geometry is None else geometry[1]
        if self.axes[0].at is not None:
            raw=project(tuple(a.raw for a in self.axes))
            value=project(tuple(a.value for a in self.axes))
            tangent=project(tuple(a.value+a.derivative*.001 for a in self.axes))
            if raw is None or value is None or tangent is None:
                self.clear()
                return
            for i,axis in enumerate(self.axes):
                axis.raw=raw[i];axis.value=value[i]
                axis.derivative=(tangent[i]-value[i])/.001
        super().acknowledge(dx,dy,size,settings)

    def observe(self, point, at, radius):
        # Base method maintains the same motion estimate and firing uncertainty.
        _,velocity=super().observe(point,at,radius)
        filtered=tuple(axis(point[i],at) for i,axis in enumerate(self.axes))
        return filtered,velocity


def make_head_filter(mode):
    if mode == 'current':return HeadFilter()
    if mode == 'one_euro':return OneEuroHeadFilter()
    raise ValueError('Unknown head filter: '+str(mode))


def scene_shift(old, new, held, scale):
    """Estimate uncommanded camera translation from two independent bodies.

    Never infer a camera movement from the held head alone. Repeated/ambiguous
    layouts fail closed instead of attaching the lock to the nearest character.
    """

    def unique(items):
        result = []
        for d in items:
            if not any(box_iou(d.box, x.box) > 0.75 for x in result):
                result.append(d)
        return result[:10]

    old = [d for d in unique(old) if held is None or box_iou(d.box, held.box) < 0.3]
    new = unique(new)
    if len(old) < 2 or len(new) < 2:
        return None
    proposals = []
    for a in old:
        for b in new:
            ratio = (b.box[3] - b.box[1]) / (a.box[3] - a.box[1])
            if not 0.85 <= ratio <= 1.18:
                continue
            dx, dy = b.head[0] - a.head[0], b.head[1] - a.head[1]
            if abs(dx) > 320 * scale or abs(dy) > 180 * scale:
                continue
            used = set()
            matches = []
            for anchor in old:
                errors = [
                    (
                        math.hypot(
                            d.head[0] - anchor.head[0] - dx,
                            d.head[1] - anchor.head[1] - dy,
                        ),
                        i,
                        d,
                    )
                    for i, d in enumerate(new)
                    if i not in used
                    and 0.85
                    <= (d.box[3] - d.box[1]) / (anchor.box[3] - anchor.box[1])
                    <= 1.18
                ]
                if errors:
                    error, i, d = min(errors, key=lambda x: x[0])
                    if error <= 8 * scale:
                        used.add(i)
                        matches.append(
                            (d.head[0] - anchor.head[0], d.head[1] - anchor.head[1])
                        )
            if len(matches) >= 2:
                mean = tuple(sum(v[i] for v in matches) / len(matches) for i in (0, 1))
                if not any(math.dist(mean, p[1]) < 8 * scale for p in proposals):
                    proposals.append((len(matches), mean))
    proposals.sort(key=lambda p: p[0], reverse=True)
    if not proposals or (len(proposals) > 1 and proposals[0][0] == proposals[1][0]):
        return None
    return proposals[0][1]


class RecordedController(Controller):
    def __init__(self, settings=None, profile=None):
        self.profile = profile or dict(duration_seconds=0.3, reference_angle_degrees=30)
        super().__init__(settings)

    def reset(self):
        super().reset()
        self.course = None
        self.acquired_at = None
        self.head_filter = make_head_filter(self.settings.head_filter)
        self.in_deadzone = False
        self.anchors = []
        self.last_observation = -math.inf
        self.last_observation_at = None
        self.last_movement = -math.inf
        self.verified_samples = 0
        self.verification_source = None
        self.shot_sent = False
        self.filtered_velocity = (0.0, 0.0)
        self.fine_tracking = False
        self.micro_limit_degrees = 0.12

    def set_head_filter(self, mode):
        replacement=make_head_filter(mode)
        self.settings.head_filter=mode
        self.head_filter=replacement
        self.course=None
        self.in_deadzone=False
        self.filtered_velocity=(0.0,0.0)
        self.verified_samples=0
        self.verification_source=None

    def movement_sent(self, at, dx=None, dy=None):
        self.verification_source = None
        small_follow = (
            self.fine_tracking
            and dx is not None
            and dy is not None
            and math.hypot(dx, dy) * self.settings.sensitivity * 0.022
            <= self.micro_limit_degrees
        )
        if not small_follow:
            self.last_movement = at
            self.verified_samples = 0
            self.verification_source = None

    def shot_emitted(self, at):
        self.shot_sent = True
        self.last_shot = at
        self.verification_source = None
        self.verified_samples = 0

    def can_emit_shot(self, source, now):
        return (
            not self.shot_sent
            and self.verification_source == source
            and source >= self.last_movement + 0.008
            and 0 <= now - source <= min(self.settings.max_age, 0.075)
        )

    def acknowledge(self, dx, dy, width, height):
        super().acknowledge(dx, dy, width, height)
        self.head_filter.acknowledge(dx, dy, (width, height), self.settings)
        projected = []
        for d in self.anchors:
            geometry = rotate_geometry(
                d.box,
                d.head,
                dx,
                dy,
                (width, height),
                self.settings.sensitivity,
                self.settings.fov_4_3,
            )
            if geometry is not None:
                projected.append(Detection(d.cls, d.score, *geometry, d.head_score))
        self.anchors = projected

    def release_after_confirmed_kill(self, now):
        return super().release_after_confirmed_kill(now)

    def step(
        self, detections, width, height, now, captured_at, *, enabled, foreground, fire
    ):
        if (self.settings.aim_point == "body" and self.shot_sent
                and now-self.last_shot >= self.settings.shot_interval):
            self.shot_sent=False
            self.verified_samples=0
            self.verification_source=None
        if enabled and foreground and captured_at <= self.last_observation:
            self.verified_samples = 0
            self.verification_source = None
            return dict(
                dx=0, dy=0, shoot=False, target=None, reason="duplicate_observation"
            )
        self.camera_delta = (0.0, 0.0)
        camera_shift = None
        single_continuation = False
        if enabled and foreground and 0 <= now - captured_at <= self.settings.max_age:
            usable = [
                d
                for d in detections
                if d.score >= self.settings.confidence
                and d.head_score >= self.settings.head_confidence
                and all(math.isfinite(v) for v in (*d.box, *d.head))
                and d.box[2] > d.box[0]
                and d.box[3] > d.box[1]
            ]
            ordinary_match = self.select([(0,d) for d in usable],width,height,captured_at) if self.projected_locked is not None else None
            if self.projected_locked is not None and (self.settings.fire_policy != "zone" or ordinary_match is None):
                camera_shift = scene_shift(
                    self.anchors, usable, self.projected_locked, width / 1280
                )
                # Bounded temporal continuity for a lone body. This is not a
                # camera-motion measurement: camera and character motion cannot
                # be separated from one detection alone.
                if (camera_shift is None and len(self.anchors) == 1
                        and len(usable) == 1 and self.locked_at is not None
                        and 0 < captured_at - self.locked_at <= 0.08):
                    old, candidate = self.projected_locked, usable[0]
                    ow, oh = old.box[2] - old.box[0], old.box[3] - old.box[1]
                    nw, nh = candidate.box[2] - candidate.box[0], candidate.box[3] - candidate.box[1]
                    distance = math.dist(old.head, candidate.head)
                    if (candidate.cls == old.cls and 0.9 <= nw / ow <= 1.1
                            and 0.9 <= nh / oh <= 1.1
                            and distance <= min(100 * width / 1280, oh)
                            and distance > max(12 * width / 1280, min(65 * width / 1280, oh * 0.35))):
                        camera_shift = (candidate.head[0] - old.head[0], candidate.head[1] - old.head[1])
                        single_continuation = True
                        self.head_filter = make_head_filter(self.settings.head_filter)
                        self.verified_samples = 0
                if camera_shift is not None:
                    sx, sy = camera_shift
                    old = self.projected_locked
                    self.projected_locked = Detection(
                        old.cls,
                        old.score,
                        (
                            old.box[0] + sx,
                            old.box[1] + sy,
                            old.box[2] + sx,
                            old.box[3] + sy,
                        ),
                        (old.head[0] + sx, old.head[1] + sy),
                        old.head_score,
                    )
                    self.head_filter.translate(sx, sy)
            self.anchors = usable
        if single_continuation:
            fire = False
        action = super().step(
            detections,
            width,
            height,
            now,
            captured_at,
            enabled=enabled,
            foreground=foreground,
            fire=False,
        )
        self.verification_source = None
        target = action.get("target")
        if (self.settings.fire_policy != "zone" and target is None and action.get("reason") == "target_temporarily_lost"
                and self.locked_at is not None and captured_at-self.locked_at >= .25):
            valid = [d for d in detections if d.cls == self.settings.enemy
                     and d.score >= self.settings.confidence
                     and d.head_score >= self.settings.head_confidence
                     and all(math.isfinite(v) for v in (*d.box,*d.head))]
            old = self.projected_locked or self.locked
            if valid and old is not None and all(math.dist(d.head,old.head)>180*width/1280 for d in valid):
                self.reset()
                result=self.step(detections,width,height,now,captured_at,
                                 enabled=enabled,foreground=foreground,fire=fire)
                result["reacquired_after_scene_change"]=True
                return result
        if target is None or "error" not in action:
            self.verified_samples = 0
            held=self.projected_locked or self.locked
            return dict(action, shoot=False, recorded=True,
                        held_box=held.box if held else None,held_head=held.head if held else None,
                        held_missing_ms=(captured_at-self.locked_at)*1000 if self.locked_at is not None else None)
        previous_at = self.last_observation_at
        self.last_observation = captured_at
        self.last_observation_at = now
        dt = (
            min(0.05, max(0.012, now - previous_at))
            if previous_at is not None
            else 0.033
        )
        if self.acquired_at is None:
            self.acquired_at = captured_at
        scale = width / 1280
        head_radius = max(
            1.5 * scale,
            min(
                (target.box[2] - target.box[0]) * 0.14,
                (target.box[3] - target.box[1]) * 0.055,
            ),
        )
        point, filtered = self.head_filter.observe(
            self.previous, captured_at, head_radius
        )
        vx, vy = filtered
        self.filtered_velocity = filtered
        age = max(0, now - captured_at)
        lead = [value * age for value in filtered]
        lead_limit = min(24 * scale, (target.box[2] - target.box[0]) * 0.45)
        lead_size = math.hypot(*lead)
        if lead_size > lead_limit:
            lead = [value * lead_limit / lead_size for value in lead]
        ex = point[0] - width / 2 + lead[0]
        ey = point[1] - height / 2 + lead[1]
        focal = height / (2 * math.tan(math.radians(self.settings.fov_4_3) / 2) * 0.75)
        yaw = math.degrees(math.atan2(ex, focal))
        pitch = math.degrees(math.atan2(ey, math.hypot(focal, ex)))
        angle = math.hypot(yaw, pitch)
        scale = width / 1280
        tolerance = max(
            0.85 * scale, min(4 * scale, head_radius * 0.5)
        )
        self.micro_limit_degrees = min(
            0.3, max(0.12, math.degrees(math.atan2(tolerance, focal)))
        )
        if self.course is None or (
            previous_at is not None and now - previous_at > 0.18
        ):
            duration = (
                self.profile["duration_seconds"]
                / 1.24
                * (max(angle, 1) / self.profile["reference_angle_degrees"]) ** 0.35
            )
            self.course = dict(
                start=now, duration=min(0.65, max(0.115, duration)), angle=angle
            )
        c = self.course
        # Bridge the measured detector interval instead of stopping every 30 ms.
        # Prediction is bounded independently from the stricter firing age.
        motion_age = max(self.settings.max_age, 0.18)
        horizon = min(0.10, max(0.065, dt * 1.5), max(0.001, motion_age - age - 0.003))
        u0 = min(1, max(0, (now - c["start"]) / c["duration"]))
        u1 = min(1, u0 + horizon / c["duration"])
        s0, s1 = minimum_jerk(u0), minimum_jerk(u1)
        gain = (
            (s1 - s0) / max(1e-8, 1 - s0)
            if u0 < 0.98
            else min(0.85, 1 - math.exp(-horizon / 0.045))
        )
        self.fine_tracking = u0 >= 0.98
        factor = 1 / (max(0.01, self.settings.sensitivity) * 0.022)
        follow = [value * horizon for value in filtered]
        follow_size = math.hypot(*follow)
        follow_limit = min(24 * scale, (target.box[2] - target.box[0]) * 0.45)
        if follow_size > follow_limit:
            follow = [value * follow_limit / follow_size for value in follow]
        desired = (
            (yaw * gain + math.degrees(math.atan2(ex + follow[0], focal)) - yaw)
            * factor,
            (
                pitch * gain
                + math.degrees(math.atan2(ey + follow[1], math.hypot(focal, ex)))
                - pitch
            )
            * factor,
        )
        peak = max(70, 1.875 * max(c["angle"], 1) / c["duration"])
        max_counts = min(self.settings.max_step, peak * horizon * factor)
        magnitude = math.hypot(*desired)
        clip = min(1, max_counts / magnitude) if magnitude else 1
        dx, dy = (round(value * clip) for value in desired)
        raw_error = math.dist(self.previous, (width / 2, height / 2))
        aligned = math.hypot(ex, ey) <= tolerance
        quiet_radius = tolerance * 0.85
        quiet_limit = quiet_radius * (1.15 if self.in_deadzone else 1)
        self.in_deadzone = math.hypot(ex, ey) <= quiet_limit and raw_error <= tolerance
        # Do not keep correcting while waiting for a second aligned source.
        # Both observed and age-compensated positions must be in the center zone.
        # A moving head leaving that zone resumes normal tracking next frame.
        center_hold = aligned and raw_error <= quiet_radius and math.hypot(*filtered) * horizon < quiet_radius
        if center_hold:
            dx = dy = 0
        if not aligned:
            self.verified_samples = 0
        fresh_after_input = captured_at >= self.last_movement + 0.008
        qualifies = (
            aligned
            and fresh_after_input
            and target.score >= max(0.5, self.settings.confidence)
            and target.head_score >= max(0.8, self.settings.head_confidence)
            and now - captured_at <= min(0.075, self.settings.max_age)
            and self.head_filter.uncertainty <= head_radius * 0.5
        )
        self.verified_samples = self.verified_samples + 1 if qualifies else 0
        # An established, low-noise track at the center need not wait another
        # detector period. Cold acquisition still requires verification.
        immediate_center = (qualifies and len(self.head_filter.samples) == 3
                            and raw_error <= tolerance * .5
                            and math.hypot(ex,ey) <= tolerance * .5
                            and self.head_filter.uncertainty <= tolerance * .25
                            and math.hypot(*filtered) * age <= tolerance * .25)
        shoot = bool(
            fire
            and not self.shot_sent
            and (self.verified_samples >= 2 or immediate_center)
            and now - self.last_shot >= self.settings.shot_interval
        )
        if shoot:
            self.verification_source = captured_at
            dx = dy = 0
        reason = "recorded_aim"
        if self.shot_sent:
            reason = "one_shot_sent_wait_for_kill_or_f7"
        elif shoot:
            reason = "observed_alignment_single_shot"
        elif aligned:
            reason = "verify_fresh_alignment"
        return dict(
            action,
            dx=dx,
            dy=dy,
            shoot=shoot,
            reason=reason,
            recorded=True,
            movement_duration=horizon if dx or dy else 0,
            motion_max_age=motion_age,
            curve=(u0, u1) if u0 < 0.98 else None,
            predicted_error=(ex, ey),
            filtered_velocity=filtered,
            verified_samples=self.verified_samples,
            shot_token=self.verification_source,
            shot_sent=self.shot_sent,
            tolerance_px=tolerance,
            head_radius_px=head_radius,
            quiet_radius_px=quiet_radius,
            head_filter=self.settings.head_filter,
            center_hold=center_hold,
            immediate_center=immediate_center,
            estimated_head=point,
            prediction_uncertainty_px=self.head_filter.uncertainty,
            scene_camera_shift=camera_shift,
            fire_checks=dict(aligned=aligned, fresh_after_input=fresh_after_input,
                             fresh_frame=now-captured_at <= min(.075,self.settings.max_age),
                             certain=self.head_filter.uncertainty <= head_radius*.5,
                             strong_body=target.score >= max(.5,self.settings.confidence),
                             strong_head=target.head_score >= max(.8,self.settings.head_confidence),
                             shot_available=not self.shot_sent),
        )


class RecordedMotionPlan(MotionPlan):
    def begin(self, action, size, rect, captured_at, now, max_age):
        if not action.get("recorded"):
            return super().begin(action, size, rect, captured_at, now, max_age)
        old = self.pending
        self.cancel()
        if action["target"] is None or action["reason"] == "await_render_after_snap":
            return
        if not (action["dx"] or action["dy"] or action["shoot"]):
            return
        motion_age = (
            max_age if action["shoot"] else action.get("motion_max_age", max_age)
        )
        duration = min(
            action.get("movement_duration", 0), max(0, captured_at + motion_age - now)
        )
        start_velocity = None
        if (
            old
            and old.get("recorded")
            and not old["shoot"]
            and not action["shoot"]
            and now <= old["deadline"]
            and old["size"] == size
            and old["rect"] == rect
        ):
            start_velocity = old.get("velocity")
        self.pending = dict(
            start_velocity=start_velocity,
            dx=action["dx"],
            dy=action["dy"],
            shoot=bool(action["shoot"]),
            sent=(0, 0),
            start=now,
            duration=duration,
            next_at=now,
            size=size,
            rect=rect,
            source=captured_at,
            deadline=captured_at + motion_age,
            curve=action.get("curve"),
            recorded=True,
            shot_token=action.get("shot_token"),
        )

    def poll(self, now):
        p = self.pending
        if p is None or not p.get("recorded"):
            return super().poll(now)
        if now > p["deadline"]:
            self.cancel()
            return None
        if now < p["next_at"]:
            return None
        fraction = (
            min(1, max(0, (now - p["start"]) / p["duration"])) if p["duration"] else 1
        )
        curve = p.get("curve")
        if curve:
            a, b = curve
            denominator = minimum_jerk(b) - minimum_jerk(a)
            if denominator > 1e-8:
                fraction = (
                    minimum_jerk(a + (b - a) * fraction) - minimum_jerk(a)
                ) / denominator
        velocity = None
        if p.get("start_velocity") is not None and p["duration"] > 0:
            u = min(1, max(0, (now - p["start"]) / p["duration"]))
            values, velocity = [], []
            for total, initial in zip((p["dx"], p["dy"]), p["start_velocity"]):
                average = total / p["duration"]
                # Monotone cubic: match the old slope without reversing/overshooting.
                initial = (
                    math.copysign(min(abs(initial), 2 * abs(average)), average)
                    if initial * average > 0
                    else 0
                )
                final = average
                a, b = initial * p["duration"], final * p["duration"]
                values.append(
                    (-2 * u**3 + 3 * u * u) * total
                    + (u**3 - 2 * u * u + u) * a
                    + (u**3 - u * u) * b
                )
                velocity.append(
                    (
                        (-6 * u * u + 6 * u) * total
                        + (3 * u * u - 4 * u + 1) * a
                        + (3 * u * u - 2 * u) * b
                    )
                    / p["duration"]
                )
            cumulative = tuple(round(v) for v in values)
        else:
            cumulative = (round(p["dx"] * fraction), round(p["dy"] * fraction))
            u = (
                min(1, max(0, (now - p["start"]) / p["duration"]))
                if p["duration"]
                else 1
            )
            slope = 1
            if curve and denominator > 1e-8:
                z = a + (b - a) * u
                slope = 30 * z * z * (1 - z) * (1 - z) * (b - a) / denominator
            velocity = tuple(
                total * slope / p["duration"] if p["duration"] else 0
                for total in (p["dx"], p["dy"])
            )
        p["velocity"] = velocity
        dx, dy = (cumulative[i] - p["sent"][i] for i in range(2))
        p["sent"] = cumulative
        p["next_at"] = now + self.tick
        done = now >= p["start"] + p["duration"]
        event = dict(
            dx=dx,
            dy=dy,
            shoot=bool(p["shoot"] and done),
            size=p["size"],
            rect=p["rect"],
            source=p["source"],
            done=done,
            glide=bool(p["duration"]),
            recorded=True,
            shot_token=p.get("shot_token"),
            deadline=p["deadline"],
        )
        if done:
            self.cancel()
        return event if dx or dy or event["shoot"] else None
