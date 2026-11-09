"""Dynamic circular firing zone and bounded predictive local-demo control."""
import math
from random import random
from mvp_core import Detection, box_iou
from mvp_recorded_aim import RecordedController
from mvp_geometry import rotate_geometry

SHOT_INTERVAL = .25
OPENING_INTERVAL = .20

class ZoneController(RecordedController):
    def reset(self):
        super().reset()
        self.zone=None
        self.observation_interval=.033
        self.accepted_at=None
        self.recovery_sample=None
        self.distance_target=None
        self.distance_band=None
        self.opening_taps=0
        self.body_fallback=False

    def tracking_detection(self, detection, *, for_rebase=False):
        if not getattr(self, 'distance_fire', False):
            return detection
        x1,y1,x2,y2 = detection.box
        hx,hy = detection.head
        if (detection.head_score >= self.settings.head_confidence
                and x1 <= hx <= x2 and y1 <= hy <= y1 + .45*(y2-y1)):
            return detection
        # This point is only for association. Firing uses the visible torso.
        anchor = ((x1+x2)*.5, y1+.12*(y2-y1))
        score = self.settings.head_confidence
        if for_rebase:
            score = detection.head_score if 0 <= detection.head_score < score else 0.
        return Detection(detection.cls, detection.score, detection.box, anchor,
                         score)

    def recover_target(self, detections, width, height, now, captured_at):
        previous = self.recovery_sample
        self.recovery_sample = None
        held = self.projected_locked or self.locked
        if held is None or self.lock_size != (width, height):
            return None, None

        choices = self.eligible_choices(detections, width, height, now)
        held_height = held.box[3] - held.box[1]
        radius = min(100 * width/1280, held_height)
        nearby = [d for _, d in choices
                  if math.dist(held.head, d.head) <= radius
                  and .75 * held_height <= d.box[3]-d.box[1] <= 1.33 * held_height]
        if nearby:
            # A plausible same target takes precedence over any new target.
            if len(nearby) != 1 or nearby[0].cls != self.settings.enemy:
                return None, None
            candidate = nearby[0]
            mode = 'reassociate'
        else:
            eligible = [(distance, d) for distance, d in choices
                        if d.cls == self.settings.enemy
                        and (self.settings.radius <= 0 or distance <= self.settings.radius * width/1280)]
            if not eligible:
                return None, None
            candidate = min(eligible, key=lambda item: item[0])[1]
            mode = 'reacquire'

        count = 1
        if previous:
            old, source, streak, old_mode = previous
            if old_mode == mode and 0 < captured_at-source <= .15 and box_iou(old.box, candidate.box) >= .30:
                count = streak + 1
        self.recovery_sample = (candidate, captured_at, count, mode)
        if count >= 3 and (mode == 'reassociate' or captured_at-self.locked_at >= .35):
            return candidate, mode
        return None, None

    def set_head_filter(self, mode):
        super().set_head_filter(mode)
        self.zone=None

    def zone_geometry(self, now):
        z=self.zone
        if z is None or not 0<=now-z['source']<=self.settings.max_age:return None
        dt=max(0,now-z['accepted'])
        center=tuple(z['center'][i]+z['velocity'][i]*dt for i in (0,1))
        return dict(center=center,radius=z['radius'],size=z['size'],source=z['source'])

    def zone_fire_ready(self, now):
        z=self.zone_geometry(now)
        interval = self.zone['interval'] if z else SHOT_INTERVAL
        return bool(z and self.zone['fire'] and now>=self.last_shot+interval
                    and math.dist(z['center'],(z['size'][0]/2,z['size'][1]/2))<=z['radius'])

    def can_emit_shot(self, source, now):
        return bool(self.zone and self.zone['source']==source and self.zone_fire_ready(now))

    def shot_emitted(self, at):
        # Cadence starts at successful submission, never at decision preparation.
        self.last_shot=at
        self.shot_sent=False

    def tap_emitted(self):
        if self.zone and self.zone['fire_mode']=='opening':
            self.opening_taps -= 1
            # Require a new observation aimed at the torso before starting spray.
            if self.opening_taps == 0:
                self.zone = None

    def acknowledge(self, dx, dy, width, height):
        super().acknowledge(dx,dy,width,height)
        if self.zone:
            point=self.zone['center'];x,y=point
            moved=rotate_geometry((x-1,y-1,x+1,y+1),point,dx,dy,(width,height),self.settings.sensitivity,self.settings.fov_4_3)
            if moved is None:self.zone=None
            else:self.zone['center']=moved[1]

    def step(self,detections,width,height,now,captured_at,*,enabled,foreground,fire):
        tracking = [self.tracking_detection(d) for d in detections]
        weak_heads = {d.box for d, normalized in zip(detections,tracking) if d is not normalized}
        detections = tracking
        action=super().step(detections,width,height,now,captured_at,enabled=enabled,foreground=foreground,fire=False)
        recovery_mode = None
        if action['reason'] == 'target_temporarily_lost':
            candidate, recovery_mode = self.recover_target(detections, width, height, now, captured_at)
            if candidate is not None:
                if recovery_mode == 'reacquire':
                    retired = self.retired
                    self.reset()
                    self.retired = retired
                else:
                    self.projected_locked = candidate
                    self.head_filter.clear()
                    self.course = None
                    self.accepted_at = None
                action = super().step([candidate], width, height, now, captured_at,
                                      enabled=enabled, foreground=foreground, fire=False)
                fire = False
        else:
            self.recovery_sample = None
        target=action.get('target')
        if target is None or 'estimated_head' not in action:
            self.zone=None
            return dict(action,shoot=False,fire_policy='zone')
        scale=width/1280;bw=target.box[2]-target.box[0];bh=target.box[3]-target.box[1]
        fire_mode = None
        interval = SHOT_INTERVAL
        if getattr(self, 'distance_fire', False):
            if self.distance_target != self.acquired_at or self.distance_band is None:
                self.distance_target = self.acquired_at
                height_ratio = bh / height
                self.distance_band = 'far' if height_ratio <= .05 else 'near' if height_ratio >= .50 else 'middle'
                self.opening_taps = (1 if random() < .5 else 2) if self.distance_band == 'middle' else 0
                self.body_fallback = False
            self.body_fallback |= target.box in weak_heads
            if self.body_fallback:
                fire_mode = 'spray'
            elif self.distance_band == 'middle':
                fire_mode = 'opening' if self.opening_taps else 'spray'
                interval = OPENING_INTERVAL
            else:
                fire_mode = 'tap' if self.distance_band == 'near' else 'spray'
        radius=max(.5*scale,min(bw*.14,bh*.055))
        body = fire_mode == 'spray' or (fire_mode is None and self.settings.aim_point=='body')
        if body:radius=max(.5*scale,min(bw*.22,bh*.12))
        age=max(0,now-captured_at)
        velocity=tuple(max(-1800*scale,min(1800*scale,v)) for v in action['filtered_velocity'])
        samples = self.head_filter.samples
        if len(samples) == 3:
            first, middle, last = (sample[1] for sample in samples)
            velocity = tuple(v if (b-a)*(c-b)>0 and v*(c-b)>0 else 0.
                             for a,b,c,v in zip(first,middle,last,velocity))
        point=action['estimated_head']
        # Reject one-frame positional lag as a reason to fire outside the circle:
        # firing center is the observed keypoint projected only by source age.
        observed=self.previous
        if fire_mode == 'spray':
            body_center = (target.box[0]+bw*.5, target.box[1]+bh*.43)
            point = tuple(p + b - o for p,b,o in zip(point,body_center,observed))
            observed = body_center
        center=tuple(observed[i]+velocity[i]*age for i in (0,1))
        if self.accepted_at is not None and 0<now-self.accepted_at<.2:
            self.observation_interval=.7*self.observation_interval+.3*(now-self.accepted_at)
        self.accepted_at=now
        # Movement can bridge detector intervals; firing still expires at max_age.
        motion_age=max(.18,self.settings.max_age)
        horizon=min(.10,max(.05,self.observation_interval*1.5),max(.001,motion_age-age-.003))
        future=tuple(point[i]+velocity[i]*(age+horizon) for i in (0,1))
        # Bound extrapolation to local body scale, not to a fixed distance in time.
        moving = math.hypot(*velocity) >= 35*scale
        limit=max(18*scale,bw*.95) if moving else max(12*scale,bw*.75)
        delta=tuple(future[i]-point[i] for i in (0,1));length=math.hypot(*delta)
        if length>limit:future=tuple(point[i]+delta[i]*limit/length for i in (0,1))
        cross=(width/2,height/2)
        gain=1-math.exp(-horizon/.07)
        if not moving and math.dist(center,cross) <= 3*radius:
            gain *= .65
        pixels=[]
        for i in (0,1):
            desired=(point[i]+velocity[i]*age-cross[i])*gain+velocity[i]*horizon
            endpoint=future[i]-cross[i]
            pixels.append(math.copysign(min(abs(desired),abs(endpoint)),endpoint) if desired*endpoint>0 else 0.)
        focal=height/(2*math.tan(math.radians(self.settings.fov_4_3)/2)*.75)
        factor=1/(max(.01,self.settings.sensitivity)*.022)
        dx=round(math.degrees(math.atan2(pixels[0],focal))*factor)
        dy=round(math.degrees(math.atan2(pixels[1],math.hypot(focal,pixels[0])))*factor)
        dx=max(-self.settings.max_step,min(self.settings.max_step,dx));dy=max(-self.settings.max_step,min(self.settings.max_step,dy))
        inside=math.dist(center,cross)<=radius
        if inside:dx=dy=0
        self.zone=dict(center=center,radius=radius,size=(width,height),source=captured_at,accepted=now,velocity=velocity,fire=fire,fire_mode=fire_mode,interval=interval)
        return dict(action,dx=dx,dy=dy,shoot=False,reason='inside_fire_zone' if inside else 'predictive_zone_aim',
                    recovered_lone_target=recovery_mode == 'reassociate', recovery_mode=recovery_mode,
                    fire_policy='zone',fire_mode=fire_mode,body_fallback=self.body_fallback,
                    zone_center=center,zone_radius_px=radius,zone_inside=inside,
                    prediction_time_ms=(age+horizon)*1000,prediction_point=future,
                    movement_duration=horizon if dx or dy else 0,motion_max_age=motion_age,
                    curve=(0.,1.),shot_sent=False,shot_interval=interval,shot_ready=self.zone_fire_ready(now),
                    fire_checks=dict(zone_inside=inside,fresh_source=0<=now-captured_at<=self.settings.max_age,
                                     cadence_ready=now>=self.last_shot+interval,fire_enabled=fire))
