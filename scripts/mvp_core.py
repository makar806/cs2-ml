"""Pure target selection and closed-loop control for a local CV prototype."""
from dataclasses import dataclass
import math
from mvp_geometry import rotate_geometry

@dataclass
class Detection:
    cls: int
    score: float
    box: tuple
    head: tuple
    head_score: float

@dataclass
class Settings:
    enemy: int = 0
    confidence: float = .35
    head_confidence: float = .55
    sensitivity: float = 1.0
    fov_4_3: float = 90.0
    radius: float = 0.0 # zero = the entire visible frame; positive = opt-in radius
    max_step: int = 4096
    max_age: float = .10
    aim_point: str = "head"
    head_filter: str = "current"
    fire_policy: str = "verified"
    memory_guard: bool = False
    stop_before_shot: bool = True
    shot_interval: float = .40


def aim_detection(d, mode):
    """Torso location is a bounding-box heuristic, not a learned body keypoint."""
    if mode == 'head':return d
    if mode != 'body':raise ValueError('Unknown aim point: '+str(mode))
    x1,y1,x2,y2=d.box
    return Detection(d.cls,d.score,d.box,((x1+x2)*.5,y1+(y2-y1)*.35),d.score)


def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


class Controller:
    """One fresh observation -> full angular correction and optional single shot.

    No multi-frame alignment gate or position EMA. Predictions remain fallible;
    firing after a calculated correction is an open-loop local-demo policy.
    """
    def __init__(self,settings=None):
        self.settings=settings or Settings()
        self.last_shot=-math.inf
        self.reset()

    def reset(self):
        self.previous=None;self.previous_at=None;self.camera_delta=(0.,0.)
        self.locked=None;self.locked_at=None;self.lock_size=None
        self.projected_locked=None
        self.aligned_samples=0
        self.retired=[]

    def release_after_confirmed_kill(self,now):
        """Explicit kill signal, never a timeout on a missing detection."""
        if self.locked is None:return False
        d=self.projected_locked or self.locked;box=d.box;head=d.head
        self.reset()
        # A post-kill animation can still look standing in the next image.
        # This short geometric exclusion is not a general corpse classifier.
        self.retired=[(now+.4,box,head)]
        return True

    def select(self,choices,width,height,now):
        """Associate the previous body/head after camera motion; never nearest-switch.

        Missing observations pause input without releasing identity. Geometry
        cannot establish identity through complete occlusion or crossing bodies.
        """
        s=self.settings;scale=width/1280
        if self.lock_size is not None and self.lock_size!=(width,height):return None
        if self.locked is not None:
            if self.projected_locked is None:return None
            old=self.projected_locked;predicted=old.head;a=old.box
            matches=[]
            for distance,d in choices:
                b=d.box
                iou=box_iou(a,b)
                head_distance=math.dist(predicted,d.head)
                gate=max(12*scale,min(65*scale,(a[3]-a[1])*.35))
                ratio=(b[3]-b[1])/(a[3]-a[1])
                ordinary_match = .75 <= ratio <= 1.33 and head_distance <= gate and iou >= .10
                # A cropped body box can change height abruptly while the head
                # stays in place. Require both strong overlap and a close head.
                resized_match = (s.fire_policy == 'zone' and .45 <= ratio <= 2.2
                                 and iou >= .45 and head_distance <= min(16*scale, gate*.5))
                if ordinary_match or resized_match:
                    matches.append((head_distance/gate+(1-iou)*.25,d))
            matches.sort(key=lambda item:item[0])
            unique=[]
            for cost,d in matches:
                # Class-aware detector NMS can output the SAME body twice,
                # once as T and once as CT. That is not two identity candidates.
                duplicate=any(d.cls!=other.cls and box_iou(d.box,other.box)>.75 and
                    math.dist(d.head,other.head)<.25*min(d.box[3]-d.box[1],other.box[3]-other.box[1])
                    for _,other in unique)
                if not duplicate:unique.append((cost,d))
            matches=unique
            # A new closer/larger body must not steal the held identity. If two
            # geometric candidates are indistinguishable, pause rather than guess.
            if matches and not (len(matches)>1 and matches[1][0]-matches[0][0]<.2):
                return matches[0][1]
            return None
        eligible=[item for item in choices if (s.radius<=0 or item[0]<=s.radius*scale) and item[1].cls==s.enemy]
        return min(eligible,key=lambda item:item[0])[1] if eligible else None

    def acknowledge(self,dx,dy,width,height):
        fy=height/(2*math.tan(math.radians(self.settings.fov_4_3)/2)*.75)
        shift=tuple(math.tan(math.radians(v*self.settings.sensitivity*.022))*fy for v in (dx,dy))
        self.camera_delta=tuple(self.camera_delta[i]+shift[i] for i in range(2))
        if self.projected_locked is not None:
            old=self.projected_locked
            geometry=rotate_geometry(old.box,old.head,dx,dy,(width,height),self.settings.sensitivity,self.settings.fov_4_3)
            self.projected_locked=Detection(old.cls,old.score,*geometry,old.head_score) if geometry is not None else None
        retired=[]
        for until,box,head in self.retired:
            geometry=rotate_geometry(box,head,dx,dy,(width,height),self.settings.sensitivity,self.settings.fov_4_3)
            if geometry is not None:retired.append((until,*geometry))
        self.retired=retired

    def eligible_choices(self, detections, width, height, now):
        s=self.settings
        cx,cy=width/2,height/2;choices=[]
        self.retired=[entry for entry in self.retired if entry[0]>now]
        for d in detections:
            if d.score<s.confidence or d.head_score<s.head_confidence:continue
            if not all(math.isfinite(float(v)) for v in (*d.box,*d.head,d.score,d.head_score)):continue
            x1,y1,x2,y2=d.box;hx,hy=d.head
            if not (0<=x1<x2<=width and 0<=y1<y2<=height and x1<=hx<=x2 and y1<=hy<=y2):continue
            # A keypoint on the torso/feet is not a usable head. Do not chase it
            # or shoot merely because its network confidence is high.
            if hy>y1+.45*(y2-y1):continue
            if any(box_iou(d.box,box)>.6 and math.dist(d.head,head)<.3*(y2-y1)
                   for _,box,head in self.retired):continue
            distance=math.hypot(hx-cx,hy-cy)
            choices.append((distance,d))
        return choices

    def step(self,detections,width,height,now,captured_at,*,enabled,foreground,fire):
        s=self.settings
        empty={'dx':0,'dy':0,'shoot':False,'target':None,'reason':'inactive_or_stale'}
        if not enabled or not foreground:
            self.reset();return empty
        if not 0<=now-captured_at<=s.max_age:
            self.aligned_samples=0;return empty
        scale=width/1280;cx,cy=width/2,height/2
        choices=self.eligible_choices(detections,width,height,now)
        target=self.select(choices,width,height,captured_at)
        if target is None:
            return dict(empty,reason='target_temporarily_lost' if self.locked is not None else 'no_valid_head')
        tracking=self.previous is not None
        raw_class=target.cls
        # Team is a property of the held identity, not a new classification
        # decision on every frame. Never acquire a friendly-class object cold.
        if tracking and target.cls!=s.enemy:
            target=Detection(s.enemy,target.score,target.box,target.head,target.head_score)
        hx,hy=target.head;vx=vy=0.
        if s.fire_policy != "zone" and tracking and 0<now-self.last_shot<.04:
            return {'dx':0,'dy':0,'shoot':False,'target':target,'reason':'recoil_settle'}
        # The FIRST correction already includes fire. Suppress only a duplicate
        # large correction against a view that has not rendered that input yet.
        # Unlike v2 this is never an alignment gate before the initial shot.
        if self.previous is not None and math.hypot(*self.camera_delta)>6*scale and math.dist(target.head,self.previous)<scale:
            return {'dx':0,'dy':0,'shoot':False,'target':target,'reason':'await_render_after_snap'}
        if self.previous is not None and self.previous_at is not None:
            dt=captured_at-self.previous_at
            predicted=self.projected_locked.head if self.projected_locked is not None else self.previous
            delta=(hx-predicted[0],hy-predicted[1])
            if .005<=dt<=.15 and math.hypot(*delta)<60*scale:
                limit=1200*scale
                vx=max(-limit,min(limit,delta[0]/dt));vy=max(-limit,min(limit,delta[1]/dt))
        self.locked=target;self.locked_at=captured_at;self.lock_size=(width,height)
        self.projected_locked=target
        self.previous=target.head;self.previous_at=captured_at;self.camera_delta=(0.,0.)
        # Extrapolate only the measured processing age, never a long future lead.
        age=min(now-captured_at,.05)
        # The old limit allowed 60px of extrapolation, several heads at range.
        # Recoil is unobserved camera motion, not enemy velocity: suppress lead
        # just after a shot, and bound it by target scale at all other times.
        lead=(vx*age,vy*age)
        limit=min(6*scale,(target.box[2]-target.box[0])*.15)
        magnitude=math.hypot(*lead)
        gain=min(1,limit/magnitude) if magnitude else 1
        if 0<=now-self.last_shot<.18:gain=0
        hx+=lead[0]*gain;hy+=lead[1]*gain
        if not (0<=hx<width and 0<=hy<height):return dict(empty,reason='prediction_outside_frame')
        target=Detection(target.cls,target.score,target.box,(hx,hy),target.head_score)
        ex,ey=hx-cx,hy-cy
        fy=height/(2*math.tan(math.radians(s.fov_4_3)/2)*.75)
        yaw=math.degrees(math.atan2(ex,fy))
        pitch=math.degrees(math.atan2(ey,math.hypot(fy,ex)))
        factor=1/(max(s.sensitivity,.01)*.022)
        desired=(round(yaw*factor),round(pitch*factor))
        dx,dy=(max(-s.max_step,min(s.max_step,v)) for v in desired)
        clamped=(dx,dy)!=desired
        if tracking:
            # Closed-loop follow corrections should not assume zero render/input
            # delay. Full correction on every delayed frame can oscillate.
            dx,dy=round(dx*.65),round(dy*.65)
            if math.hypot(ex,ey)<.7*scale:dx=dy=0
        tolerance=max(2*scale,min(6*scale,(target.box[2]-target.box[0])*.08))
        # Follow-up shots require observed alignment, not an optimistic flick
        # that assumes recoil/camera response will match the command perfectly.
        observed_error=math.dist(self.previous,(cx,cy))
        self.aligned_samples=self.aligned_samples+1 if observed_error<=tolerance else 0
        verified=not tracking or self.aligned_samples>=2
        shoot=bool(fire and not clamped and verified and now-self.last_shot>=s.shot_interval)
        if shoot:self.last_shot=now
        return {'dx':dx,'dy':dy,'shoot':shoot,'target':target,'error':(ex,ey),
                'stable':1,'reason':'snap_fire' if shoot else ('limited_step' if clamped else 'snap_aim'),
                'velocity':(vx,vy),'tracking':tracking,'lead':(lead[0]*gain,lead[1]*gain),
                'aligned_samples':self.aligned_samples,'raw_class':raw_class}


def decode_result(result):
    detections=[]
    if result.keypoints is None or result.keypoints.conf is None:return detections
    import torch
    # One device-to-host transfer instead of five synchronizations per frame.
    packed=torch.cat((result.boxes.xyxy,result.boxes.conf[:,None],result.boxes.cls[:,None],result.keypoints.data[:,0,:]),dim=1).cpu().tolist()
    for row in packed:
        x1,y1,x2,y2,score,cls,hx,hy,confidence=row
        detections.append(Detection(int(cls),float(score),(x1,y1,x2,y2),(hx,hy),float(confidence)))
    return detections
