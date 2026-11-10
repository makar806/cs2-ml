"""Small CPU background-motion estimator; never reads or emits game input."""
from collections import deque
import math
from mvp_core import Detection


class CameraMotion:
    def __init__(self):
        self.previous = None
        self.steps = deque(maxlen=32)
        self.cut_pending = False

    def update(self, image, detections, at):
        import cv2
        import numpy as np
        h, w = image.shape[:2]
        size = (256, max(32, round(h * 256 / w)))
        gray = cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2GRAY)
        previous = self.previous
        self.previous = (gray, detections, at, (w, h))
        if previous is None:
            return list(self.steps)
        old, bodies, source, old_size = previous
        step = dict(source=source, at=at, matrix=None, cut=False)
        if old_size == (w, h) and 0 < at-source <= 1.5:
            mask = np.zeros_like(old)
            mask[8:int(size[1]*.78), 12:-12] = 255
            sx, sy = size[0]/w, size[1]/h
            for d in bodies:
                x1,y1,x2,y2 = d.box
                cv2.rectangle(mask, (int(x1*sx)-2,int(y1*sy)-2),
                              (int(x2*sx)+2,int(y2*sy)+2), 0, -1)
            points = cv2.goodFeaturesToTrack(old, 64, .015, 7, mask=mask)
            matrix = None
            if at-source <= .25 and points is not None and len(points) >= 10:
                moved, valid, error = cv2.calcOpticalFlowPyrLK(
                    old, gray, points, None, winSize=(15,15), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 12, .05))
                good = (valid.ravel() > 0) & (error.ravel() < 25)
                a, b = points.reshape(-1,2)[good], moved.reshape(-1,2)[good]
                if len(a) >= 10:
                    estimate, inliers = cv2.estimateAffinePartial2D(
                        a, b, method=cv2.RANSAC, ransacReprojThreshold=1.8, maxIters=100)
                    if estimate is not None and inliers.sum() >= max(9, len(a)*.65):
                        # Reject object-local fits: background features must span the scene.
                        support = a[inliers.ravel() > 0]
                        spread = np.ptp(support, axis=0)
                        scale = math.hypot(estimate[0,0], estimate[0,1])
                        if spread[0] > size[0]*.30 and spread[1] > size[1]*.18 and .85 < scale < 1.18:
                            matrix = estimate
            if matrix is not None:
                # Convert resized coordinates back to the captured image coordinates.
                m = np.eye(3); m[:2] = matrix
                resize = np.diag([sx,sy,1.])
                step['matrix'] = (np.linalg.inv(resize) @ m @ resize).tolist()
                step['cut'] = self.cut_pending
                self.cut_pending = False
            else:
                a = old[8:int(size[1]*.78),12:-12].astype(float)
                b = gray[8:int(size[1]*.78),12:-12].astype(float)
                difference = float(np.mean(np.abs(a-b)))
                correlation = float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if min(a.std(),b.std()) > 8 else 1.
                changed = difference > 35 and correlation < .35
                step['cut'] = bool(changed and self.cut_pending)
                self.cut_pending = changed
        else:
            self.cut_pending = False
        self.steps.append(step)
        return list(self.steps)


class CameraLock:
    """Carry the held raw geometry through camera steps, including skipped packets."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.target = None
        self.source = None

    def anchor(self, target, source):
        self.target, self.source = target, source

    def advance(self, steps, at, size):
        if self.target is None:
            return None, None
        chain = [s for s in steps if self.source < s['at'] <= at]
        if any(s['cut'] for s in chain):
            self.clear()
            return None, 'camera_scene_cut'
        d = self.target
        for step in chain:
            if step['source'] != self.source or step['matrix'] is None:
                return None, None  # No evidence of camera motion: keep controller identity.
            m = step['matrix']
            def transform(x,y):
                return (m[0][0]*x+m[0][1]*y+m[0][2],m[1][0]*x+m[1][1]*y+m[1][2])
            x1,y1,x2,y2 = d.box
            corners = [transform(x,y) for x,y in ((x1,y1),(x2,y1),(x1,y2),(x2,y2))]
            box = (min(p[0] for p in corners), min(p[1] for p in corners),
                   max(p[0] for p in corners), max(p[1] for p in corners))
            d = Detection(d.cls,d.score,box,transform(*d.head),d.head_score)
            self.source = step['at']
            self.target = d
        self.target = d
        w,h = size
        margin=16*w/1280
        if (d.box[2] < 0 or d.box[0] > w or d.box[3] < 0 or d.box[1] > h
                or not -margin<=d.head[0]<=w+margin or not -margin<=d.head[1]<=h+margin):
            self.clear()
            return None, 'camera_target_left_view'
        return d, None

    def update_controller(self, controller, history, steps, at, size):
        if controller.locked is None:
            self.clear()
            return None
        target,reason=self.advance(steps,at,size)
        if reason:
            controller.reset()
        elif target is not None:
            rebased=history.rebase([target],at,size)
            old=controller.projected_locked
            if rebased and old is not None:
                shift=tuple(rebased[0].head[i]-old.head[i] for i in (0,1))
                if sum(v*v for v in shift)>(6*size[0]/1280)**2:
                    controller.projected_locked=rebased[0]
                    controller.head_filter.translate(*shift)
                    if hasattr(controller,'body_filter'):
                        controller.body_filter.translate(*shift)
                    controller.anchors=[]
        return reason
