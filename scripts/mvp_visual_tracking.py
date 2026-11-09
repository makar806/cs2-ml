"""CPU appearance correspondence across adjacent captured images."""
import cv2
import numpy as np

class VisualTracking:
    def __init__(self):
        self.previous = None

    def update(self, image, detections, at):
        h, w = image.shape[:2]
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (w//8, h//8))
        result = []
        previous = self.previous
        self.previous = (gray, detections, at, (w,h))
        if previous is None:
            return result
        old, bodies, source, size = previous
        if size != (w,h) or not 0 < at-source <= .18:
            return result
        for index, d in enumerate(bodies[:10]):
            x1,y1,x2,y2 = d.box
            # Include local background; a head alone is a poor identity template.
            x1=max(0,int(x1/8)-6); y1=max(0,int(y1/8)-6)
            x2=min(w//8,int(x2/8)+6); y2=min(h//8,int(y2/8)+6)
            patch=old[y1:y2,x1:x2]
            if min(patch.shape)<8 or patch.std()<10:
                continue
            scores=cv2.matchTemplate(gray,patch,cv2.TM_CCOEFF_NORMED)
            _,score,_,point=cv2.minMaxLoc(scores)
            px,py=point
            scores[max(0,py-4):py+5,max(0,px-4):px+5]=-1
            if score < .82 or score-float(scores.max()) < .05:
                continue
            dx,dy=(px-x1)*8,(py-y1)*8
            if abs(dx)>w*.35 or abs(dy)>h*.3:
                continue
            result.append(dict(index=index,source=source,shift=(dx,dy),score=score))
        return result
