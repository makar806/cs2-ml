"""Small CPU HUD silhouette matcher. No capture, inference model or game API."""
from pathlib import Path
import json
import cv2
import numpy as np
from mvp_weapon_policy import aim_point_for_weapon

def normalized(mask):
    ys,xs=np.where(mask>190)
    if not len(xs):return np.zeros((32,96),bool)
    mask=(mask[ys.min():ys.max()+1,xs.min():xs.max()+1]>190).astype(np.uint8)*255
    return cv2.resize(mask,(96,32),interpolation=cv2.INTER_AREA)>100

class WeaponVisual:
    def __init__(self,directory,interval=.15):
        self.templates=[]
        self.palette=None
        palette=Path(directory)/'palette.json'
        if palette.exists():
            color=json.loads(palette.read_text(encoding='utf-8'))['bgr']
            if len(color)!=3 or not all(isinstance(v,(int,float)) and 0<=v<=255 for v in color):
                raise ValueError('Invalid calibrated HUD color')
            self.palette=np.array(color,dtype=np.float32)
        for path in sorted(Path(directory).glob('*.png')):
            mask=cv2.imread(str(path),0)
            if mask is not None:self.templates.append(('weapon_'+path.stem,normalized(mask)))
        self.interval=interval;self.at=None;self.candidate=None;self.count=0;self.weapon=None
    def classify(self,image):
        h,w=image.shape[:2]
        roi=image[int(h*.55):int(h*.94),int(w*.78):]
        roi=cv2.resize(roi,(max(1,round(roi.shape[1]*720/h)),281))
        mask=(roi.max(axis=2)>190).astype(np.uint8)*255
        # A calibrated icon color separates the silhouette from bright scene
        # geometry. Text under the active entry can have a different color.
        icon_mask=mask
        if self.palette is not None:
            icon_mask=(np.abs(roi.astype(np.float32)-self.palette).max(axis=2)<=35).astype(np.uint8)*255
        joined=cv2.morphologyEx(icon_mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(joined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        found=[]
        for contour in contours:
            x,y,cw,ch=cv2.boundingRect(contour)
            if not (30<=cw<=150 and 5<=ch<=45 and cw/ch>=1.7):continue
            # Active inventory entry has a label immediately beneath its icon.
            label=mask[y+ch+3:min(mask.shape[0],y+ch+24),max(0,x-10):x+cw]
            if label.size==0 or np.count_nonzero(label)<12:continue
            sample=normalized(icon_mask[y:y+ch,x:x+cw])
            scores=sorted(((2*np.logical_and(sample,t).sum()/max(1,sample.sum()+t.sum()),name) for name,t in self.templates),reverse=True)
            if len(scores)>1 and scores[0][0]>=.78 and scores[0][0]-scores[1][0]>=.06:
                found.append(scores[0])
        names=set(name for _,name in found)
        return next(iter(names)) if len(names)==1 else None
    def update(self,image,at):
        if self.at is not None and 0<=at-self.at<self.interval:return self.snapshot(at)
        if self.at is not None and (at-self.at>.5 or at<self.at):self.candidate=None;self.count=0
        self.at=at
        name=self.classify(image)
        self.count=self.count+1 if name is not None and name==self.candidate else 1
        self.candidate=name
        self.weapon=name if name is not None and self.count>=2 else None
        return self.snapshot(at)
    def snapshot(self,now):
        name=self.weapon if self.at is not None and 0<=now-self.at<=.5 else None
        return dict(weapon_name=name,weapon_aim_point=aim_point_for_weapon(name),weapon_sample_at=self.at)
