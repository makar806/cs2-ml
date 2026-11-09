"""Offline synthetic occlusion probe; does not alter source dataset/annotations."""
import argparse,json,hashlib
from pathlib import Path
import cv2
import numpy as np

def prepare(dataset,output,limit=8):
    dataset=Path(dataset);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cases=[]
    for label in sorted((dataset/'labels/test').glob('*.txt')):
        candidates=list((dataset/'images/test').glob(label.stem+'.*'))
        if not candidates:continue
        image=cv2.imread(str(candidates[0]))
        if image is None:continue
        h,w=image.shape[:2];objects=[]
        for line in label.read_text().splitlines():
            row=list(map(float,line.split()))
            if len(row)!=8 or row[7]!=2:continue
            cls,cx,cy,bw,bh,hx,hy,vis=row
            objects.append(dict(cls=int(cls),box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h],head=[hx*w,hy*h]))
        if not objects:continue
        target=max(objects,key=lambda d:(d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))
        box=target['box'];hx,hy=target['head'];radius=max(3,min((box[2]-box[0])*.3,(box[3]-box[1])*.08))
        original=output/(label.stem+'-original.png');cv2.imwrite(str(original),image)
        x1,y1=max(0,int(hx-radius)),max(0,int(hy-radius));x2,y2=min(w,int(hx+radius)+1),min(h,int(hy+radius)+1)
        occluded=image.copy();occluded[y1:y2,x1:x2]=(80,80,80)
        changed=output/(label.stem+'-occluded.png');cv2.imwrite(str(changed),occluded)
        cases.append(dict(id=label.stem,split='test',source=str(candidates[0].resolve()),source_sha256=hashlib.sha256(candidates[0].read_bytes()).hexdigest(),original=original.name,occluded=changed.name,target=target,occluder=[x1,y1,x2,y2]))
        if len(cases)>=limit:break
    manifest=dict(cases=cases,note='Existing test split has already been inspected historically; this is not a new independent test. Visibility=2 comes from saved labels, not new visual verification. Gray square covers an estimated head region; not a calibrated head box. Ground truth is a point/body box; no amodal mask IoU can be computed. No SAMEO output is generated.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=8);a=p.parse_args()
    print('prepared',len(prepare(a.dataset,a.output,a.limit)['cases']))
