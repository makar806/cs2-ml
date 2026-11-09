"""Fixed-threshold V4/V5 evaluation on identical held-out images."""
import collections
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from ultralytics import YOLO
from evaluate_pose import iou

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/fifth-training'
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    report={'protocol':{'confidence':.25,'iou':.5,'imgsz':640,'class_required':True,'limitations':'New check: six held-out matches, 93 correlated frames, 31 manually labeled targets. Historical third-party overlap unknown; labels were assisted by V4 predictions, creating potential verification bias. Old checks repeatedly inspected. Head error is conditional on successful detections.'}}
    for version in ('v4','v5'):
        model=YOLO(str(ROOT/f'runs/pose-{version}/weights/best.pt'))
        report[version]={}
        for split in ('newcheck','check','test'):
            total_gt=total_pred=matched=0; errors=[]; details=[]
            class_gt=collections.Counter();class_match=collections.Counter(); small_gt=small_match=0
            paths=sorted((ROOT/'data/pose-v5/images'/split).glob('*'))
            for path in paths:
                rows=[list(map(float,line.split())) for line in (ROOT/'data/pose-v5/labels'/split/(path.stem+'.txt')).read_text().splitlines()]
                result=model.predict(str(path),device=0,imgsz=640,conf=.25,verbose=False)[0]
                h,w=result.orig_shape;boxes=result.boxes.xyxy.cpu().numpy();classes=result.boxes.cls.cpu().numpy().astype(int)
                ground=[(int(c),np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]),np.array([hx*w,hy*h]),vis) for c,x,y,bw,bh,hx,hy,vis in rows]
                candidates=sorted([(iou(b,g[1]),pi,gi) for pi,b in enumerate(boxes) for gi,g in enumerate(ground) if classes[pi]==g[0]],reverse=True)
                used_p,used_g=set(),set()
                for overlap,pi,gi in candidates:
                    if overlap<.5 or pi in used_p or gi in used_g:continue
                    used_p.add(pi);used_g.add(gi);class_match[ground[gi][0]]+=1
                    if ground[gi][1][3]-ground[gi][1][1]<60:small_match+=1
                    if ground[gi][3]>0:errors.append(float(np.linalg.norm(result.keypoints.xy.cpu().numpy()[pi,0]-ground[gi][2])))
                for g in ground:
                    class_gt[g[0]]+=1
                small_gt+=sum(int(g[1][3]-g[1][1]<60) for g in ground)
                total_gt+=len(ground);total_pred+=len(boxes);matched+=len(used_g)
                details.append({'image':path.name,'gt':len(ground),'predictions':len(boxes),'matched':len(used_g)})
                if split=='newcheck' and len(ground):
                    result.save(filename=str(OUT/f'{version}-{path.name}'))
            report[version][split]={'images':len(paths),'gt':total_gt,'predictions':total_pred,'matched':matched,'false_positives':total_pred-matched,'precision':matched/total_pred if total_pred else 0,'recall':matched/total_gt if total_gt else 0,'class_gt':dict(class_gt),'class_matched':dict(class_match),'small_under_60px':{'gt':small_gt,'matched':small_match},'head_conditional_mean_pixels':float(np.mean(errors)) if errors else None,'head_matches':len(errors),'details':details}
        print(version,json.dumps({s:{k:v for k,v in r.items() if k!='details'} for s,r in report[version].items()}),flush=True)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    samples=[r['image'] for r in report['v5']['newcheck']['details'] if r['gt']][:12]
    for version in ('v4','v5'):
        canvas=Image.new('RGB',(1280,4*265),'white');draw=ImageDraw.Draw(canvas)
        for idx,name in enumerate(samples):
            im=Image.open(OUT/f'{version}-{name}');im.thumbnail((426,240));x,y=idx%3*426,idx//3*265
            canvas.paste(im,(x,y));draw.text((x+3,y+242),name,fill='black')
        canvas.save(OUT/f'preview-{version}.jpg')
