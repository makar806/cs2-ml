"""Choose confidence on validation only; report frozen choice on diagnostic checks."""
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from evaluate_pose import iou

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/sixth-experiment'

def score(cache,threshold):
    gt=pred=matched=0
    for item in cache:
        keep=item['scores']>=threshold;boxes=item['boxes'][keep];classes=item['classes'][keep]
        candidates=sorted([(iou(b,g),pi,gi) for pi,b in enumerate(boxes) for gi,(c,g) in enumerate(item['ground']) if classes[pi]==c],reverse=True)
        used_p,used_g=set(),set()
        for overlap,pi,gi in candidates:
            if overlap<.5 or pi in used_p or gi in used_g:continue
            used_p.add(pi);used_g.add(gi)
        gt+=len(item['ground']);pred+=len(boxes);matched+=len(used_g)
    precision=matched/pred if pred else 0;recall=matched/gt if gt else 0
    return {'confidence':threshold,'gt':gt,'predictions':pred,'matched':matched,'fp':pred-matched,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0}

if __name__=='__main__':
    model=YOLO(str(ROOT/'runs/pose-v6/weights/best.pt'));cache={}
    for split in ('val','newcheck','check','test'):
        cache[split]=[]
        for path in sorted((ROOT/'data/pose-v5/images'/split).glob('*')):
            result=model.predict(str(path),device=0,imgsz=960,conf=.05,verbose=False)[0]
            h,w=result.orig_shape
            rows=[list(map(float,line.split())) for line in (ROOT/'data/pose-v5/labels'/split/(path.stem+'.txt')).read_text().splitlines()]
            ground=[(int(c),np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])) for c,x,y,bw,bh,*other in rows]
            cache[split].append({'scores':result.boxes.conf.cpu().numpy(),'boxes':result.boxes.xyxy.cpu().numpy(),'classes':result.boxes.cls.cpu().numpy().astype(int),'ground':ground})
    grid=[score(cache['val'],round(t,2)) for t in np.arange(.10,.91,.05)]
    chosen=max(grid,key=lambda r:(r['f1'],r['precision'],r['confidence']))
    checks={s:{'default':score(cache[s],.25),'selected':score(cache[s],chosen['confidence'])} for s in ('newcheck','check','test')}
    # Ensure low-confidence caching reproduced direct .25 predictions on every check.
    sweep=json.loads((OUT/'resolution-results/results.json').read_text())
    original=next(r for r in sweep['results'] if r['version']=='v6' and r['imgsz']==960)
    for split,pair in checks.items():
        assert pair['default']['matched']==original['sets'][split]['matched'] and pair['default']['fp']==original['sets'][split]['fp']
    result={'selection':'Maximize class-matched box F1 on validation only at 960. Fixed .10-.90 grid step .05; no thresholds selected from diagnostic checks. This is confidence threshold selection, not calibrated probability. Same val already used for checkpoint selection.','chosen':chosen,'validation_grid':grid,'checks':checks}
    (OUT/'confidence-selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
