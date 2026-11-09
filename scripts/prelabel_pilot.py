"""Create 50 training-only prediction hints; never create human annotations."""
import collections
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/prelabel-pilot'

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--from-task',type=int)
    args=parser.parse_args()
    if args.from_task is not None:
        OUT=ROOT.parent/'2026-10-17/new-chat/outputs/prelabel-all'
    OUT.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(f'file:{(ROOT/".label-studio/label_studio.sqlite3").as_posix()}?mode=ro',uri=True)
    rows=db.execute('SELECT id,data FROM task WHERE project_id=3 AND NOT EXISTS (SELECT 1 FROM task_completion c WHERE c.task_id=task.id AND c.was_cancelled=0) AND NOT EXISTS (SELECT 1 FROM prediction p WHERE p.task_id=task.id) ORDER BY id').fetchall()
    groups=collections.defaultdict(list)
    manifest=json.loads((ROOT/'data/raw/multimap_batch_001/manifest.json').read_text())
    lookup={r['file']:r for r in manifest['images']}
    for tid,raw in rows:
        data=json.loads(raw)
        if 'multi_train_' not in data['image']:continue
        name=data['image'][data['image'].index('multi_train_'):]
        if name in lookup:groups[lookup[name]['map']].append((tid,data,name))
    chosen=[]
    # Spread frames over the full map list instead of choosing consecutive timestamps.
    for map_name,items in groups.items():
        indices=np.linspace(0,len(items)-1,min(8,len(items)),dtype=int)
        groups[map_name]=[items[i] for i in indices]
    while len(chosen)<50 and any(groups.values()):
        for map_name in sorted(groups):
            if groups[map_name] and len(chosen)<50:chosen.append(groups[map_name].pop(0))
    if args.from_task is not None:
        chosen=[]
        for tid,raw in rows:
            if tid<args.from_task:continue
            data=json.loads(raw)
            chosen.append((tid,data,Path(data['image']).name))
    else:
        assert len(chosen)==50
    model=YOLO(str(ROOT/'runs/pose-v4/weights/best.pt'))
    payload=[]; summary=[]
    for tid,data,name in chosen:
        path=ROOT/'.label-studio/media'/data['image'].split('/data/')[1]
        prediction=model.predict(str(path),device=0,imgsz=960,conf=.20,verbose=False)[0]
        h,w=prediction.orig_shape;results=[];scores=[]
        for i,box in enumerate(prediction.boxes):
            x1,y1,x2,y2=box.xyxy[0].cpu().tolist();score=float(box.conf[0]);scores.append(score)
            bid=f'p4-{tid}-{i}'
            common={'original_width':w,'original_height':h,'image_rotation':0,'to_name':'image'}
            results.append(dict(common,id=bid,from_name='label',type='rectanglelabels',value={'x':x1/w*100,'y':y1/h*100,'width':(x2-x1)/w*100,'height':(y2-y1)/h*100,'rotation':0,'rectanglelabels':[['T','CT'][int(box.cls[0])]]}))
            hx,hy=prediction.keypoints.xy[i,0].cpu().tolist()
            certainty=float(prediction.keypoints.conf[i,0]) if prediction.keypoints.conf is not None else 0
            if certainty>=.5 and x1<=hx<=x2 and y1<=hy<=y2:
                results.append(dict(common,id=f'h4-{tid}-{i}',parentID=bid,from_name='head',type='keypointlabels',value={'x':hx/w*100,'y':hy/h*100,'width':.6,'keypointlabels':['Head']}))
        payload.append({'task':tid,'model_version':'V4-pilot-960-conf020','score':sum(scores)/len(scores) if scores else 0,'result':results})
        summary.append({'task':tid,'file':name,'boxes':len(scores),'heads':sum(r['type']=='keypointlabels' for r in results),'url':f'http://127.0.0.1:8080/projects/3/data?task={tid}'})
        if len(summary)%50==0:print('Processed',len(summary),'/',len(chosen),flush=True)
    (OUT/'predictions.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    (OUT/'tasks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'tasks':len(summary),'boxes':sum(r['boxes'] for r in summary),'empty':sum(r['boxes']==0 for r in summary)}))
