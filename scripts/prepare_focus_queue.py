"""Mine unsubmitted training-only images for a short positive-focused review queue."""
import collections
import json
from pathlib import Path
import sqlite3
import numpy as np
from PIL import Image,ImageDraw
from ultralytics import YOLO
from evaluate_pose import iou

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/sixth-experiment/focus-review'
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/'data/raw/multimap_batch_001/manifest.json').read_text())
    lookup={r['file']:r for r in source['images']}
    training=json.loads((ROOT/'data/pose-v5/manifest.json').read_text())
    excluded={int(k) for k,v in training['new_match_assignment'].items() if v!='train'}
    db=sqlite3.connect(f'file:{(ROOT/".label-studio/label_studio.sqlite3").as_posix()}?mode=ro',uri=True)
    rows=db.execute('SELECT t.id,t.data FROM task t WHERE t.project_id=3 AND NOT EXISTS (SELECT 1 FROM task_completion a WHERE a.task_id=t.id AND a.was_cancelled=0) ORDER BY t.id').fetchall()
    models={v:YOLO(str(ROOT/f'runs/pose-{v}/weights/best.pt')) for v in ('v4','v6')}
    groups=collections.defaultdict(list)
    for count,(tid,raw) in enumerate(rows):
        data=json.loads(raw)
        if 'multi_train_' not in data['image']:continue
        name=data['image'][data['image'].index('multi_train_'):]
        row=lookup[name]
        if row['match_id'] in excluded:continue
        path=ROOT/'.label-studio/media'/data['image'].split('/data/')[1]
        results={v:model.predict(str(path),imgsz=960,device=0,conf=.3,verbose=False)[0] for v,model in models.items()}
        prediction=results['v6'];h,w=prediction.orig_shape
        other=results['v4'].boxes.xyxy.cpu().numpy()
        boxes=prediction.boxes.xyxy.cpu().numpy()
        candidates=[]
        for idx,box in enumerate(boxes):
            x1,y1,x2,y2=box
            # Suppress obvious own-weapon-like lower-frame proposals from queue selection.
            if y2-y1<25 or x2-x1<8 or y1>h*.75 or (x1>w*.5 and y1>h*.58):continue
            agreement=any(iou(box,b)>=.4 and int(prediction.boxes.cls[idx])==int(results['v4'].boxes.cls[j]) for j,b in enumerate(other))
            confidence=float(prediction.boxes.conf[idx])
            if confidence<.5 and not agreement:continue
            candidates.append((idx,agreement,confidence))
        if not candidates:continue
        rank=max(conf+(.3 if agreed else 0) for _,agreed,conf in candidates)
        result=[]
        for idx,agreed,confidence in candidates:
            x1,y1,x2,y2=boxes[idx].tolist();bid=f'focus-{tid}-{idx}'
            common={'original_width':w,'original_height':h,'image_rotation':0,'to_name':'image'}
            result.append(dict(common,id=bid,from_name='label',type='rectanglelabels',value={'x':x1/w*100,'y':y1/h*100,'width':(x2-x1)/w*100,'height':(y2-y1)/h*100,'rotation':0,'rectanglelabels':[['T','CT'][int(prediction.boxes.cls[idx])]]}))
            hx,hy=prediction.keypoints.xy[idx,0].cpu().tolist()
            if float(prediction.keypoints.conf[idx,0])>=.5 and x1<=hx<=x2 and y1<=hy<=y2:
                result.append(dict(common,id=f'focus-h-{tid}-{idx}',parentID=bid,from_name='head',type='keypointlabels',value={'x':hx/w*100,'y':hy/h*100,'width':.6,'keypointlabels':['Head']}))
        record={'source_task_id':tid,'file':name,'map':row['map'],'match_id':row['match_id'],'source_url':row['source_url'],'source_image_url':data['image'],'sha256':row['sha256'],'rank':rank,'result':result}
        groups[row['map']].append(record)
    selected=[]
    for key in groups:groups[key].sort(key=lambda r:r['rank'],reverse=True)
    while len(selected)<60 and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected)<60:selected.append(groups[key].pop(0))
    (OUT/'candidates.json').write_text(json.dumps(selected,indent=2))
    for page in range((len(selected)+15)//16):
        canvas=Image.new('RGB',(1600,4*245),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(selected[page*16:(page+1)*16]):
            path=ROOT/'.label-studio/media'/row['source_image_url'].split('/data/')[1]
            im=Image.open(path).convert('RGB');d=ImageDraw.Draw(im);w,h=im.size
            for region in row['result']:
                if region['type']!='rectanglelabels':continue
                b=region['value'];xy=[b['x']*w/100,b['y']*h/100,(b['x']+b['width'])*w/100,(b['y']+b['height'])*h/100]
                d.rectangle(xy,outline='red',width=3)
            im.thumbnail((400,225));x,y=j%4*400,j//4*245;canvas.paste(im,(x,y));draw.text((x+3,y+227),f'{row["source_task_id"]} {row["map"]}',fill='black')
        canvas.save(OUT/f'candidates-{page}.jpg')
    print('Candidates',len(selected),dict(collections.Counter(r['map'] for r in selected)),flush=True)
