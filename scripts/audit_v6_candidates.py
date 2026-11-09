"""Flag annotation disagreements for visual review, never auto-accept predictions."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from ultralytics import YOLO
from evaluate_pose import iou

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/sixth-experiment'
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/pose-v5/manifest.json').read_text())
    model=YOLO(str(ROOT/'runs/pose-v5/weights/best.pt'))
    flags=[]
    for row in manifest['images']:
        if 'multi-match-' not in row['group']:continue
        name='multi_'+Path(row['source']).name.split('multi_')[1]
        path=ROOT/'data/pose-v5/images'/row['split']/name
        labels=ROOT/'data/pose-v5/labels'/row['split']/(path.stem+'.txt')
        gt=[list(map(float,line.split())) for line in labels.read_text().splitlines()]
        frame=cv2.imread(str(path));h,w=frame.shape[:2]
        result=model.predict(frame,imgsz=1280,device=0,conf=.25,verbose=False)[0]
        ground=[np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]) for c,x,y,bw,bh,*other in gt]
        pred=result.boxes.xyxy.cpu().numpy()
        unmatched=[]
        for i,box in enumerate(pred):
            if any(iou(box,g)>=.5 and int(result.boxes.cls[i])==int(gt[j][0]) for j,g in enumerate(ground)):continue
            unmatched.append(i)
        if not unmatched:continue
        index=len(flags)
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im)
        for j,g in enumerate(ground):
            draw.rectangle(g.tolist(),outline='lime',width=3);draw.text((g[0],g[1]),'GT '+['T','CT'][int(gt[j][0])],fill='lime')
        for i in unmatched:
            b=pred[i];draw.rectangle(b.tolist(),outline='red',width=3);draw.text((b[0],b[1]),f'P{i} '+['T','CT'][int(result.boxes.cls[i])],fill='red')
        im.save(OUT/f'audit-{row["task"]}.jpg')
        flags.append({'task':row['task'],'split':row['split'],'file':name,'source':str(path),'gt':gt,'unmatched':[{'box':pred[i].tolist(),'class':int(result.boxes.cls[i]),'confidence':float(result.boxes.conf[i])} for i in unmatched]})
    (OUT/'audit-candidates.json').write_text(json.dumps(flags,indent=2))
    for page in range((len(flags)+15)//16):
        canvas=Image.new('RGB',(1600,4*245),'white');d=ImageDraw.Draw(canvas)
        for j,row in enumerate(flags[page*16:(page+1)*16]):
            im=Image.open(OUT/f'audit-{row["task"]}.jpg');im.thumbnail((400,225));x,y=j%4*400,j//4*245
            canvas.paste(im,(x,y));d.text((x+3,y+227),f'{row["task"]} {row["split"]} GT {len(row["gt"])}',fill='black')
        canvas.save(OUT/f'audit-sheet-{page}.jpg')
    print('Flagged images:',len(flags))
