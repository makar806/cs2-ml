"""Sample one new recording and compare inference sizes without claiming accuracy."""
from pathlib import Path
import collections
import json
import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work/hard-frames'
SOURCE=Path(r'C:\Users\makar\Videos\Captures\Counter-Strike 2 2026-10-18 02-17-26.mp4')

if __name__=='__main__':
    WORK.mkdir(parents=True,exist_ok=True)
    model=YOLO(str(ROOT/'runs/pose-v2/weights/best.pt'))
    sizes=(640,960,1280)
    for size in sizes:
        for _ in range(3):
            model.predict(np.zeros((1080,1440,3),dtype=np.uint8),imgsz=size,device=0,conf=.25,verbose=False)
    cap=cv2.VideoCapture(str(SOURCE));fps=cap.get(cv2.CAP_PROP_FPS)
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    report=[]
    for sample,idx in enumerate(range(0,frame_count,round(fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
        assert ok
        cv2.imwrite(str(WORK/f'sample-{sample:03d}.png'),frame)
        entry={'sample':sample,'frame':idx,'approx_seconds':idx/fps,
               'source_position_ms':cap.get(cv2.CAP_PROP_POS_MSEC),'predictions':{}}
        for size in sizes:
            torch.cuda.synchronize();start=time.perf_counter()
            r=model.predict(frame,imgsz=size,device=0,conf=.25,verbose=False)[0]
            torch.cuda.synchronize();duration=(time.perf_counter()-start)*1000
            detections=[]
            for i,b in enumerate(r.boxes):
                detections.append({'class':r.names[int(b.cls.item())],'confidence':float(b.conf.item()),
                    'bbox':b.xyxy[0].cpu().tolist(),'head':r.keypoints.xy[i,0].cpu().tolist()})
            entry['predictions'][str(size)]={'detections':detections,'call_ms':duration}
            cv2.imwrite(str(WORK/f'pred-{sample:03d}-{size}.jpg'),r.plot(kpt_radius=4,line_width=2))
        report.append(entry)
        if sample%10==0:print(f'Compared {sample+1} frames',flush=True)
    cap.release()
    (WORK/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    summary={str(size):{'total_predictions':sum(len(x['predictions'][str(size)]['detections']) for x in report),
        'median_predict_call_ms':float(np.median([x['predictions'][str(size)]['call_ms'] for x in report]))} for size in sizes}
    (WORK/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for page in range((len(report)+15)//16):
        canvas=Image.new('RGB',(1280,4*265),'white');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(report[page*16:(page+1)*16]):
            im=Image.open(WORK/f'sample-{row["sample"]:03d}.png');im.thumbnail((320,240))
            x,y=i%4*320,i//4*265;canvas.paste(im,(x,y))
            counts='/'.join(str(len(row['predictions'][str(s)]['detections'])) for s in sizes)
            draw.text((x+3,y+242),f'{row["sample"]:03d} t~{row["approx_seconds"]:.0f}s boxes {counts}',fill='black')
        canvas.save(WORK/f'overview-{page}.jpg')
    print(json.dumps(summary,indent=2))
