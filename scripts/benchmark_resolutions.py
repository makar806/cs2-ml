"""Exploratory resolution sweep; identical labels and fixed matching thresholds."""
import csv
import argparse
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from ultralytics import YOLO
from evaluate_pose import iou

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'2026-10-17/new-chat/outputs/resolution-comparison'

def match(result,rows):
    h,w=result.orig_shape
    ground=[(int(c),np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]),np.array([hx*w,hy*h]),vis) for c,x,y,bw,bh,hx,hy,vis in rows]
    boxes=result.boxes.xyxy.cpu().numpy();classes=result.boxes.cls.cpu().numpy().astype(int)
    pairs=sorted([(iou(b,g[1]),pi,gi) for pi,b in enumerate(boxes) for gi,g in enumerate(ground) if classes[pi]==g[0]],reverse=True)
    used_p,used_g=set(),set();head=[]
    for overlap,pi,gi in pairs:
        if overlap<.5 or pi in used_p or gi in used_g:continue
        used_p.add(pi);used_g.add(gi)
        if ground[gi][3]>0:head.append(float(np.linalg.norm(result.keypoints.xy.cpu().numpy()[pi,0]-ground[gi][2])))
    return {'gt':len(ground),'predictions':len(boxes),'matched':len(used_g),'fp':len(boxes)-len(used_p),'small_gt':sum(int(g[1][3]-g[1][1]<60) for g in ground),'small_matched':sum(int(ground[gi][1][3]-ground[gi][1][1]<60) for gi in used_g),'heads':head}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--versions',nargs='+',default=['v4','v5'])
    parser.add_argument('--sizes',nargs='+',type=int,default=[640,960,1280])
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True)
    data=[]
    for split in ('newcheck','check','test'):
        for path in sorted((ROOT/'data/pose-v5/images'/split).glob('*')):
            labels=ROOT/'data/pose-v5/labels'/split/(path.stem+'.txt')
            rows=[list(map(float,line.split())) for line in labels.read_text().splitlines()]
            data.append((split,path,cv2.imread(str(path)),rows))
    # Display two positive examples per new map when available, chosen before sweep results.
    previews=[];maps={}
    for split,path,frame,rows in data:
        if split!='newcheck' or not rows:continue
        map_name=path.name.split('-de_')[1].split('-r')[0]
        if maps.get(map_name,0)<2:previews.append(path.name);maps[map_name]=maps.get(map_name,0)+1
    configs=[(v,s) for v in args.versions for s in args.sizes]
    random.Random(42).shuffle(configs)
    reports=[]
    for version,size in configs:
        model=YOLO(str(ROOT/f'runs/pose-{version}/weights/best.pt'))
        params={'device':0,'imgsz':size,'conf':.25,'verbose':False}
        # Both source shapes (720p/4:3) warmed; timing excludes disk, plotting, and matching.
        for _ in range(5):
            for shape in ((720,1280),(1080,1440)):
                model.predict(np.zeros((*shape,3),dtype=np.uint8),**params)
        details=[];timings=[]
        canvas=Image.new('RGB',(1280,4*265),'white');draw=ImageDraw.Draw(canvas)
        for split,path,frame,rows in data:
            torch.cuda.synchronize();start=time.perf_counter()
            result=model.predict(frame,**params)[0]
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            item=match(result,rows);item.update({'split':split,'image':path.name,'predict_ms':elapsed})
            details.append(item)
            if path.name in previews:
                idx=previews.index(path.name);im=Image.fromarray(result.plot()[...,::-1]);im.thumbnail((426,240))
                x,y=idx%3*426,idx//3*265;canvas.paste(im,(x,y));draw.text((x+3,y+242),f'{version} {size} | GT {item["gt"]} matched {item["matched"]} FP {item["fp"]}',fill='black')
        # A second timing pass in reverse order to expose warm/cache/order variation.
        for split,path,frame,rows in reversed(data):
            if split!='newcheck':continue
            torch.cuda.synchronize();start=time.perf_counter();model.predict(frame,**params);torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
        first=[r['predict_ms'] for r in details if r['split']=='newcheck']
        timing={'pass1_median_ms':float(np.median(first)),'pass2_median_ms':float(np.median(timings)),'median_ms':float(np.median(first+timings)),'p95_ms':float(np.percentile(first+timings,95)),'samples':len(first+timings)}
        stats={}
        for split in ('newcheck','check','test'):
            part=[r for r in details if r['split']==split]
            total={key:sum(r[key] for r in part) for key in ('gt','predictions','matched','fp','small_gt','small_matched')}
            heads=[e for r in part for e in r['heads']]
            total.update({'images':len(part),'precision':total['matched']/total['predictions'] if total['predictions'] else 0,'recall':total['matched']/total['gt'] if total['gt'] else 0,'conditional_head_mean_pixels':float(np.mean(heads)) if heads else None,'head_matches':len(heads)})
            stats[split]=total
        reports.append({'version':version,'imgsz':size,'timing_720p_predict':timing,'sets':stats,'details':details})
        canvas.save(OUT/f'preview-{version}-{size}.jpg')
        (OUT/'results.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),'protocol':'FP32 PyTorch, batch1, conf .25, class IoU .5, warmup 10; synchronized wall time around predict on predecoded BGR arrays. No disk, capture, overlay, video encoding or game load. Newcheck used for tuning, not final test. 93 correlated newcheck frames and 31 labels. Small: body height<60px in original image; head error conditional. Historical source overlap and V4 assisted-label bias remain possible.','results':reports},indent=2),encoding='utf-8')
        print(version,size,json.dumps({'timing':timing,'sets':stats}),flush=True)
    # Check previous baseline reproduced before drawing conclusions.
    baseline=json.loads((ROOT.parent/'2026-10-17/new-chat/outputs/fifth-training/comparison.json').read_text())
    for run in reports:
        if run['imgsz']!=640 or run['version'] not in baseline:continue
        for split,metrics in run['sets'].items():
            old=baseline[run['version']][split]
            assert metrics['matched']==old['matched'] and metrics['fp']==old['false_positives'],(run['version'],split,metrics,old)
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=['version','imgsz','split','gt','matched','fp','precision','recall','small_gt','small_matched','median_predict_ms'])
        writer.writeheader()
        for run in sorted(reports,key=lambda r:(r['version'],r['imgsz'])):
            for split,m in run['sets'].items():
                writer.writerow(dict(version=run['version'],imgsz=run['imgsz'],split=split,**{k:m[k] for k in ('gt','matched','fp','precision','recall','small_gt','small_matched')},median_predict_ms=run['timing_720p_predict']['median_ms']))
    print('COMPLETED; available historical 640 baselines checked',flush=True)
