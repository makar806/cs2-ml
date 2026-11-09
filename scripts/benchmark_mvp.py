"""Repeatable inference/quality and virtual controller comparisons; never injects input."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from benchmark_resolutions import match
from mvp_core import Controller, Settings, Detection

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/mvp-optimization'

def controller_benchmark():
    spec=importlib.util.spec_from_file_location('baseline_core',OUT/'baseline_core.py')
    baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    reports={}
    # Camera fixture uses the same documented angular projection as the controller.
    # Fresh measurements, no inference delay: isolates controller convergence, not end-to-end latency.
    for name,module in [('baseline',baseline),('current',sys.modules[Controller.__module__])]:
        items=[]
        for hz in (20,30,60):
            for width,height in ((1280,720),(1440,1080)):
                for error in ((180,60),(-130,-70),(24,15),(7,5)):
                    for body_height in (40,200):
                        controller=module.Controller(module.Settings())
                        point=np.array((width/2+error[0],height/2+error[1]),dtype=float)
                        for step in range(90):
                            x,y=point;d=module.Detection(0,.95,(x-12,y-8,x+12,y+body_height),(x,y),.99)
                            t=10+step/hz
                            a=controller.step([d],width,height,t,t,enabled=True,foreground=True,fire=True)
                            if a['shoot']:break
                            point-=np.tan(np.radians(np.array((a['dx'],a['dy']))*.022))*(height/1.5)
                        items.append({'hz':hz,'size':[width,height],'initial_error':error,'body_height':body_height,
                                      'steps':step+1,'acquire_ms':step/hz*1000 if a['shoot'] else None,
                                      'actual_error_px':float(np.linalg.norm(point-(width/2,height/2))),
                                      'shot':a['shoot']})
        reports[name]=items
    (OUT/'controller.json').write_text(json.dumps(reports,indent=2))
    for name,items in reports.items():
        ms=[x['acquire_ms'] for x in items if x['shot']]
        print(name,{'shots':len(ms),'cases':len(items),'median_ms':float(np.median(ms)),'p95_ms':float(np.percentile(ms,95))},flush=True)

def inference_benchmark(engine=None):
    torch.set_num_threads(2)
    data=[]
    for split in ('newcheck','check','test'):
        for p in sorted((ROOT/f'data/pose-v7/images/{split}').glob('*')):
            rows=[list(map(float,line.split())) for line in (ROOT/f'data/pose-v7/labels/{split}/{p.stem}.txt').read_text().splitlines()]
            data.append((split,p.name,cv2.imread(str(p)),rows))
    configs=[('pytorch-fp32',ROOT/'runs/pose-v7/weights/best.pt',1280,False),
             ('pytorch-fp16',ROOT/'runs/pose-v7/weights/best.pt',1280,True),
             ('pytorch-fp16-960',ROOT/'runs/pose-v7/weights/best.pt',960,True)]
    if engine:configs.append(('tensorrt-fp16',Path(engine),1280,True))
    reports=[]
    for name,weights,size,half in configs:
        model=YOLO(str(weights),task='pose')
        params=dict(imgsz=size,device=0,conf=.35,quantize=16 if half else None,verbose=False,rect=True)
        for shape in ((720,1280),(1080,1440)):
            for _ in range(6):model.predict(np.zeros((*shape,3),np.uint8),**params)
        details=[]
        for split,filename,frame,rows in data:
            torch.cuda.synchronize();start=time.perf_counter()
            result=model.predict(frame,**params)[0]
            torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
            item=match(result,rows);item.update(split=split,image=filename,ms=ms);details.append(item)
        timing=[]
        for _,_,frame,_ in data[::3]*2:
            torch.cuda.synchronize();start=time.perf_counter();model.predict(frame,**params);torch.cuda.synchronize()
            timing.append((time.perf_counter()-start)*1000)
        stats={}
        for split in ('newcheck','check','test'):
            items=[x for x in details if x['split']==split]
            s={k:sum(x[k] for x in items) for k in ('gt','matched','fp','small_gt','small_matched')}
            heads=[v for x in items for v in x['heads']]
            s.update(images=len(items),head_mean_px=float(np.mean(heads)) if heads else None)
            stats[split]=s
        report=dict(name=name,imgsz=size,median_ms=float(np.median(timing)),p95_ms=float(np.percentile(timing,95)),sets=stats,details=details)
        reports.append(report)
        (OUT/'inference.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),
            'protocol':'Batch1, conf .35, warmed shapes, synchronized wall time, predecoded frames; excludes game/capture/overlay. Historical diagnostic sets, not fresh unbiased test.',
            'results':reports},indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='details'}),flush=True)
        del model
        torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller',action='store_true');p.add_argument('--engine');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.controller:controller_benchmark()
    else:inference_benchmark(a.engine)
