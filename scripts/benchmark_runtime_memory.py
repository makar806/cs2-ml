"""Compare runtime on saved images in separate processes, never capture the PC."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time
import cv2
import psutil

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('backend',choices=['native','legacy']);args=parser.parse_args()
    files=sorted((ROOT/'data/pose-v7/images/val').glob('*'))
    if not files:files=sorted((ROOT/'data/pose-v7/images/train').glob('*'))
    files=files[::max(1,len(files)//12)][:12]
    process=psutil.Process();start=time.perf_counter()
    if args.backend=='native':
        from mvp_native_trt import NativeTensorRT
        model=NativeTensorRT(ROOT/'runs/pose-v7/weights/best.engine');predict=model.predict
    else:
        from mvp_runtime import load_model
        from mvp_core import decode_result
        model,_=load_model('tensorrt',1280)
        predict=lambda image:decode_result(model.predict(image,imgsz=1280,conf=.35,device=0,verbose=False)[0])
    import numpy as np
    for _ in range(3):predict(np.zeros((720,1280,3),np.uint8))
    warmup=time.perf_counter()-start;durations=[];records=[]
    for path in files:
        image=cv2.imread(str(path));pred=predict(image)
        records.append(dict(path=str(path),detections=[asdict(d) for d in pred]))
    image=cv2.resize(cv2.imread(str(files[0])),(1280,720))
    for _ in range(40):
        start=time.perf_counter();predict(image);durations.append((time.perf_counter()-start)*1000)
    mem=process.memory_info()
    report=dict(backend=args.backend,torch_loaded='torch' in sys.modules,warmup_seconds=warmup,
                working_set_mb=mem.rss/1048576,private_mb=getattr(mem,'private',mem.vms)/1048576,
                peak_working_set_mb=getattr(mem,'peak_wset',mem.rss)/1048576,
                predict_median_ms=statistics.median(durations),predict_p95_ms=sorted(durations)[38],records=records)
    out=ROOT/'runs/mvp-v3.4';out.mkdir(parents=True,exist_ok=True)
    (out/f'{args.backend}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
    if args.backend=='native':model.close()
