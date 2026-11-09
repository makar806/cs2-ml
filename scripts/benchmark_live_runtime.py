"""Read-only local-game runtime benchmark. Never imports or invokes Mouse."""
import argparse
import json
from pathlib import Path
import statistics
import time
import psutil
from mvp_native_trt import NativeTensorRT
from mvp_windows import find_window,foreground,client_rect,Capture
from mvp_wgc import WindowCapture

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=960);p.add_argument('--duration',type=float,default=12);p.add_argument('--capture',choices=['wgc','dxgi'],default='wgc');args=p.parse_args()
    hwnd=find_window('local-bots')
    if not hwnd or not foreground(hwnd):raise RuntimeError('Keep local CS2 foreground for benchmark')
    capture=None;model=None;samples=[]
    try:
        model=NativeTensorRT(ROOT/'runs/pose-v7/weights/best.engine',args.size)
        if args.capture=='wgc':capture=WindowCapture(hwnd,30)
        else:
            capture=Capture(30,'dxcam');capture.set_region(client_rect(hwnd))
        until=time.perf_counter()+args.duration;last=-1;next_at=0
        while time.perf_counter()<until:
            if not foreground(hwnd):raise RuntimeError('Game lost focus; benchmark invalid')
            frame=capture.get();now=time.perf_counter()
            if frame is None or frame[0]==last or now<next_at:
                time.sleep(.001);continue
            last,source,image,rect=frame;start=time.perf_counter();ds=model.predict(image);end=time.perf_counter()
            next_at=start+max(1/30,(end-start)/.55)
            samples.append(dict(predict_ms=(end-start)*1000,age_ms=(end-source)*1000,capture_ms=capture.last_ms,detections=len(ds)))
        if not samples:raise RuntimeError('No frames captured: '+capture.error)
        mem=psutil.Process().memory_info();summary={}
        for key in ('predict_ms','age_ms','capture_ms'):
            values=sorted(s[key] for s in samples[3:]);summary[key]=dict(median=statistics.median(values),p95=values[int(len(values)*.95)])
        result=dict(size=args.size,capture=args.capture,n=len(samples),fps=len(samples)/args.duration,
                    worker_rss_mb=mem.rss/1048576,system_ram_percent=psutil.virtual_memory().percent,metrics=summary)
        out=ROOT/'runs/mvp-v3.4';out.mkdir(parents=True,exist_ok=True)
        (out/f'live-{args.capture}-{args.size}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        if capture:capture.close()
        if model:model.close()
