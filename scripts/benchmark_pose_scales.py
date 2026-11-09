"""Compare class/Head on one immutable live fixture; no input, no saved screenshot."""
from dataclasses import asdict
import json
from pathlib import Path
import time
from mvp_capture import DesktopCapture
from mvp_native_trt import NativeTensorRT
from mvp_windows import find_window,foreground

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    hwnd=find_window('local-bots')
    if not foreground(hwnd):raise RuntimeError('Keep the local fixture in foreground')
    capture=DesktopCapture(hwnd,30);model=None
    try:
        end=time.perf_counter()+3;frame=None
        while frame is None and time.perf_counter()<end:frame=capture.get();time.sleep(.005)
        if frame is None:raise RuntimeError(capture.error or 'No frame')
        image=frame[2].copy();capture.close();capture=None
        result={}
        for size in (960,1280):
            model=NativeTensorRT(ROOT/'runs/pose-v7/weights/best.engine',size)
            result[str(size)]=[asdict(d) for d in model.predict(image)]
            model.close();model=None
        out=ROOT/'runs/mvp-v3.4/pose-scale-fixture.json';out.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        if capture:capture.close()
        if model:model.close()
