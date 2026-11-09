"""Run the actual input thread against fake camera/Mouse, never Windows input."""
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
from queue import Queue
import statistics
import threading
import time
from types import SimpleNamespace
import live_mvp as live
from mvp_core import Controller,Settings,Detection
from mvp_motion import MotionPlan,MotionHistory

ROOT=Path(__file__).resolve().parents[1]
SIZE=(1280,720);RECT=(0,0,1280,720)

class FakeMouse:
    def __init__(self):self.camera=0.;self.events=[]
    def act(self,dx,dy,shoot=False):
        self.camera+=math.radians(dx*.022)
        self.events.append((time.perf_counter(),dx,dy,shoot))
    def tick(self):pass
    def release(self):pass


def main():
    # Replacing these references is process-local and makes the engine incapable
    # of keyboard polling or injecting input; no Engine constructor is called.
    live.USER=SimpleNamespace(GetAsyncKeyState=lambda _:0)
    live.foreground=lambda _:True;live.client_rect=lambda _:RECT
    e=live.Engine.__new__(live.Engine)
    e.lock=threading.RLock();e.stop=threading.Event();e.worker_stop=threading.Event()
    e.worker=SimpleNamespace(is_alive=lambda:True);e.packets=Queue();e.messages=Queue()
    e.mode='simulator';e.pipeline='full';e.hwnd=123;e.armed=True;e.fire=True;e.ready=True
    e.last_motion_at=0;e.last_snap_at=0;e.settings=Settings()
    e.controller=Controller(e.settings);e.motion=MotionPlan();e.history=MotionHistory();e.mouse=FakeMouse()
    e.state=dict(input_error='',moves=0,shots=0,status='',glide_steps=0)
    e.output=ROOT/'runs/mvp-v3.3/fake-input-thread';e.output.mkdir(parents=True,exist_ok=True)
    thread=threading.Thread(target=e.inputs);thread.start()
    start=time.perf_counter();next_frame=start;waiting=[];frame=0
    try:
        while time.perf_counter()-start<3:
            now=time.perf_counter()
            if now>=next_frame:
                elapsed=now-start;world=80+90*math.sin(elapsed*1.5)
                hx=640+480*math.tan(math.atan2(world,480)-e.mouse.camera)
                d=Detection(0,.99,(hx-20,350,hx+20,450),(hx,360),.99)
                frame+=1
                packet=dict(id=frame,captured_at=now,hwnd=123,rect=RECT,size=SIZE,detections=[asdict(d)],
                            fps=45,predict_ms=25,latency_ms=25,capture_ms=0,frames=frame)
                waiting.append((now+.025,packet));next_frame+=1/45
            while waiting and waiting[0][0]<=now:e.packets.put(waiting.pop(0)[1])
            time.sleep(.001)
    finally:e.stop.set();thread.join(timeout=2)
    if thread.is_alive():raise RuntimeError('Input thread did not stop')
    actions=[json.loads(line) for line in (e.output/'actions.jsonl').read_text().splitlines()]
    glide=[a for a in actions if a['glide']]
    counts=Counter(a['source'] for a in glide)
    gaps=[(b['time']-a['time'])*1000 for a,b in zip(glide,glide[1:]) if a['source']==b['source']]
    result=dict(fake_mouse=True,actual_engine_thread=True,version='3.3',state=e.state,
                actions=len(actions),smooth_sources=len(counts),
                median_steps_per_source=statistics.median(counts.values()),
                median_same_source_gap_ms=statistics.median(gaps),
                p95_same_source_gap_ms=sorted(gaps)[int(len(gaps)*.95)])
    (e.output/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert not e.state['input_error']
    assert result['median_steps_per_source']>=3,'Actual engine still collapses glide steps'
    assert result['median_same_source_gap_ms']<8,'Actual engine still wakes too slowly'

if __name__=='__main__':main()
