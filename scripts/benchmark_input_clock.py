"""Measure actual Windows wait/scheduler timing without any mouse input."""
import json
from pathlib import Path
import statistics
import threading
import time
from collections import Counter
from mvp_motion import MotionPlan


def summary(values):
    values=sorted(values)
    return dict(n=len(values),median=statistics.median(values),p95=values[min(len(values)-1,int(len(values)*.95))],max=max(values))


def measure(wait):
    delays=[]
    for _ in range(100):
        start=time.perf_counter();wait(.001);delays.append((time.perf_counter()-start)*1000)
    counts=[];gaps=[]
    for _ in range(25):
        p=MotionPlan();start=time.perf_counter();times=[]
        a=dict(dx=100,dy=-20,shoot=False,tracking=True,target=True,error=(20,0),reason='snap_aim')
        p.begin(a,(1280,720),(0,0,1280,720),start,start,.1)
        while p.pending:
            if p.poll(time.perf_counter()):times.append(time.perf_counter())
            wait(.001)
        counts.append(len(times));gaps.extend((b-a)*1000 for a,b in zip(times,times[1:]))
    return dict(wait_1ms=summary(delays),steps_per_18ms=summary(counts),step_gap_ms=summary(gaps) if gaps else None)


if __name__=='__main__':
    result={name:measure(wait) for name,wait in [('event_wait',threading.Event().wait),('time_sleep',time.sleep)]}
    baseline=Path(__file__).resolve().parents[1]/'runs/live-mvp/20261025-032256/actions.jsonl'
    if baseline.exists():
        actions=[json.loads(line) for line in baseline.read_text().splitlines()]
        smooth=[a for a in actions if a.get('glide')]
        counts=Counter(a['source'] for a in smooth)
        gaps=[(b['time']-a['time'])*1000 for a,b in zip(smooth,smooth[1:]) if a['source']==b['source']]
        result['v3.2_real_log']=dict(actions=len(actions),smooth_actions=len(smooth),
            smooth_steps_per_source=summary(list(counts.values())),same_source_gap_ms=summary(gaps))
    output=Path(__file__).resolve().parents[1]/'runs/mvp-v3.3';output.mkdir(parents=True,exist_ok=True)
    (output/'clock.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
