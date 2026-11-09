"""Static/moving 3-D camera fixture; no OS injection and no model accuracy claim."""
import json
import math
from pathlib import Path
import time
import numpy as np
from mvp_core import Controller,Detection,Settings

ROOT=Path(__file__).resolve().parents[1]

def ry(t):
    c,s=math.cos(t),math.sin(t);return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rx(t):
    c,s=math.cos(t),math.sin(t);return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def run():
    rows=[]
    for hz in (30,45,60):
        for age in (0,.02,.05):
            for speed in (0,120,-240,600):
                for initial in ((160,50),(-180,-60),(20,10)):
                    camera=np.eye(3);controller=Controller(Settings());errors=[];shots=[];costs=[]
                    for frame in range(round(hz*.6)):
                        now=10+frame/hz
                        # World target translates at a known velocity, observed age seconds ago.
                        point=np.array([(initial[0]+speed*(frame/hz-age))/480,initial[1]/480,1.])
                        ray=camera.T@point
                        if ray[2]<=0:break
                        x,y=640+480*ray[0]/ray[2],360+480*ray[1]/ray[2]
                        d=Detection(0,.99,(max(0,x-20),max(0,y-10),min(1280,x+20),min(720,y+100)),(x,y),.99)
                        start=time.perf_counter()
                        action=controller.step([d],1280,720,now,now-age,enabled=True,foreground=True,fire=True)
                        costs.append((time.perf_counter()-start)*1000)
                        camera=camera@ry(math.radians(action['dx']*.022))@rx(-math.radians(action['dy']*.022))
                        controller.acknowledge(action['dx'],action['dy'],1280,720)
                        actual=camera.T@np.array([(initial[0]+speed*frame/hz)/480,initial[1]/480,1.])
                        error=float(np.linalg.norm(actual[:2]/actual[2]*480));errors.append(error)
                        if action['shoot']:shots.append({'frame':frame,'error_px':error})
                    rows.append(dict(hz=hz,age=age,speed_px_s=speed,initial=initial,first_fire_frame=shots[0]['frame'] if shots else None,
                                     first_shot_error_px=shots[0]['error_px'] if shots else None,
                                     tracking_error_median_px=float(np.median(errors[2:])) if len(errors)>2 else None,
                                     control_p95_ms=float(np.percentile(costs,95)),shots=shots))
    static=[r for r in rows if r['speed_px_s']==0]
    assert all(r['first_fire_frame']==0 and r['first_shot_error_px']<.3 for r in static)
    assert all(r['first_fire_frame']==0 for r in rows)
    report={'protocol':'Synthetic exact detections, 3-D yaw/pitch camera, 108 combinations; excludes capture/inference/OS/game. First moving shot has no velocity history.',
            'cases':len(rows),'all_fire_on_first_valid_frame':True,
            'max_static_shot_error_px':max(r['first_shot_error_px'] for r in static),
            'control_p95_ms':float(np.percentile([r['control_p95_ms'] for r in rows],95)),
            'moving_tracking_median_px':float(np.median([r['tracking_error_median_px'] for r in rows if r['speed_px_s']])),
            'rows':rows}
    out=ROOT/'runs/mvp-v3';out.mkdir(exist_ok=True)
    (out/'snap-camera.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))

if __name__=='__main__':run()
