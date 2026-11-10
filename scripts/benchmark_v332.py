"""Before/after synthetic checks against the exact handoff commit."""
import argparse
import json
from pathlib import Path
import subprocess
import types
from benchmark_recorded_aim import run
from mvp_ak_spray import AKSpray
from mvp_zone_aim import ZoneController

ROOT=Path(__file__).resolve().parents[1]
BASE='b3833262a70bf2851c164d46fa0fb7a2bf80694e'


def historical(path,revision):
    source=subprocess.check_output(['git','show',f'{revision}:{path}'],cwd=ROOT,text=True)
    module=types.ModuleType('historical_'+Path(path).stem)
    module.__file__=str(ROOT/path)
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module


def benchmark(revision):
    old_zone=historical('scripts/mvp_zone_aim.py',revision).ZoneController
    old_spray=historical('scripts/mvp_ak_spray.py',revision).AKSpray
    profile=dict(duration_seconds=.2492,reference_angle_degrees=34.459)
    cases=[]
    for filter_name in ('current','one_euro'):
        for scenario,kwargs in (
            ('stationary',dict(offset=250,speed=0)),
            ('moving_xy',dict(offset=250,speed=-60,offset_y=-80,speed_y=40)),
            ('turn',dict(offset=100,speed=100,turn_at=1.5)),
            ('large_transfer',dict(offset=450,speed=0)),
        ):
            for delay in (.03,.07):
                common=dict(delay=delay,noise=.5,head_filter=filter_name,fire_policy='zone',**kwargs)
                before=run(profile,controller_type=old_zone,**common)
                after=run(profile,controller_type=ZoneController,**common)
                keys=('first_within_5px_seconds','settled_within_5px_for_200ms_seconds',
                      'tracking_error_p50_px','tracking_error_p95_px','tracking_error_p99_px',
                      'min_signed_error_x','max_signed_error_x','max_input_counts')
                cases.append(dict(scenario=scenario,filter=filter_name,delay=delay,
                    before={k:before[k] for k in keys},after={k:after[k] for k in keys}))
    old=old_spray(.97)
    new=AKSpray(.97)
    old.start(0.,True,None)
    new.start(0.,True,None)
    differences=[]
    for i in range(1,3051):
        elapsed=i/1000
        a=old.correction(elapsed)
        b=new.correction(elapsed)
        if a!=b:differences.append(dict(elapsed=elapsed,before=a,after=b))
    return dict(baseline=revision,synthetic_only=True,aim_cases=cases,
                unchanged_head_controller=all(c['before']==c['after'] for c in cases),
                standing_wall_profile=dict(ticks=3050,exact_count_mismatches=len(differences),
                                           first_mismatches=differences[:5]),
                caveat='No real game hit rate, zoom calibration, recoil measurements or laptop FPS are inferred')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',default=BASE)
    p.add_argument('--output',type=Path,default=ROOT/'work/v332-control-comparison.json')
    args=p.parse_args()
    result=benchmark(args.baseline)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='aim_cases'},indent=2))
