"""Offline identical-input comparison; synthetic results are never CS2 accuracy."""
import argparse,json,math,random,statistics,time
from pathlib import Path
from mvp_one_euro import OneEuro
from mvp_recorded_aim import HeadFilter

def trajectories(seed):
    rng=random.Random(seed);cases=[]
    for mode in ('stationary','linear','turns','stop_start'):
        rows=[];t=0.
        for i in range(360):
            t+=rng.uniform(.025,.045)
            if mode=='stationary':truth=(640.,360.)
            elif mode=='linear':truth=(400+35*t,360+12*t)
            elif mode=='turns':truth=(640+100*math.sin(t*1.8),360+40*math.cos(t*1.2))
            else:
                phase=t%4
                truth=(640+60*min(2,phase) if phase<2 else 760.,360.)
            rows.append(dict(t=t,truth=truth,point=tuple(v+rng.gauss(0,1.5) for v in truth),radius=5.))
        cases.append((mode,rows))
    return cases

def evaluate(cases,method,cutoff=1.,beta=.01):
    result={};cost=[]
    for name,rows in cases:
        f=HeadFilter();e=[OneEuro(cutoff,beta),OneEuro(cutoff,beta)];errors=[];outputs=[];refs=[]
        for i,row in enumerate(rows):
            start=time.perf_counter_ns()
            if method=='raw':point=row['point']
            elif method=='current':point=f.observe(tuple(row['point']),row['t'],row['radius'])[0]
            else:point=tuple(e[j](row['point'][j],row['t']) for j in (0,1))
            cost.append((time.perf_counter_ns()-start)/1000)
            if i>=30:
                outputs.append(point);refs.append(row['truth']);errors.append(math.dist(point,row['truth']))
        result[name]=dict(rmse_px=math.sqrt(statistics.mean(x*x for x in errors)),p95_error_px=sorted(errors)[int(.95*(len(errors)-1))],samples=len(errors))
        if name=='stationary':
            result[name]['jitter_rms_px']=math.sqrt(sum(statistics.pvariance(p[j] for p in outputs) for j in (0,1)))
    return dict(cases=result,median_cost_us=statistics.median(cost))

def run():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    train=trajectories(42);test=trajectories(20261031)
    grid=[]
    for cutoff in (.5,1,2,4,8):
        for beta in (0,.005,.01,.03,.1,.3):
            r=evaluate(train,'euro',cutoff,beta)
            score=statistics.mean(x['rmse_px'] for x in r['cases'].values())
            grid.append((score,cutoff,beta))
    _,cutoff,beta=min(grid)
    report=dict(kind='synthetic identical-input comparison',parameters=dict(min_cutoff=cutoff,beta=beta,derivative_cutoff=1.),training_seed=42,test_seed=20261031,noise_std_px=1.5,radius_px=5.,results={m:evaluate(test,m,cutoff,beta) for m in ('raw','current','euro')},limitations=['No CS2 ground-truth trajectories available in these live logs.','No camera movement, source age prediction or control is simulated here.','Parameters selected on synthetic training trajectories only.','RMSE includes both smoothing error and lag; no claim of real-world improvement.'])
    Path(a.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
