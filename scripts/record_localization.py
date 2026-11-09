"""Read-only localization recording: inference worker only, no controller/input loop."""
import argparse,json,multiprocessing as mp,time
from pathlib import Path
from queue import Empty
from datetime import datetime
from mvp_runtime import inference_process
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',required=True,choices=['static','player-motion','camera-motion','both-motion']);p.add_argument('--duration',type=float,default=60);p.add_argument('--graphics',default='unspecified');a=p.parse_args()
    if not 1<=a.duration<=3600:p.error('duration must be 1..3600 seconds')
    out=ROOT/'runs/localization'/datetime.now().strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True)
    config=dict(mode='local-bots',pipeline='full',model_backend='native',imgsz=1280,hz=30,capture_backend='dxgi',blocking_wait=False,death_source='manual',output=str(out))
    (out/'experiment.json').write_text(json.dumps(dict(scenario=a.scenario,graphics=a.graphics,control_enabled=False,configuration=config),indent=2),encoding='utf-8')
    ctx=mp.get_context('spawn');stop=ctx.Event();active=ctx.Event();active.set();packets=ctx.Queue(maxsize=1);messages=ctx.Queue(maxsize=2)
    worker=ctx.Process(target=inference_process,args=(config,stop,packets,messages,active))
    worker.start();deadline=time.perf_counter()+a.duration
    try:
        with (out/'predictions.jsonl').open('w',encoding='utf-8') as log:
            while time.perf_counter()<deadline:
                try:
                    while True:
                        m=messages.get_nowait()
                        if 'error' in m:raise RuntimeError(m['error'])
                except Empty:pass
                if not worker.is_alive():raise RuntimeError('Inference worker stopped; inspect error.txt')
                try:packet=packets.get(timeout=.2)
                except Empty:continue
                log.write(json.dumps(packet)+'\n');log.flush()
    except KeyboardInterrupt:pass
    finally:
        stop.set();worker.join(timeout=10)
        if worker.is_alive():print('Worker still shutting down; not launching another.')
    print(out)
if __name__=='__main__':mp.freeze_support();main()
