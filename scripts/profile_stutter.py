"""Read-only per-present A/B profiler. Does not inject input or change settings.

Run record-performance.cmd manually as administrator: PresentMon needs ETW rights.
Only this run's own child processes/trace sessions are created and terminated.
"""
import argparse
from contextlib import contextmanager, nullcontext
import csv
import ctypes
import json
import math
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
from queue import Empty
import psutil

ROOT=Path(__file__).resolve().parents[1]
RUN_OUTPUT=None


@contextmanager
def system_capture_trace(directory):
    """Save one named WPR session across both A/B stages, even on stage failure."""
    instance = ['-instancename', f'CS2ML_capture_{directory.name}']
    with (directory / 'wpr.log').open('w', encoding='utf-8') as log:
        # GPU.Light already collects thread switches, ready events and GPU activity.
        # The previous combined verbose profiles dropped millions of events.
        subprocess.run(['wpr', '-start', 'GPU.light', '-filemode', *instance],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
        try:
            yield
        finally:
            print('Saving the system trace. Wait for the completion sound.', flush=True)
            try:
                subprocess.run(['wpr', '-stop', str(directory / 'system.etl'),
                                'CS2 capture isolation', '-skipPdbGen', *instance],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            except (subprocess.SubprocessError, OSError):
                subprocess.run(['wpr', '-cancel', *instance], stdout=log,
                               stderr=subprocess.STDOUT, timeout=30)
                raise


def panel_running():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenMutexW.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_wchar_p]
    kernel.OpenMutexW.restype=ctypes.c_void_p;kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenMutexW(0x00100000,False,'Local\\CS2MLMVPPanelV2')
    if handle:kernel.CloseHandle(handle);return True
    return ctypes.get_last_error()==5 # access denied still indicates an existing panel


def wait_for_game(find,is_foreground,timeout=90,clock=time.monotonic,sleep=time.sleep):
    deadline=clock()+timeout
    while clock()<deadline:
        hwnd=find('local-bots')
        if hwnd and is_foreground(hwnd):return hwnd
        sleep(.1)
    raise RuntimeError(f'CS2 was not ready within {timeout:g} seconds; no capture was started')


def wait_for_live_state(read_state, is_focused, timeout=180, clock=time.monotonic, sleep=time.sleep):
    deadline=clock()+timeout
    while clock()<deadline:
        state=read_state()
        if state.get('ready') and state.get('armed') and is_focused():
            return state
        sleep(.2)
    raise RuntimeError('MVP did not become ready and active with CS2 focused; no trace was started')


def latest_panel_state():
    paths=sorted((ROOT/'runs/live-mvp').glob('*/status.json'))
    try:return json.loads(paths[-1].read_text(encoding='utf-8'))
    except (OSError,ValueError,IndexError):return {}


def percentile(values,p):
    values=sorted(values);index=(len(values)-1)*p
    low=math.floor(index);high=math.ceil(index)
    return values[low]+(values[high]-values[low])*(index-low)


def summarize_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as source:rows=list(csv.DictReader(source))
    if not rows:raise ValueError('No CS2 present events recorded')
    groups={}
    for row in rows:groups.setdefault(row.get('SwapChainAddress','main'),[]).append(row)
    rows=max(groups.values(),key=len)
    fields={name.lower():name for name in rows[0]}
    key=next((fields[name] for name in ('msbetweenpresents','frametime') if name in fields),None)
    if key is None:raise ValueError(f'Unknown PresentMon CSV fields: {list(rows[0])}')
    values=[]
    for row in rows:
        try:value=float(row[key])
        except (ValueError,TypeError):continue
        if math.isfinite(value) and value>0:values.append(value)
    if not values:raise ValueError('No positive frame times')
    return dict(n=len(values),field=key,mean_ms=sum(values)/len(values),
                median_ms=percentile(values,.5),p95_ms=percentile(values,.95),
                p99_ms=percentile(values,.99),max_ms=max(values),
                above_33ms=sum(v>33.333 for v in values),above_50ms=sum(v>50 for v in values),
                above_100ms=sum(v>100 for v in values),approx_fps=1000/(sum(values)/len(values)))


def own_worker(stage,output,hz=30):
    from mvp_runtime import inference_process
    ctx=mp.get_context('spawn');stop=ctx.Event();packets=ctx.Queue(1);messages=ctx.Queue(4)
    mode,backend=('queries','none') if stage=='queries' else stage.split('_',1)
    config=dict(mode='local-bots',imgsz=960,hz=hz,pipeline=mode,
                model_backend='native',capture_backend=backend,output=str(output))
    target=(replay_worker if stage=='inference_replay' else
            isolation_worker if stage=='queries' or backend.endswith('_fixed') else inference_process)
    process=ctx.Process(target=target,args=(config,stop,packets,messages),daemon=True)
    process.start()
    until=time.monotonic()+40;ready=False
    try:
        while time.monotonic()<until and process.is_alive():
            try:
                message=messages.get(timeout=.2)
                if message.get('error'):raise RuntimeError(message['error'])
                if message.get('ready'):ready=True;break
            except Empty:pass
        if not ready:raise RuntimeError('Benchmark worker did not become ready')
    except BaseException:
        stop.set();process.join(3)
        if process.is_alive():process.terminate();process.join(2)
        raise
    return process,stop,packets,messages


def replay_worker(config,stop,packets,messages):
    """GPU competition without any screen-capture API or input controller."""
    from mvp_runtime import load_model,latest_put
    import cv2
    model=None
    try:
        import os
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
        files=sorted((ROOT/'data/pose-v7/images/val').glob('*'))
        if not files:raise RuntimeError('No saved validation image for inference-only isolation')
        image=cv2.imread(str(files[0]))
        if image is None:raise RuntimeError('Cannot decode replay image')
        image=cv2.resize(image,(1920,1080))
        model,_=load_model('native',960)
        for _ in range(3):model.predict(image)
        latest_put(messages,{'ready':True})
        with (Path(config['output'])/'api.jsonl').open('w',encoding='utf-8') as log:
            while not stop.is_set():
                before=time.perf_counter();cpu=time.thread_time();model.predict(image)
                elapsed=time.perf_counter()-before
                log.write(json.dumps(dict(time=time.time(),api='inference_replay',ms=elapsed*1000,
                    cpu_ms=(time.thread_time()-cpu)*1000,stages=getattr(model,'last_timings',{})))+'\n');log.flush()
                # Fixed requested 30 Hz: isolate inference itself, without the
                # production adaptive governor hiding slow predictions.
                time.sleep(max(.001,1/config['hz']-elapsed))
    except Exception:
        import traceback
        error=traceback.format_exc();(Path(config['output'])/'error.txt').write_text(error)
        latest_put(messages,{'error':error})
    finally:
        if model:model.close()
        for queue in (packets,messages):queue.cancel_join_thread()


def isolation_worker(config,stop,packets,messages):
    """Separate window queries from capture; frozen geometry is TEST-ONLY."""
    from mvp_windows import find_window,foreground,client_rect,Capture
    from mvp_runtime import latest_put
    capture=None;mss_capture=None;sampler=None
    import logging
    logger=logging.getLogger('dxcam');previous_level=logger.level
    handler=logging.FileHandler(Path(config['output'])/'capture-driver.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logger.addHandler(handler);logger.setLevel(logging.DEBUG)
    try:
        hwnd=find_window('local-bots');rect=client_rect(hwnd)
        if not hwnd or rect is None:raise RuntimeError('No local game window')
        backend=config['capture_backend']
        hz=config['hz']
        if config['pipeline']=='queries':pass
        elif backend in ('dxgi_fixed','dxgi_legacy_fixed'):
            # Process-local alternative implemented by the installed DXcam.
            # No package patch, registry, game or driver setting is changed.
            if backend=='dxgi_legacy_fixed':
                import os
                os.environ['DXCAM_USE_DUPLICATE_OUTPUT1']='0'
            capture=Capture(hz,'dxcam');capture.set_region(rect)
        elif backend=='wgc_fixed':
            import mvp_wgc
            mvp_wgc.client_rect=lambda _:rect
            capture=mvp_wgc.WindowCapture(hwnd,hz)
        elif backend=='mss_fixed':
            import mss
            mss_capture=mss.mss()
        else:raise ValueError(backend)
        latest_put(messages,{'ready':True})
        output=Path(config['output']);last_find=0;last_frame=-1
        with (output/'api.jsonl').open('w',encoding='utf-8') as log:
            while not stop.is_set():
                start=time.perf_counter()
                if config['pipeline']=='queries':
                    if start-last_find>1:
                        before=time.perf_counter();find_window('local-bots');last_find=start
                        log.write(json.dumps(dict(time=time.time(),api='find_window',ms=(time.perf_counter()-before)*1000))+'\n')
                    before=time.perf_counter();client_rect(hwnd);foreground(hwnd)
                    log.write(json.dumps(dict(time=time.time(),api='geometry',ms=(time.perf_counter()-before)*1000))+'\n')
                elif mss_capture:
                    x,y,x2,y2=rect;before=time.perf_counter()
                    mss_capture.grab(dict(left=x,top=y,width=x2-x,height=y2-y))
                    log.write(json.dumps(dict(time=time.time(),api='mss',ms=(time.perf_counter()-before)*1000))+'\n')
                elif capture:
                    if capture.error:raise RuntimeError(capture.error)
                    frame=capture.get()
                    if frame is not None and frame[0]!=last_frame:
                        last_frame=frame[0]
                        log.write(json.dumps(dict(time=time.time(),api='capture',ms=capture.last_ms,
                                                  capture_id=frame[0],captured_at=frame[1]))+'\n')
                log.flush();stop.wait(max(.001,1/hz-(time.perf_counter()-start)))
    except Exception:
        import traceback
        error=traceback.format_exc();(Path(config['output'])/'error.txt').write_text(error)
        latest_put(messages,{'error':error})
    finally:
        if capture:capture.close()
        if mss_capture:mss_capture.close()
        logger.removeHandler(handler);handler.close();logger.setLevel(previous_level)
        for queue in (packets,messages):queue.cancel_join_thread()


def main():
    global RUN_OUTPUT
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,default=20)
    parser.add_argument('--suite',choices=['full','isolation','live','gpu-isolation','capture-trace','capture-rate'],default='full')
    args=parser.parse_args()
    if not 10<=args.seconds<=60:raise ValueError('Stage duration must be 10–60 seconds')
    if not ctypes.windll.shell32.IsUserAnAdmin():
        raise RuntimeError('PresentMon requires ETW rights. Right-click record-performance.cmd and Run as administrator.')
    from mvp_windows import find_window,foreground,USER
    from mvp_metrics import MemorySampler
    # A second active MVP instance would invalidate all baseline comparisons.
    runs=sorted((ROOT/'runs/live-mvp').glob('*/status.json'))
    if runs:
        status=json.loads(runs[-1].read_text(encoding='utf-8'))
        if args.suite!='live' and status.get('ready') and status.get('armed') and panel_running():
            raise RuntimeError('Stop the model using F6 before recording')
    directory=ROOT/'runs/stutter-profile'/time.strftime('%Y%m%d-%H%M%S')
    directory.mkdir(parents=True,exist_ok=False)
    RUN_OUTPUT=directory
    results=[]
    if args.suite=='live':
        print('Keep the current MVP panel open and ready. Return to LOCAL CS2; toggle F8 yourself during this recording.',flush=True)
        print('Recorder never injects input. Recording duration:',args.seconds,'seconds. Results:',directory,flush=True)
    else:
        print('Return to CS2 and keep the local scene steady. No aiming or firing will run.',flush=True)
        duration = 'approximately one minute plus saving' if args.suite == 'capture-trace' else 'approximately two minutes'
        print('Bounded isolated stages,', duration, '. Results:',directory,flush=True)
        print('Close the ordinary MVP panel completely. Do not press F8 during this recording.',flush=True)
        deadline=time.monotonic()+90
        while panel_running() and time.monotonic()<deadline:time.sleep(.2)
        if panel_running():raise RuntimeError('Close the ordinary MVP panel, not only F6, then rerun the recording')
    print('Waiting up to 90 seconds for CS2 to be restored and focused...',flush=True)
    hwnd=wait_for_game(find_window,foreground)
    if args.suite=='live':
        print('Waiting for MVP Ready and F8 ACTIVE (up to 180 seconds). The timed recording has NOT started yet.',flush=True)
        wait_for_live_state(latest_panel_state,lambda:foreground(hwnd))
    pid=ctypes.c_ulong();USER.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
    time.sleep(3)
    stages=(['dxgi_30_first','dxgi_20','dxgi_30_repeat'] if args.suite=='capture-rate' else
            ['baseline','capture_dxgi_fixed'] if args.suite=='capture-trace' else
            ['baseline','capture_dxgi_fixed','capture_dxgi_legacy_fixed','inference_replay'] if args.suite=='gpu-isolation' else
            ['live'] if args.suite=='live' else
            (['baseline','capture_dxgi','inference_dxgi','capture_wgc','inference_wgc'] if args.suite=='full'
             else ['baseline','queries','capture_dxgi_fixed','capture_wgc_fixed','capture_mss_fixed']))
    recording = system_capture_trace(directory) if args.suite == 'capture-trace' else nullcontext()
    with recording:
        for number,stage in enumerate(stages):
            stage_seconds = 12 if args.suite == 'capture-trace' and stage == 'baseline' else args.seconds
            output=directory/stage;output.mkdir();process=stop=packets=messages=None
            monitor=trace=None;trace_log=None
            try:
                if args.suite=='capture-rate':
                    capture_hz = 20 if stage=='dxgi_20' else 30
                    process,stop,packets,messages=own_worker('capture_dxgi_fixed',output,capture_hz)
                elif stage not in ('baseline','live'):
                    process,stop,packets,messages=own_worker(stage,output)
                time.sleep(3) # exclude initialization/warmup from per-present capture
                if not foreground(hwnd):raise RuntimeError('Keep CS2 foreground through all stages')
                print(f'{number+1}/{len(stages)} {stage}',flush=True)
                path=output/'present.csv'
                trace_log=(output/'presentmon.log').open('w',encoding='utf-8')
                command=[str(ROOT/'tools/PresentMon-2.6.0-x64.exe'),'--process_id',str(pid.value),
                         '--session_name',f'CS2ML_{directory.name}_{number}','--output_file',str(path),
                         '--timed',str(stage_seconds),'--terminate_after_timed','--no_console_stats','--v1_metrics']
                if args.suite in ('live','capture-trace','capture-rate'):command.append('--qpc_time_ms')
                trace=subprocess.Popen(command,stdout=trace_log,stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW,cwd=ROOT)
                if args.suite=='live':
                    live_state=latest_panel_state()
                    if not live_state.get('panel_pid') or not live_state.get('worker_pid'):
                        raise RuntimeError('Restart the MVP panel with v3.20 before live recording: process IDs are required')
                    monitor=MemorySampler(live_state['worker_pid'],panel_pid=live_state['panel_pid'])
                else:
                    monitor=MemorySampler(process.pid if process else __import__('os').getpid())
                samples=[];deadline=time.monotonic()+stage_seconds+12
                while trace.poll() is None and time.monotonic()<deadline:
                    sample=dict(time=time.time(),perf_counter=time.perf_counter(),foreground=foreground(hwnd),external_panel=panel_running(),**monitor.sample())
                    if args.suite=='live':
                        try:
                            paths=sorted((ROOT/'runs/live-mvp').glob('*/status.json'))
                            mvp=json.loads(paths[-1].read_text(encoding='utf-8'))
                            sample['mvp']={key:mvp.get(key) for key in ('version','armed','fire','ready','frames','shots','capture_creations','latency_ms','status','panel_pid','worker_pid')}
                        except (OSError,ValueError,IndexError):pass
                    samples.append(sample)
                    if process is not None and not process.is_alive():raise RuntimeError('Workload process exited unexpectedly')
                    time.sleep(.5)
                if trace.poll() is None:raise RuntimeError('PresentMon did not finish its bounded capture')
                if trace.returncode:raise RuntimeError('PresentMon capture failed; see '+str(output/'presentmon.log'))
                summary=summarize_csv(path)
                summary.update(stage=stage,foreground_fraction=sum(s['foreground'] for s in samples)/len(samples),
                               ram_peak_percent=max(s['ram_percent'] for s in samples),
                               page_input_peak=max(s.get('pages_input_per_sec',0) for s in samples))
                if args.suite=='capture-rate':summary['capture_target_hz']=capture_hz
                if summary['foreground_fraction']<.95:summary['invalid_reason']='Game lost focus; comparison is invalid'
                if args.suite!='live' and any(s['external_panel'] for s in samples):summary['invalid_reason']='Ordinary MVP panel was opened during recording; comparison is invalid'
                if args.suite=='live':
                    states=[s['mvp'] for s in samples if 'mvp' in s]
                    summary['active_samples']=sum(bool(s.get('armed')) for s in states)
                    summary['versions']=sorted({s['version'] for s in states if s.get('version')})
                    summary['capture_creations_max']=max((s.get('capture_creations') or 0 for s in states),default=0)
                    if not summary['active_samples']:summary['invalid_reason']='F8 was never active in the sampled recording'
                (output/'memory.json').write_text(json.dumps(samples,indent=2))
                (output/'summary.json').write_text(json.dumps(summary,indent=2))
                results.append(summary);(directory/'summary.json').write_text(json.dumps(results,indent=2))
                print(json.dumps(summary),flush=True)
            finally:
                if trace is not None and trace.poll() is None:
                    # Only our explicitly named bounded recording, never another tool's session.
                    subprocess.run([str(ROOT/'tools/PresentMon-2.6.0-x64.exe'),'--session_name',
                                    f'CS2ML_{directory.name}_{number}','--terminate_existing_session'],
                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW,timeout=8)
                    trace.wait(timeout=8)
                if monitor:monitor.close()
                if trace_log:trace_log.close()
                if stop:
                    stop.set();process.join(4)
                    if process.is_alive():process.terminate();process.join(2)
                    for queue in (packets,messages):queue.close();queue.cancel_join_thread()
    print('DONE. Results saved:',directory,flush=True)
    import winsound
    winsound.MessageBeep()


if __name__=='__main__':
    mp.freeze_support()
    try:main()
    except Exception:
        import traceback
        error=traceback.format_exc();print(error,flush=True)
        if RUN_OUTPUT:(RUN_OUTPUT/'error.txt').write_text(error,encoding='utf-8')
        raise SystemExit(1)
