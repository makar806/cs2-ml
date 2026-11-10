"""Isolated capture/inference process. Never injects mouse or keyboard events."""
import hashlib
import json
from pathlib import Path
from queue import Empty,Full
from mvp_logging import open_log
import re
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]

def latest_put(queue,value):
    try:queue.put_nowait(value);return
    except Full:pass
    try:queue.get_nowait()
    except Empty:pass
    try:queue.put_nowait(value)
    except Full:pass


def memory_paused(available_mb,was_paused=False,enabled=True):
    return bool(enabled) and available_mb<(900 if was_paused else 600)

def load_model(backend,size,blocking_wait=False):
    manifest=ROOT/'configs/mvp-runtime.json'
    runtime=json.loads(manifest.read_text()) if manifest.exists() else {}
    model_run = runtime.get('model_run', 'pose-v7')
    if not isinstance(model_run, str) or not re.fullmatch(r'pose-v[0-9]+', model_run):
        raise ValueError('Invalid model_run in the runtime manifest')
    model_dir = ROOT/'runs'/model_run/'weights'
    engine, weights = model_dir/'best.engine', model_dir/'best.pt'
    approved=(runtime.get('engine_approved') and engine.exists() and weights.exists() and
              hashlib.sha256(engine.read_bytes()).hexdigest()==runtime.get('engine_sha256') and
              hashlib.sha256(weights.read_bytes()).hexdigest()==runtime.get('weights_sha256'))
    if backend in ('auto','native'):
        if not approved:raise RuntimeError(f'Native runtime requires a verified engine and weights for {model_run}')
        import numpy as np
        from mvp_native_trt import NativeTensorRT
        model=NativeTensorRT(engine,size,blocking_wait=blocking_wait)
        for _ in range(3):model.predict(np.zeros((720,1280,3),np.uint8))
        return model,f'Native TensorRT FP16 (no torch) / {model_run}'+(' / blocking wait' if blocking_wait else '')
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1)
    use_engine=backend=='tensorrt' or (backend=='auto' and approved)
    try:
        model=YOLO(str(engine if use_engine else weights),task='pose')
        for _ in range(3):model.predict(np.zeros((720,1280,3),np.uint8),imgsz=size,conf=.35,device=0,verbose=False)
    except Exception:
        if backend!='auto' or not use_engine:raise
        model=YOLO(str(weights),task='pose');use_engine=False
        model.predict(np.zeros((720,1280,3),np.uint8),imgsz=size,conf=.35,device=0,verbose=False)
    return model,'TensorRT FP16' if use_engine else 'PyTorch FP32'

def inference_process(config,stop,packets,status,active=None):
    import os
    import cv2
    cv2.setNumThreads(1)
    from mvp_visual_tracking import VisualTracking
    from mvp_camera_motion import CameraMotion
    visual_tracking = VisualTracking()
    camera_motion = CameraMotion()
    # Scope numerical thread pools to this inference process, not the system or
    # training environment. Large idle BLAS pools needlessly reserve memory.
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    from dataclasses import asdict
    from collections import deque
    import statistics
    from mvp_windows import find_window,foreground,client_rect,USER
    from mvp_capture import create_capture
    from mvp_capture_session import CaptureSession
    from mvp_core import decode_result
    capture=None;model=None;output=Path(config['output']);log=None;events=None;session=None
    import logging
    capture_logger=logging.getLogger('dxcam');old_level=capture_logger.level
    capture_handler=logging.FileHandler(output/'capture-driver.log',encoding='utf-8')
    capture_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    capture_logger.addHandler(capture_handler);capture_logger.setLevel(logging.INFO)
    try:
        if config['pipeline']=='capture':model=None;backend='capture only'
        else:model,backend=load_model(config['model_backend'],config['imgsz'],config.get('blocking_gpu_wait',False))
        latest_put(status,{'ready':False,'model_backend':backend,'status':'Подготовка захвата. Не включай управление до «Готово».'})
        capture_backend=config.get('capture_backend','dxgi')
        warmup=config.get('capture_warmup_seconds',10 if config['pipeline']=='full' else 0)
        session=CaptureSession(lambda hwnd:create_capture(hwnd,config['hz'],capture_backend),
                               window_bound=capture_backend=='wgc',warmup=warmup)
        hwnd=None;last_find=0;last_frame=-1;last_predict=0;total=0;previous=None;last_memory=0;low_memory=False
        warm_predictions=0;warm_size=None;announced_ready=False;last_status=0;durations=deque(maxlen=9)
        feed=None;last_feed_frame=-1;feed_ms=0
        feed_status='disabled'
        hud=None;hud_signal={};hud_ms=0.
        if config.get('weapon_hud'):
            from mvp_hud import HudReader
            hud=HudReader(ROOT/'work/hud')
        if config.get('death_source','manual')=='visual' and config['pipeline']=='full' and config['mode']=='local-bots':
            from mvp_killfeed import VisualKillFeed
            template=ROOT/'work/killfeed/self-name.png'
            if template.exists():
                feed=VisualKillFeed.load(template)
                feed_status='template_loaded'
                latest_put(status,{'visual_feed_status':'Плашка: шаблон CT загружен'})
            else:
                feed_status='template_missing'
                latest_put(status,{'visual_feed_status':'Плашка: нужна калибровка ника; F7 — ручной сброс'})
        log=open_log(output/'frames.jsonl',config.get('async_logs',True))
        events=open_log(output/'capture-events.jsonl',config.get('async_logs',True))
        while not stop.is_set():
            now=time.perf_counter()
            if now-last_find>1 and (not hwnd or not USER.IsWindow(hwnd)):
                current=find_window(config['mode'],include_minimized=True);last_find=now
                if current!=hwnd:
                    hwnd=current;last_frame=-1
                    session.bind(hwnd)
                    latest_put(status,{'ready':announced_ready,'hwnd':hwnd})
            if config.get('foreground_only') and hwnd is not None and not foreground(hwnd):
                if session.capture is not None:
                    session.close()
                    capture=None
                    announced_ready=False
                    visual_tracking=VisualTracking()
                    camera_motion=CameraMotion()
                    events.write(json.dumps(dict(time=time.time(),event='release_unfocused'))+'\n');events.flush()
                if now-last_status>.25:
                    latest_put(status,{'ready':False,'hwnd':hwnd,'capture_active':False,
                                       'status':'Захват на паузе. Вернись в игру.'})
                    last_status=now
                stop.wait(.05);continue
            if hwnd is None:
                # Hybrid laptops may expose no usable NVIDIA desktop while
                # CS2 is closed. Do not create/retry duplication against that
                # unrelated desktop; initialize when an actual game exists.
                if session.capture is not None:
                    session.close();capture=None;announced_ready=False
                if now-last_status>.5:
                    latest_put(status,{'ready':False,'hwnd':None,'capture_active':False,'status':'Ожидание окна CS2' if config['mode']=='local-bots' else 'Ожидание окна симулятора'})
                    last_status=now
                stop.wait(.05);continue
            before=session.creations;capture=session.ensure()
            if session.creations!=before:
                last_frame=-1;warm_predictions=0;warm_size=None;announced_ready=False
                events.write(json.dumps(dict(time=time.time(),event='create',count=session.creations,hwnd=hwnd))+'\n');events.flush()
            if capture is None:
                stop.wait(.03);continue
            if hasattr(capture,'drain_slow_grabs'):
                for started,ended in capture.drain_slow_grabs():
                    events.write(json.dumps(dict(event='slow_grab',time=time.time(),started=started,
                                                 ended=ended,ms=(ended-started)*1000))+'\n')
                    events.flush()
            if capture.error:raise RuntimeError(capture.error)
            focused=foreground(hwnd)
            # The interactive panel captures only the foreground game. Other
            # callers can still prewarm desktop capture for their diagnostics.
            frame=capture.get() if config.get('foreground_only') else (capture.warm_frame() if hasattr(capture,'warm_frame') else capture.get())
            if frame is None:
                time.sleep(.003);continue
            frame_id,captured,image,rect=frame
            if hud is not None and focused and (active is None or active.is_set()):
                hud_start=time.perf_counter()
                hud_signal=hud.update(image,captured)
                hud_ms=(time.perf_counter()-hud_start)*1000
            if feed is not None and focused and frame_id!=last_feed_frame:
                last_feed_frame=frame_id
                before_feed=time.perf_counter();previous_sample=feed.last_at
                feed.update(image,captured,active is not None and active.is_set())
                if previous_sample!=feed.last_at:feed_ms=(time.perf_counter()-before_feed)*1000
            if frame_id!=last_frame:session.note_frame()
            size=(image.shape[1],image.shape[0])
            if size!=warm_size:
                was_ready=announced_ready;warm_size=size;warm_predictions=0;announced_ready=False
                if was_ready:latest_put(status,{'ready':False,'suspend_input':True,'status':'Изменился размер кадра. Подготовка модели; затем F8.'})
            model_warm=model is None or warm_predictions>=3
            ready=session.ready and model_warm
            if now-last_status>.25 or ready!=announced_ready:
                message=dict(ready=ready,hwnd=hwnd,capture_creations=session.creations,
                             capture_warm_remaining=session.remaining,capture=capture_backend.upper(),capture_active=True)
                if not ready:message['status']=f'Подготовка захвата: {session.remaining:.0f} с. Дождись «Готово».'
                elif not announced_ready:message['status']='Готово. Захват сохраняется; F8 включает вычисления и управление.'
                latest_put(status,message);last_status=now;announced_ready=ready
            need_warm=not model_warm
            enabled=active is None or config['pipeline']!='full' or active.is_set()
            if not need_warm and (not enabled or not focused or not ready):
                last_frame=frame_id;time.sleep(.003);continue
            if not need_warm and frame_id==last_frame:
                time.sleep(.001);continue
            if now-last_memory>1:
                import psutil
                available=psutil.virtual_memory().available/1048576;last_memory=now
                low_memory=memory_paused(available,low_memory,enabled=config.get("memory_guard",False))
                if low_memory:
                    latest_put(status,{'ready':True,'suspend_input':True,'status':f'Остановлено: свободно только {available:.0f} МБ RAM. Освободи память перед F8.'})
            if low_memory:
                stop.wait(.1);continue
            interval=max(1/config['hz'],statistics.median(durations)/.55 if durations else 0)
            if now-last_predict<interval:
                time.sleep(.001);continue
            last_frame=frame_id;start=time.perf_counter();last_predict=start
            # Bound age before spending GPU time. A blocked capture never creates backlog.
            if not need_warm and (start-captured>.10 or rect!=client_rect(hwnd)):continue
            detections=[]
            if model is not None:
                if backend.startswith('Native'):detections=model.predict(image)
                else:
                    result=model.predict(image,imgsz=config['imgsz'],conf=.35,device=0,verbose=False)[0]
                    detections=decode_result(result)
            end=time.perf_counter()
            if need_warm:
                warm_predictions+=1
                events.write(json.dumps(dict(time=time.time(),event='warm_model',size=size,ms=(end-start)*1000))+'\n');events.flush()
                continue
            durations.append(end-start);total+=1
            packet={'id':frame_id,'captured_at':captured,'hwnd':hwnd,'rect':rect,
                    'inference_started_at':start,'inference_ended_at':end,
                    'detections':[asdict(d) for d in detections],'size':(image.shape[1],image.shape[0]),
                    'frames':total,'predict_ms':(end-start)*1000,'capture_ms':capture.last_ms,
                    'latency_ms':(end-captured)*1000,'fps':1/(end-previous) if previous else 0,
                    'model_backend':backend,'budget_hz':1/interval,'capture':capture_backend.upper(),
                    'capture_creations':session.creations,'predict_stages':getattr(model,'last_timings',{})}
            packet.update(hud_signal,weapon_hud_ms=hud_ms,visual_feed_status=feed_status)
            if feed is not None:
                signal=feed.snapshot()
                packet.update(visual_kill_sequence=signal['sequence'],visual_kill_at=signal['last_event'],
                              visual_feed_rows=signal['rows'],visual_feed_ms=feed_ms,
                              visual_feed_sample_at=signal['sampled_at'])
            tracking_start=time.perf_counter()
            packet["visual_tracks"]=visual_tracking.update(image,detections,captured)
            packet["visual_tracking_ms"]=(time.perf_counter()-tracking_start)*1000
            camera_start=time.perf_counter()
            packet['camera_steps']=camera_motion.update(image,detections,captured)
            packet['camera_motion_ms']=(time.perf_counter()-camera_start)*1000
            packet['camera_motion_valid']=bool(packet['camera_steps'] and packet['camera_steps'][-1]['matrix'] is not None)
            packet['camera_scene_cut']=bool(packet['camera_steps'] and packet['camera_steps'][-1]['cut'])
            packet['packet_ready_at']=time.perf_counter()
            packet['capture_to_packet_ms']=(packet['packet_ready_at']-captured)*1000
            if hasattr(log,'stats'):
                packet['worker_log_health']={'frames':log.stats(),'capture':events.stats()}
            previous=end
            latest_put(packets,packet)
            log.write(json.dumps({'time':time.time(),**{k:v for k,v in packet.items() if k not in ('detections','camera_steps')},'detections':len(detections)})+'\n')
            if total%30==0:log.flush()
    except Exception:
        error=traceback.format_exc();(output/'error.txt').write_text(error,encoding='utf-8')
        latest_put(status,{'ready':False,'status':'Ошибка: '+error.splitlines()[-1],'error':error})
    finally:
        capture_logger.removeHandler(capture_handler);capture_handler.close();capture_logger.setLevel(old_level)
        if session:
            try:session.close()
            except Exception:pass
        if log:log.close()
        if events:events.close()
        if hasattr(log,'stats') and hasattr(events,'stats'):
            (output/'worker-log-health.json').write_text(json.dumps(
                {'frames':log.stats(),'capture':events.stats()},indent=2),encoding='utf-8')
        if model is not None and hasattr(model,'close'):model.close()
        packets.cancel_join_thread();status.cancel_join_thread()
