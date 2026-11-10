"""Local-bot CV demonstration with isolated inference and optional zone overlay."""
import argparse
from collections import deque
from dataclasses import asdict
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import threading
import time
import traceback
from mvp_core import Controller,Settings,Detection,aim_detection
from mvp_windows import key_down,release_movement_keys,Mouse,foreground,client_rect,input_environment,USER,C,W,KERNEL
from mvp_runtime import inference_process
from mvp_recorded_aim import visual_kill_matches
from mvp_motion import MotionPlan,MotionHistory
from mvp_camera_motion import CameraLock
from mvp_ak_spray import AKSpray
from mvp_windows import set_key
from mvp_weapon_state import WeaponState
from mvp_logging import open_log

ROOT=Path(__file__).resolve().parents[1]
VERSION='3.32'
TITLE=f'CS2 ML — MVP v{VERSION}'

class Engine:
    def __init__(self,mode,settings,imgsz,hz,pipeline='full',model_backend='auto',smooth_tracking=True,capture_backend='dxgi',blocking_gpu_wait=False,death_source='visual',aim_profile=None,spray_button=None,crouch_recoil_scale=.75,async_logs=True):
        self.mode=mode;self.settings=settings;self.pipeline=pipeline
        self.async_logs=async_logs
        self.stop=threading.Event();self.lock=threading.RLock()
        self.armed=False;self.fire=True;self.ready=False;self.hwnd=None;self.last_motion_at=0
        profile=None
        if aim_profile and settings.fire_policy!="zone":
            from mvp_recorded_aim import RecordedController,load_profile
            profile=load_profile(aim_profile)
            self.controller=RecordedController(settings,profile)
        else:self.controller=Controller(settings)
        if settings.fire_policy=='zone':
            from mvp_zone_aim import ZoneController
            self.controller=ZoneController(settings,profile)
            self.controller.distance_fire = spray_button is not None
        self.mouse=Mouse()
        self.spray = AKSpray(settings.sensitivity,crouch_recoil_scale) if spray_button is not None else None
        self.spray_button = 5 if spray_button is not None else None
        self.weapons = WeaponState() if self.spray else None
        self.physical_keys = None
        if profile or settings.fire_policy=="zone":
            from mvp_recorded_aim import RecordedMotionPlan
            self.motion=RecordedMotionPlan()
        else:self.motion=MotionPlan(duration=.018 if smooth_tracking else 0)
        self.zone_context=None
        self.history=MotionHistory();self.last_snap_at=0
        self.state={'status':'Загрузка модели в отдельном процессе…','frames':0,'shots':0,'moves':0,'fps':0,'latency_ms':0,'predict_ms':0,'capture':'WGC','input_error':'',
                    'version':VERSION,'blocking_gpu_wait':blocking_gpu_wait,'smooth_tracking':smooth_tracking,'input_loop_p95_ms':0,'input_loop_hz':0,'glide_steps':0,
                    'acquisition':'whole frame' if settings.radius<=0 else f'radius {settings.radius}',
                    'death_source':death_source,'visual_feed_status':'Плашка: ожидание захвата',
                    'confirmed_kill_releases':0}
        self.state['capture']=capture_backend.upper()
        if self.spray:
            self.state.update(spray_weapon=None, spray_button=5, spray_active=False,
                              crouch_recoil_scale=crouch_recoil_scale)
        self.state.update(aim_style='zone' if settings.fire_policy=='zone' else ('recorded' if profile else 'classic'),
                          fire_policy=settings.fire_policy,shot_interval=.25 if settings.fire_policy=='zone' else settings.shot_interval,memory_guard=settings.memory_guard,
                          aim_duration_ms=profile['duration_seconds']/1.24*1000 if profile and settings.fire_policy!='zone' else None,
                          single_shot_per_target=bool(profile) and settings.fire_policy!="zone",head_filter=settings.head_filter if profile or settings.fire_policy=="zone" else "none",
                          one_euro_parameters=dict(min_cutoff=.5,beta=.3,derivative_cutoff=1.))
        self.output=ROOT/'runs/live-mvp'/time.strftime('%Y%m%d-%H%M%S');self.output.mkdir(parents=True,exist_ok=False)
        config=dict(mode=mode,imgsz=imgsz,hz=hz,pipeline=pipeline,model_backend=model_backend,capture_backend=capture_backend,blocking_gpu_wait=blocking_gpu_wait,death_source=death_source,memory_guard=settings.memory_guard,foreground_only=True,weapon_hud=bool(self.spray),async_logs=async_logs,output=str(self.output))
        (self.output/'settings.json').write_text(json.dumps(dict(config,controller=asdict(settings),version=VERSION,manual_spray_button=self.spray_button,crouch_recoil_scale=crouch_recoil_scale,one_euro_parameters=self.state['one_euro_parameters'],capture_warmup_seconds=10,input_wait='CPython high-resolution time.sleep',tracking_duration_ms=self.motion.duration*1000,tracking_tick_ms=4),indent=2))
        (self.output/'diagnostics.json').write_text(json.dumps(input_environment(),indent=2))
        ctx=mp.get_context('spawn');self.worker_stop=ctx.Event();self.worker_active=ctx.Event();self.packets=ctx.Queue(maxsize=1);self.messages=ctx.Queue(maxsize=2)
        self.worker=ctx.Process(target=inference_process,args=(config,self.worker_stop,self.packets,self.messages,self.worker_active),daemon=True)
        self.worker.start()
        self.state.update(panel_pid=__import__('os').getpid(),worker_pid=self.worker.pid)
        from mvp_metrics import memory_monitor
        self.metrics_thread=threading.Thread(target=memory_monitor,args=(self.stop,self.state,self.output,self.worker.pid),daemon=True)
        self.metrics_thread.start()
        from mvp_stop import StopBeforeShot
        self.stop_before_shot=StopBeforeShot()
        self.last_snapshot=dict(self.state,armed=False,fire=True,ready=False,mode=self.mode,aim_zone=None,zone_rect=None)
        self.thread=threading.Thread(target=self.inputs,daemon=True);self.thread.start()

    def snapshot(self):
        # UI must not wait behind a delayed SendInput call holding the input lock.
        if not self.lock.acquire(blocking=False):
            return dict(self.last_snapshot,aim_zone=None,input_busy=True)
        try:
            zone=self.controller.zone_geometry(time.perf_counter()) if hasattr(self.controller,'zone_geometry') and self.armed else None
            context=getattr(self,'zone_context',None)
            self.last_snapshot=dict(self.state,armed=self.armed,fire=self.fire,ready=self.ready,mode=self.mode,aim_zone=zone,zone_rect=context['rect'] if context else None,input_busy=False)
            return dict(self.last_snapshot)
        finally:self.lock.release()

    def set_head_filter(self, mode):
        with self.lock:
            if not hasattr(self.controller,'set_head_filter'):
                raise ValueError('Выбор фильтра доступен с плавным профилем')
            if self.settings.head_filter==mode:return
            self.motion.cancel()
            self.release_fire()
            self.controller.set_head_filter(mode)
            self.state['head_filter']=mode
            settings_path=self.output/'settings.json'
            saved=json.loads(settings_path.read_text(encoding='utf-8'))
            saved['controller']=asdict(self.settings)
            settings_path.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
            with (self.output/'filter-events.jsonl').open('a',encoding='utf-8') as events:
                events.write(json.dumps(dict(time=time.time(),head_filter=mode,one_euro_parameters=self.state['one_euro_parameters']))+'\n')

    def disarm(self,reason=None):
        with self.lock:
            self.armed=False;self.motion.cancel();self.history.clear();self.controller.reset()
            if hasattr(self,'worker_active'):self.worker_active.clear()
            try:self.release_fire()
            except OSError as error:self.state['input_error']=str(error)
            if reason:self.state['status']=reason

    def update_weapon(self,packet,now):
        if self.weapons is None:
            return
        if self.weapons.update(packet):
            self.release_fire(reason='weapon_changed')
            self.motion.cancel()
            self.controller.reset()
        profile=self.weapons.profile(now)
        self.controller.weapon_aim_point=profile.aim_point if profile else None
        self.state['spray_weapon']=profile.name if profile else None
        self.state['weapon_pause_reason']=self.weapons.reason(now)

    def release_fire(self, *, automatic_only=False, reason='disabled', preserve_mode=False):
        spray = getattr(self, 'spray', None)
        started = spray.started if spray else None
        try:
            self.mouse.release()
        finally:
            if spray:
                if spray.started is not None:
                    self.state['spray_stop_reason'] = reason
                spray.stop(time.perf_counter())
                self.state['spray_active'] = False
                if started is not None:
                    ended=time.perf_counter()
                    self.motion.cancel()
                    self.last_snap_at=ended
                    self.controller.last_shot=max(self.controller.last_shot,ended)
                    if hasattr(self.controller,'zone'):
                        self.controller.zone=None
                    for name in ('head_filter','body_filter'):
                        if hasattr(self.controller,name):
                            getattr(self.controller,name).clear()
                    if hasattr(self.controller,'course'):
                        self.controller.course=None
                if not (preserve_mode or automatic_only):
                    spray.enabled = False
                    spray.awaiting_reload = False
                    if hasattr(self.controller,'restart_series'):
                        self.controller.restart_series()
                if automatic_only:
                    spray.wall_mode = False
                if spray.crouched:
                    # A real Ctrl press transfers ownership to the user. Never
                    # send key-up while the physical key remains down.
                    physical = getattr(self, 'physical_keys', None)
                    if physical is None or not physical.left_ctrl_down():
                        set_key(0xA2, False)
                    spray.crouched = False
            if started is not None and hasattr(self,'action_log'):
                self.action_log.write(json.dumps(dict(event='spray_stop',reason=reason,
                    spray_started_at=started,input_submitted_at=time.perf_counter(),time=time.time()))+'\n')

    def start_spray(self, manual):
        if not manual or not self.spray.enabled:
            return False
        weapons = getattr(self, 'weapons', None)
        if weapons is None or weapons.reason(time.perf_counter(),spray=True):
            return False
        if self.settings.stop_before_shot:
            held = [code for code in (0x57,0x41,0x53,0x44) if key_down(code)]
            if not self.stop_before_shot.ready(time.perf_counter(),held,release_movement_keys):
                self.state['shot_wait_reason']='movement_settle'
                return False
        rect = client_rect(self.hwnd)
        if rect is None:
            return False
        self.mouse.act(0, 0, True, hold=True)
        now = time.perf_counter()
        self.spray.start(now, manual, getattr(self.controller, 'acquired_at', None))
        self.spray.rect = rect
        self.spray.last_ammo = weapons.sample['weapon_ammo']
        physical = getattr(self,'physical_keys',None)
        self.spray.ctrl_generation = physical.ctrl_generation if physical else None
        self.spray.user_stance_override = physical.ctrl_down() if physical else True
        self.controller.shot_emitted(now)
        self.state['shots'] += 1
        self.state['spray_active'] = True
        self.motion.cancel()
        if hasattr(self,'action_log'):
            self.action_log.write(json.dumps(dict(event='spray_start',manual=True,
                input_submitted_at=now,time=time.time(),target=self.spray.target,
                source=(self.controller.zone or {}).get('source'),
                input_timing=self.mouse.last_call))+'\n')
        return True

    def update_spray(self, now):
        spray = self.spray
        pressed = key_down(self.spray_button)
        toggle = pressed and not spray.manual_pressed
        spray.manual_pressed = pressed
        if toggle and spray.enabled:
            self.release_fire(reason='manual_toggle')
            return True
        if toggle and self.fire:
            spray.enabled = True
            spray.wall_mode = self.controller.zone_geometry(now) is None
            spray.require_body = False
        if not spray.enabled:
            return False
        if not self.fire:
            self.release_fire()
            return True

        weapons = getattr(self, 'weapons', None)
        reason = weapons.reason(now,spray=True) if weapons else 'weapon_unknown_or_unsupported'
        sample = weapons.sample if weapons else {}
        ammo = sample.get('weapon_ammo')
        if sample.get('weapon_reloading') is True or ammo == 0:
            spray.reload_seen = True
            spray.awaiting_reload = True
            spray.require_body = True
        if (spray.awaiting_reload and type(ammo) is int and ammo > 0
                and (spray.reload_seen or (spray.last_ammo is not None and ammo > spray.last_ammo))
                and reason is None):
            spray.awaiting_reload = False
            spray.reload_seen = False
        if reason:
            self.release_fire(reason=reason,preserve_mode=True)
            self.state['shot_wait_reason'] = reason
            return True
        if type(ammo) is int:
            spray.last_ammo = ammo
        geometry = self.controller.zone_geometry(now)
        if spray.started is not None and spray.wall_mode and geometry is not None:
            spray.wall_mode = False
            spray.target = self.controller.acquired_at
        if spray.started is None:
            if now < spray.ready_at or spray.awaiting_reload:
                return True
            if (not spray.wall_mode or getattr(spray,'require_body',False)) and geometry is None:
                return True
            self.start_spray(manual=True)
            return True
        elapsed = now - spray.started
        if not spray.wall_mode and (self.controller.acquired_at != spray.target or geometry is None):
            self.release_fire(reason='target_not_fresh',preserve_mode=True)
            return True
        if elapsed >= spray.max_duration:
            spray.awaiting_reload = True
            spray.require_body = True
            self.release_fire(reason='profile_complete',preserve_mode=True)
            return True

        rect = client_rect(self.hwnd)
        if rect != spray.rect:
            self.release_fire(reason='geometry_changed')
            return True
        physical = getattr(self, 'physical_keys', None)
        user_crouched = physical.ctrl_down() if physical else key_down(0x11)
        if physical and physical.ctrl_generation != spray.ctrl_generation:
            spray.ctrl_generation = physical.ctrl_generation
            spray.user_stance_override = True
            # Physical down/up has taken ownership of the actual key state.
            if spray.crouched and not physical.left_ctrl_down():
                set_key(0xA2,False)
            spray.crouched = False
        if (elapsed >= spray.crouch_at and not spray.crouched and not user_crouched
                and physical is not None and not spray.user_stance_override):
            set_key(0xA2, True)
            spray.crouched = True
        dx, dy = spray.correction(elapsed,user_crouched or spray.crouched)
        if geometry is not None:
            fx, fy = spray.follow(elapsed,geometry['center'],self.controller.zone['velocity'],
                                  geometry['size'],self.settings)
            dx += fx
            dy += fy
        if dx or dy:
            self.mouse.act(dx, dy)
            size = (rect[2]-rect[0], rect[3]-rect[1])
            sent = time.perf_counter()
            self.controller.acknowledge(dx, dy, *size)
            self.history.record(sent, dx, dy, size, self.settings)
            self.state['moves'] += 1
            if hasattr(self,'action_log'):
                self.action_log.write(json.dumps(dict(event='spray_move',dx=dx,dy=dy,
                    source=geometry['source'] if geometry else None,
                    input_submitted_at=sent,time=time.time(),input_timing=self.mouse.last_call))+'\n')
        self.motion.cancel()
        return True

    def send_pending(self,log,frame):
        with self.lock:
            if self.stop.is_set() or not self.armed or not foreground(self.hwnd):
                spray = getattr(self, 'spray', None)
                if spray and (spray.enabled or spray.started is not None or spray.crouched):
                    self.release_fire()
                self.motion.cancel();return
            now=time.perf_counter()
            spray = getattr(self, 'spray', None)
            try:
                if spray and self.update_spray(now):
                    return
            except OSError as error:
                self.state['input_error']=str(error)
                self.disarm('Ошибка ввода; управление отключено')
                return
            context=getattr(self,'zone_context',None)
            zone = getattr(self.controller, 'zone', None)
            weapons = getattr(self,'weapons',None)
            weapon_reason = weapons.reason(now) if weapons else ('weapon_unknown_or_unsupported' if spray else None)
            spray_ready = weapon_reason is None
            self.state['shot_wait_reason'] = weapon_reason or ''
            if context and self.fire and spray_ready and hasattr(self.controller,'zone_fire_ready') and self.controller.zone_fire_ready(now):
                self.motion.cancel()
                event=dict(dx=0,dy=0,shoot=True,recorded=True,glide=False,done=True,
                           size=context['size'],rect=context['rect'],source=self.controller.zone['source'],
                           deadline=self.controller.zone['source']+self.settings.max_age,fire_policy='zone')
            else:event=self.motion.poll(now)
            if event is None:return
            if client_rect(self.hwnd)!=event['rect'] or time.perf_counter()>event.get('deadline',event['source']+self.settings.max_age):
                self.motion.cancel();return
            try:
                if event.get('recorded') and event['shoot']:
                    if (weapon_reason or (weapons and weapons.reason(time.perf_counter()))
                            or not self.fire or not self.controller.can_emit_shot(event['source'],time.perf_counter())):
                        self.motion.cancel();return
                if event['shoot'] and self.settings.stop_before_shot:
                    held=[code for code in (0x57,0x41,0x53,0x44) if key_down(code)]
                    if not self.stop_before_shot.ready(time.perf_counter(),held,release_movement_keys):
                        self.state['shot_wait_reason']='movement_settle'
                        self.motion.cancel();return
                    self.state['shot_wait_reason']=''
                start=time.perf_counter()
                self.mouse.act(event['dx'],event['dy'],event['shoot'])
                sent=time.perf_counter()
                api_ms=getattr(self.mouse,'last_call',{}).get('api_ms',0)
                self.state['max_input_api_ms']=max(self.state.get('max_input_api_ms',0),api_ms)
                if api_ms>30:self.state['input_stalls']=self.state.get('input_stalls',0)+1
                if event['dx'] or event['dy']:
                    self.last_motion_at=sent;self.state['moves']+=1
                    if event['glide']:self.state['glide_steps']=self.state.get('glide_steps',0)+1
                    self.controller.acknowledge(event['dx'],event['dy'],*event['size'])
                    if hasattr(self.controller,'movement_sent'):self.controller.movement_sent(sent,event['dx'],event['dy'])
                    self.history.record(sent,event['dx'],event['dy'],event['size'],self.settings)
                    if not event['glide']:self.last_snap_at=sent
                if event['shoot']:
                    self.state['shots']+=1;self.controller.last_shot=sent
                    if hasattr(self.controller,'shot_emitted'):self.controller.shot_emitted(sent)
                    if spray:self.controller.tap_emitted()
                log.write(json.dumps(dict(time=time.time(),frame=frame,capture_to_input_ms=(sent-event['source'])*1000,
                                         acquired_at=getattr(self.controller,'acquired_at',None),
                                         input_submitted_at=sent,decision_at=start,
                                         control_ms=(sent-start)*1000,input_timing=getattr(self.mouse,'last_call',{}),**event))+'\n')
            except OSError as error:
                self.state['input_error']=str(error);self.disarm('Windows отклонила ввод. Наблюдение продолжается.')
                (self.output/'input-error.txt').write_text(traceback.format_exc(),encoding='utf-8')

    def inputs(self):
        previous={0x77:False,0x78:False,0x75:False,0x76:False};seen=None;last_accepted=0;last_flush=0
        visual_reference=None
        camera_lock=CameraLock()
        loop_times=deque(maxlen=256);last_loop=None;last_metrics=0;visual_sequence=0
        log=open_log(self.output/'actions.jsonl',self.async_logs)
        self.action_log=log
        decisions=open_log(self.output/'control.jsonl',self.async_logs)
        input_timings=open_log(self.output/'input-timing.jsonl',self.async_logs)
        try:
            if self.spray:
                from mvp_windows import PhysicalKeys
                try:
                    self.physical_keys=PhysicalKeys()
                    self.state['physical_ctrl_observer']=True
                except OSError:
                    self.state['physical_ctrl_observer']=False
            while not self.stop.is_set():
                now=time.perf_counter()
                if self.armed and last_loop is not None:loop_times.append((now-last_loop)*1000)
                last_loop=now if self.armed else None
                if loop_times and now-last_metrics>.5:
                    samples=sorted(loop_times)
                    self.state['input_loop_p95_ms']=samples[min(len(samples)-1,int(len(samples)*.95))]
                    self.state['input_loop_hz']=1000/(sum(samples)/len(samples));last_metrics=now
                    if self.async_logs:
                        self.state['input_log_health']={name:writer.stats() for name,writer in
                                                       (('actions',log),('control',decisions),('input',input_timings))}
                try:
                    while True:
                        message=self.messages.get_nowait();self.state.update({k:v for k,v in message.items() if k!='suspend_input'});self.ready=message.get('ready',self.ready)
                        if 'hwnd' in message:self.hwnd=message['hwnd']
                        if message.get('suspend_input'):self.disarm(message['status'])
                        if 'error' in message:self.disarm();self.stop.set()
                except Empty:pass
                if not self.worker.is_alive() and not self.worker_stop.is_set():
                    self.ready=False;self.disarm('Процесс вычислений остановлен; смотри error.txt');self.stop.set();break
                # Rising edges cover physical holds; the transition bit also
                # catches a very short press between polling intervals.
                for key in previous:
                    value=USER.GetAsyncKeyState(key);down=bool(value&0x8000)
                    edge=bool(value&1) or (down and not previous[key]);previous[key]=down
                    if not edge:continue
                    if key==0x75:self.disarm('Остановлено F6');self.stop.set();break
                    if key==0x78:self.fire=not self.fire;self.motion.cancel();self.release_fire()
                    if key==0x76:
                        with self.lock:
                            self.motion.cancel();self.history.clear();self.controller.reset();self.release_fire()
                            self.last_snap_at=time.perf_counter();self.state['status']='Цель отпущена вручную. Ожидание нового кадра.'
                    if key==0x77:
                        if self.armed:self.disarm('Выключено. F8 — включить')
                        elif self.state['input_error']:self.state['status']='Ошибка ввода; перезапусти панель после устранения причины'
                        elif self.pipeline!='full':self.state['status']='Диагностика: управление отключено'
                        elif self.ready and foreground(self.hwnd):
                            self.armed=True;self.controller.reset();self.state['status']='ACTIVE'
                            self.last_snap_at=time.perf_counter()
                            if hasattr(self,'worker_active'):self.worker_active.set()
                        else:self.state['status']='Дождись Готово, активируй целевое окно и нажми F8'
                if self.armed and not foreground(self.hwnd):self.disarm('Потерян фокус. Для возобновления F8')
                packet=None
                try:
                    while True:packet=self.packets.get_nowait()
                except Empty:pass
                if packet is not None and packet['id']!=seen:
                    seen=packet['id'];self.hwnd=packet['hwnd']
                    now=time.perf_counter();age=now-packet['captured_at']
                    self.state.update({k:v for k,v in packet.items() if k not in ('detections','hwnd','id','size','rect','captured_at','camera_steps')})
                    self.state['latency_ms']=age*1000
                    self.state['queue_age_ms']=(now-packet['packet_ready_at'])*1000 if 'packet_ready_at' in packet else None
                    self.update_weapon(packet,now)
                    sequence=packet.get('visual_kill_sequence',visual_sequence)
                    if sequence!=visual_sequence:
                        visual_sequence=sequence;event_at=packet.get('visual_kill_at')
                        accepted=bool(self.armed and foreground(self.hwnd) and visual_kill_matches(self.controller,event_at,now))
                        decisions.write(json.dumps(dict(time=time.time(),frame=seen,reason='visual_feed_diagnostic',
                            sequence=sequence,event_at=event_at,accepted=accepted,armed=self.armed,
                            acquired_at=getattr(self.controller,'acquired_at',None),last_shot=self.controller.last_shot,
                            held=self.controller.locked is not None))+'\n')
                        if accepted:
                            with self.lock:
                                if self.controller.release_after_confirmed_kill(now):
                                    self.motion.cancel();self.release_fire(automatic_only=True,reason='visual_kill')
                                    # Keep history: this very image can already
                                    # contain the next enemy, rebased by in-flight input.
                                    self.last_snap_at=max(self.last_snap_at,event_at-1e-6)
                                    self.state['confirmed_kill_releases']+=1
                                    decisions.write(json.dumps(dict(time=time.time(),frame=seen,reason='visual_kill_confirmed',
                                        heads=[],selected=None,dx=0,dy=0,shoot=False))+'\n')
                    if not self.armed:self.state['status']='Пауза. F8 — включить'
                    geometry_valid=foreground(self.hwnd) and packet['rect']==client_rect(self.hwnd)
                    valid=(self.armed and geometry_valid and packet['captured_at']>self.last_snap_at
                           and 0<=age<=self.settings.max_age)
                    if valid:
                        aim_point = 'head' if self.spray else self.settings.aim_point
                        detections=[aim_detection(Detection(d['cls'],d['score'],tuple(d['box']),tuple(d['head']),d['head_score']),aim_point) for d in packet['detections']]
                        with self.lock:
                            start=time.perf_counter()
                            if not self.stop.is_set() and self.armed:
                                if self.spray:
                                    detections = [self.controller.tracking_detection(d, for_rebase=True)
                                                  for d in detections]
                                raw_detections=detections
                                visual_registered=False
                                camera_reason=camera_lock.update_controller(self.controller,self.history,
                                    packet.get('camera_steps',[]),packet['captured_at'],packet['size'])
                                if camera_reason:
                                    # Leaving the field of view is a camera event, not a kill.
                                    self.motion.cancel();self.release_fire(automatic_only=True,reason=camera_reason);visual_reference=None
                                current_detections=self.history.rebase(detections,packet['captured_at'],packet['size'])
                                candidates = current_detections
                                if self.spray:
                                    candidates = [self.controller.tracking_detection(d) for d in candidates]
                                normal_match=self.controller.select([(0,d) for d in candidates if d.score>=self.settings.confidence and d.head_score>=self.settings.head_confidence],*packet['size'],packet['captured_at'])
                                if visual_reference is not None and self.controller.locked is not None and normal_match is None:
                                    source,index,old=visual_reference
                                    for track in packet.get('visual_tracks',[]):
                                        if track['source']==source and track['index']==index:
                                            sx,sy=track['shift']
                                            projected=Detection(old.cls,old.score,(old.box[0]+sx,old.box[1]+sy,old.box[2]+sx,old.box[3]+sy),(old.head[0]+sx,old.head[1]+sy),old.head_score)
                                            rebased=self.history.rebase([projected],packet['captured_at'],packet['size'])
                                            if rebased:
                                                prior=self.controller.projected_locked
                                                if hasattr(self.controller,'head_filter'):
                                                    self.controller.head_filter.clear()
                                                    self.controller.course=None
                                                    self.controller.verified_samples=0
                                                self.controller.projected_locked=rebased[0]
                                                visual_registered=True
                                            break
                                detections=current_detections
                                if visual_registered and hasattr(self.controller,'anchors'):
                                    self.controller.anchors=[]
                                weapon_ok = self.weapons is None or self.weapons.reason(start) is None
                                action=self.controller.step(detections,*packet['size'],start,packet['captured_at'],enabled=True,foreground=True,fire=self.fire and weapon_ok)
                                self.zone_context=dict(size=packet['size'],rect=packet['rect']) if action.get('fire_policy')=='zone' and action.get('target') else None
                                action['aim_point']=action.get('aim_point',aim_point)
                                action['weapon_pause_reason']=self.weapons.reason(start) if self.weapons else None
                                action['visual_registered']=visual_registered
                                action['camera_release_reason']=camera_reason
                                visual_reference=None
                                if action['target'] is not None:
                                    from mvp_core import box_iou
                                    candidates=[]
                                    for index,raw in enumerate(raw_detections):
                                        rebased=self.history.rebase([raw],packet['captured_at'],packet['size'])
                                        if rebased:candidates.append((box_iou(rebased[0].box,action['target'].box),index,raw))
                                    if candidates:
                                        overlap,index,raw=max(candidates,key=lambda item:item[0])
                                        if overlap>.5:
                                            visual_reference=(packet['captured_at'],index,raw)
                                            camera_lock.anchor(raw,packet['captured_at'])
                                decisions.write(json.dumps(dict(time=time.time(),frame=seen,age_ms=age*1000,
                                    captured_at=packet['captured_at'],decision_at=start,
                                    packet_ready_at=packet.get('packet_ready_at'),
                                    heads=[dict(cls=d.cls,box=d.box,head=d.head,confidence=d.head_score) for d in detections],
                                    selected=asdict(action['target']) if action['target'] else None,
                                    **{k:v for k,v in action.items() if k!='target'}))+'\n')
                                self.motion.begin(action,packet['size'],packet['rect'],packet['captured_at'],start,self.settings.max_age)
                                self.state['status']='Цель потеряна: удержание сохранено. F7 — отпустить.' if action['reason']=='target_temporarily_lost' else action['reason'];last_accepted=start
                    elif self.armed:
                        if not geometry_valid:self.motion.cancel()
                        self.state['status']='Ожидание свежего кадра'
                self.send_pending(log,seen)
                self.mouse.tick()
                while self.mouse.timings:input_timings.write(json.dumps(self.mouse.timings.popleft())+'\n')
                if self.armed and now-last_accepted>.2 and (not self.spray or self.spray.started is None):
                    self.release_fire(reason='frame_stale',preserve_mode=True)
                if now-last_flush>.25:log.flush();decisions.flush();input_timings.flush();last_flush=now
                # Event.wait(1ms) measured ~15.5ms here: it collapsed a complete
                # glide into one or two jumps. CPython 3.11 time.sleep uses a
                # high-resolution waitable timer on Windows. No global timer or
                # priority settings; stop is checked again after this short wait.
                if self.armed or self.motion.pending:time.sleep(.001)
                else:self.stop.wait(.01)
        except Exception:
            self.state['input_error']=traceback.format_exc();self.stop.set()
            (self.output/'input-error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finally:
            self.disarm();self.ready=False;self.worker_stop.set();log.close();decisions.close()
            if self.physical_keys:
                self.physical_keys.close()
                self.physical_keys=None
            while self.mouse.timings:input_timings.write(json.dumps(self.mouse.timings.popleft())+'\n')
            input_timings.close()
            if self.async_logs:
                self.state['input_log_health']={name:writer.stats() for name,writer in
                                               (('actions',log),('control',decisions),('input',input_timings))}

    def close(self):
        self.stop.set();self.worker_stop.set();self.disarm()
        self.thread.join(timeout=2);self.worker.join(timeout=3)
        self.metrics_thread.join(timeout=2)
        if self.worker.is_alive():self.worker.terminate();self.worker.join(timeout=2)
        self.ready=False
        (self.output/'summary.json').write_text(json.dumps(self.snapshot(),ensure_ascii=False,indent=2),encoding='utf-8')
        for q in (self.packets,self.messages):q.close();q.cancel_join_thread()
        if self.physical_keys:
            self.physical_keys.close()
            self.physical_keys=None

class App:
    def __init__(self,args):
        import tkinter as tk
        from tkinter import ttk
        self.ak_spray = getattr(args, 'ak_spray', False)
        self.args=args;self.engine=None;self.root=tk.Tk();self.root.title(TITLE);self.root.geometry('690x465' if self.ak_spray else '690x425')
        frame=ttk.Frame(self.root,padding=18);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text=f'MVP v{VERSION}',font=('Arial',14,'bold')).pack(anchor='w')
        self.mode=tk.StringVar(value=args.mode);self.sens=tk.StringVar(value=str(args.sensitivity));self.size=tk.StringVar(value=str(args.imgsz));self.hz=tk.StringVar(value=str(args.hz));self.enemy=tk.StringVar(value='T');self.pipeline=tk.StringVar(value=args.pipeline);self.backend=tk.StringVar(value=args.model_backend)
        self.aim_point=tk.StringVar(value='Голова')
        self.head_filter=tk.StringVar(value='1€')
        self.memory_guard=tk.BooleanVar(value=False)
        self.stop_before_shot=tk.BooleanVar(value=True)
        self.smooth=tk.BooleanVar(value=True)
        self.blocking_wait=tk.BooleanVar(value=args.blocking_gpu_wait)
        self.capture_backend=tk.StringVar(value=args.capture_backend)
        self.death_source=tk.StringVar(value=args.death_source)
        self.spray_button=tk.StringVar(value='Mouse 4')
        form=ttk.Frame(frame);form.pack(fill='x',pady=10)
        fields = [
            ('Фильтр координат', self.head_filter, ['Текущий', '1€']),
            ('Чувствительность CS2', self.sens, None),
            ('Размер входа модели (при запуске)', self.size, ['960', '1280']),
            ('Противник', self.enemy, ['T', 'CT']),
        ]
        if self.ak_spray:
            ttk.Label(frame,text='Одиночные автоматически; Mouse 4 — включить / выключить ручную очередь').pack(anchor='w')
        else:
            fields.insert(1, ('Точка наведения', self.aim_point, ['Голова', 'Корпус']))
        for row, (label, variable, values) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky='w', pady=4)
            widget = (ttk.Combobox(form, textvariable=variable, values=values, state='readonly', width=22)
                      if values else ttk.Entry(form, textvariable=variable, width=25))
            widget.grid(row=row, column=1, sticky='w', padx=12)
            if variable is self.head_filter:
                widget.bind('<<ComboboxSelected>>', self.change_head_filter)
        ttk.Checkbutton(frame, text='Остановка WASD перед выстрелом',
                        variable=self.stop_before_shot).pack(anchor='w')
        buttons=ttk.Frame(frame);buttons.pack(fill='x',pady=12)
        ttk.Button(buttons,text='Запустить мониторинг',command=self.start).pack(side='left')
        ttk.Button(buttons,text='Стоп',command=self.stop).pack(side='left',padx=10)
        ttk.Label(frame,text='F8 — включить/выключить коррекцию и стрельбу\nF9 — только коррекция / коррекция + выстрел; F7 — отпустить цель\nF6 — остановить; Alt+Tab отключает управление').pack(anchor='w')
        self.status=tk.StringVar(value='Готов к запуску');ttk.Label(frame,textvariable=self.status,wraplength=650).pack(anchor='w',pady=12)
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        if args.autostart:self.root.after(300,self.start)
        if args.duration:self.root.after(int(args.duration*1000),self.close)
        self.tick();self.root.mainloop()
    def change_head_filter(self, event=None):
        if self.engine:
            try:self.engine.set_head_filter('one_euro' if self.head_filter.get()=='1€' else 'current')
            except (ValueError,OSError) as error:self.status.set(str(error))

    def start(self):
        self.stop()
        try:
            sens=float(self.sens.get().replace(',','.'))
            if not .01<=sens<=20:raise ValueError('Sensitivity: 0.01–20')
            interval=.25
            spray_button = 5 if self.ak_spray else None
            self.engine=Engine(self.mode.get(),Settings(enemy=0 if self.enemy.get()=='T' else 1,sensitivity=sens,shot_interval=interval,fire_policy="zone",memory_guard=self.memory_guard.get(),stop_before_shot=self.stop_before_shot.get(),head_filter="one_euro" if self.head_filter.get()=="1€" else "current",aim_point="body" if self.aim_point.get()=="Корпус" else "head"),int(self.size.get()),int(self.hz.get()),self.pipeline.get(),self.backend.get(),self.smooth.get(),self.capture_backend.get(),self.blocking_wait.get(),self.death_source.get(),aim_profile=self.args.aim_profile,spray_button=spray_button,crouch_recoil_scale=getattr(self.args,'crouch_recoil_scale',.75),async_logs=not getattr(self.args,'sync_logs',False))
        except Exception as error:self.status.set(str(error))
    def stop(self):
        if self.engine:self.engine.close();self.engine=None
        self.status.set('Остановлено')
    def close(self):
        self.stop()
        self.root.destroy()
    def tick(self):
        if self.engine:
            if (self.engine.output/'close.request').exists():self.close();return
            if (self.engine.output/'stop.request').exists():self.stop()
            if self.engine:
                s=self.engine.snapshot()
                mode = ('Наведение и стрельба' if s['fire'] else 'Только наведение') if s['armed'] else 'Ожидание'
                if s.get('spray_active'):
                    mode = 'Спрей AK-47'
                status = s['input_error'] or s['status']
                if status in ('predictive_zone_aim', 'inside_fire_zone'):
                    status = 'Сопровождение цели'
                if s['armed'] and s['fire'] and not s['input_error']:
                    pause=s.get('shot_wait_reason') or s.get('weapon_pause_reason')
                    labels={'weapon_stale':'Огонь на паузе: нет свежего HUD',
                            'weapon_unknown_or_unsupported':'Огонь на паузе: оружие не определено или не поддержано',
                            'reload_unknown_or_active':'Огонь на паузе: перезарядка или её состояние неизвестно',
                            'ammo_unknown':'Огонь на паузе: боезапас не определён',
                            'magazine_empty':'Огонь на паузе: магазин пуст',
                            'zoom_unverified':'Огонь на паузе: режим прицела не проверен'}
                    status=labels.get(pause,status)
                    if s.get('weapon_hud_status') in ('hud_calibration_missing','hud_calibration_incomplete'):
                        status='Огонь на паузе: нужна калибровка HUD'
                self.status.set(f"{mode} | {status}\nДетектор {s['fps']:.1f} кадр/с | задержка {s['latency_ms']:.0f} мс" if s['armed'] else status)
                if time.perf_counter()-getattr(self,'last_status_write',0)>=.25:
                    (self.engine.output/'status.json').write_text(json.dumps(s,ensure_ascii=False),encoding='utf-8')
                    self.last_status_write=time.perf_counter()
        self.root.after(50,self.tick)

if __name__=='__main__':
    mp.freeze_support()
    KERNEL.CreateMutexW.argtypes=[C.c_void_p,W.BOOL,W.LPCWSTR];KERNEL.CreateMutexW.restype=W.HANDLE
    C.set_last_error(0);instance_mutex=KERNEL.CreateMutexW(None,False,'Local\\CS2MLMVPPanelV2')
    if not instance_mutex:raise C.WinError(C.get_last_error())
    if C.get_last_error()==183:
        USER.FindWindowW.argtypes=[W.LPCWSTR,W.LPCWSTR];USER.FindWindowW.restype=W.HWND
        USER.ShowWindow.argtypes=[W.HWND,C.c_int];USER.SetForegroundWindow.argtypes=[W.HWND]
        hwnd=USER.FindWindowW(None,TITLE)
        if hwnd:USER.ShowWindow(hwnd,9);USER.SetForegroundWindow(hwnd)
        raise SystemExit(0)
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['simulator','local-bots'],default='local-bots');p.add_argument('--sensitivity',type=float,default=.97);p.add_argument('--imgsz',type=int,default=960);p.add_argument('--hz',type=int,default=30);p.add_argument('--autostart',action='store_true');p.add_argument('--pipeline',choices=['full','capture','inference'],default='full');p.add_argument('--model-backend',choices=['auto','native','tensorrt','pytorch'],default='auto');p.add_argument('--capture-backend',choices=['dxgi','wgc'],default='dxgi');p.add_argument('--duration',type=float,default=0)
    p.add_argument('--blocking-gpu-wait',action='store_true')
    p.add_argument('--death-source',choices=['visual','manual'],default='visual')
    p.add_argument('--aim-profile',help='JSON timing profile produced by fit_aim_profile.py')
    p.add_argument('--ak-spray',action='store_true',help='Visual weapon gating and manual Mouse 4 spray in local bots')
    p.add_argument('--crouch-recoil-scale',type=float,default=.75,help='Experimental crouch multiplier; 1 disables the hypothesis')
    p.add_argument('--sync-logs',action='store_true',help='Revert asynchronous diagnostics for controlled A/B measurement')
    App(p.parse_args())
