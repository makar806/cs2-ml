"""Windows capture and ordinary mouse input; no process memory access."""
import ctypes as C
import os
from ctypes import wintypes as W
from pathlib import Path
import threading
import time
from collections import deque

class _WindowsOnly:
    """Importable on offline hosts; no operating-system call can silently succeed."""
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        def unavailable(*args, **kwargs):
            raise OSError(f'{self.name}.{name} requires Windows; use injected offline input')
        setattr(self, name, unavailable)
        return unavailable


USER = C.WinDLL('user32', use_last_error=True) if os.name == 'nt' else _WindowsOnly('user32')
KERNEL = C.WinDLL('kernel32', use_last_error=True) if os.name == 'nt' else _WindowsOnly('kernel32')
try:USER.SetProcessDpiAwarenessContext(C.c_void_p(-4))
except Exception:pass

USER.GetForegroundWindow.restype=W.HWND
USER.GetWindowTextW.argtypes=[W.HWND,W.LPWSTR,C.c_int]
USER.GetAncestor.argtypes=[W.HWND,W.UINT];USER.GetAncestor.restype=W.HWND
USER.GetWindowThreadProcessId.argtypes=[W.HWND,C.POINTER(W.DWORD)]
USER.GetClientRect.argtypes=[W.HWND,C.POINTER(W.RECT)]
USER.ClientToScreen.argtypes=[W.HWND,C.POINTER(W.POINT)]
USER.IsIconic.argtypes=[W.HWND];USER.IsWindowVisible.argtypes=[W.HWND]
USER.IsWindow.argtypes=[W.HWND];USER.IsWindow.restype=W.BOOL
USER.GetAsyncKeyState.argtypes=[C.c_int];USER.GetAsyncKeyState.restype=C.c_short
KERNEL.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];KERNEL.OpenProcess.restype=W.HANDLE
KERNEL.QueryFullProcessImageNameW.argtypes=[W.HANDLE,W.DWORD,W.LPWSTR,C.POINTER(W.DWORD)]
KERNEL.CloseHandle.argtypes=[W.HANDLE]

def input_environment():
    """Read service state only. Never stop, disable or reconfigure a driver."""
    api=C.WinDLL('advapi32',use_last_error=True)
    api.OpenSCManagerW.argtypes=[W.LPCWSTR,W.LPCWSTR,W.DWORD];api.OpenSCManagerW.restype=W.HANDLE
    api.OpenServiceW.argtypes=[W.HANDLE,W.LPCWSTR,W.DWORD];api.OpenServiceW.restype=W.HANDLE
    api.QueryServiceStatus.argtypes=[W.HANDLE,C.c_void_p];api.QueryServiceStatus.restype=W.BOOL
    api.CloseServiceHandle.argtypes=[W.HANDLE]
    class SERVICE_STATUS(C.Structure):
        _fields_=[(name,W.DWORD) for name in ('type','state','controls','win32_exit','service_exit','checkpoint','wait_hint')]
    states={}
    manager=api.OpenSCManagerW(None,None,1)
    if not manager:return {'drivers':states,'blocked':False,'query_error':C.get_last_error()}
    try:
        for name in ('FACEIT','FACEIT_IOMMU'):
            service=api.OpenServiceW(manager,name,4)
            if not service:continue
            try:
                status=SERVICE_STATUS()
                if api.QueryServiceStatus(service,C.byref(status)):states[name]=int(status.state)
            finally:api.CloseServiceHandle(service)
    finally:api.CloseServiceHandle(manager)
    return {'drivers':states,'blocked':False,'note':'Driver presence is diagnostic metadata, not proof that ordinary input is blocked. Actual SendInput failures disarm the controller.'}

class MOUSEINPUT(C.Structure):
    _fields_=[('dx',W.LONG),('dy',W.LONG),('mouseData',W.DWORD),('dwFlags',W.DWORD),('time',W.DWORD),('dwExtraInfo',C.c_size_t)]
class KEYBDINPUT(C.Structure):
    _fields_=[('wVk',W.WORD),('wScan',W.WORD),('dwFlags',W.DWORD),('time',W.DWORD),('dwExtraInfo',C.c_size_t)]
class HARDWAREINPUT(C.Structure):
    _fields_=[('uMsg',W.DWORD),('wParamL',W.WORD),('wParamH',W.WORD)]
class INPUTUNION(C.Union):
    _fields_=[('mi',MOUSEINPUT),('ki',KEYBDINPUT),('hi',HARDWAREINPUT)]
class INPUT(C.Structure):
    _anonymous_=('u',)
    _fields_=[('type',W.DWORD),('u',INPUTUNION)]
USER.SendInput.argtypes=[W.UINT,C.POINTER(INPUT),C.c_int];USER.SendInput.restype=W.UINT

def key_down(code):return bool(USER.GetAsyncKeyState(code)&0x8000)


class PhysicalKeys:
    """Observe physical Ctrl edges, excluding our own SendInput events.

    This is an ordinary Windows low-level keyboard hook in this process, not
    a hook/injection into the game. Its message loop never performs control.
    """
    def __init__(self):
        if os.name != 'nt':
            raise OSError('Physical keyboard observation requires Windows')
        self.keys = {key for key in (0xA2, 0xA3) if key_down(key)}
        self.ctrl_generation = 0
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.error = None
        self.thread_id = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(2) or self.error:
            self.close()
            raise OSError(self.error or 'Physical keyboard observer did not start')

    def ctrl_down(self):
        return bool(self.keys)

    def left_ctrl_down(self):
        return 0xA2 in self.keys

    def _run(self):
        hook = None
        try:
            class KeyboardEvent(C.Structure):
                _fields_ = [('vkCode', W.DWORD), ('scanCode', W.DWORD),
                            ('flags', W.DWORD), ('time', W.DWORD),
                            ('extra', C.c_size_t)]
            callback_type = C.WINFUNCTYPE(C.c_ssize_t, C.c_int, W.WPARAM, W.LPARAM)
            USER.SetWindowsHookExW.argtypes = [C.c_int, callback_type, W.HINSTANCE, W.DWORD]
            USER.SetWindowsHookExW.restype = W.HANDLE
            USER.CallNextHookEx.argtypes = [W.HANDLE, C.c_int, W.WPARAM, W.LPARAM]
            USER.CallNextHookEx.restype = C.c_ssize_t
            USER.UnhookWindowsHookEx.argtypes = [W.HANDLE]
            USER.GetMessageW.argtypes = [C.POINTER(W.MSG), W.HWND, W.UINT, W.UINT]
            USER.PeekMessageW.argtypes = [C.POINTER(W.MSG), W.HWND, W.UINT, W.UINT, W.UINT]
            USER.PostThreadMessageW.argtypes = [W.DWORD, W.UINT, W.WPARAM, W.LPARAM]
            KERNEL.GetCurrentThreadId.restype = W.DWORD
            KERNEL.GetModuleHandleW.argtypes = [W.LPCWSTR]
            KERNEL.GetModuleHandleW.restype = W.HMODULE
            self.thread_id = KERNEL.GetCurrentThreadId()

            @callback_type
            def callback(code, message, data):
                if code >= 0:
                    event = C.cast(data, C.POINTER(KeyboardEvent)).contents
                    if not event.flags & 0x10 and event.vkCode in (0x11, 0xA2, 0xA3):
                        key = 0xA3 if event.flags & 1 else 0xA2
                        if message in (0x100, 0x104) and key not in self.keys:
                            self.keys.add(key)
                            self.ctrl_generation += 1
                        elif message in (0x101, 0x105) and key in self.keys:
                            self.keys.discard(key)
                            self.ctrl_generation += 1
                return USER.CallNextHookEx(hook, code, message, data)

            hook = USER.SetWindowsHookExW(13, callback, KERNEL.GetModuleHandleW(None), 0)
            if not hook:
                raise C.WinError(C.get_last_error())
            message = W.MSG()
            # Create the thread message queue before close() can post WM_QUIT.
            USER.PeekMessageW(C.byref(message), None, 0, 0, 0)
            self.ready.set()
            while not self.stopping.is_set() and USER.GetMessageW(C.byref(message), None, 0, 0) > 0:
                pass
        except Exception as error:
            self.error = str(error)
        finally:
            if hook:
                USER.UnhookWindowsHookEx(hook)
            self.ready.set()

    def close(self):
        self.stopping.set()
        if self.thread_id is not None:
            USER.PostThreadMessageW(self.thread_id, 0x12, 0, 0)
        self.thread.join(timeout=2)


def set_key(code, down):
    event = INPUT(type=1, ki=KEYBDINPUT(code, 0, 0 if down else 2, 0, 0))
    if USER.SendInput(1, C.byref(event), C.sizeof(INPUT)) != 1:
        raise OSError(f'Windows rejected ordinary keyboard input, GetLastError={C.get_last_error()}')

def process_name(hwnd):
    pid=W.DWORD();USER.GetWindowThreadProcessId(hwnd,C.byref(pid))
    handle=KERNEL.OpenProcess(0x1000,False,pid.value)
    if not handle:return ''
    try:
        size=W.DWORD(32768);buffer=C.create_unicode_buffer(size.value)
        return Path(buffer.value).name.lower() if KERNEL.QueryFullProcessImageNameW(handle,0,buffer,C.byref(size)) else ''
    finally:KERNEL.CloseHandle(handle)

def find_window(mode,include_minimized=False):
    found=[]
    callback=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
    @callback
    def enum(hwnd,unused):
        if not USER.IsWindowVisible(hwnd) or (not include_minimized and USER.IsIconic(hwnd)):return True
        buffer=C.create_unicode_buffer(512);USER.GetWindowTextW(hwnd,buffer,512)
        name=buffer.value
        match=(mode=='simulator' and name=='CS2 ML MVP Simulator') or (mode=='local-bots' and process_name(hwnd)=='cs2.exe')
        if match:
            rect=W.RECT();USER.GetClientRect(hwnd,C.byref(rect))
            if rect.right>=320 and rect.bottom>=200:found.append(hwnd)
        return True
    USER.EnumWindows.argtypes=[callback,W.LPARAM]
    USER.EnumWindows(enum,0)
    return found[0] if len(found)==1 else None

def client_rect(hwnd):
    rect=W.RECT();point=W.POINT(0,0)
    if not USER.GetClientRect(hwnd,C.byref(rect)) or not USER.ClientToScreen(hwnd,C.byref(point)):return None
    return point.x,point.y,point.x+rect.right,point.y+rect.bottom

def foreground(hwnd):
    return bool(hwnd and USER.GetForegroundWindow()==hwnd and not USER.IsIconic(hwnd))

def mouse_event(dx=0,dy=0,flags=1):
    event=INPUT(type=0,mi=MOUSEINPUT(dx,dy,0,flags,0,0))
    C.set_last_error(0)
    cpu=time.thread_time();started=time.perf_counter()
    sent=USER.SendInput(1,C.byref(event),C.sizeof(INPUT))
    ended=time.perf_counter();cpu_ms=(time.thread_time()-cpu)*1000
    if sent!=1:
        code=C.get_last_error()
        raise OSError(f'Windows rejected mouse input: SendInput={sent}/1, GetLastError={code}, INPUT={C.sizeof(INPUT)} bytes. Input disabled for this session; see diagnostics.json. This return value alone does not identify the blocking component.')
    return dict(api_ms=(ended-started)*1000,cpu_ms=cpu_ms,api_started=started,api_ended=ended)

class Mouse:
    def __init__(self):
        self.down=False;self.release_at=0;self.lock=threading.RLock()
        self.timings=deque(maxlen=512);self.last_call={}
    def act(self,dx,dy,shoot=False,*,hold=False):
        requested=time.perf_counter()
        with self.lock:self._act(dx,dy,shoot,(time.perf_counter()-requested)*1000,hold)
    def _act(self,dx,dy,shoot=False,lock_ms=0,hold=False):
        """Serialize the correction and button-down in one ordinary SendInput batch."""
        building=time.perf_counter();events=[]
        if dx or dy:events.append(INPUT(type=0,mi=MOUSEINPUT(dx,dy,0,1,0,0)))
        press=shoot and not self.down
        if press:events.append(INPUT(type=0,mi=MOUSEINPUT(0,0,0,2,0,0)))
        if not events:return
        batch=(INPUT*len(events))(*events)
        C.set_last_error(0);cpu=time.thread_time();started=time.perf_counter()
        sent=USER.SendInput(len(events),batch,C.sizeof(INPUT))
        ended=time.perf_counter()
        self.last_call=dict(time=time.time(),kind='move+down' if press and (dx or dy) else 'down' if press else 'move',
                            api_started=started,api_ended=ended,
                            lock_ms=lock_ms,build_ms=(started-building)*1000,api_ms=(ended-started)*1000,
                            cpu_ms=(time.thread_time()-cpu)*1000,requested=len(events),sent=int(sent))
        self.timings.append(self.last_call)
        if press and sent==len(events):self.down=True;self.release_at=float('inf') if hold else time.perf_counter()+.025
        if sent!=len(events):
            if press and sent>0:self.down=True
            raise OSError(f'Windows rejected ordinary input: SendInput={sent}/{len(events)}, GetLastError={C.get_last_error()}')
    def move(self,dx,dy):
        if dx or dy:mouse_event(dx,dy)
    def shoot(self):
        if not self.down:
            mouse_event(flags=2);self.down=True;self.release_at=time.perf_counter()+.025
    def release(self):
        requested=time.perf_counter()
        with self.lock:
            if self.down:
                acquired=time.perf_counter()
                try:
                    result=mouse_event(flags=4)
                    self.timings.append(dict(time=time.time(),kind='up',lock_ms=(acquired-requested)*1000,
                        total_ms=(time.perf_counter()-acquired)*1000,**(result or {})))
                finally:self.down=False
    def tick(self):
        if self.down and time.perf_counter()>=self.release_at:self.release()

class Capture:
    """Bounded latest-frame slot. DXGI first, MSS fallback for unsupported desktops."""
    def __init__(self,fps=30,backend='auto'):
        self.fps=fps;self.backend=backend;self.region=None;self.latest=None
        self.lock=threading.Lock();self.stop=threading.Event();self.error='';self.name='starting';self.counter=0;self.last_ms=0
        self.slow_grabs=deque(maxlen=32)
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def set_region(self,region):
        with self.lock:
            if self.region!=region:self.latest=None
            self.region=region
    def get(self):
        with self.lock:return self.latest
    def run(self):
        import numpy as np
        camera=None;sct=None
        try:
            if self.backend!='mss':
                try:
                    import dxcam
                    camera=dxcam.create(output_color='BGR');self.name='DXGI'
                except Exception as error:
                    if self.backend=='dxcam':raise
                    self.error='DXGI fallback: '+str(error)
            if camera is None:
                import mss
                sct=mss.mss();self.name='MSS'
            while not self.stop.is_set():
                started=time.perf_counter()
                with self.lock:region=self.region
                if region:
                    x1,y1,x2,y2=region
                    if x2>x1 and y2>y1:
                        grab_started=time.perf_counter()
                        try:
                            if camera is not None:
                                # DXGI copies/converts only the game client. copy=True owns the
                                # array until inference completes; never reuse a mapped surface.
                                frame=camera.grab(region=region,copy=True)
                            else:frame=np.asarray(sct.grab({'left':x1,'top':y1,'width':x2-x1,'height':y2-y1}))[:,:,:3].copy()
                            if frame is not None:
                                self.last_ms=(time.perf_counter()-started)*1000
                                self.counter+=1
                                with self.lock:self.latest=(self.counter,started,frame,region)
                        except Exception as error:
                            self.error=str(error)
                            if camera is not None and self.backend=='auto':
                                camera.release();camera=None
                                import mss
                                sct=mss.mss();self.name='MSS'
                        finally:
                            grab_ended=time.perf_counter()
                            if grab_ended-grab_started >= .05:
                                self.slow_grabs.append((grab_started,grab_ended))
                # Event.wait coarsened a 33ms capture period to ~47ms here.
                # This worker does not poll input; bounded high-resolution sleep
                # keeps capture rate predictable without a busy loop.
                time.sleep(max(.001,1/self.fps-(time.perf_counter()-started)))
        except Exception as error:self.error=str(error)
        finally:
            if camera is not None:camera.release()
            if sct is not None:sct.close()
    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise TimeoutError('Capture is still stopping; refusing concurrent recreation')


def release_movement_keys(codes):
    events=(INPUT*len(codes))()
    for i,code in enumerate(codes):
        events[i].type=1
        events[i].ki=KEYBDINPUT(code,0,2,0,0)
    if codes and USER.SendInput(len(codes),events,C.sizeof(INPUT))!=len(codes):
        raise OSError(C.get_last_error(),'Movement key release failed')
