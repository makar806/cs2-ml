"""Window Graphics Capture, bounded latest owned client frame, cursor disabled."""
import ctypes as C
from ctypes import wintypes as W
import threading
import time
import numpy as np
from mvp_windows import USER,client_rect,foreground

class GeometryPending(RuntimeError):pass

class WindowCapture:
    def __init__(self,hwnd,hz):
        from windows_capture import WindowsCapture
        self.hwnd=hwnd;self.latest=None;self.error='';self.counter=0
        self.lock=threading.Lock();self.last_ms=0;self.name='WGC';self.geometry_since=None;self.crop_cache=None
        self.capture=WindowsCapture(window_hwnd=hwnd,cursor_capture=False,draw_border=False,
                                    minimum_update_interval=max(1,round(1000/hz)))
        @self.capture.event
        def on_frame_arrived(frame,control):
            entered=time.perf_counter()
            try:
                region=client_rect(hwnd)
                if region is None:return
                image=self._client(frame.frame_buffer,region)
                self.geometry_since=None
                # Native timespan is WGC SystemRelativeTime in 100 ns units.
                source=frame.timespan/10_000_000
                # WGC can report a scheduled compositor timestamp a few ms
                # ahead of callback QPC. Bound that skew and never use a future
                # timestamp to undercount the age of a decision.
                if not -.025<=entered-source<5:
                    raise RuntimeError(f'WGC timestamp/QPC mismatch: raw={frame.timespan}, converted={source}, now={entered}')
                source=min(source,entered)
                self.last_ms=(time.perf_counter()-entered)*1000
                self.counter+=1
                with self.lock:self.latest=(self.counter,source,image,region)
            except GeometryPending as error:
                with self.lock:self.latest=None
                if self.geometry_since is None:self.geometry_since=entered
                if entered-self.geometry_since>2:self.error=str(error);control.stop()
            except Exception as error:
                self.error=str(error);control.stop()
        @self.capture.event
        def on_closed():self.error=self.error or 'Целевое окно закрыто'
        self.control=self.capture.start_free_threaded()

    def _client(self,pixels,region):
        h,w=pixels.shape[:2];x,y,x2,y2=region;cw,ch=x2-x,y2-y
        if (w,h)==(cw,ch):return np.ascontiguousarray(pixels[:,:,:3])
        key=(w,h,region)
        if self.crop_cache is not None and self.crop_cache[0]==key:
            ox,oy=self.crop_cache[1]
            return np.ascontiguousarray(pixels[oy:oy+ch,ox:ox+cw,:3])
        bounds=W.RECT()
        dwm=C.WinDLL('dwmapi');dwm.DwmGetWindowAttribute.argtypes=[W.HWND,W.DWORD,C.c_void_p,W.DWORD]
        candidates=[]
        if dwm.DwmGetWindowAttribute(self.hwnd,9,C.byref(bounds),C.sizeof(bounds))==0:
            candidates.append((bounds.left,bounds.top,bounds.right,bounds.bottom))
        USER.GetWindowRect.argtypes=[W.HWND,C.POINTER(W.RECT)]
        if USER.GetWindowRect(self.hwnd,C.byref(bounds)):
            candidates.append((bounds.left,bounds.top,bounds.right,bounds.bottom))
        for left,top,right,bottom in candidates:
            ox,oy=x-left,y-top
            if (right-left,bottom-top)==(w,h) and 0<=ox and 0<=oy and ox+cw<=w and oy+ch<=h:
                self.crop_cache=(key,(ox,oy))
                return np.ascontiguousarray(pixels[oy:oy+ch,ox:ox+cw,:3])
        raise GeometryPending(f'WGC/client geometry mismatch: frame={w}x{h}, client={cw}x{ch}. Use windowed mode.')

    def get(self):
        with self.lock:return self.latest

    def close(self):
        self.control.stop()
