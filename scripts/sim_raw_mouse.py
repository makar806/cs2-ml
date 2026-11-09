"""Receive relative mouse counts in the simulator's own window only.

No global hooks, drivers or system mouse setting changes. Desktop cursor pixels
include Windows pointer acceleration and cannot stand in for game mouse counts.
"""
import ctypes as C
from ctypes import wintypes as W

class RawMouse:
    def __init__(self,hwnd,on_move):
        self.user=C.WinDLL('user32',use_last_error=True);self.hwnd=hwnd
        class Header(C.Structure):
            _fields_=[('kind',W.DWORD),('size',W.DWORD),('device',W.HANDLE),('param',W.WPARAM)]
        class Mouse(C.Structure):
            _fields_=[('flags',W.USHORT),('buttons',W.ULONG),('raw_buttons',W.ULONG),('x',W.LONG),('y',W.LONG),('extra',W.ULONG)]
        class Input(C.Structure):_fields_=[('header',Header),('mouse',Mouse)]
        class Device(C.Structure):_fields_=[('page',W.USHORT),('usage',W.USHORT),('flags',W.DWORD),('target',W.HWND)]
        self.user.RegisterRawInputDevices.argtypes=[C.POINTER(Device),W.UINT,W.UINT]
        self.user.GetRawInputData.argtypes=[W.HANDLE,W.UINT,C.c_void_p,C.POINTER(W.UINT),W.UINT]
        self.user.GetRawInputData.restype=W.UINT
        self.user.SetWindowLongPtrW.argtypes=[W.HWND,C.c_int,C.c_void_p];self.user.SetWindowLongPtrW.restype=C.c_void_p
        self.user.CallWindowProcW.argtypes=[C.c_void_p,W.HWND,W.UINT,W.WPARAM,W.LPARAM];self.user.CallWindowProcW.restype=C.c_ssize_t
        proc=C.WINFUNCTYPE(C.c_ssize_t,W.HWND,W.UINT,W.WPARAM,W.LPARAM)
        self.events=0
        @proc
        def receive(window,message,param,data):
            if message==0xff: # WM_INPUT, delivered only to this registered window
                packet=Input();size=W.UINT(C.sizeof(packet))
                n=self.user.GetRawInputData(W.HANDLE(data),0x10000003,C.byref(packet),C.byref(size),C.sizeof(Header))
                if n!=0xffffffff and packet.header.kind==0 and not packet.mouse.flags&1:
                    if packet.mouse.x or packet.mouse.y:
                        self.events+=1;on_move(packet.mouse.x,packet.mouse.y)
            return self.user.CallWindowProcW(self.original,window,message,param,data)
        self.callback=receive
        self.original=self.user.SetWindowLongPtrW(hwnd,-4,C.cast(receive,C.c_void_p))
        if not self.original:raise C.WinError(C.get_last_error())
        self.Device=Device
        device=Device(1,2,0,hwnd) # generic desktop mouse, foreground only
        if not self.user.RegisterRawInputDevices(C.byref(device),1,C.sizeof(Device)):raise C.WinError(C.get_last_error())

    def close(self):
        device=self.Device(1,2,1,None) # RIDEV_REMOVE
        self.user.RegisterRawInputDevices(C.byref(device),1,C.sizeof(self.Device))
        self.user.SetWindowLongPtrW(self.hwnd,-4,self.original)
