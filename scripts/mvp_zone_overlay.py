"""Optional ordinary transparent, noninteractive window for a local demo."""
import tkinter as tk
from mvp_windows import USER,W

class ZoneOverlay:
    def __init__(self, parent):
        self.window=tk.Toplevel(parent);self.window.withdraw();self.window.overrideredirect(True)
        self.window.attributes('-topmost',True);self.window.attributes('-transparentcolor','#010101')
        self.canvas=tk.Canvas(self.window,bg='#010101',highlightthickness=0);self.canvas.pack(fill='both',expand=True)
        self.circle=self.canvas.create_oval(0,0,1,1,outline='#20ff50',width=1)
        self.window.update_idletasks()
        hwnd=self.window.winfo_id();hwnd=USER.GetAncestor(hwnd,2) or hwnd
        USER.GetWindowLongW.argtypes=[W.HWND, W.INT];USER.GetWindowLongW.restype=W.LONG
        USER.SetWindowLongW.argtypes=[W.HWND,W.INT,W.LONG];USER.SetWindowLongW.restype=W.LONG
        style=USER.GetWindowLongW(hwnd,-20)
        USER.SetWindowLongW(hwnd,-20,style|0x20|0x80000|0x80|0x08000000)
        self.last_rect=None
        self.visible=False
    def update(self,zone,rect):
        x,y,x2,y2=rect
        if tuple(rect)!=self.last_rect:
            self.window.geometry(f'{x2-x}x{y2-y}{x:+d}{y:+d}');self.last_rect=tuple(rect)
        cx,cy=zone['center'];r=zone['radius']
        self.canvas.coords(self.circle,cx-r,cy-r,cx+r,cy+r)
        if not self.visible:
            self.window.deiconify();self.visible=True
    def hide(self):
        if self.visible:self.window.withdraw();self.visible=False
    def close(self):self.window.destroy()
