import sys,tkinter as tk
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mvp_zone_overlay import ZoneOverlay
from mvp_windows import USER

def test_overlay_is_clickthrough_and_cannot_take_focus(monkeypatch):
    root=tk.Tk();root.withdraw()
    try:
        overlay=ZoneOverlay(root)
        hwnd=overlay.window.winfo_id();hwnd=USER.GetAncestor(hwnd,2) or hwnd
        style=USER.GetWindowLongW(hwnd,-20)
        assert style & 0x20 and style & 0x08000000
        calls=[]
        monkeypatch.setattr(overlay.window,'deiconify',lambda:calls.append('show'))
        overlay.update(dict(center=(100,200),radius=5),(0,0,1280,720))
        assert overlay.canvas.coords(overlay.circle)==[95.,195.,105.,205.]
        overlay.update(dict(center=(110,200),radius=5),(0,0,1280,720))
        assert calls==['show']
        overlay.close()
    finally:root.destroy()
