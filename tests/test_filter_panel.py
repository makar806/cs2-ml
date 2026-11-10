"""Real Tk widget wiring with hidden window and mocked engine; no capture/input."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
tk = pytest.importorskip('tkinter', reason='Tk is required for the real GUI check')
from tkinter import ttk
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import live_mvp as live


def test_panel_selector_applies_without_restart_and_default_is_one_euro(monkeypatch):
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('A display is required for the real GUI check')
    root.withdraw()
    monkeypatch.setattr(tk,'Tk',lambda:root)
    monkeypatch.setattr(root,'mainloop',lambda:None)
    class FakeEngine:
        def __init__(self,*args,**kwargs):
            self.settings=args[1];self.changes=[];self.closed=False
        def set_head_filter(self,mode):self.changes.append(mode)
        def close(self):self.closed=True
    monkeypatch.setattr(live,'Engine',FakeEngine)
    args=SimpleNamespace(mode='simulator',sensitivity=.97,imgsz=960,hz=30,pipeline='full',model_backend='native',aim_profile='profile.json',blocking_gpu_wait=False,capture_backend='dxgi',death_source='manual',autostart=False,duration=0)
    try:
        app=live.App(args)
        assert app.head_filter.get()=='1€'
        app.start();engine=app.engine
        assert engine.settings.head_filter=='one_euro'
        assert engine.settings.fire_policy=='zone'
        assert engine.settings.shot_interval==.25
        assert not engine.settings.memory_guard
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        selector=next(w for w in descendants(root) if isinstance(w,ttk.Combobox) and str(w.cget('textvariable'))==str(app.head_filter))
        selector.set('Текущий');selector.event_generate('<<ComboboxSelected>>')
        assert app.engine is engine and not engine.closed
        assert engine.changes==['current']
        selector.set('1€');selector.event_generate('<<ComboboxSelected>>')
        assert engine.changes==['current','one_euro']
        app.stop()
    finally:root.destroy()
