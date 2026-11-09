"""Local closed-loop fixture using an existing labeled screenshot as a visual scene."""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from mvp_core import Controller,Settings,decode_result

ROOT=Path(__file__).resolve().parents[1]

class Scene:
    def __init__(self,sensitivity=1,width=960,height=540):
        candidates=[]
        for path in (ROOT/'data/pose-v1/labels/train').glob('*.txt'):
            rows=[list(map(float,line.split())) for line in path.read_text().splitlines()]
            if len(rows)==1 and rows[0][0]==0 and rows[0][-1]>0 and rows[0][4]>.22:
                candidates.append((rows[0][4],path,rows[0]))
        assert candidates,'No labeled T screenshot for the simulator'
        # A clear, front-facing arena fixture (not the smoke-heavy screenshot).
        chosen=next((c for c in candidates if c[1].stem.startswith('127_')),None)
        _,label,row=chosen or sorted(candidates,reverse=True)[0]
        path=next((ROOT/'data/pose-v1/images/train').glob(label.stem+'.*'))
        self.image=cv2.imread(str(path));ih,iw=self.image.shape[:2]
        self.head=(row[5]*iw,row[6]*ih);self.width=width;self.height=height;self.sensitivity=sensitivity
        self.head_radius=row[4]*ih*.07
        self.source=str(path);self.hits=0;self.shots=0;self.offset=np.zeros(2,dtype=float);self.reset()
    def reset(self):
        shift=[(130,-60),(-140,50),(90,80)][self.hits%3]
        self.offset=np.array(self.head)-np.array((self.width/2+shift[0],self.height/2+shift[1]))
    def render(self,show_hud=True):
        matrix=np.float32([[1,0,-self.offset[0]],[0,1,-self.offset[1]]])
        frame=cv2.warpAffine(self.image,matrix,(self.width,self.height))
        x,y=self.width//2,self.height//2
        cv2.line(frame,(x-7,y),(x+7,y),(40,255,40),1);cv2.line(frame,(x,y-7),(x,y+7),(40,255,40),1)
        if show_hud:
            cv2.putText(frame,f'LOCAL SIM | hits {self.hits} / shots {self.shots} | {time.perf_counter():.2f}',(10,self.height-15),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        return frame
    def move(self,dx,dy):
        focal=self.height/1.5
        self.offset+=np.tan(np.radians(np.array([dx,dy])*self.sensitivity*.022))*focal
    def shoot(self):
        self.shots+=1
        actual=np.array(self.head)-self.offset-np.array((self.width/2,self.height/2))
        error=float(np.linalg.norm(actual))
        if error<=self.head_radius:self.hits+=1;self.reset();return True,error
        return False,error

def headless_test(weights=None,baseline=False):
    from ultralytics import YOLO
    scene=Scene();model=YOLO(str(weights or ROOT/'runs/pose-v7/weights/best.pt'),task='pose')
    controller=Controller(Settings(sensitivity=1));trace=[]
    if baseline:
        import importlib.util,sys
        spec=importlib.util.spec_from_file_location('baseline_core',ROOT/'runs/mvp-optimization/baseline_core.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        controller=module.Controller(module.Settings(sensitivity=1))
    for step in range(180):
        frame=scene.render(False);captured=time.perf_counter()
        prediction=model.predict(frame,imgsz=1280,conf=.35,device=0,verbose=False)[0]
        action=controller.step(decode_result(prediction),scene.width,scene.height,time.perf_counter(),captured,enabled=True,foreground=True,fire=True)
        scene.move(action['dx'],action['dy'])
        if hasattr(controller,'acknowledge'):controller.acknowledge(action['dx'],action['dy'],scene.width,scene.height)
        hit=error=None
        if action['shoot']:hit,error=scene.shoot()
        trace.append({'step':step,'dx':action['dx'],'dy':action['dy'],'shoot':action['shoot'],'target':action['target'] is not None,'hit':hit,'actual_head_error':error})
        if scene.hits>=3:break
    out=ROOT/'runs/live-mvp-tests';out.mkdir(exist_ok=True)
    report={'source':scene.source,'steps':len(trace),'hits':scene.hits,'shots':scene.shots,'head_proxy_radius_px':scene.head_radius,'trace':trace,'note':'Actual V7 inference and controller on a panned training screenshot. Hit zone is a proxy circle around annotated head, radius 7% of body height; no head box exists in labels. Functional closed-loop test, not an accuracy, real headshot or game input benchmark.'}
    (out/('closed-loop-baseline.json' if baseline else 'closed-loop.json')).write_text(json.dumps(report,indent=2))
    cv2.imwrite(str(out/'last-frame.jpg'),scene.render())
    print(json.dumps({k:v for k,v in report.items() if k!='trace'}))
    # Immediate-fire mode trades the former multi-frame alignment gate for
    # responsiveness. Keep every miss in the report instead of treating this
    # training-image fixture as a perfect-accuracy requirement.
    assert scene.hits>=3,'Closed-loop acquisition/shooting failed'

def gui():
    # Native window interaction here is application behavior, not agent UI automation.
    from mvp_windows import USER,C,W
    import tkinter as tk
    from PIL import Image,ImageTk
    root=tk.Tk();root.title('CS2 ML MVP Simulator');root.resizable(False,False)
    scene=Scene();canvas=tk.Canvas(root,width=scene.width,height=scene.height,highlightthickness=0);canvas.pack()
    canvas.create_text(10,10,anchor='nw',text='loading')
    root.update_idletasks();hwnd=USER.GetAncestor(root.winfo_id(),2)
    USER.GetCursorPos.argtypes=[C.POINTER(W.POINT)];USER.SetCursorPos.argtypes=[C.c_int,C.c_int]
    USER.SetCursorPos.restype=W.BOOL
    locked=False;events=[];target_since=time.perf_counter()
    from sim_raw_mouse import RawMouse
    raw=RawMouse(hwnd,lambda dx,dy:scene.move(dx,dy) if locked and USER.GetForegroundWindow()==hwnd else None)
    log=ROOT/'runs/live-mvp-tests/native-simulator.json';log.parent.mkdir(exist_ok=True)
    def lock_toggle(event=None):
        nonlocal locked,target_since
        locked=not locked
        if locked:
            target_since=time.perf_counter()
            point=W.POINT(scene.width//2,scene.height//2);USER.ClientToScreen(hwnd,C.byref(point));USER.SetCursorPos(point.x,point.y)
        canvas.configure(cursor='none' if locked else '')
    def shoot(event=None):
        nonlocal target_since
        hit,error=scene.shoot()
        now=time.perf_counter()
        events.append({'hit':hit,'error_px':error,'acquire_ms':(now-target_since)*1000})
        if hit:target_since=now
        log.write_text(json.dumps({'hits':scene.hits,'shots':scene.shots,'last_hit':hit,'last_error':error,'time':time.time(),'events':events,'raw_mouse_events':raw.events},indent=2))
    def tick():
        nonlocal locked
        if USER.GetForegroundWindow()!=hwnd:locked=False
        if locked:
            center=W.POINT(scene.width//2,scene.height//2);USER.ClientToScreen(hwnd,C.byref(center))
            cursor=W.POINT();USER.GetCursorPos(C.byref(cursor))
            if (cursor.x,cursor.y)!=(center.x,center.y):USER.SetCursorPos(center.x,center.y)
        image=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(scene.render(),cv2.COLOR_BGR2RGB)))
        canvas.delete('all');canvas.create_image(0,0,anchor='nw',image=image);canvas.image=image
        root.after(16,tick)
    def close(event=None):raw.close();root.destroy()
    root.protocol('WM_DELETE_WINDOW',close)
    root.bind('<F8>',lock_toggle);root.bind('<ButtonPress-1>',shoot);root.bind('<Escape>',close)
    tick();root.mainloop()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--headless-test',action='store_true');parser.add_argument('--weights');parser.add_argument('--baseline',action='store_true');args=parser.parse_args()
    if args.headless_test:headless_test(args.weights,args.baseline)
    else:gui()
