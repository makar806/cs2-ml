"""Render raw frames for model-blind annotation (no predictions or game metadata)."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--roi', type=int, nargs=4, default=[0,0,1280,720])
    parser.add_argument('--width', type=int, default=400)
    parser.add_argument('--columns', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    x,y,r,s=args.roi
    cells=[]
    for index in range(args.start,args.stop+1,args.step):
        capture.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=capture.read()
        if not ok:
            raise ValueError(index)
        frame=cv2.resize(frame[y:s,x:r],(args.width,round(args.width*(s-y)/(r-x))),interpolation=cv2.INTER_NEAREST)
        frame=cv2.copyMakeBorder(frame,25,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(frame,f'{index} [{x},{y},{r},{s}]',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
        cells.append(frame)
    capture.release()
    while len(cells)%args.columns:
        cells.append(np.zeros_like(cells[0]))
    sheet=np.concatenate([np.concatenate(cells[i:i+args.columns],axis=1) for i in range(0,len(cells),args.columns)])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(args.output),sheet)


if __name__=='__main__':
    main()
