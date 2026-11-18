"""Render ground-truth first fragments with raw scores; no predicted box is presented as GT."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from protocol import ROOT
from replay import read_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media',type=Path,required=True)
    parser.add_argument('--audit',type=Path,default=ROOT/'evidence/next/raw-firsts-unfiltered.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    annotations=json.loads((ROOT/'data/next/benchmark.json').read_text())
    audit=json.loads(args.audit.read_text())
    sources={r['case']:r for r in json.loads((ROOT/'data/next/sources.json').read_text())}
    cells=[]
    for index,row in enumerate(annotations['appearances']):
        image=next(read_frames(args.media/sources[row['case']]['file'],[row['first']]))[2]
        x,y,r,s=row['box']
        cx,cy=(x+r)//2,(y+s)//2
        left,top=max(0,min(cx-90,1100)),max(0,min(cy-70,580))
        cv2.rectangle(image,(x,y),(r,s),(70,240,70),1)
        crop=cv2.resize(image[top:top+140,left:left+180],(450,350),interpolation=cv2.INTER_NEAREST)
        cell=np.zeros((440,480,3),np.uint8)
        cell[75:425,15:465]=crop
        scores=[audit['models'][name]['cases'][index]['best_presence'] for name in ['v18','v21']]
        for line,text in enumerate([f"case {row['case']}, frame {row['first']} | green: visible GT",
                                     f'Best raw presence: V18 {scores[0]:.6f} / V21 {scores[1]:.6f}',
                                     'Raw scores before threshold / NMS']):
            cv2.putText(cell,text,(10,20+22*line),cv2.FONT_HERSHEY_SIMPLEX,.41,(235,235,235),1,cv2.LINE_AA)
        cells.append(cell)
    sheet=np.concatenate([np.concatenate(cells[i:i+2],axis=1) for i in range(0,len(cells),2)])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(args.output),sheet):
        raise IOError(args.output)


if __name__=='__main__':
    main()
