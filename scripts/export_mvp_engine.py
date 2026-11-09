"""Rebuild the local FP16 engine; benchmark approval is a separate step."""
import os
import argparse
os.environ['YOLO_AUTOINSTALL']='false'
from pathlib import Path
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT/'runs/pose-v7/weights/best.pt')
    args = parser.parse_args()
    YOLO(str(args.weights.resolve())).export(
        format='engine',imgsz=(736,1280),quantize=16,dynamic=True,
        batch=1,workspace=1,device=0,simplify=True)
