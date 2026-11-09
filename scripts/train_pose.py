"""First controlled training run: body class and one head keypoint."""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--name', default='pose-baseline')
    parser.add_argument('--data', default=str(ROOT / 'configs/pose.yaml'))
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--weights',default=str(ROOT/'models/yolo11n-pose.pt'))
    parser.add_argument('--lr0',type=float,default=.001)
    parser.add_argument('--optimizer',default='auto')
    args = parser.parse_args()
    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs,
                batch=args.batch, imgsz=args.imgsz, device=0, workers=0, seed=42,
                deterministic=True, amp=True, patience=20, cache=False,
                project=str(ROOT / 'runs'), name=args.name,
                degrees=0, translate=.1, scale=.3, fliplr=.5, flipud=0,
                mosaic=.5, close_mosaic=10, mixup=0, plots=True,
                optimizer=args.optimizer,lr0=args.lr0)
