import argparse
from pathlib import Path

from dataset_config import resolve_dataset
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "configs/dataset.yaml"))
    parser.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="training")
    args = parser.parse_args()
    YOLO(args.weights, task="pose").train(
        data=str(resolve_dataset(args.data)),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=0,
        seed=42,
        deterministic=True,
        amp=True,
        patience=20,
        cache=False,
        project=str(ROOT / "runs"),
        name=args.name,
        degrees=0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0,
        plots=True,
        optimizer="AdamW",
        lr0=0.001,
    )


if __name__ == "__main__":
    main()
