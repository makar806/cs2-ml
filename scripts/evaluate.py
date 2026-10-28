import argparse
import json
from pathlib import Path

from dataset_config import resolve_dataset
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    parser.add_argument("--data", default=str(ROOT / "configs/dataset.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    metrics = YOLO(args.weights, task="pose").val(
        data=str(resolve_dataset(args.data)),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        workers=0,
        batch=8,
        project=str(ROOT / "runs"),
        name="evaluation",
        plots=True,
    )
    print(json.dumps(metrics.results_dict, indent=2))


if __name__ == "__main__":
    main()
