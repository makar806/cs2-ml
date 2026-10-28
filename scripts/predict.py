import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Image, directory, or video")
    parser.add_argument("--weights", default=str(ROOT / "models/best.pt"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0", help="GPU index or cpu")
    args = parser.parse_args()
    model = YOLO(args.weights, task="pose")
    for result in model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        conf=0.35,
        save=True,
        stream=True,
        project=str(ROOT / "runs"),
        name="predictions",
    ):
        print(f"{result.path}: {len(result.boxes)} detections; {result.speed}")


if __name__ == "__main__":
    main()
