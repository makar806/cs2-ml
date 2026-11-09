import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Image, image directory, or video file')
    parser.add_argument('--weights', default=str(ROOT / 'models/yolo11n.pt'))
    args = parser.parse_args()
    model = YOLO(args.weights)
    for result in model.predict(source=args.source, device=0, save=True, stream=True,
                                project=str(ROOT / 'runs'), name='predictions'):
        print(f'{result.path}: {len(result.boxes)} detections; {result.speed}')
