import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / 'configs/dataset.yaml').read_text())
    config['path'] = str(ROOT / 'data/dataset')
    for split in ('train', 'val'):
        folder = Path(config['path']) / config[split]
        if not folder.is_dir() or not any(folder.iterdir()):
            raise SystemExit(f'Prepare labeled dataset first: {folder}')
    resolved = ROOT / 'runs/dataset-resolved.yaml'
    resolved.parent.mkdir(exist_ok=True)
    resolved.write_text(yaml.safe_dump(config), encoding='utf-8')
    model = YOLO(str(ROOT / 'models/yolo11n.pt'))
    model.train(data=str(resolved), epochs=args.epochs, batch=args.batch,
                imgsz=640, device=0, workers=0, seed=42,
                project=str(ROOT / 'runs'), name='detector')
