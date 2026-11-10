"""Compare both models on the same preserved holdout and explicit PyTorch device."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from evaluate_pose import iou

root = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=root/'data/pose-v10')
    parser.add_argument('--output',type=Path,default=root/'runs/pose-v10/comparison-v9.json')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--imgsz',type=int,default=960)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    data=args.data
    source=[]
    for split in ('newcheck','check','test'):
        images=sorted((data/'images'/split).glob('*'))
        if not images:
            raise ValueError(f'Missing holdout split: {split}')
        for image in images:
            label=data/'labels'/split/(image.stem+'.txt')
            source.append(dict(split=split,image=image.name,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
    started=time.perf_counter()
    report = {'confidence': .35, 'iou': .5, 'imgsz': args.imgsz, 'backend': 'PyTorch', 'device': args.device, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'latency_not_comparable_to_target_laptop': True, 'models': {}}
    for version in ('v9', 'v10'):
        model = YOLO(str(root / f'runs/pose-{version}/weights/best.pt'))
        if model.names != {0: 'T', 1: 'CT'}:
            raise ValueError(f'Unexpected label mapping: {model.names}')
        scores = report['models'][version] = {}
        for split in ('newcheck', 'check', 'test'):
            gt_count = pred_count = matches = 0
            head_errors = []
            correct_heads = 0
            images = sorted((data / 'images' / split).iterdir())
            results = model.predict([str(p) for p in images], stream=True, batch=8,
                                    conf=.35, imgsz=args.imgsz, device=args.device, verbose=False)
            for path, result in zip(images, results, strict=True):
                assert Path(result.path).name == path.name
                labels = (data / 'labels' / split / f'{path.stem}.txt').read_text().splitlines()
                h, w = result.orig_shape
                targets = []
                for row in labels:
                    cls, x, y, bw, bh, hx, hy, vis = map(float, row.split())
                    box = np.array([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
                    targets.append((int(cls), box, np.array([hx*w, hy*h]), vis))
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                heads = result.keypoints.xy.cpu().numpy()
                confidence = result.keypoints.conf.cpu().numpy()
                candidates = sorted(((iou(b, t[1]), pi, gi)
                                     for pi, b in enumerate(boxes) for gi, t in enumerate(targets)
                                     if classes[pi] == t[0]), reverse=True)
                used_p, used_g = set(), set()
                for overlap, pi, gi in candidates:
                    if overlap < .5 or pi in used_p or gi in used_g:
                        continue
                    used_p.add(pi)
                    used_g.add(gi)
                    if targets[gi][3] > 0 and confidence[pi, 0] >= .55:
                        correct_heads += 1
                        head_errors.append(float(np.linalg.norm(heads[pi, 0] - targets[gi][2])))
                gt_count += len(targets)
                pred_count += len(boxes)
                matches += len(used_g)
            scores[split] = dict(images=len(images), gt=gt_count, predictions=pred_count,
                                 matches=matches, false_positives=pred_count-matches,
                                 heads_above_threshold=correct_heads,
                                 median_head_error_px=float(np.median(head_errors)) if head_errors else None)
        print(version, scores, flush=True)
    report['elapsed_seconds']=time.perf_counter()-started
    report['host']=dict(system=platform.system(),machine=platform.machine(),threads=args.threads)
    report['inputs']=source
    report['weights_sha256']={v:hashlib.sha256((root/f'runs/pose-{v}/weights/best.pt').read_bytes()).hexdigest() for v in ('v9','v10')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Saved {args.output}')

if __name__=='__main__':
    main()
